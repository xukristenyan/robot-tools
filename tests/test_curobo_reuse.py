"""Scene reuse preserves state transitions and recovers from partial updates."""
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

from robot_tools.services.curobo import Cuboid, World
from robot_tools.services.curobo.contract import AllowedContact, Attachment, PutMeshRequest, StageContext

RUNTIME = Path(os.environ.get('CUROBO_TEST_RUNTIME',
    str(Path(__file__).resolve().parents[1]/'runtimes/curobo')))


class IndexArray(np.ndarray):
    def long(self):
        return self.astype(np.int64)


@pytest.fixture
def runtime(monkeypatch):
    # Exercise the real backend's state management without constructing GPU solvers.
    # Numerical IK, collision and full planning are covered by live GPU checks.
    names = {
        'torch': {'empty': lambda n,dtype=None,device=None:np.empty(n,dtype=dtype).view(IndexArray),
                  'cat': lambda rows:np.concatenate(rows).view(IndexArray), 'long':np.int64,
                  'tensor':lambda rows:np.asarray(rows).view(IndexArray)},
        'storage': {'configure_cache': lambda: None},
        'contact_collision': {'ContactCollisionChecker': object},
        'curobo.collision_checking': {'RobotCollisionChecker': object, 'RobotCollisionCheckerCfg': object},
        'curobo.config_io': {'load_yaml': lambda _: {}},
        'curobo.inverse_kinematics': {'InverseKinematics': object, 'InverseKinematicsCfg': object},
        'curobo.motion_planner': {'MotionPlanner': object, 'MotionPlannerCfg': object},
        'curobo.scene': {'Cuboid': object, 'Mesh': object, 'Scene': object},
        'curobo.types': {k: object for k in ('ContentPath','GoalToolPose','JointState','Pose')},
    }
    for name, values in names.items():
        module=ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules,name,module)
    spec=importlib.util.spec_from_file_location('reuse_test_engine',RUNTIME/'engine.py')
    engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
    monkeypatch.setitem(sys.modules,'engine',engine)
    spec=importlib.util.spec_from_file_location('reuse_test_backend',RUNTIME/'backend.py')
    backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)
    return engine,backend


class Geometry:
    def __init__(self):
        self.kinematics_config=self
        self.spheres=None
        self.updates=0
        self.resets=0

    def reset_link_spheres(self, name):
        self.spheres=None
        self.resets+=1

    def update_link_spheres(self, name, spheres):
        self.spheres=spheres.copy()
        self.updates+=1


def backend_fixture(runtime):
    module=runtime[1]
    backend=module.CuroboBackend.__new__(module.CuroboBackend)
    backend._prepared_scene=None;backend._attachment_states={};backend._active_context=None
    backend._batch_ik_solvers={};backend.attachment_frame='attached_object';backend.margin=.01
    backend._tensor=lambda a:np.array(a,dtype=np.float32,copy=True)
    resets=[];loaded=[];partitions=[]
    solver=NS(reset_cuda_graph=lambda:resets.append(True))
    backend.planner=NS(kinematics=Geometry(),ik_solver=solver,trajopt_solver=solver,
        graph_planner=solver,update_world=lambda scene:loaded.append(scene))
    backend.checker=NS(kinematics=Geometry())
    backend.contact_checker=NS(configure=lambda groups:partitions.append(groups) or False)
    backend._scene=lambda world,**kwargs:world.model_dump()
    return backend,loaded,partitions


def world(x=3.,name='object'):
    return World(obstacles=[Cuboid(name=name,pose=np.array([x,3,3,1,0,0,0.]),dims=(.1,.1,.1))])


def carried(radius=.02):
    return StageContext(attachment=Attachment(object_name='object',spheres=np.array([[0.,0.,.2,radius]])))


def test_identical_context_reuses_scene_and_attachment_but_detach_is_applied(runtime):
    b,loaded,_=backend_fixture(runtime)
    b.prepare(world(),carried(),True)
    first=b.planner.kinematics.spheres.copy()
    b.reset()
    assert b._active_context is None
    b.prepare(world(),carried(),True)
    assert len(loaded)==1 and b.planner.kinematics.updates==1
    np.testing.assert_allclose(first[:,3],[.03])
    b.prepare(world(),carried(.04),True)
    assert len(loaded)==1
    np.testing.assert_allclose(b.checker.kinematics.spheres[:,3],[.05])
    b.reset();b.prepare(world(),StageContext(),True)
    assert b.planner.kinematics.spheres is None and b.checker.kinematics.spheres is None


def test_new_batch_consumer_receives_current_attachment(runtime):
    b,_,_=backend_fixture(runtime)
    b.prepare(world(),carried(),True)
    added=Geometry();b._batch_ik_solvers[True]=NS(kinematics=added)
    b._attach(b._active_context)
    np.testing.assert_array_equal(added.spheres,b.planner.kinematics.spheres)
    assert b.planner.kinematics.updates==1 and added.updates==1


def test_pose_membership_contacts_and_mutated_input_invalidate_scene(runtime,monkeypatch):
    b,loaded,_=backend_fixture(runtime)
    import torch
    b.planner.kinematics.get_sphere_index_from_link_name=lambda name:torch.tensor([0])
    value=world();b.prepare(value,StageContext(),True)
    value.obstacles[0].pose[0]=4.
    b.prepare(value,StageContext(),True)
    contact=StageContext(allowed_contacts=[AllowedContact(obstacle='object',robot_links=['panda_link1'])])
    b.prepare(value,contact,True)
    b.prepare(value,StageContext(),True)
    b.prepare(world(name='other'),StageContext(),True)
    assert len(loaded)==5


def test_partial_scene_failure_forces_reload_of_previous_context(runtime):
    b,loaded,_=backend_fixture(runtime)
    b.prepare(world(),StageContext(),True)
    original=b.contact_checker.configure
    b.contact_checker.configure=lambda groups:(_ for _ in ()).throw(RuntimeError('partial update'))
    with pytest.raises(RuntimeError,match='partial update'):
        b.prepare(world(4.),StageContext(),True)
    b.reset();b.contact_checker.configure=original
    b.prepare(world(),StageContext(),True)
    assert len(loaded)==3


def test_raw_ik_context_does_not_discard_cached_world_or_keep_attachment(runtime):
    b,loaded,_=backend_fixture(runtime)
    b.prepare(world(),carried(),True)
    b.reset();b.prepare(World(),StageContext(),False)
    assert b.planner.kinematics.spheres is None
    b.reset();b.prepare(world(),carried(),True)
    assert len(loaded)==1
    np.testing.assert_allclose(b.checker.kinematics.spheres[:,3],[.03])


def test_failed_attachment_update_is_not_reused(runtime):
    b,_,_=backend_fixture(runtime)
    b.prepare(world(),carried(),True)
    original=b.planner.kinematics.update_link_spheres
    b.planner.kinematics.update_link_spheres=lambda *a:(_ for _ in ()).throw(RuntimeError('attachment'))
    with pytest.raises(RuntimeError,match='attachment'):
        b.prepare(world(),carried(.04),True)
    b.reset();b.planner.kinematics.update_link_spheres=original
    b.prepare(world(),carried(),True)
    np.testing.assert_allclose(b.planner.kinematics.spheres[:,3],[.03])


def test_disk_mesh_arrays_are_reused_bounded_and_missing_files_fail(runtime,tmp_path,monkeypatch):
    engine=runtime[0];store=engine.SceneStore(tmp_path,capacity=2)
    request=PutMeshRequest(vertices=np.array([[0.,0,0],[1.,0,0],[0.,1,0]],np.float32),
                           faces=np.array([[0,1,2]],np.int32))
    ids=[store.put_mesh(request).mesh_id for _ in range(3)]
    original=np.load;reads=[]
    def load(path,**kwargs):
        reads.append(path);return original(path,**kwargs)
    monkeypatch.setattr(np,'load',load)
    first=store.mesh(ids[0]);again=store.mesh(ids[0])
    assert first[0] is again[0] and len(reads)==1
    store.mesh(ids[1]);store.mesh(ids[2]);store.mesh(ids[0])
    assert len(reads)==4 and len(store.meshes)==2
    np.testing.assert_array_equal(store.mesh(ids[0])[0],request.vertices)
    (tmp_path/f'{ids[0]}.npz').unlink()
    with pytest.raises(engine.InputError,match='mesh_not_found'):store.mesh(ids[0])
