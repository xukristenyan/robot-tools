"""Headless adapter around pinned AnyPlace model and native diffusion routine."""

from __future__ import annotations

import os
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np

from robot_tools.services.anyplace.contract import PredictPlacementsRequest, PredictPlacementsResponse

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "upstream"
DEFAULT_WEIGHTS = Path("/data/xkyan3/robot-tools-models/anyplace")


def upstream_imports():
    # This sets only source lookup; no simulation, dataset, or robot is started.
    sys.path.insert(0, str(UPSTREAM)) if str(UPSTREAM) not in sys.path else None
    os.environ["ANYPLACE_SOURCE_DIR"] = str(UPSTREAM / "anyplace")
    from anyplace.model.transformer.policy import NSMTransformerSingleTransformationRegression
    from anyplace.utils import config_util, util
    from anyplace.utils.anyplace.multistep_pose_regression_anyplace import multistep_regression_scene

    return NSMTransformerSingleTransformationRegression, config_util, util, multistep_regression_scene


def resolve_checkpoint(checkpoint=None) -> Path:
    if checkpoint is not None:
        path = Path(checkpoint).expanduser()
    else:
        path = Path(os.environ.get("ANYPLACE_WEIGHTS_DIR", DEFAULT_WEIGHTS)) / "anyplace_multitask/model.pth"
    if not path.is_file():
        raise FileNotFoundError(f"AnyPlace checkpoint missing: {path}. Run robot-tools download.")
    return path


class AnyPlaceBackend:
    def __init__(self, checkpoint=None):
        import torch

        model_class, config_util, util, self.inference = upstream_imports()
        if not torch.cuda.is_available():
            raise RuntimeError("AnyPlace upstream inference requires CUDA")
        self.checkpoint_path = resolve_checkpoint(checkpoint)
        # Official checkpoints contain training config in addition to tensors.
        checkpoint_data = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        config_path = UPSTREAM / "anyplace/config/full_eval_cfgs/anyplace_inference/vial_insertion"
        config = config_util.load_config(
            str(config_path / "anyplace_diffusion_molmocrop_multitask.yaml"), demo_train_eval="eval"
        )
        config_util.update_recursive(config["model"]["refine_pose"], checkpoint_data["args"]["model"]["refine_pose"])
        refine = config["model"]["refine_pose"]
        if refine["type"] != "nsm_transformer":
            raise ValueError("Only the official AnyPlace diffusion checkpoints are supported")
        model_kwargs = dict(config["model"]["nsm_transformer"])
        model_kwargs.update(refine["model_kwargs"]["nsm_transformer"])
        self.model = model_class(feat_dim=refine["feat_dim"], mc_vis=None, **model_kwargs).cuda().eval()
        self.model.load_state_dict(checkpoint_data["refine_pose_model_state_dict"], strict=True)
        self.rot_grid = util.generate_healpix_grid(size=config["data"]["rot_grid_samples"])
        self.options = config["experiment"]["eval"]
        self._lock = threading.Lock()

    def predict_placements(self, request: PredictPlacementsRequest) -> PredictPlacementsResponse:
        import torch
        from geometry import point_cloud

        parent = point_cloud(request.place)
        child = point_cloud(request.pick)
        # The compatibility patch initializes planar regions at their mean;
        # a line or single point does not provide enough surface geometry.
        for cloud in (parent, child):
            if np.linalg.matrix_rank(cloud - cloud.mean(axis=0), tol=1e-8) < 2:
                raise ValueError("Masked depth must provide at least a two-dimensional surface")
        options = self.options
        start = time.perf_counter()
        # Upstream uses global NumPy/Python/Torch RNGs. Serialize requests and
        # restore host state so a seeded call does not contaminate other work.
        with self._lock, torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.inference_mode():
            numpy_state, python_state = np.random.get_state(), random.getstate()
            try:
                np.random.seed(request.seed)
                random.seed(request.seed)
                torch.random.default_generator.manual_seed(request.seed)
                torch.cuda.manual_seed(request.seed)
                relative = self.inference(
                    None,
                    parent,
                    child,
                    None,
                    self.model,
                    None,
                    scene_scale=1 / 1.2,
                    scene_mean=[0.35, 0, 0],
                    grid_pts=None,
                    rot_grid=self.rot_grid,
                    viz=False,
                    n_iters=request.n_refine_iters,
                    init_k_val=request.num_samples,
                    no_sc_score=True,
                    with_coll=False,
                    run_affordance=False,
                    return_top=True,
                    no_parent_crop=True,
                    init_parent_mean=options["init_parent_mean_pos"],
                    init_orig_ori=options["init_orig_ori"],
                    refine_anneal=options["refine_anneal"],
                    add_per_iter_noise=options["add_per_iter_noise"],
                    per_iter_noise_kwargs=options["per_iter_noise_kwargs"],
                    variable_size_crop=options["variable_size_crop"],
                    timestep_emb_decay_factor=options["timestep_emb_decay_factor"],
                    remove_redundant_pose=False,
                )
            finally:
                np.random.set_state(numpy_state)
                random.setstate(python_state)
        return PredictPlacementsResponse(
            relative=np.asarray(relative, dtype=np.float64),
            timing_ms=(time.perf_counter() - start) * 1000,
        )
