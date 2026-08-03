from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from robot_tools.services.graspgenx import GraspGenXClient
from robot_tools.services.graspgenx.contract import InferRequest, SweepVolumeParams


def _sweep_volume() -> SweepVolumeParams:
    return SweepVolumeParams(
        extents_open=[0.085, 0.04, 0.12],
        offset_open=[0.0, 0.0, 0.06],
        extents_mid=[0.045, 0.04, 0.12],
        offset_mid=[0.0, 0.0, 0.06],
    )


def _call_inference(
    client: GraspGenXClient,
    method_name: str,
    *,
    grasp_threshold: Any,
    topk_num_grasps: Any,
):
    selection = {
        "grasp_threshold": grasp_threshold,
        "topk_num_grasps": topk_num_grasps,
    }
    point_cloud = np.zeros((4, 3), dtype=np.float32)
    sweep_volume = _sweep_volume()

    if method_name == "infer":
        return client.infer(point_cloud, **selection)
    if method_name == "infer_object":
        return client.infer_object(point_cloud, sweep_volume, **selection)
    if method_name == "infer_scene_depth":
        return client.infer_scene_depth(
            np.ones((2, 2), dtype=np.float32),
            np.eye(3, dtype=np.float32),
            np.ones((2, 2), dtype=np.int32),
            sweep_volume,
            **selection,
        )
    if method_name == "infer_scene_pc":
        return client.infer_scene_pc(
            point_cloud,
            np.ones(4, dtype=np.int32),
            sweep_volume,
            **selection,
        )
    raise AssertionError(f"unhandled method: {method_name}")


@pytest.mark.parametrize(
    "method_name",
    ["infer", "infer_object", "infer_scene_depth", "infer_scene_pc"],
)
@pytest.mark.parametrize(
    ("grasp_threshold", "topk_num_grasps"),
    [
        (float("nan"), 100),
        (float("inf"), 100),
        (float("-inf"), 100),
        (True, 100),
        (False, 100),
        (np.bool_(True), 100),
        (np.bool_(False), 100),
        (-1.0, 0),
        (-1.0, -2),
        (-1.0, True),
        (-1.0, False),
        (-1.0, np.bool_(True)),
        (-1.0, np.bool_(False)),
        (-1.0, 1.5),
        (-1.0, "3"),
    ],
)
def test_invalid_selection_is_rejected_before_request(
    monkeypatch,
    method_name,
    grasp_threshold,
    topk_num_grasps,
):
    calls = []
    with GraspGenXClient(wait=False) as client:
        monkeypatch.setattr(client, "_request", lambda *args: calls.append(args))

        with pytest.raises((TypeError, ValueError)):
            _call_inference(
                client,
                method_name,
                grasp_threshold=grasp_threshold,
                topk_num_grasps=topk_num_grasps,
            )

    assert calls == []


def test_selection_accepts_numpy_scalars_before_request(monkeypatch):
    sentinel = object()
    calls = []
    with GraspGenXClient(wait=False) as client:
        monkeypatch.setattr(
            client,
            "_request",
            lambda *args: calls.append(args) or sentinel,
        )

        result = _call_inference(
            client,
            "infer",
            grasp_threshold=np.float32(0.25),
            topk_num_grasps=np.int64(3),
        )

    assert result is sentinel
    assert len(calls) == 1
    request = calls[0][0]
    assert request.grasp_threshold == pytest.approx(0.25)
    assert request.topk_num_grasps == 3


@pytest.mark.parametrize("field", ["grasp_threshold", "topk_num_grasps"])
@pytest.mark.parametrize("boolean", [True, False, np.bool_(True), np.bool_(False)])
def test_infer_contract_rejects_boolean_selection(field, boolean):
    with pytest.raises(ValidationError, match=rf"{field} must not be boolean"):
        InferRequest(
            point_cloud=np.zeros((4, 3), dtype=np.float32),
            **{field: boolean},
        )
