from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from robot_tools.services.sam3.contract import SegmentByPointRequest, SegmentByTextRequest


def _load_sam3_server_class():
    path = Path(__file__).resolve().parents[1] / "runtimes" / "sam3" / "server.py"
    spec = importlib.util.spec_from_file_location("robot_tools_test_sam3_server", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SAM3Server


def test_text_fake_returns_a_pipeline_usable_instance():
    server = _load_sam3_server_class()(fake=True)
    image = np.zeros((20, 40, 3), dtype=np.uint8)

    response = server.segment_by_text(SegmentByTextRequest(image=image, text_prompt="mug"))

    assert len(response.detections) == 1
    detection = response.detections[0]
    assert detection.label == "mug"
    assert detection.box_xyxy == [10.0, 5.0, 30.0, 15.0]
    assert detection.mask.shape == image.shape[:2]
    assert detection.mask.dtype == np.bool_
    assert detection.mask.any()


def test_point_fake_is_centered_on_absolute_xy_and_respects_multimask():
    server = _load_sam3_server_class()(fake=True)
    image = np.zeros((32, 48, 3), dtype=np.uint8)

    response = server.segment_by_point(
        SegmentByPointRequest(
            image=image,
            point_xy=(12.0, 9.0),
            multimask_output=True,
        )
    )

    assert response.masks.shape == (3, 32, 48)
    assert response.masks[:, 9, 12].all()
    assert response.scores == [0.95, 0.8, 0.65]
