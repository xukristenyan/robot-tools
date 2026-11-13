"""Real HTTP inference plus cached-decoder equivalence to pinned upstream predict_masks.

Synthetic clouds exercise execution and point order, not semantic segmentation accuracy.
Run with the runtime's real environment; starts and stops its own loopback server.
"""

import json
import socket
import threading
import time

import numpy as np
import torch
import uvicorn
from server import PointSAMServer

from robot_tools.services.pointsam import PointSAMClient


def main():
    rng = np.random.default_rng(42)
    xyz = rng.normal(size=(2048, 3))
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    xyz = (xyz * 0.08 + [0.5, -0.2, 0.9]).astype(np.float64)
    rgb = np.tile(np.array([210, 80, 40], np.uint8), (len(xyz), 1))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    start = time.perf_counter()
    service = PointSAMServer(port=port)
    load_seconds = time.perf_counter() - start
    http = uvicorn.Server(uvicorn.Config(service._build_app(), host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=http.run, daemon=True)
    thread.start()
    try:
        with PointSAMClient(host="127.0.0.1", port=port) as client:
            info = client.check_compatibility()
            encoded = client.encode_cloud(xyz, rgb)
            first = client.segment(encoded.cloud_id, xyz[:1], np.array([1]))
            refined = client.segment(encoded.cloud_id, xyz[:2], np.array([1, 0]), prompt_mask=first.logits[0])
            assert first.masks.shape == (3, len(xyz))
            assert refined.masks.shape == (1, len(xyz))
            assert first.masks.dtype == bool and first.logits.dtype == np.float32
            # Compare adapter output with the public upstream function, including
            # re-encoding. This detects parameter ordering, normalization and cache errors.
            saved = service.clouds[encoded.cloud_id]
            state = saved["encoded"]
            prompts = torch.as_tensor(
                (xyz[:1] - saved["center"]) / saved["scale"], dtype=torch.float32, device=service.backend.device
            )[None]
            labels = torch.ones((1, 1), dtype=torch.int64, device=service.backend.device)
            with torch.inference_mode():
                reference, scores = service.backend.model.predict_masks(
                    state["coords"], state["aux"].features, prompts, labels, multimask_output=True
                )
            order = torch.argsort(scores[0], descending=True)
            reference = reference[0, order].cpu().numpy()
            np.testing.assert_allclose(first.logits, reference, rtol=1e-4, atol=1e-4)
            np.testing.assert_array_equal(first.masks, reference > 0)
            assert client.release_cloud(encoded.cloud_id).released
            assert not client.release_cloud(encoded.cloud_id).released
            assert not service.clouds
            print(
                json.dumps(
                    {
                        "service": info["service_id"],
                        "gpu": torch.cuda.get_device_name(0),
                        "torch": torch.__version__,
                        "points": len(xyz),
                        "mask_shape": list(first.masks.shape),
                        "refined_shape": list(refined.masks.shape),
                        "scores": first.scores.tolist(),
                        "load_seconds": load_seconds,
                        "encode_ms": encoded.timing_ms,
                        "decode_ms": first.timing_ms,
                        "refine_ms": refined.timing_ms,
                        "upstream_max_abs_error": float(np.abs(first.logits - reference).max()),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "released": True,
                    },
                    indent=2,
                )
            )
    finally:
        http.should_exit = True
        thread.join(timeout=10)
        assert not thread.is_alive()


if __name__ == "__main__":
    main()
