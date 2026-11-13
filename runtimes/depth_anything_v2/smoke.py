"""Real HTTP inference and parity with the official RGB/BGR inference path."""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import uvicorn
from server import DepthAnythingV2Server

from robot_tools.services.depth_anything_v2 import DepthAnythingV2Client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise ValueError(f"Cannot read image: {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    service = DepthAnythingV2Server(port=port, checkpoint=args.checkpoint, input_size=args.input_size)
    server = uvicorn.Server(uvicorn.Config(service._build_app(), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        if not server.started:
            raise RuntimeError("Smoke server failed to start")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with DepthAnythingV2Client(host="127.0.0.1", port=port) as client:
            result = client.estimate_depth(rgb)
        prediction = result.relative_inverse_depth
        expected = service.backend.model.infer_image(bgr, input_size=args.input_size)
        np.testing.assert_allclose(prediction, expected, rtol=1e-5, atol=1e-5)
        assert prediction.shape == rgb.shape[:2]
        assert prediction.dtype == np.float32 and np.isfinite(prediction).all()
        assert prediction.max() > prediction.min()
        print(
            json.dumps(
                {
                    "device": service.backend.device,
                    "image": str(args.image),
                    "shape": list(prediction.shape),
                    "dtype": str(prediction.dtype),
                    "input_size": args.input_size,
                    "range": [float(prediction.min()), float(prediction.max())],
                    "http_inference_ms": result.timing_ms,
                    "upstream_max_abs_difference": float(np.abs(prediction - expected).max()),
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20
                    if torch.cuda.is_available()
                    else None,
                },
                indent=2,
            )
        )
    finally:
        server.should_exit = True
        thread.join(timeout=10)


if __name__ == "__main__":
    main()
