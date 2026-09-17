"""Real checkpoint smoke test on synthetic RGBD; not a placement-quality test.

Run with the installed uv environment, e.g.:
    uv run --no-sync python smoke.py
    uv run --no-sync python smoke.py --host 127.0.0.1 --port 5560
"""

import argparse
import json

import numpy as np

from robot_tools.services.anyplace import AnyPlaceClient, masked_rgbd
from robot_tools.services.anyplace.contract import PredictPlacementsRequest


def observations(planar=False):
    v, u = np.mgrid[:40, :40]
    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    k = np.array([[200.0, 0, 20], [0, 200.0, 20], [0, 0, 1]])
    depth = (0.5 + 0.008 * ((u - 20) ** 2 + (v - 20) ** 2) / 400).astype(np.float32)
    pick = masked_rgbd(rgb, depth, ((u - 20) ** 2 + (v - 20) ** 2) < 225, k)
    pose = np.eye(4)
    pose[0, 3] = 0.15
    place_depth = np.full_like(depth, 0.5) if planar else depth
    place = masked_rgbd(rgb, place_depth, np.ones((40, 40), dtype=bool), k, camera_pose=pose)
    return pick, place


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", help="Use the real HTTP server instead of constructing the backend")
    parser.add_argument("--port", type=int, default=5560)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--n-refine-iters", type=int, default=50)
    parser.add_argument("--planar", action="store_true")
    parser.add_argument("--repeat", action="store_true", help="Assert repeated seeded calls agree")
    args = parser.parse_args()
    pick, place = observations(args.planar)
    request = PredictPlacementsRequest(
        pick=pick,
        place=place,
        num_samples=args.num_samples,
        n_refine_iters=args.n_refine_iters,
    )
    if args.host:
        with AnyPlaceClient(host=args.host, port=args.port, timeout_ms=120_000) as client:

            def infer():
                return client.predict_placements(
                    pick,
                    place,
                    num_samples=args.num_samples,
                    n_refine_iters=args.n_refine_iters,
                )

            result = infer()
            if args.repeat:
                np.testing.assert_allclose(infer().relative, result.relative, atol=1e-6)
    else:
        from backend import AnyPlaceBackend

        backend = AnyPlaceBackend()
        result = backend.predict_placements(request)
        if args.repeat:
            np.testing.assert_allclose(backend.predict_placements(request).relative, result.relative, atol=1e-6)
    assert result.relative.shape == (args.num_samples, 4, 4)
    print(
        json.dumps(
            {
                "shape": list(result.relative.shape),
                "dtype": str(result.relative.dtype),
                "timing_ms": result.timing_ms,
                "finite": bool(np.isfinite(result.relative).all()),
                "planar": args.planar,
                "repeat_checked": args.repeat,
            }
        )
    )


if __name__ == "__main__":
    main()
