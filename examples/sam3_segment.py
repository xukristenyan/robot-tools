"""SAM 3 segmentation — both prompt types in one script.

`segment_by_text` returns ranked detections (mask + box_xyxy + score per object
matching the text). `segment_by_point` returns up to 3 candidate masks at
increasing scales for a single foreground click.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from _config import load_endpoint, load_params
from PIL import Image

from robot_tools.services.sam3 import SAM3Client


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--text", default="the red mug")
    p.add_argument("--click", type=float, nargs=2, metavar=("X", "Y"), default=None)
    p.add_argument("--out-dir", type=Path, default=Path("sam3_out"))
    args = p.parse_args()

    params = load_params("sam3")
    image = np.asarray(Image.open(args.image).convert("RGB"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with SAM3Client.from_endpoint(load_endpoint("sam3")) as client:
        text_resp = client.segment_by_text(image, args.text, score_threshold=params["score_threshold"])
        print(f"[text] '{args.text}' → {len(text_resp.detections)} detection(s)")
        for i, det in enumerate(text_resp.detections):
            print(f"  #{i} score={det.score:.3f} box_xyxy={det.box_xyxy} pixels={int(det.mask.sum())}")
            np.save(args.out_dir / f"text_mask_{i}.npy", det.mask)

        if args.click is not None:
            point_resp = client.segment_by_point(
                image,
                tuple(args.click),
                multimask_output=params["multimask_output"],
            )
            print(f"[click] xy={args.click} → {len(point_resp.scores)} candidate mask(s)")
            for i, score in enumerate(point_resp.scores):
                print(f"  #{i} score={score:.3f} pixels={int(point_resp.masks[i].sum())}")
                np.save(args.out_dir / f"point_mask_{i}.npy", point_resp.masks[i])


if __name__ == "__main__":
    main()
