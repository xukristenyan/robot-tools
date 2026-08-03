"""Headless SAM 3 model adapter for the HTTP runtime.

Two modes (image-only for now; video tracking can be added later):
  - segment_by_text(image, text) → instances with mask + box + score
  - segment_by_point(image, (x, y), multimask_output) → up to 3 candidate masks

Adapted from cap-x's `capx/serving/launch_sam3_server.py` patterns.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def _to_numpy(t: Any) -> np.ndarray:
    if hasattr(t, "detach"):
        x = t.detach().cpu()
        if x.dtype == torch.bfloat16:
            x = x.float()
        return x.numpy()
    if hasattr(t, "cpu"):
        x = t.cpu()
        if hasattr(x, "dtype") and x.dtype == torch.bfloat16:
            x = x.float()
        return x.numpy()
    return np.asarray(t)


class SAM3Backend:
    """Loads SAM 3 once, runs single-image inference."""

    def __init__(self, device: str = "cuda"):
        # Imports deferred so this module is importable without sam3 installed.
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.device = device
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if "cuda" in device:
                idx = int(device.split(":")[-1]) if ":" in device else 0
                torch.cuda.set_device(idx)

        logger.info("loading SAM 3 model (device=%s)...", device)
        model = build_sam3_image_model(enable_inst_interactivity=True)
        if hasattr(model, "to"):
            model = model.to(device)
        self.model = model
        self.processor = Sam3Processor(model, device=device, confidence_threshold=0.0)
        self._device_type = "cuda" if "cuda" in device else "cpu"

    # ── text prompt: returns list of (mask, box, score, label) ─────────────
    def segment_by_text(self, image: np.ndarray, text_prompt: str) -> list[dict]:
        pil = self._as_pil(image)
        with torch.autocast(self._device_type, dtype=torch.bfloat16):
            state = self.processor.set_image(pil)
            output = self.processor.set_text_prompt(state=state, prompt=text_prompt)

        masks_t = output.get("masks")
        boxes_t = output.get("boxes")
        scores_t = output.get("scores")
        if masks_t is None or boxes_t is None or scores_t is None:
            return []

        masks = _to_numpy(masks_t)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)
        boxes = _to_numpy(boxes_t)
        scores = _to_numpy(scores_t)

        results: list[dict] = []
        for i in range(len(scores)):
            results.append(
                {
                    "mask": (masks[i] > 0).astype(bool),
                    "box_xyxy": [float(v) for v in boxes[i].tolist()],
                    "score": float(scores[i]),
                    "label": text_prompt,
                }
            )
        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    # ── point prompt: returns (masks (N,H,W) bool, scores list[float]) ─────
    def segment_by_point(
        self,
        image: np.ndarray,
        point_xy: tuple[float, float],
        multimask_output: bool = True,
    ) -> tuple[np.ndarray, list[float]]:
        if getattr(self.model, "inst_interactive_predictor", None) is None:
            raise RuntimeError("Instance interactivity not enabled on SAM 3 model.")

        pil = self._as_pil(image)
        with torch.autocast(self._device_type, dtype=torch.bfloat16):
            state = self.processor.set_image(pil)
            point_coords = np.array([list(point_xy)], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int64)  # foreground
            masks, scores, _ = self.model.predict_inst(
                state,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=multimask_output,
            )

        masks = np.asarray(masks)
        scores = np.asarray(scores)
        if masks.size == 0:
            return np.zeros((0,) + image.shape[:2], dtype=bool), []

        order = np.argsort(scores)[::-1]
        masks = masks[order].astype(bool)
        scores = scores[order]
        return masks, [float(s) for s in scores]

    @staticmethod
    def _as_pil(image: np.ndarray) -> Image.Image:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        return Image.fromarray(image[..., :3]).convert("RGB")
