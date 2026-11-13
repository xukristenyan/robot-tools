"""Pinned Point-SAM model, with the encoder split out for reusable cloud embeddings.

The decoder below follows upstream PointCloudSAM.predict_masks exactly for B=1.
Use ordinary PyTorch LayerNorm rather than the optional Apex fused replacement.
Weights, epsilon, and normalization semantics are unchanged; no training code runs.
"""

from __future__ import annotations

import getpass
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

UPSTREAM_REVISION = "25f4fd9675479f3c951e830771595855ff7ed466"
WEIGHTS_REVISION = "fb2cd5cd8047e3681491d82fc41b880a9a914fc8"


def data_dir():
    root = os.environ.get("ROBOT_TOOLS_MODELS_ROOT", f"/data/{getpass.getuser()}/robot-tools-models")
    return Path(os.environ.get("POINTSAM_DATA_DIR", str(Path(root) / "pointsam"))).expanduser()


def upstream_dir():
    return data_dir() / "source" / "Point-SAM"


def import_upstream():
    source = upstream_dir()
    if not (source / "pc_sam/model/pc_sam.py").is_file():
        raise FileNotFoundError("Point-SAM source is missing; run robot-tools install with the pointsam profile")
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise RuntimeError("Point-SAM source does not match its pinned revision")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from pc_sam.model.pc_sam import PointCloudSAM

    return PointCloudSAM


class PointSAMBackend:
    def __init__(self, *, checkpoint=None, device="cuda"):
        root = data_dir()
        for key, subdir in {
            "TORCH_HOME": "cache/torch",
            "CUDA_CACHE_PATH": "cache/cuda",
            "HF_HOME": "cache/huggingface",
        }.items():
            os.environ.setdefault(key, str(root / subdir))
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from safetensors.torch import load_model

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Point-SAM real runtime requires a CUDA GPU")
        path = Path(checkpoint) if checkpoint else root / "weights/model.safetensors"
        if not path.is_file():
            raise FileNotFoundError("Point-SAM weights are missing; run robot-tools download")
        import_upstream()
        config = OmegaConf.load(upstream_dir() / "configs/model/default.yaml")
        self.model = instantiate(config)
        # Strict loading catches an incorrect model/configuration or partial checkpoint.
        load_model(self.model, str(path), strict=True, device="cpu")
        self.model.eval().to(self.device)

    def encode(self, xyz, rgb, num_groups, group_size):
        torch = self.torch
        from pc_sam.model.mask_decoder import AuxInputs

        with torch.inference_mode():
            coords = torch.from_numpy(np.ascontiguousarray(xyz)).to(self.device)[None]
            features = torch.from_numpy(np.array(rgb, dtype=np.float32, copy=True)).to(self.device)[None] / 255
            grouper = self.model.pc_encoder.patch_embed.grouper
            grouper.num_groups, grouper.group_size = num_groups, group_size
            embeddings, patches = self.model.pc_encoder(coords, features)
            centers = patches["centers"]
            positional = self.model.point_encoder.pe_layer(centers)
            aux = AuxInputs(coords=coords, features=features, centers=centers)
            torch.cuda.synchronize(self.device)
        # Do not retain the large temporary patch features in the cloud cache.
        return {
            "coords": coords,
            "embeddings": embeddings,
            "centers": centers,
            "knn_idx": patches["knn_idx"],
            "positional": positional,
            "aux": aux,
        }

    def segment(self, cloud, prompts, labels, mask, multiple):
        torch = self.torch
        with torch.inference_mode():
            points = torch.from_numpy(np.array(prompts, dtype=np.float32, copy=True)).to(self.device)[None]
            labels = torch.from_numpy(np.array(labels, dtype=np.int64, copy=True)).to(self.device)[None]
            mask = (
                None
                if mask is None
                else torch.from_numpy(np.array(mask, dtype=np.float32, copy=True)).to(self.device)[None]
            )
            sparse = self.model.point_encoder(points, labels)
            dense = self.model.mask_encoder(mask, cloud["coords"], cloud["centers"], cloud["knn_idx"])
            logits, scores = self.model.mask_decoder(
                cloud["embeddings"],
                cloud["positional"],
                sparse,
                dense,
                aux_inputs=cloud["aux"],
                multimask_output=multiple,
            )
            return logits[0].float().cpu().numpy(), scores[0].float().cpu().numpy()
