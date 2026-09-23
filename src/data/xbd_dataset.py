"""
Phase 3, Sub-step 5: PyTorch Dataset for the xBD building damage dataset.

Loads pre-disaster image, post-disaster image, and the corresponding
segmentation mask for each tile. Applies dataset-specific normalization
(computed in Phase 3, see DECISIONS.md) to images only, never to masks.

Un-classified pixels (raw mask value 5) are remapped to ignore_index=255,
per the Phase 3 decision to exclude them from loss/metrics while keeping
them visible in raw mask files for debugging.
"""
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

IGNORE_INDEX = 255
UNCLASSIFIED_RAW_VALUE = 5


class XBDDataset(Dataset):
    def __init__(self, split: str, config_path: str = "configs/config.yaml"):
        assert split in ("train", "val", "test"), f"Invalid split: {split}"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.images_dir = Path("data/raw/xbd/train/images")
        self.masks_dir = Path("data/processed/masks")
        splits_dir = Path("data/interim/splits")

        with open(splits_dir / f"{split}.txt") as f:
            self.base_names = [line.strip() for line in f if line.strip()]

        self.mean = torch.tensor(config["normalization"]["mean"]).view(3, 1, 1)
        self.std = torch.tensor(config["normalization"]["std"]).view(3, 1, 1)

    def __len__(self):
        return len(self.base_names)

    def _load_image(self, base: str, suffix: str) -> torch.Tensor:
        img_path = self.images_dir / f"{base}_{suffix}.png"
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
        img_tensor = (img_tensor - self.mean) / self.std
        return img_tensor

    def _load_mask(self, base: str) -> torch.Tensor:
        mask_path = self.masks_dir / f"{base}_mask.png"
        mask = np.array(Image.open(mask_path))
        mask = mask.copy()
        mask[mask == UNCLASSIFIED_RAW_VALUE] = IGNORE_INDEX
        return torch.from_numpy(mask).long()

    def __getitem__(self, idx: int):
        base = self.base_names[idx]
        pre_img = self._load_image(base, "pre_disaster")
        post_img = self._load_image(base, "post_disaster")
        mask = self._load_mask(base)
        return {
            "pre_image": pre_img,
            "post_image": post_img,
            "mask": mask,
            "base_name": base,
        }