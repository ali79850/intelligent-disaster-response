"""
Phase 3, Sub-step 5/6: PyTorch Dataset for the xBD building damage dataset.

Loads pre-disaster image, post-disaster image, and the corresponding
segmentation mask for each tile. Applies dataset-specific normalization
(computed in Phase 3, see DECISIONS.md) to images only, never to masks.

Un-classified pixels (raw mask value 5) are remapped to ignore_index=255.

Augmentation (train split only): horizontal flip, vertical flip, and
90-degree rotations, applied identically to pre-image, post-image, and mask
using a single shared random choice per sample - never independent random
calls per tensor, which would misalign them.
"""
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

IGNORE_INDEX = 255
UNCLASSIFIED_RAW_VALUE = 5


class XBDDataset(Dataset):
    def __init__(self, split: str, config_path: str = "configs/config.yaml", augment: bool = None):
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

        # Default: augment only on train, unless explicitly overridden
        self.augment = augment if augment is not None else (split == "train")

    def __len__(self):
        return len(self.base_names)

    def _load_image_array(self, base: str, suffix: str) -> np.ndarray:
        img_path = self.images_dir / f"{base}_{suffix}.png"
        return np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0

    def _load_mask_array(self, base: str) -> np.ndarray:
        mask_path = self.masks_dir / f"{base}_mask.png"
        mask = np.array(Image.open(mask_path)).copy()
        mask[mask == UNCLASSIFIED_RAW_VALUE] = IGNORE_INDEX
        return mask

    @staticmethod
    def _apply_shared_transform(pre_arr, post_arr, mask_arr, hflip, vflip, k_rot90):
        if hflip:
            pre_arr = np.ascontiguousarray(pre_arr[:, ::-1, :])
            post_arr = np.ascontiguousarray(post_arr[:, ::-1, :])
            mask_arr = np.ascontiguousarray(mask_arr[:, ::-1])
        if vflip:
            pre_arr = np.ascontiguousarray(pre_arr[::-1, :, :])
            post_arr = np.ascontiguousarray(post_arr[::-1, :, :])
            mask_arr = np.ascontiguousarray(mask_arr[::-1, :])
        if k_rot90:
            pre_arr = np.ascontiguousarray(np.rot90(pre_arr, k=k_rot90, axes=(0, 1)))
            post_arr = np.ascontiguousarray(np.rot90(post_arr, k=k_rot90, axes=(0, 1)))
            mask_arr = np.ascontiguousarray(np.rot90(mask_arr, k=k_rot90, axes=(0, 1)))
        return pre_arr, post_arr, mask_arr

    def __getitem__(self, idx: int):
        base = self.base_names[idx]
        pre_arr = self._load_image_array(base, "pre_disaster")
        post_arr = self._load_image_array(base, "post_disaster")
        mask_arr = self._load_mask_array(base)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            k_rot90 = random.choice([0, 1, 2, 3])
            pre_arr, post_arr, mask_arr = self._apply_shared_transform(
                pre_arr, post_arr, mask_arr, hflip, vflip, k_rot90
            )

        pre_tensor = torch.from_numpy(pre_arr).permute(2, 0, 1)
        post_tensor = torch.from_numpy(post_arr).permute(2, 0, 1)
        pre_tensor = (pre_tensor - self.mean) / self.std
        post_tensor = (post_tensor - self.mean) / self.std
        mask_tensor = torch.from_numpy(mask_arr).long()

        return {
            "pre_image": pre_tensor,
            "post_image": post_tensor,
            "mask": mask_tensor,
            "base_name": base,
        }