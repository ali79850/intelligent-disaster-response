"""
Phase 9: Inference engine - separates trained model from any request-time
logic. Load once, call repeatedly. Reuses the exact preprocessing/
normalization from XBDDataset (Phase 3) to avoid train/inference drift.
"""
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from src.models.unet import UNetResNet18

CLASS_NAMES = ["background", "no-damage", "minor-damage", "major-damage", "destroyed"]
IGNORE_INDEX = 255


class DamageInferenceEngine:
    def __init__(self, checkpoint_path: str, config_path: str = "configs/config.yaml", device: str = None):
        self.checkpoint_path = checkpoint_path
        self.model_version = Path(checkpoint_path).stem  # e.g. "unet_epoch10"

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.mean = torch.tensor(config["normalization"]["mean"]).view(3, 1, 1)
        self.std = torch.tensor(config["normalization"]["std"]).view(3, 1, 1)

        self.model = UNetResNet18(num_classes=5, pretrained=False).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Exact same normalization as XBDDataset (Phase 3) - reused, not
        reimplemented, to avoid train/inference drift."""
        arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        tensor = (tensor - self.mean) / self.std
        return tensor

    def predict(self, pre_image: Image.Image, post_image: Image.Image) -> dict:
        """
        Runs inference on a pre/post image pair. Returns a structured
        result: predicted mask, per-class probabilities, and summary stats.
        """
        pre_tensor = self._preprocess_image(pre_image).unsqueeze(0).to(self.device)
        post_tensor = self._preprocess_image(post_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(pre_tensor, post_tensor)  # (1, 5, H, W)
            probs = torch.softmax(logits, dim=1)
            pred_mask = probs.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)
            probs = probs.squeeze(0).cpu().numpy()  # (5, H, W)

        total_pixels = pred_mask.size
        class_stats = {}
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            count = int((pred_mask == cls_idx).sum())
            class_stats[cls_name] = {
                "pixel_count": count,
                "percentage": round(100 * count / total_pixels, 2),
            }

        # Mean confidence (max softmax prob) for the predicted class, per pixel
        max_probs = probs.max(axis=0)
        mean_confidence = float(max_probs.mean())

        return {
            "model_version": self.model_version,
            "predicted_mask": pred_mask,          # (H, W) int array, for visualization/further processing
            "class_probabilities": probs,          # (5, H, W) float array, for uncertainty analysis
            "class_statistics": class_stats,
            "mean_confidence": round(mean_confidence, 4),
            "image_size": pred_mask.shape,
        }