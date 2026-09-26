"""
Phase 7: Grad-CAM adapted for dense segmentation output.

Standard Grad-CAM backpropagates from a single classification score.
Here, we backpropagate from the SUM of a target class's logits over a
specific region of interest (e.g., "the destroyed-class logit, summed
over exactly the pixels where the model wrongly predicted destroyed"),
letting us directly explain a specific observed decision rather than a
generic class-level explanation.

WHAT THIS SHOWS: which spatial regions, at the chosen intermediate layer,
had the largest gradient influence on the target class score in the
region of interest - i.e., where the network's internal representation
was most sensitive for this specific prediction.

WHAT THIS DOES NOT PROVE: Grad-CAM does not establish causal necessity
(the highlighted region is influential, not necessarily "the reason" in
any complete sense), does not capture the full decoder/skip-connection
pathway's contribution (only the chosen bottleneck layer), and can
sometimes highlight plausible-looking regions that are not the true
mechanism. Treat this as a diagnostic aid, not proof.
"""
import torch
import torch.nn.functional as F


class SegmentationGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, pre_img, post_img, target_class: int, region_mask: torch.Tensor = None):
        """
        region_mask: optional (H, W) boolean tensor restricting which pixels'
        logits to sum over. If None, sums over the whole image (standard
        "does this class matter anywhere" explanation). Pass a specific
        region to explain a specific localized decision instead.
        """
        self.model.zero_grad()
        output = self.model(pre_img, post_img)  # (1, 5, H, W)

        class_logits = output[:, target_class, :, :]  # (1, H, W)
        if region_mask is not None:
            score = (class_logits * region_mask.float()).sum()
        else:
            score = class_logits.sum()

        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=pre_img.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = cam * 0.0  # flat/zero gradient case - explicit, not silently NaN

        return cam