"""
Phase 4, Baseline 3: A deliberately simple CNN for per-pixel damage
classification, trained end-to-end on concatenated pre+post images.

This is NOT a U-Net (no skip connections) - that distinction is intentional
and is what justifies a more sophisticated architecture in Phase 5.

Revised after measuring ~4.2s/batch (see DECISIONS.md) with the original
full-resolution-first design: now downsamples early (stride-4 stem) so
the expensive conv stack operates on a much smaller feature map, only
upsampling back to full resolution at the very end.
"""
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        # Stem: aggressively downsample BEFORE the expensive conv stack.
        # stride=4 conv takes 1024x1024 -> 256x256 immediately (16x fewer
        # pixels per subsequent layer), which is the fix for the measured
        # ~4.2s/batch slowness of the original full-res-first design.
        self.stem = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )  # -> (B, 16, 256, 256)

        self.enc1 = conv_block(16, 32)
        self.pool1 = nn.MaxPool2d(2)   # -> 128x128
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)   # -> 64x64

        self.bottleneck = conv_block(64, 64)  # stays at 64x64 - cheap now

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)   # -> 128x128
        self.dec2 = conv_block(32, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)   # -> 256x256
        self.dec1 = conv_block(16, 16)

        # Final upsample back to full 1024x1024 resolution, only once,
        # at the very end - not repeated at every layer.
        self.final_upsample = nn.ConvTranspose2d(16, 16, kernel_size=4, stride=4)  # -> 1024x1024
        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, pre_img, post_img):
        x = torch.cat([pre_img, post_img], dim=1)  # (B, 6, H, W)

        x = self.stem(x)        # (B, 16, 256, 256)

        x = self.enc1(x)        # (B, 32, 256, 256)
        x = self.pool1(x)       # (B, 32, 128, 128)
        x = self.enc2(x)        # (B, 64, 128, 128)
        x = self.pool2(x)       # (B, 64, 64, 64)

        x = self.bottleneck(x)  # (B, 64, 64, 64)

        x = self.up2(x)         # (B, 32, 128, 128)
        x = self.dec2(x)
        x = self.up1(x)         # (B, 16, 256, 256)
        x = self.dec1(x)

        x = self.final_upsample(x)  # (B, 16, 1024, 1024)
        return self.classifier(x)