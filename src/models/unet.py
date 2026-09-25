"""
Phase 5: U-Net with a pretrained ResNet-18 encoder and skip connections.

Directly addresses Baseline 3's diagnosed failure (Phase 4): aggressive
downsampling with no skip connections caused complete collapse on
minority damage classes. Skip connections here let fine spatial detail
from early encoder layers reach the decoder directly, bypassing the
bottleneck compression.

Input: 6 channels (3 pre-disaster + 3 post-disaster, concatenated).
ResNet-18 is pretrained on ImageNet (natural images) - used here for
general low/mid-level visual feature transfer (edges, textures), not
domain-specific satellite imagery knowledge. First conv layer is modified
to accept 6 channels instead of ResNet's native 3; the pretrained weights
for that layer are duplicated across the extra channels as a reasonable
initialization, not discarded.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class UNetResNet18(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()

        resnet = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        # Modify first conv to accept 6 channels (3 pre + 3 post) instead of 3.
        # Duplicate pretrained weights across the extra channels rather than
        # discarding pretraining entirely - a documented, reasonable choice.
        original_conv1 = resnet.conv1
        new_conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                new_conv1.weight[:, :3] = original_conv1.weight
                new_conv1.weight[:, 3:] = original_conv1.weight
        resnet.conv1 = new_conv1

        # Encoder stages (standard ResNet-18 layout)
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)   # /2,  64 ch
        self.pool0 = resnet.maxpool                                        # /4
        self.enc1 = resnet.layer1                                          # /4,  64 ch
        self.enc2 = resnet.layer2                                          # /8,  128 ch
        self.enc3 = resnet.layer3                                          # /16, 256 ch
        self.enc4 = resnet.layer4                                          # /32, 512 ch

        def up_block(in_ch, skip_ch, out_ch):
            return nn.ModuleDict({
                "up": nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                "conv": nn.Sequential(
                    nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            })

        self.up4 = up_block(512, 256, 256)   # /32 -> /16, skip from enc3
        self.up3 = up_block(256, 128, 128)   # /16 -> /8,  skip from enc2
        self.up2 = up_block(128, 64, 64)     # /8  -> /4,  skip from enc1
        self.up1 = up_block(64, 64, 64)      # /4  -> /2,  skip from enc0

        # Final upsample /2 -> full resolution
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def _apply_up_block(self, block, x, skip):
        x = block["up"](x)
        x = torch.cat([x, skip], dim=1)
        return block["conv"](x)

    def forward(self, pre_img, post_img):
        x = torch.cat([pre_img, post_img], dim=1)  # (B, 6, H, W)

        e0 = self.enc0(x)          # /2,  64 ch  - skip
        p0 = self.pool0(e0)        # /4
        e1 = self.enc1(p0)         # /4,  64 ch  - skip
        e2 = self.enc2(e1)         # /8,  128 ch - skip
        e3 = self.enc3(e2)         # /16, 256 ch - skip
        e4 = self.enc4(e3)         # /32, 512 ch - bottleneck

        d4 = self._apply_up_block(self.up4, e4, e3)   # -> /16
        d3 = self._apply_up_block(self.up3, d4, e2)   # -> /8
        d2 = self._apply_up_block(self.up2, d3, e1)   # -> /4
        d1 = self._apply_up_block(self.up1, d2, e0)   # -> /2

        out = self.final_up(d1)     # -> full res
        out = self.final_conv(out)
        return self.classifier(out)