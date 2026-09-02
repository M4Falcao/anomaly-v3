"""Shared multi-layer feature backbone.

Uses pretrained AlexNet (matches the rest of this codebase) and extracts
feature maps after three semantic levels, upsampled to a common spatial
size. Returns both the concatenated tensor and per-layer maps (for
methods like FastFlow/CFLOW that operate per-layer).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet


class MultiLayerAlexNet(nn.Module):
    """Extract features from AlexNet at 3 semantic levels.

    Levels (index into alexnet.features sequential):
        L1: after features[2]  -> 64 ch, ~stride 4
        L2: after features[5]  -> 192 ch, ~stride 8
        L3: after features[12] -> 256 ch, ~stride 32

    All maps are bilinearly upsampled to `out_size` (default 56).
    """

    def __init__(self, out_size=56, freeze=True):
        super().__init__()
        self.alexnet = alexnet(pretrained=True)
        self.out_size = out_size
        self.layer_channels = [64, 192, 256]
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        feats = []
        for i, layer in enumerate(self.alexnet.features):
            x = layer(x)
            if i == 2:
                feats.append(x)
            elif i == 5:
                feats.append(x)
            elif i == 12:
                feats.append(x)
                break

        # Resize all to common spatial size
        resized = [F.interpolate(f, size=(self.out_size, self.out_size),
                                 mode='bilinear', align_corners=False)
                   for f in feats]
        concat = torch.cat(resized, dim=1)  # (B, 64+192+256=512, H, W)
        return concat, resized

    @property
    def concat_channels(self):
        return sum(self.layer_channels)
