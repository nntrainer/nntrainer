"""Reconstructed ReID embedding head (global pool -> linear embedding).

Upstream ``models_pose/head/reid.py`` was unavailable; this mirrors a standard
ReID embedding head and matches the reference ``inference.py`` which consumes
``feature`` as a [B, embed_dim] vector. Module names here are mirrored by the
weight_converter.
"""

import torch
import torch.nn as nn


class ReID(nn.Module):
    def __init__(self, nc, embed_dim=128, ch=None, img_size=320, featuremap_ratio=32):
        super().__init__()
        in_ch = ch[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, embed_dim, bias=True)

    def forward(self, feat):
        x = self.pool(feat).flatten(1)   # [B, in_ch]
        return self.fc(x)                # [B, embed_dim]
