"""Reconstructed RTMCC (RTMPose SimCC) keypoint head.

The upstream ``models_pose/head/keypoint.py`` was unavailable; this mirrors the
standard RTMPose RTMCC head (final conv -> flatten -> ScaleNorm+Linear MLP ->
GAU -> per-axis SimCC classifiers) and is consistent with the pose output
decoded by the reference ``inference.py``:

    pose : [B, 2*nkpt, simcc_bins]   (cls_x rows stacked over cls_y rows)

Hyper-parameters (gau_hidden, gau_s, expansion, final_kernel, simcc split) are
exposed so they can be matched to the real checkpoint. The weight_converter
auto-derives every dimension from the checkpoint tensor shapes, so only the
module *names* here need to stay in sync with the real hierarchy.
"""

import torch
import torch.nn as nn

from models_pose.common import RTMCCBlock, ScaleNorm


class RTMCC(nn.Module):
    def __init__(
        self,
        nc,
        ch,
        img_size=320,
        featuremap_ratio=32,
        nkpt=87,
        gau_hidden=256,
        gau_s=128,
        expansion=2,
        simcc_split_ratio=2.0,
        final_kernel=7,
    ):
        super().__init__()
        in_ch = ch[-1]
        feat = img_size // featuremap_ratio
        self.nkpt = nkpt
        self.simcc_bins = int(img_size * simcc_split_ratio)

        self.final_layer = nn.Conv2d(
            in_ch, nkpt, final_kernel, 1, final_kernel // 2, bias=True
        )
        flatten_dims = feat * feat
        self.mlp = nn.Sequential(
            ScaleNorm(flatten_dims),
            nn.Linear(flatten_dims, gau_hidden, bias=False),
        )
        self.gau = RTMCCBlock(
            nkpt, gau_hidden, gau_hidden, expansion_factor=expansion, s=gau_s
        )
        self.cls_x = nn.Linear(gau_hidden, self.simcc_bins, bias=False)
        self.cls_y = nn.Linear(gau_hidden, self.simcc_bins, bias=False)

    def forward(self, feat):
        x = self.final_layer(feat)          # [B, nkpt, H, W]
        x = torch.flatten(x, 2)             # [B, nkpt, H*W]
        x = self.mlp(x)                     # [B, nkpt, gau_hidden]
        x = self.gau(x)                     # [B, nkpt, gau_hidden]
        pred_x = self.cls_x(x)              # [B, nkpt, simcc_bins]
        pred_y = self.cls_y(x)              # [B, nkpt, simcc_bins]
        return torch.cat([pred_x, pred_y], dim=1)  # [B, 2*nkpt, simcc_bins]
