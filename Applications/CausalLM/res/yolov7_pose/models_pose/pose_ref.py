"""Exact reference model for pose_base_v311.pt (pose-only YOLOv7 tiny).

Reconstructed from the checkpoint's parameter hierarchy/shapes so its
state_dict keys match the real model exactly (used by make_reference.py to
produce a PyTorch reference for parity). Assembled from the real building
blocks in common_v2.py; the wiring/channels below mirror the checkpoint.

Structure:
  backbone.backbone.blocks.0.0            StemS0 conv 3->48 (s2)
  backbone.backbone.blocks.1.base         DownConv 48->96 (s2)
  backbone.backbone.blocks.1.elan         ELAN_S0 96->96 (bn 48)
  backbone.backbone.blocks.{2,3,4}.elan   maxpool(s2)+ELAN_S0 doubling
  backbone.features.spp                   SPPCSPCTiny 768->384
  backbone.features.feature_up.{0,1}      UpSampleBlock(base)+ELAN_S0
  backbone.features.feature_down.{0,1}    DownSampleRouteS0(base)+ELAN_S0
  backbone.features.ends.0                Conv 384->768
  head                                    RTMCC SimCC pose head
"""

import torch
import torch.nn as nn

from models_pose.common import RTMCCBlock, ScaleNorm
from models_pose.common_v2 import (
    Conv,
    DownConv,
    DownSampleRouteS0,
    ELAN_S0,
    SPPCSPCTiny,
    StemS0,
    UpSampleBlock,
)


class _Block1(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = DownConv(48, 96)
        self.elan = ELAN_S0(96, 96, 48, 2)

    def forward(self, x):
        return self.elan(self.base(x))


class _BlockMP(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.mp = nn.MaxPool2d(2, 2)
        self.elan = ELAN_S0(c_in, c_out, c_in, 2)

    def forward(self, x):
        return self.elan(self.mp(x))


class CSPTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            StemS0(3, 48),
            _Block1(),
            _BlockMP(96, 192),
            _BlockMP(192, 384),
            _BlockMP(384, 768),
        ])

    def forward(self, x, return_intermediate_nodes=False):
        nodes = []
        for b in self.blocks:
            x = b(x)
            nodes.append(x)
        return (x, nodes) if return_intermediate_nodes else x


class _BlockS2(nn.Module):
    def __init__(self, base, elan):
        super().__init__()
        self.base = base
        self.elan = elan

    def forward(self, x, route):
        return self.elan(self.base(x, route))


class Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.spp = SPPCSPCTiny(768, 384)
        self.feature_up = nn.ModuleList([
            _BlockS2(UpSampleBlock(384, 384, self._r(384)), ELAN_S0(384, 192, 96, 2)),
            _BlockS2(UpSampleBlock(192, 192, self._r(192)), ELAN_S0(192, 96, 48, 2)),
        ])
        self.feature_down = nn.ModuleList([
            _BlockS2(DownSampleRouteS0(96, 384, self._r(192)), ELAN_S0(384, 192, 96, 2)),
            _BlockS2(DownSampleRouteS0(192, 768, self._r(384)), ELAN_S0(768, 384, 192, 2)),
        ])
        self.ends = nn.ModuleList([Conv(384, 768, 3, 1)])

    @staticmethod
    def _r(c):
        m = nn.Module()
        m.c_out = c
        return m

    def forward(self, x, features):
        spp = self.spp(x)
        up0 = self.feature_up[0](spp, features[3])
        up1 = self.feature_up[1](up0, features[2])
        down0 = self.feature_down[0](up1, up0)
        down1 = self.feature_down[1](down0, spp)
        return self.ends[0](down1)


class YOLOv7TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CSPTiny()
        self.features = Neck()

    def forward(self, x):
        x, feats = self.backbone(x, return_intermediate_nodes=True)
        return self.features(x, feats)


class RTMCC(nn.Module):
    def __init__(self, nkpt=87, img=320, ratio=32, gau_hidden=256, s=128,
                 expansion=2, split=2.0, final_k=7):
        super().__init__()
        feat = img // ratio
        self.final_layer = nn.Conv2d(768, nkpt, final_k, 1, final_k // 2)
        self.mlp = nn.Sequential(
            ScaleNorm(feat * feat),
            nn.Linear(feat * feat, gau_hidden, bias=False),
        )
        self.gau = RTMCCBlock(nkpt, gau_hidden, gau_hidden,
                              expansion_factor=expansion, s=s)
        bins = int(img * split)
        self.cls_x = nn.Linear(gau_hidden, bins, bias=False)
        self.cls_y = nn.Linear(gau_hidden, bins, bias=False)

    def forward(self, feat):
        x = torch.flatten(self.final_layer(feat), 2)
        x = self.gau(self.mlp(x))
        return torch.cat([self.cls_x(x), self.cls_y(x)], dim=1)


class YOLOv7Posetiny(nn.Module):
    """Pose-only model matching pose_base_v311.pt."""

    def __init__(self, nkpt=87, img_size=320):
        super().__init__()
        self.backbone = YOLOv7TinyBase()
        self.head = RTMCC(nkpt=nkpt, img=img_size)
        self.model = nn.Sequential(self.backbone, self.head)

    def forward(self, x):
        return self.head(self.backbone(x))
