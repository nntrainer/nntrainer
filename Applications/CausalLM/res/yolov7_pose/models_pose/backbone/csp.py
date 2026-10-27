"""Reconstructed CSPTiny / BlockS2 for YOLOv7ReIDtiny (models_pose).

NOTE
----
The upstream ``models_pose/backbone/csp.py`` was not available when this port
was written. This file is a structural reconstruction derived from the public
constraints in ``yolo_v2.py`` / ``yolo_backbone.py``:

  * ``CSPTiny(c_in, c_stem, c_block, num_repeat)`` exposes ``.blocks`` whose
    ``.c_out`` values are read by ``IFeatureYOLOv7Tiny`` and must satisfy
    ``blocks[0].c_out == c_stem`` and ``blocks[-1].c_out == c_stem * 16``.
  * ``forward(x, return_intermediate_nodes=True)`` returns ``(deepest, nodes)``
    where ``nodes[i]`` is aligned channel-wise with ``blocks[i]``.
  * ``BlockS2(sample_cls, elan_cls, c_in, c_sample_out, c_elan_out, n, route)``
    applies ``sample_cls`` then ``elan_cls`` (used by the neck FPN).

If the real checkpoint's parameter hierarchy differs, only the module names in
THIS file need to change; the weight_converter mirrors these names.
"""

import torch
import torch.nn as nn

from models_pose.common_v2 import (
    Conv,
    DownConv,
    DownSampleRouteS0,
    ELAN_S0,
    StemS0,
)


class BlockS2(nn.Module):
    """Sample (down/up) block followed by an ELAN block."""

    @property
    def c_out(self):
        return self.elan.c_out

    def __init__(self, sample_cls, elan_cls, c_in, c_sample_out, c_elan_out,
                 n_repeat, route=None):
        super().__init__()
        # UpSampleBlock/DownSampleRouteS0 both emit c_sample_out channels; the
        # former has no ``c_out`` property, so use the argument directly.
        self.sample = sample_cls(c_in, c_sample_out, route=route)
        self.elan = elan_cls(c_sample_out, c_elan_out, c_elan_out, n_repeat)

    def forward(self, x, route=None):
        x = self.sample(x, route) if route is not None else self.sample(x)
        return self.elan(x)


class CSPTiny(nn.Module):
    """YOLOv7-tiny CSP backbone producing 5 stage nodes (strides 2..32)."""

    def __init__(self, c_in: int, c_stem: int, c_block: int, num_repeat: int):
        super().__init__()

        # blocks[0]: stem, stride 2, c_out = c_stem
        stem = StemS0(c_in, c_stem)

        # blocks[1..4]: ELAN stages at strides 4, 8, 16, 32.
        # Channels double each stage: c_stem*2, *4, *8, *16.
        stage1 = _Stage(c_stem, c_stem * 2, c_block, num_repeat)
        stage2 = _Stage(c_stem * 2, c_stem * 4, c_block * 2, num_repeat)
        stage3 = _Stage(c_stem * 4, c_stem * 8, c_block * 4, num_repeat)
        stage4 = _Stage(c_stem * 8, c_stem * 16, c_block * 8, num_repeat)

        self.blocks = nn.ModuleList([stem, stage1, stage2, stage3, stage4])

    def forward(self, x: torch.Tensor, return_intermediate_nodes: bool = False):
        nodes = []
        for block in self.blocks:
            x = block(x)
            nodes.append(x)
        if return_intermediate_nodes:
            return x, nodes
        return x


class _Stage(nn.Module):
    """One downsample (stride 2) + ELAN block."""

    @property
    def c_out(self):
        return self.elan.c_out

    def __init__(self, c_in: int, c_out: int, c_bottleneck: int, num_repeat: int):
        super().__init__()
        self.down = DownConv(c_in, c_in)
        self.elan = ELAN_S0(c_in, c_out, c_bottleneck, num_repeat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.elan(self.down(x))
