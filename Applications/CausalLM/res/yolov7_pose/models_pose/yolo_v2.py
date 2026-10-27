import torch
import torch.nn as nn

from models_pose.backbone.csp import BlockS2, CSPTiny
from models_pose.common_v2 import (
    Conv,
    DownConv,
    DownSampleRouteS0,
    ELAN_S0,
    MPBlock,
    SPPCSPCTiny,
    UpSampleBlock,
)


class IFeatureYOLOv7Tiny(nn.Module):
    @property
    def num_features(self) -> int:
        return 3

    def __init__(self, ends: nn.ModuleList, c_block: int, num_repeat: int, num_ends: int):
        super().__init__()
        c_stem = ends[0].c_out
        self.num_ends = num_ends

        net = ends[-1]
        assert (net.c_out // 2) == (c_stem * 8)
        net = SPPCSPCTiny(net.c_out, net.c_out // 2)
        blocks = [net]
        self.spp = net

        ends_reversed = list(reversed(ends))
        feature_up = []
        for block, (sw, se) in zip(ends_reversed[1 : self.num_features], ((4, 2), (2, 1))):
            net = BlockS2(UpSampleBlock, ELAN_S0, net.c_out, c_stem * sw, c_block * se, num_repeat, route=block)
            feature_up.append(net)
        blocks += feature_up
        self.feature_up = nn.ModuleList(feature_up)

        blocks_reversed = list(reversed(blocks))
        feature_down = []
        # NOTE(port): the deepest down-route concatenates the SPP output (8*c_stem
        # channels). For DownSampleRouteS0's 2-way concat to balance, that stage
        # must target 16*c_stem, so its (sw, se) is (16, 8) -- this also makes the
        # final end channel 2 * 8*c_stem = 16*c_stem = int(512*widen), matching
        # YOLOv7ReIDtiny.c_ends_const. (Reconstructed from channel constraints.)
        for block, (sw, se) in zip(blocks_reversed[1 : self.num_features], ((4, 2), (16, 8))):
            net = BlockS2(DownSampleRouteS0, ELAN_S0, net.c_out, c_stem * sw, c_block * se, num_repeat, route=block)
            feature_down.append(net)
        blocks += feature_down
        self.feature_down = nn.ModuleList(feature_down)

        self.ends = nn.ModuleList([Conv(block.c_out, block.c_out * 2, 3, 1) for block in blocks[-self.num_ends :]])

    def forward(self, x: torch.Tensor, features):
        x = self.spp(x)
        xs = [x]

        features_reversed = list(reversed(features))
        for block, x_route in zip(self.feature_up, features_reversed[1 : self.num_features]):
            x = block(x, x_route)
            xs.append(x)

        xs_reversed = list(reversed(xs))
        for block, x_route in zip(self.feature_down, xs_reversed[1 : self.num_features]):
            x = block(x, x_route)
            xs.append(x)

        return [block(x) for block, x in zip(self.ends, xs[-self.num_ends :])]


class YOLOv7TinyBase(nn.Module):
    @property
    def _num_repeat(self) -> int:
        return 2

    def __init__(self, c_in: int, widen_factor: float, num_ends: int):
        super().__init__()
        c_stem = int(round(32 * widen_factor))
        c_block = int(round(32 * widen_factor))
        self.backbone = CSPTiny(c_in, c_stem, c_block, self._num_repeat)
        self.features = IFeatureYOLOv7Tiny(self.backbone.blocks, c_block, self._num_repeat, num_ends)
        self.features_feat = IFeatureYOLOv7Tiny(self.backbone.blocks, c_block, self._num_repeat, num_ends)

    def forward(self, x: torch.Tensor):
        x, features = self.backbone(x, return_intermediate_nodes=True)
        x_pose = self.features(x, features)
        x_feat = self.features_feat(x, features)
        return x_pose, x_feat


def create_predefined_YOLOv7_backbone(
    ch, model_name="YOLOv7-tiny", widen_factor=1.0, output_stage_num=1, disable_aux=False
):
    if model_name != "YOLOv7-tiny":
        raise NotImplementedError(f"models_pose only includes YOLOv7-tiny, got {model_name!r}")
    return YOLOv7TinyBase(ch, widen_factor, output_stage_num)
