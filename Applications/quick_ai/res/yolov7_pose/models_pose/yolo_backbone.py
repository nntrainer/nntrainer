import torch
from torch import nn

from models_pose.fuse import IFuse
from models_pose.head.keypoint import RTMCC
from models_pose.head.reid import ReID
from models_pose.yolo_v2 import create_predefined_YOLOv7_backbone


class IReIDYOLOv7(IFuse):
    @property
    def c_ends(self):
        return [self.backbone.features.ends[-1].c_out]

    @property
    def output_stage_num(self) -> int:
        return 1

    @property
    def c_ends_const(self):
        raise NotImplementedError()

    def __init__(
        self,
        ch=3,
        nc=None,
        img_size=640,
        embed_dim=256,
        backbone_name="YOLOv7-tiny",
        widen_factor=1.0,
        featuremap_ratio=32,
        **kwargs,
    ):
        super().__init__()
        assert nc is not None

        self.ch = ch
        self.stride = torch.tensor([1])
        self.widen_factor = widen_factor
        self.backbone = create_predefined_YOLOv7_backbone(
            ch,
            backbone_name,
            widen_factor=widen_factor,
            output_stage_num=self.output_stage_num,
            disable_aux=True,
        )
        self._verify_end()
        self.head, self.head_feat = self._create_end(
            nc,
            self.c_ends,
            img_size=img_size,
            featuremap_ratio=featuremap_ratio,
            embed_dim=embed_dim,
            nkpt=kwargs["nkpt"],
        )
        self.model = nn.Sequential(self.backbone, self.head, self.head_feat)

    def _create_end(self, nc: int, c_ends, img_size: int = None, featuremap_ratio: int = None, embed_dim=256, nkpt=0):
        return (
            RTMCC(nc=nc, ch=c_ends, img_size=img_size, featuremap_ratio=featuremap_ratio, nkpt=nkpt),
            ReID(nc, embed_dim=embed_dim, ch=c_ends, img_size=img_size, featuremap_ratio=featuremap_ratio),
        )

    def _verify_end(self):
        for actual, expected in zip(self.c_ends, self.c_ends_const):
            assert actual == int(expected * self.widen_factor), f"{actual}=={int(expected * self.widen_factor)}"

    def forward_no_aug(self, x):
        x0, x1 = self.backbone(x)
        x0 = self.head(x0[-1])
        x1 = self.head_feat(x1[-1])
        return x0, x1

    def forward(self, x, augment=False, profile=False):
        return self.forward_no_aug(x)

    def fuse(self):
        for module in self.model.modules():
            if isinstance(module, IFuse):
                module.fuse()
        return self


class YOLOv7ReIDtiny(IReIDYOLOv7):
    @property
    def c_ends_const(self):
        return [512]

    def __init__(self, ch=3, nc=None, img_size=640, featuremap_ratio=32, **kwargs):
        super().__init__(
            ch,
            nc,
            img_size,
            backbone_name="YOLOv7-tiny",
            featuremap_ratio=featuremap_ratio,
            **kwargs,
        )
