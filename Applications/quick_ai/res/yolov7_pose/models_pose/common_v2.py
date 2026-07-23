import torch
import torch.nn as nn
import torch.nn.functional as F

from models_pose.common import ReOrg
from models_pose.fuse import IFuse


def fuse_conv_and_bn(conv, bn):
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def _iter_blocks(blocks, x):
    for block in blocks:
        x = block(x)
        yield x


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class MP(nn.MaxPool2d):
    def __init__(self, k: int = 2, **kwargs):
        super().__init__(k, stride=k, **kwargs)

    def forward(self, x: torch.Tensor, *args: torch.Tensor) -> torch.Tensor:
        return super().forward(x)


class MPBlock(MP):
    def __init__(self, *args, **kwargs):
        super().__init__()


class Conv(IFuse):
    @property
    def c_out(self) -> int:
        return self.conv.out_channels

    def __init__(self, c_in, c_out, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self._c_out = c_out
        self.conv = nn.Conv2d(c_in, c_out, k, s, autopad(k, p), groups=g, bias=True)
        self.bn = nn.BatchNorm2d(c_out, eps=1e-5)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x: torch.Tensor, *args: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor, *args: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))

    def fuse_operations(self):
        self.conv = fuse_conv_and_bn(self.conv, self.bn)
        delattr(self, "bn")


class DownConv(Conv):
    def __init__(self, c_in: int, c_out: int, **kwargs):
        super().__init__(c_in, c_out, k=3, s=2)


class StemS0(nn.Sequential):
    @property
    def c_out(self):
        return self._c_out

    def __init__(self, c_in: int, c_out: int):
        super().__init__(Conv(c_in, c_out, 3, s=2))
        self._c_out = c_out


class SPPCSPCTiny(nn.Module):
    @property
    def c_out(self):
        return self.cv4.c_out

    def __init__(self, c_in: int, c_out: int, e: float = 0.5, k=None):
        super().__init__()
        if k is None:
            k = (5, 9, 13)
        c_hidden = int(2 * c_out * e)
        self.cv1 = Conv(c_in, c_hidden, 1, 1)
        self.cv2 = Conv(c_in, c_hidden, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(x, stride=1, padding=x // 2) for x in k])
        self.cv3 = Conv(4 * c_hidden, c_hidden, 1, 1)
        self.cv4 = Conv(2 * c_hidden, c_out, 1, 1)

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.cv3(torch.cat([x1] + [m(x1) for m in self.m], 1))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))


class ELAN_S0(nn.Module):
    def _sample_block(self, xs):
        return xs

    def _create_block(self, c_out: int, n: int, prev=Conv):
        for _ in range(n):
            conv = Conv(prev.c_out, c_out, 3, 1)
            yield conv
            prev = conv

    def _c_in_last(self) -> int:
        convs = [self.conv1, self.conv2] + list(self._sample_block(self.conv_blocks))
        return sum(c.c_out for c in convs)

    def _c_out_block(self) -> int:
        return self.conv2.c_out

    @property
    def c_out(self):
        return self.last_conv.c_out

    def __init__(self, c_in: int, c_out: int, c_bottleneck: int, n_blocks: int):
        super().__init__()
        self.conv1 = Conv(c_in, c_bottleneck, 1, 1)
        self.conv2 = Conv(c_in, c_bottleneck, 1, 1)
        self.conv_blocks = nn.ModuleList(self._create_block(self._c_out_block(), n_blocks, self.conv2))
        self.last_conv = Conv(self._c_in_last(), c_out, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x = self.conv2(x)
        xs_for_cat = [x1, x] + [v for v in _iter_blocks(self.conv_blocks, x)]
        return self.last_conv(torch.cat(xs_for_cat[::-1], dim=1))


class DownSampleRouteS0(nn.Module):
    @property
    def c_out(self):
        return self._c_out

    def __init__(self, c_in: int, c_out: int, route: nn.Module = None):
        super().__init__()
        self._c_out = c_out
        c_hidden = c_out // 2
        self.conv = Conv(c_in, c_hidden, 3, 2)
        assert c_out == sum(x.c_out for x in [self.conv, route] if x)

    def forward(self, x: torch.Tensor, *args: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.conv(x), *args), dim=1)


class UpSampleBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, route: nn.Module = None):
        super().__init__()
        c_hidden = c_out // 2
        self.conv1 = Conv(c_in, c_hidden, 1, 1)
        self.up = nn.Upsample(None, 2, "nearest")
        self.conv2 = Conv(route.c_out, c_hidden, 1, 1)

    def forward(self, x: torch.Tensor, x_route: torch.Tensor) -> torch.Tensor:
        x = self.up(self.conv1(x))
        x1 = self.conv2(x_route)
        return torch.cat((x1, x), dim=1)


class RepConv(IFuse):
    @property
    def c_out(self) -> int:
        return self.rbr_dense.c_out

    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, p=None, g: int = 1, act=True):
        super().__init__()
        assert k == 3
        assert autopad(k, p) == 1
        assert g == 1
        padding_11 = autopad(k, p) - k // 2
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.rbr_dense = Conv(c_in, c_out, k, s, autopad(k, p), act=False)
        self.rbr_1x1 = Conv(c_in, c_out, 1, s, padding_11, act=False)
        self.rbr_identity = nn.BatchNorm2d(c_in) if c_out == c_in and s == 1 else self.identity

    def identity(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((1, self.c_out, 1, 1), dtype=x.dtype, device=x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.rbr_dense(x) + self.rbr_1x1(x) + self.rbr_identity(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.rbr_reparam(x))

    def fuse_conv_bn(self, conv, bn):
        return fuse_conv_and_bn(conv, bn)

    def fuse_operations(self):
        self.rbr_dense = fuse_conv_and_bn(self.rbr_dense.conv, self.rbr_dense.bn)
        self.rbr_1x1 = fuse_conv_and_bn(self.rbr_1x1.conv, self.rbr_1x1.bn)

        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = F.pad(self.rbr_1x1.weight, [1, 1, 1, 1])

        if isinstance(self.rbr_identity, nn.Module):
            identity_conv_1x1 = nn.Conv2d(
                self.rbr_dense.in_channels,
                self.rbr_dense.out_channels,
                kernel_size=1,
                bias=False,
            )
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)
            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = F.pad(identity_conv_1x1.weight, [1, 1, 1, 1])
        else:
            bias_identity_expanded = torch.nn.Parameter(torch.zeros_like(rbr_1x1_bias))
            weight_identity_expanded = torch.nn.Parameter(torch.zeros_like(weight_1x1_expanded))

        self.rbr_dense.weight = torch.nn.Parameter(
            self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded
        )
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)
        self.rbr_reparam = self.rbr_dense

        delattr(self, "rbr_identity")
        delattr(self, "rbr_1x1")
        delattr(self, "rbr_dense")
