"""E2E tests for TIER 2 layer mappers.

Tests that GroupNorm, InstanceNorm, topk, and argsort are correctly mapped
through the full conversion pipeline (trace -> map -> cleanup).

Includes model-level tests using architectures that exercise these operations.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from decomposer import AdaptiveConverter


# ============================================================================
# Unit tests: individual layer mapper correctness
# ============================================================================

def test_group_norm():
    """nn.GroupNorm -> group_normalization with correct properties."""
    model = nn.Sequential()
    model.add_module('gn', nn.GroupNorm(num_groups=4, num_channels=16))
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"input": torch.randn(1, 16, 8, 8)})

    gn = [l for l in result.layers if l.layer_type == "group_normalization"]
    assert len(gn) == 1, f"Expected 1 group_normalization, got {len(gn)}"
    props = gn[0].properties
    assert props["num_groups"] == 4
    assert "epsilon" in props
    assert gn[0].has_weight
    assert gn[0].has_bias
    print("  PASS: GroupNorm mapped correctly")


def test_group_norm_no_affine():
    """nn.GroupNorm with affine=False."""
    model = nn.Sequential()
    model.add_module('gn', nn.GroupNorm(num_groups=2, num_channels=8,
                                        affine=False))
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"input": torch.randn(1, 8, 4, 4)})

    gn = [l for l in result.layers if l.layer_type == "group_normalization"]
    assert len(gn) == 1
    assert not gn[0].has_weight
    assert not gn[0].has_bias
    print("  PASS: GroupNorm(affine=False) mapped correctly")


def test_instance_norm_2d():
    """nn.InstanceNorm2d -> instance_normalization."""
    model = nn.Sequential()
    model.add_module('in_', nn.InstanceNorm2d(16, affine=True))
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"input": torch.randn(1, 16, 8, 8)})

    ins = [l for l in result.layers if l.layer_type == "instance_normalization"]
    assert len(ins) == 1, f"Expected 1 instance_normalization, got {len(ins)}"
    assert ins[0].has_weight
    assert ins[0].has_bias
    print("  PASS: InstanceNorm2d mapped correctly")


def test_instance_norm_1d():
    """nn.InstanceNorm1d -> instance_normalization."""
    model = nn.Sequential()
    model.add_module('in_', nn.InstanceNorm1d(32, affine=True))
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"input": torch.randn(1, 32, 16)})

    ins = [l for l in result.layers if l.layer_type == "instance_normalization"]
    assert len(ins) == 1, f"Expected 1 instance_normalization, got {len(ins)}"
    assert ins[0].has_weight
    print("  PASS: InstanceNorm1d mapped correctly")


def test_instance_norm_no_affine():
    """nn.InstanceNorm2d with affine=False."""
    model = nn.Sequential()
    model.add_module('in_', nn.InstanceNorm2d(8))
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"input": torch.randn(1, 8, 4, 4)})

    ins = [l for l in result.layers if l.layer_type == "instance_normalization"]
    assert len(ins) == 1
    assert not ins[0].has_weight
    assert not ins[0].has_bias
    print("  PASS: InstanceNorm2d(affine=False) mapped correctly")


def test_topk_function():
    """torch.topk -> topk layer."""
    class TopKModel(nn.Module):
        def forward(self, x):
            values, indices = torch.topk(x, k=3, dim=-1)
            return values

    model = TopKModel()
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"x": torch.randn(1, 10)})

    topk = [l for l in result.layers if l.layer_type == "topk"]
    assert len(topk) >= 1, f"Expected >= 1 topk, got {len(topk)}"
    print("  PASS: torch.topk mapped correctly")


def test_argsort_method():
    """Tensor.argsort -> argsort layer."""
    class ArgsortModel(nn.Module):
        def forward(self, x):
            return x.argsort(dim=-1)

    model = ArgsortModel()
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"x": torch.randn(1, 10)})

    argsort = [l for l in result.layers if l.layer_type == "argsort"]
    assert len(argsort) >= 1, f"Expected >= 1 argsort, got {len(argsort)}"
    print("  PASS: Tensor.argsort mapped correctly")


# ============================================================================
# Model-level E2E tests
# ============================================================================

class StyleTransferBlock(nn.Module):
    """Style transfer block using InstanceNorm2d."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = F.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return F.relu(out + residual)


class MiniStyleTransferNet(nn.Module):
    """Mini style transfer network with InstanceNorm."""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
        )
        self.block = StyleTransferBlock(32)
        self.dec = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, x):
        x = self.enc(x)
        x = self.block(x)
        return self.dec(x)


def test_style_transfer_e2e():
    """Style transfer model with InstanceNorm2d."""
    model = MiniStyleTransferNet()
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"x": torch.randn(1, 3, 32, 32)})

    result.summary()

    assert len(result.unknown_layers) == 0, \
        f"Style transfer has unknowns: {[l.layer_type for l in result.unknown_layers]}"

    ins = [l for l in result.layers if l.layer_type == "instance_normalization"]
    assert len(ins) == 3, f"Expected 3 instance_normalization, got {len(ins)}"

    conv = [l for l in result.layers if l.layer_type == "conv2d"]
    assert len(conv) == 4, f"Expected 4 conv2d, got {len(conv)}"

    add = [l for l in result.layers if l.layer_type == "addition"]
    assert len(add) == 1, f"Expected 1 addition (residual), got {len(add)}"

    print("  PASS: Style transfer E2E conversion")
    print(f"    instance_normalization: {len(ins)}")
    print(f"    conv2d: {len(conv)}")
    print(f"    addition: {len(add)}")
    print(f"    Total layers: {len(result.layers)}")


class DiffusionUNetBlock(nn.Module):
    """Diffusion model block with GroupNorm (common in Stable Diffusion)."""
    def __init__(self, channels, num_groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        residual = x
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + residual


class MiniDiffusionUNet(nn.Module):
    """Mini diffusion U-Net with GroupNorm."""
    def __init__(self, channels=32, num_groups=8):
        super().__init__()
        self.conv_in = nn.Conv2d(3, channels, 3, padding=1)
        self.block1 = DiffusionUNetBlock(channels, num_groups)
        self.block2 = DiffusionUNetBlock(channels, num_groups)
        self.norm_out = nn.GroupNorm(num_groups, channels)
        self.conv_out = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.block1(x)
        x = self.block2(x)
        x = F.silu(self.norm_out(x))
        return self.conv_out(x)


def test_diffusion_unet_e2e():
    """Diffusion U-Net model with GroupNorm + SiLU."""
    model = MiniDiffusionUNet(channels=32, num_groups=8)
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"x": torch.randn(1, 3, 32, 32)})

    result.summary()

    assert len(result.unknown_layers) == 0, \
        f"Diffusion UNet has unknowns: {[l.layer_type for l in result.unknown_layers]}"

    gn = [l for l in result.layers if l.layer_type == "group_normalization"]
    assert len(gn) == 5, f"Expected 5 group_normalization, got {len(gn)}"
    for g in gn:
        assert g.properties["num_groups"] == 8

    conv = [l for l in result.layers if l.layer_type == "conv2d"]
    assert len(conv) == 6, f"Expected 6 conv2d, got {len(conv)}"

    # SiLU activations: block1(2) + block2(2) + out(1) = 5
    swish = [l for l in result.layers
             if l.layer_type == "activation"
             and l.properties.get("activation") == "swish"]
    assert len(swish) == 5, f"Expected 5 SiLU/swish, got {len(swish)}"

    add = [l for l in result.layers if l.layer_type == "addition"]
    assert len(add) == 2, f"Expected 2 addition (residual), got {len(add)}"

    print("  PASS: Diffusion UNet E2E conversion")
    print(f"    group_normalization: {len(gn)}")
    print(f"    conv2d: {len(conv)}")
    print(f"    activation(swish): {len(swish)}")
    print(f"    addition: {len(add)}")
    print(f"    Total layers: {len(result.layers)}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TIER 2 LAYER MAPPER TESTS")
    print("=" * 70)

    print("\n--- Unit Tests ---")
    test_group_norm()
    test_group_norm_no_affine()
    test_instance_norm_2d()
    test_instance_norm_1d()
    test_instance_norm_no_affine()
    test_topk_function()
    test_argsort_method()

    print("\n--- Model-Level E2E Tests ---")
    test_style_transfer_e2e()
    test_diffusion_unet_e2e()

    print("\n" + "=" * 70)
    print("ALL TIER 2 LAYER MAPPER TESTS PASSED!")
    print("=" * 70)
