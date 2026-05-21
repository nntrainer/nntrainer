"""E2E tests for TIER 1 layer mappers.

Tests that ConvTranspose2d, DepthwiseConv2d, Upsample2d, MultiheadAttention,
ChannelShuffle, F.interpolate, and F.normalize are correctly mapped through
the full conversion pipeline (trace -> map -> cleanup).

Includes model-level tests using ResNet-like, MobileNet-like, U-Net-like,
and ShuffleNet-like architectures to verify real-world usage patterns.
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

def test_conv2d_transpose():
    """nn.ConvTranspose2d -> conv2dtranspose with correct properties."""
    model = nn.Sequential()
    model.add_module('deconv', nn.ConvTranspose2d(
        16, 8, kernel_size=4, stride=2, padding=1))
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"input": torch.randn(1, 16, 8, 8)})

    deconv = [l for l in result.layers if l.layer_type == "conv2dtranspose"]
    assert len(deconv) == 1, f"Expected 1 conv2dtranspose, got {len(deconv)}"
    props = deconv[0].properties
    assert props["filters"] == 8
    assert props["kernel_size"] == "4,4"
    assert props["stride"] == "2,2"
    assert props["padding"] == "1,1"
    assert deconv[0].has_weight
    assert deconv[0].has_bias
    print("  PASS: ConvTranspose2d mapped correctly")


def test_depthwise_conv2d():
    """nn.Conv2d with groups==in_channels -> depthwiseconv2d."""
    model = nn.Sequential()
    model.add_module('dw', nn.Conv2d(
        32, 32, kernel_size=3, padding=1, groups=32, bias=False))
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"input": torch.randn(1, 32, 16, 16)})

    dw = [l for l in result.layers if l.layer_type == "depthwiseconv2d"]
    assert len(dw) == 1, f"Expected 1 depthwiseconv2d, got {len(dw)}"
    assert dw[0].properties["filters"] == 32
    assert dw[0].properties["kernel_size"] == "3,3"
    assert dw[0].has_weight
    assert not dw[0].has_bias
    print("  PASS: DepthwiseConv2d (groups==in_channels) mapped correctly")


def test_regular_conv2d_with_groups_not_depthwise():
    """nn.Conv2d with groups>1 but groups!=in_channels -> regular conv2d."""
    model = nn.Sequential()
    model.add_module('gconv', nn.Conv2d(32, 64, kernel_size=3, padding=1, groups=4))
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"input": torch.randn(1, 32, 16, 16)})

    # Should NOT be depthwiseconv2d (groups=4 != in_channels=32)
    dw = [l for l in result.layers if l.layer_type == "depthwiseconv2d"]
    assert len(dw) == 0, f"Grouped conv (groups!=in_ch) should NOT be depthwiseconv2d"
    conv = [l for l in result.layers if l.layer_type == "conv2d"]
    assert len(conv) == 1, f"Expected 1 conv2d, got {len(conv)}"
    print("  PASS: Grouped conv (groups!=in_channels) stays as conv2d")


def test_upsample_module():
    """nn.Upsample -> upsample2d with correct mode and scale."""
    model = nn.Sequential()
    model.add_module('up', nn.Upsample(scale_factor=2, mode='bilinear',
                                        align_corners=False))
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"input": torch.randn(1, 16, 8, 8)})

    up = [l for l in result.layers if l.layer_type == "upsample2d"]
    assert len(up) == 1, f"Expected 1 upsample2d, got {len(up)}"
    assert up[0].properties["upsample"] == "bilinear"
    assert up[0].properties["kernel_size"] == "2,2"
    print("  PASS: nn.Upsample mapped correctly")


def test_upsample_nearest():
    """nn.Upsample with nearest mode."""
    model = nn.Sequential()
    model.add_module('up', nn.Upsample(scale_factor=4, mode='nearest'))
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"input": torch.randn(1, 8, 4, 4)})

    up = [l for l in result.layers if l.layer_type == "upsample2d"]
    assert len(up) == 1
    assert up[0].properties["upsample"] == "nearest"
    assert up[0].properties["kernel_size"] == "4,4"
    print("  PASS: nn.Upsample(nearest) mapped correctly")


def test_f_interpolate():
    """F.interpolate -> upsample2d."""
    class InterpModel(nn.Module):
        def forward(self, x):
            return F.interpolate(x, scale_factor=3, mode='nearest')

    model = InterpModel()
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"x": torch.randn(1, 8, 4, 4)})

    up = [l for l in result.layers if l.layer_type == "upsample2d"]
    assert len(up) == 1, f"Expected 1 upsample2d from F.interpolate, got {len(up)}"
    assert up[0].properties["upsample"] == "nearest"
    assert up[0].properties["kernel_size"] == "3,3"
    print("  PASS: F.interpolate mapped correctly")


def test_f_interpolate_bilinear():
    """F.interpolate with bilinear mode."""
    class InterpModel(nn.Module):
        def forward(self, x):
            return F.interpolate(x, scale_factor=2, mode='bilinear',
                                 align_corners=False)

    model = InterpModel()
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"x": torch.randn(1, 8, 4, 4)})

    up = [l for l in result.layers if l.layer_type == "upsample2d"]
    assert len(up) == 1
    assert up[0].properties["upsample"] == "bilinear"
    print("  PASS: F.interpolate(bilinear) mapped correctly")


def test_multihead_attention():
    """nn.MultiheadAttention -> multi_head_attention."""
    class MHAModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.mha = nn.MultiheadAttention(
                embed_dim=64, num_heads=4, batch_first=True)

        def forward(self, x):
            out, _ = self.mha(x, x, x)
            return out

    model = MHAModel()
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"x": torch.randn(1, 10, 64)})

    mha = [l for l in result.layers if l.layer_type == "multi_head_attention"]
    assert len(mha) == 1, f"Expected 1 multi_head_attention, got {len(mha)}"
    props = mha[0].properties
    assert props["num_heads"] == 4
    assert props["projected_key_dim"] == 16  # 64 / 4
    assert props["projected_value_dim"] == 16
    assert props["output_shape"] == 64
    assert mha[0].has_weight
    print("  PASS: nn.MultiheadAttention mapped correctly")


def test_channel_shuffle():
    """nn.ChannelShuffle -> channel_shuffle."""
    model = nn.Sequential()
    model.add_module('shuffle', nn.ChannelShuffle(groups=4))
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"input": torch.randn(1, 16, 8, 8)})

    cs = [l for l in result.layers if l.layer_type == "channel_shuffle"]
    assert len(cs) == 1, f"Expected 1 channel_shuffle, got {len(cs)}"
    assert cs[0].properties["split_number"] == 4
    print("  PASS: nn.ChannelShuffle mapped correctly")


def test_f_normalize():
    """F.normalize -> preprocess_l2norm."""
    class NormModel(nn.Module):
        def forward(self, x):
            return F.normalize(x, p=2, dim=-1)

    model = NormModel()
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"x": torch.randn(1, 10, 64)})

    l2 = [l for l in result.layers if l.layer_type == "preprocess_l2norm"]
    assert len(l2) == 1, f"Expected 1 preprocess_l2norm, got {len(l2)}"
    assert l2[0].properties["epsilon"] == 1e-12
    print("  PASS: F.normalize mapped correctly")


# ============================================================================
# Model-level E2E tests: real architecture patterns
# ============================================================================

class ResNetBlock(nn.Module):
    """Standard ResNet BasicBlock with conv2d + batchnorm + residual."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class MiniResNet(nn.Module):
    """Minimal ResNet-like model (conv -> blocks -> pool -> fc)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.block1 = ResNetBlock(32)
        self.block2 = ResNetBlock(32)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def test_resnet_e2e():
    """ResNet-like model: conv2d + batchnorm + pooling + residual + FC.

    Verifies end-to-end conversion of a vision backbone with:
    - Conv2d layers with correct kernel/stride/padding
    - BatchNorm2d
    - MaxPool2d, AdaptiveAvgPool2d
    - ReLU activations (from F.relu)
    - Residual addition
    - Final FC classifier
    """
    model = MiniResNet(num_classes=10)
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"x": torch.randn(1, 3, 32, 32)})

    result.summary()

    # No unknowns
    assert len(result.unknown_layers) == 0, \
        f"ResNet has unknowns: {[l.layer_type for l in result.unknown_layers]}"

    layer_types = [l.layer_type for l in result.layers]

    # Conv2d: stem(1) + block1(2) + block2(2) = 5
    conv_layers = [l for l in result.layers if l.layer_type == "conv2d"]
    assert len(conv_layers) == 5, f"Expected 5 conv2d layers, got {len(conv_layers)}"

    # BatchNorm: stem(1) + block1(2) + block2(2) = 5
    bn_layers = [l for l in result.layers if l.layer_type == "batch_normalization"]
    assert len(bn_layers) == 5, f"Expected 5 batchnorm layers, got {len(bn_layers)}"

    # Pooling: MaxPool(1) + AdaptiveAvgPool(1) = 2
    pool_layers = [l for l in result.layers if l.layer_type == "pooling2d"]
    assert len(pool_layers) == 2, f"Expected 2 pooling layers, got {len(pool_layers)}"

    # FC: 1 classifier
    fc_layers = [l for l in result.layers if l.layer_type == "fully_connected"]
    assert len(fc_layers) == 1, f"Expected 1 FC layer, got {len(fc_layers)}"
    assert fc_layers[0].properties["unit"] == 10

    # Residual additions: block1(1) + block2(1) = 2
    add_layers = [l for l in result.layers if l.layer_type == "addition"]
    assert len(add_layers) == 2, f"Expected 2 addition layers, got {len(add_layers)}"

    # ReLU activations: stem(1) + block1(2) + block2(2) = 5
    relu_layers = [l for l in result.layers
                   if l.layer_type == "activation"
                   and l.properties.get("activation") == "relu"]
    assert len(relu_layers) == 5, f"Expected 5 relu activations, got {len(relu_layers)}"

    # Weight verification
    for conv in conv_layers:
        assert conv.has_weight, f"Conv {conv.name} should have weight"
        assert not conv.has_bias, f"Conv {conv.name} should have no bias (bias=False)"

    print("  PASS: ResNet E2E conversion")
    print(f"    conv2d: {len(conv_layers)}")
    print(f"    batch_normalization: {len(bn_layers)}")
    print(f"    pooling2d: {len(pool_layers)}")
    print(f"    activation(relu): {len(relu_layers)}")
    print(f"    addition (residual): {len(add_layers)}")
    print(f"    fully_connected: {len(fc_layers)}")
    print(f"    Total layers: {len(result.layers)}")


class MobileNetBlock(nn.Module):
    """MobileNet-style depthwise separable convolution block."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # Depthwise conv (groups == in_channels)
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                            groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        # Pointwise conv (1x1)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.dw(x)))
        x = F.relu(self.bn2(self.pw(x)))
        return x


class MiniMobileNet(nn.Module):
    """Minimal MobileNet-like model with depthwise separable convolutions."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.block1 = MobileNetBlock(16, 32, stride=2)
        self.block2 = MobileNetBlock(32, 64, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def test_mobilenet_e2e():
    """MobileNet-like model: depthwise separable conv + BN + pooling + FC.

    Verifies:
    - Depthwise conv (groups==in_channels) -> depthwiseconv2d
    - Pointwise conv (1x1) -> conv2d
    - BatchNorm2d
    - AdaptiveAvgPool2d
    """
    model = MiniMobileNet(num_classes=10)
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"x": torch.randn(1, 3, 32, 32)})

    result.summary()

    assert len(result.unknown_layers) == 0, \
        f"MobileNet has unknowns: {[l.layer_type for l in result.unknown_layers]}"

    # Depthwise: block1(1) + block2(1) = 2
    dw_layers = [l for l in result.layers if l.layer_type == "depthwiseconv2d"]
    assert len(dw_layers) == 2, f"Expected 2 depthwiseconv2d, got {len(dw_layers)}"

    # Regular conv: stem(1) + block1_pw(1) + block2_pw(1) = 3
    conv_layers = [l for l in result.layers if l.layer_type == "conv2d"]
    assert len(conv_layers) == 3, f"Expected 3 conv2d, got {len(conv_layers)}"

    # BN: stem(1) + block1(2) + block2(2) = 5
    bn_layers = [l for l in result.layers if l.layer_type == "batch_normalization"]
    assert len(bn_layers) == 5, f"Expected 5 batchnorm, got {len(bn_layers)}"

    print("  PASS: MobileNet E2E conversion")
    print(f"    depthwiseconv2d: {len(dw_layers)}")
    print(f"    conv2d: {len(conv_layers)}")
    print(f"    batch_normalization: {len(bn_layers)}")
    print(f"    Total layers: {len(result.layers)}")


class MiniUNet(nn.Module):
    """Minimal U-Net-like model with encoder-decoder + skip connections.

    Tests ConvTranspose2d (decoder) and Upsample/interpolate.
    """
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU())

        # Decoder with ConvTranspose2d
        self.dec2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU())  # 32 = 16+16 (skip)

        self.out_conv = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)

        # Decoder
        d2 = F.relu(self.dec2(e2))
        d2 = torch.cat([d2, e1], dim=1)  # skip connection
        d1 = self.dec1(d2)
        return self.out_conv(d1)


def test_unet_e2e():
    """U-Net-like model: ConvTranspose2d decoder + skip connections.

    Verifies:
    - ConvTranspose2d -> conv2dtranspose
    - torch.cat -> concat (skip connections)
    - Encoder-decoder architecture with correct layer counts
    """
    model = MiniUNet()
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"x": torch.randn(1, 3, 32, 32)})

    result.summary()

    assert len(result.unknown_layers) == 0, \
        f"U-Net has unknowns: {[l.layer_type for l in result.unknown_layers]}"

    # ConvTranspose2d: dec2(1)
    deconv = [l for l in result.layers if l.layer_type == "conv2dtranspose"]
    assert len(deconv) == 1, f"Expected 1 conv2dtranspose, got {len(deconv)}"
    assert deconv[0].properties["stride"] == "2,2"

    # Regular conv: enc1(1) + enc2(1) + dec1(1) + out(1) = 4
    conv_layers = [l for l in result.layers if l.layer_type == "conv2d"]
    assert len(conv_layers) == 4, f"Expected 4 conv2d, got {len(conv_layers)}"

    # Concat: 1 (skip connection)
    concat_layers = [l for l in result.layers if l.layer_type == "concat"]
    assert len(concat_layers) == 1, f"Expected 1 concat, got {len(concat_layers)}"

    print("  PASS: U-Net E2E conversion")
    print(f"    conv2dtranspose: {len(deconv)}")
    print(f"    conv2d: {len(conv_layers)}")
    print(f"    concat: {len(concat_layers)}")
    print(f"    Total layers: {len(result.layers)}")


class MiniShuffleNetBlock(nn.Module):
    """ShuffleNet-style block with channel shuffle."""
    def __init__(self, channels, groups=4):
        super().__init__()
        self.groups = groups
        mid = channels // 4
        self.compress = nn.Conv2d(channels, mid, 1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.shuffle = nn.ChannelShuffle(groups)
        self.dw = nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.expand = nn.Conv2d(mid, channels, 1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.compress(x)))
        out = self.shuffle(out)
        out = self.bn2(self.dw(out))
        out = self.bn3(self.expand(out))
        return F.relu(out + residual)


class MiniShuffleNet(nn.Module):
    """Minimal ShuffleNet-like model."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(),
        )
        self.block1 = MiniShuffleNetBlock(16, groups=4)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def test_shufflenet_e2e():
    """ShuffleNet-like model: channel_shuffle + depthwiseconv2d.

    Verifies:
    - ChannelShuffle -> channel_shuffle with correct split_number
    - Depthwise conv in ShuffleNet block
    - Grouped convolution (groups>1 but groups!=in_ch) -> conv2d
    """
    model = MiniShuffleNet(num_classes=10)
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"x": torch.randn(1, 3, 32, 32)})

    result.summary()

    assert len(result.unknown_layers) == 0, \
        f"ShuffleNet has unknowns: {[l.layer_type for l in result.unknown_layers]}"

    # ChannelShuffle
    cs = [l for l in result.layers if l.layer_type == "channel_shuffle"]
    assert len(cs) == 1, f"Expected 1 channel_shuffle, got {len(cs)}"
    assert cs[0].properties["split_number"] == 4

    # Depthwise conv
    dw = [l for l in result.layers if l.layer_type == "depthwiseconv2d"]
    assert len(dw) == 1, f"Expected 1 depthwiseconv2d, got {len(dw)}"

    print("  PASS: ShuffleNet E2E conversion")
    print(f"    channel_shuffle: {len(cs)}")
    print(f"    depthwiseconv2d: {len(dw)}")
    print(f"    Total layers: {len(result.layers)}")


class EmbeddingWithL2Norm(nn.Module):
    """Embedding model that uses F.normalize for L2 normalization."""
    def __init__(self, vocab_size=100, embed_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        e = self.embed(x)
        h = self.proj(e)
        return F.normalize(h, p=2, dim=-1)


def test_l2norm_embedding_e2e():
    """Embedding + FC + F.normalize pipeline.

    Verifies F.normalize -> preprocess_l2norm in a realistic context
    (sentence embedding / retrieval model).
    """
    model = EmbeddingWithL2Norm(vocab_size=100, embed_dim=64)
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"x": torch.randint(0, 100, (1, 16))})

    assert len(result.unknown_layers) == 0, \
        f"L2Norm model has unknowns: {[l.layer_type for l in result.unknown_layers]}"

    l2 = [l for l in result.layers if l.layer_type == "preprocess_l2norm"]
    assert len(l2) == 1, f"Expected 1 preprocess_l2norm, got {len(l2)}"

    emb = [l for l in result.layers if l.layer_type == "embedding_layer"]
    assert len(emb) == 1

    fc = [l for l in result.layers if l.layer_type == "fully_connected"]
    assert len(fc) == 1

    print("  PASS: L2Norm embedding E2E conversion")
    print(f"    embedding_layer: {len(emb)}")
    print(f"    fully_connected: {len(fc)}")
    print(f"    preprocess_l2norm: {len(l2)}")


class SuperResolutionModel(nn.Module):
    """Super-resolution model using F.interpolate for upsampling."""
    def __init__(self):
        super().__init__()
        self.feat = nn.Conv2d(3, 32, 3, padding=1)
        self.res1 = nn.Conv2d(32, 32, 3, padding=1)
        self.out = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.feat(x))
        x = F.relu(self.res1(x))
        x = F.interpolate(x, scale_factor=4, mode='nearest')
        return self.out(x)


def test_super_resolution_e2e():
    """Super-resolution model with F.interpolate.

    Verifies F.interpolate -> upsample2d in a real upscaling model.
    """
    model = SuperResolutionModel()
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"x": torch.randn(1, 3, 16, 16)})

    assert len(result.unknown_layers) == 0, \
        f"SR model has unknowns: {[l.layer_type for l in result.unknown_layers]}"

    up = [l for l in result.layers if l.layer_type == "upsample2d"]
    assert len(up) == 1
    assert up[0].properties["kernel_size"] == "4,4"

    conv_layers = [l for l in result.layers if l.layer_type == "conv2d"]
    assert len(conv_layers) == 3

    print("  PASS: Super-resolution E2E conversion")
    print(f"    conv2d: {len(conv_layers)}")
    print(f"    upsample2d: {len(up)}")


class TransformerWithStdMHA(nn.Module):
    """Standard Transformer encoder using nn.MultiheadAttention."""
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(32, d_model)
        self.layers_ = nn.ModuleList()
        for _ in range(num_layers):
            self.layers_.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(d_model, nhead, batch_first=True),
                'norm1': nn.LayerNorm(d_model),
                'ff': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.ReLU(),
                    nn.Linear(d_model * 4, d_model),
                ),
                'norm2': nn.LayerNorm(d_model),
            }))

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers_:
            attn_out, _ = layer['attn'](x, x, x)
            x = layer['norm1'](x + attn_out)
            ff_out = layer['ff'](x)
            x = layer['norm2'](x + ff_out)
        return x


def test_transformer_mha_e2e():
    """Standard Transformer with nn.MultiheadAttention.

    Verifies:
    - nn.MultiheadAttention -> multi_head_attention
    - LayerNorm
    - FC layers in FFN
    - Residual additions
    """
    model = TransformerWithStdMHA(d_model=64, nhead=4, num_layers=2)
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"x": torch.randn(1, 10, 32)})

    result.summary()

    assert len(result.unknown_layers) == 0, \
        f"Transformer has unknowns: {[l.layer_type for l in result.unknown_layers]}"

    mha = [l for l in result.layers if l.layer_type == "multi_head_attention"]
    assert len(mha) == 2, f"Expected 2 MHA layers, got {len(mha)}"
    for m in mha:
        assert m.properties["num_heads"] == 4
        assert m.properties["projected_key_dim"] == 16

    ln = [l for l in result.layers if l.layer_type == "layer_normalization"]
    assert len(ln) == 4, f"Expected 4 LayerNorm, got {len(ln)}"

    # FC: embed(1) + FFN(2*2=4) = 5
    fc = [l for l in result.layers if l.layer_type == "fully_connected"]
    assert len(fc) == 5, f"Expected 5 FC, got {len(fc)}"

    print("  PASS: Transformer+MHA E2E conversion")
    print(f"    multi_head_attention: {len(mha)}")
    print(f"    layer_normalization: {len(ln)}")
    print(f"    fully_connected: {len(fc)}")
    print(f"    Total layers: {len(result.layers)}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TIER 1 LAYER MAPPER TESTS")
    print("=" * 70)

    print("\n--- Unit Tests ---")
    test_conv2d_transpose()
    test_depthwise_conv2d()
    test_regular_conv2d_with_groups_not_depthwise()
    test_upsample_module()
    test_upsample_nearest()
    test_f_interpolate()
    test_f_interpolate_bilinear()
    test_multihead_attention()
    test_channel_shuffle()
    test_f_normalize()

    print("\n--- Model-Level E2E Tests ---")
    test_resnet_e2e()
    test_mobilenet_e2e()
    test_unet_e2e()
    test_shufflenet_e2e()
    test_l2norm_embedding_e2e()
    test_super_resolution_e2e()
    test_transformer_mha_e2e()

    print("\n" + "=" * 70)
    print("ALL TIER 1 LAYER MAPPER TESTS PASSED!")
    print("=" * 70)
