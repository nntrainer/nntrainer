#!/usr/bin/env python3
"""
@file convert_weights.py
@brief Convert FastViTKeyword PyTorch checkpoint into a single nntrainer-compatible
safetensors file, with tensor names matching the nntrainer model's weight names.

The model is fused (reparameterized) before export so all RepConv/BN pairs become
single biased convolutions. Layer_scale gamma values are folded into the preceding
conv weights/biases at conversion time (W' = gamma * W, b' = gamma * b), eliminating
the need for multiply layers in the nntrainer graph.

Naming scheme (matches nntrainer Weight::getName()):
  conv2d weight : "{layer}/conv:filter"        shape [out, in, kh, kw]
  conv2d bias   : "{layer}/conv:bias"          shape [1, C, 1, 1]
  batchnorm     : "{layer}:gamma"               shape [C]
  batchnorm bias: "{layer}:beta"                shape [C]
  batchnorm mean: "{layer}:moving_mean"          shape [C]
  batchnorm var : "{layer}:moving_variance"      shape [C]
  layernorm     : "{layer}:gamma"               shape [C]
  layernorm bias: "{layer}:beta"                shape [C]

Usage:
  python convert_weights.py --weights /path/to/ckpt.pth --out ../res/fastvit_keyword.safetensors
"""
import os
import json
import struct
import argparse
import sys
import types

# Mock overrides decorator to disable method signature validation
overrides_stub = types.ModuleType('overrides')
def overrides_decorator(method=None, *args, **kwargs):
    if callable(method):
        return method
    return lambda f: f
overrides_stub.overrides = overrides_decorator
sys.modules['overrides'] = overrides_stub

import numpy as np
import torch

class DummyModule1(torch.nn.Module):
    def __init__(self, *args, **kwargs): super().__init__()

class DummyModule2(torch.nn.Module):
    def __init__(self, *args, **kwargs): super().__init__()

class DummyModule3(torch.nn.Module):
    def __init__(self, *args, **kwargs): super().__init__()

class DummyModule4(torch.nn.Module):
    def __init__(self, *args, **kwargs): super().__init__()

class DummyModule5(torch.nn.Module):
    def __init__(self, *args, **kwargs): super().__init__()

class DummyModule6(torch.nn.Module):
    def __init__(self, *args, **kwargs): super().__init__()

class DummyModule7(torch.nn.Module):
    def __init__(self, *args, **kwargs): super().__init__()

class DummyModule(torch.nn.Module):
    def __init__(self, *args, **kwargs): super().__init__()

# Mock transformers to prevent version and import conflicts
transformers_stub = types.ModuleType('transformers')
transformers_stub.__path__ = []
sys.modules['transformers'] = transformers_stub

models_stub = types.ModuleType('transformers.models')
models_stub.__path__ = []
sys.modules['transformers.models'] = models_stub

bert_stub = types.ModuleType('transformers.models.bert')
bert_stub.__path__ = []
bert_stub.BertConfig = DummyModule
bert_stub.BertLMHeadModel = DummyModule2
bert_stub.BertTokenizer = DummyModule3
sys.modules['transformers.models.bert'] = bert_stub

modeling_bert_stub = types.ModuleType('transformers.models.bert.modeling_bert')
modeling_bert_stub.BertModel = DummyModule1
modeling_bert_stub.BertLMHeadModel = DummyModule2
modeling_bert_stub.BertAttention = DummyModule3
modeling_bert_stub.BertEncoder = DummyModule4
modeling_bert_stub.BertLayer = DummyModule5
modeling_bert_stub.BertPreTrainedModel = DummyModule6
modeling_bert_stub.BertSelfAttention = DummyModule7
sys.modules['transformers.models.bert.modeling_bert'] = modeling_bert_stub

sys.modules['transformers.models.detr'] = types.ModuleType('transformers.models.detr')
detr_modeling_stub = types.ModuleType('transformers.models.detr.modeling_detr')
detr_modeling_stub.DetrAttention = DummyModule
sys.modules['transformers.models.detr.modeling_detr'] = detr_modeling_stub

# Mock other transformers submodules using DummyModule
attn_mask_stub = types.ModuleType('transformers.modeling_attn_mask_utils')
attn_mask_stub._prepare_4d_attention_mask_for_sdpa = DummyModule
attn_mask_stub._prepare_4d_causal_attention_mask_for_sdpa = DummyModule
sys.modules['transformers.modeling_attn_mask_utils'] = attn_mask_stub

modeling_outputs_stub = types.ModuleType('transformers.modeling_outputs')
modeling_outputs_stub.BaseModelOutputWithPastAndCrossAttentions = DummyModule
modeling_outputs_stub.BaseModelOutputWithPoolingAndCrossAttentions = DummyModule
modeling_outputs_stub.CausalLMOutputWithCrossAttentions = DummyModule
sys.modules['transformers.modeling_outputs'] = modeling_outputs_stub

pytorch_utils_stub = types.ModuleType('transformers.pytorch_utils')
pytorch_utils_stub.apply_chunking_to_forward = DummyModule
sys.modules['transformers.pytorch_utils'] = pytorch_utils_stub

_HERE = os.path.dirname(os.path.abspath(__file__))


class Pack:
    """Accumulates {name: float32 ndarray} and writes one safetensors file."""

    def __init__(self, sd):
        self.sd = sd
        self.T = {}

    def _np(self, key):
        return self.sd[key].detach().cpu().float().numpy().astype(np.float32)

    def _has(self, key):
        return key in self.sd

    def _resolve(self, pt):
        """Resolve a PyTorch path that may or may not have .reparam_conv suffix.

        After model.fuse(), some modules keep reparam_conv as a submodule,
        while others get inlined. Try both paths.
        """
        if self._has(f"{pt}.weight"):
            return pt
        if self._has(f"{pt}.reparam_conv.weight"):
            return f"{pt}.reparam_conv"
        raise KeyError(f"Cannot find weight for {pt} or {pt}.reparam_conv")

    def conv(self, pt, nn):
        """Export a Conv2d with weight and bias (e.g. reparam_conv)."""
        resolved = self._resolve(pt)
        w = self._np(f"{resolved}.weight")
        b = self._np(f"{resolved}.bias")
        self.T[f"{nn}/conv:filter"] = w
        self.T[f"{nn}/conv:bias"] = b.reshape(1, w.shape[0], 1, 1)

    def conv_no_bias(self, pt, nn):
        """Export a Conv2d without bias (e.g. qkv)."""
        resolved = self._resolve(pt)
        w = self._np(f"{resolved}.weight")
        self.T[f"{nn}/conv:filter"] = w

    def conv_bn_fuse(self, conv_pt, bn_pt, nn):
        """Fuse Conv2d + BatchNorm2d into a single biased conv.

        After model.fuse(), ConvNorm (conv+bn) may be replaced by a plain Conv2d.
        Try both paths: {conv_pt}.weight (fused) or {conv_pt}.conv.weight (unfused).
        If BN params exist, fold them; otherwise export conv as-is.
        """
        # Resolve conv weight path (fused ConvNorm → Conv2d, or unfused ConvNorm.conv)
        if self._has(f"{conv_pt}.weight"):
            w = self._np(f"{conv_pt}.weight")
            b = self._np(f"{conv_pt}.bias") if self._has(f"{conv_pt}.bias") else np.zeros(w.shape[0], dtype=np.float32)
            bn_already_fused = True
        elif self._has(f"{conv_pt}.conv.weight"):
            w = self._np(f"{conv_pt}.conv.weight")
            b = self._np(f"{conv_pt}.conv.bias") if self._has(f"{conv_pt}.conv.bias") else np.zeros(w.shape[0], dtype=np.float32)
            bn_already_fused = False
        else:
            raise KeyError(f"Cannot find conv weight at {conv_pt}.weight or {conv_pt}.conv.weight")

        # Fold BN if it exists and wasn't already fused
        if not bn_already_fused and self._has(f"{bn_pt}.weight"):
            gamma = self._np(f"{bn_pt}.weight")
            beta = self._np(f"{bn_pt}.bias")
            mean = self._np(f"{bn_pt}.running_mean")
            var = self._np(f"{bn_pt}.running_var")
            eps = 1e-5
            scale = gamma / np.sqrt(var + eps)
            w = w * scale.reshape(-1, 1, 1, 1)
            b = (b - mean) * scale + beta

        self.T[f"{nn}/conv:filter"] = w
        self.T[f"{nn}/conv:bias"] = b.reshape(1, w.shape[0], 1, 1)

    def conv_fold_scale(self, conv_pt, ls_pt, nn):
        """Export a Conv2d with layer_scale gamma folded into weights and bias.

        W' = gamma * W, b' = gamma * b
        """
        resolved = self._resolve(conv_pt)
        w = self._np(f"{resolved}.weight")
        b = self._np(f"{resolved}.bias")
        gamma = self._np(f"{ls_pt}.gamma").squeeze()  # [C, 1, 1] -> [C]

        fused_w = w * gamma.reshape(-1, 1, 1, 1)
        fused_b = b * gamma

        self.T[f"{nn}/conv:filter"] = fused_w
        self.T[f"{nn}/conv:bias"] = fused_b.reshape(1, w.shape[0], 1, 1)

    def linear_to_conv(self, pt, nn):
        """Convert a Linear layer to 1x1 Conv2d (weight reshape [out,in] -> [out,in,1,1])."""
        w = self._np(f"{pt}.weight")  # [out, in]
        w = w.reshape(w.shape[0], w.shape[1], 1, 1)
        self.T[f"{nn}/conv:filter"] = w
        if self._has(f"{pt}.bias"):
            b = self._np(f"{pt}.bias")
            self.T[f"{nn}/conv:bias"] = b.reshape(1, w.shape[0], 1, 1)

    def batchnorm(self, pt, nn):
        """Export a BatchNorm2d layer (gamma, beta, moving_mean, moving_variance)."""
        if f"{pt}.weight" not in self.sd:
            print(f"Skipping BN {pt} -> {nn} (not in fused state_dict)")
            return
        self.T[f"{nn}:gamma"] = self._np(f"{pt}.weight")
        self.T[f"{nn}:beta"] = self._np(f"{pt}.bias")
        self.T[f"{nn}:moving_mean"] = self._np(f"{pt}.running_mean")
        self.T[f"{nn}:moving_variance"] = self._np(f"{pt}.running_var")

    def layernorm(self, pt, nn):
        """Export a LayerNorm layer (gamma, beta)."""
        if f"{pt}.weight" not in self.sd:
            print(f"Skipping LN {pt} -> {nn} (not in fused state_dict)")
            return
        self.T[f"{nn}:gamma"] = self._np(f"{pt}.weight")
        self.T[f"{nn}:beta"] = self._np(f"{pt}.bias")

    def write(self, path):
        names = list(self.T.keys())
        header, blob, offset = {}, bytearray(), 0
        for n in names:
            arr = np.ascontiguousarray(self.T[n], dtype="<f4")
            b = arr.tobytes()
            header[n] = {"dtype": "F32", "shape": list(arr.shape),
                         "data_offsets": [offset, offset + len(b)]}
            blob += b
            offset += len(b)
        hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(hjson)))
            f.write(hjson)
            f.write(blob)
        return len(names)


def _setup_stubs():
    """Stub out deep-vision-models modules that have heavy dependencies."""
    import sys, types
    import torch.nn as nn

    class CreateFromParent:
        @classmethod
        def from_parent(cls, parent):
            obj = cls.__new__(cls)
            obj.__dict__.update(parent.__dict__)
            return obj

    def convert_linear2conv1x1(linear, in_ch=None, out_ch=None):
        in_features = in_ch if in_ch is not None else linear.in_features
        out_features = out_ch if out_ch is not None else linear.out_features
        conv = nn.Conv2d(in_features, out_features, 1, bias=linear.bias is not None)
        w = linear.weight.data.reshape(out_features, in_features, 1, 1)
        conv.weight.data = w
        if linear.bias is not None:
            conv.bias.data = linear.bias.data
        return conv

    def reparameterize_bn_conv(bn, conv):
        """Fold BN (applied before conv) into conv.

        BN(x) = scale * (x - mean) / sqrt(var + eps) + beta = scale * x + shift
        where scale = gamma / sqrt(var + eps), shift = beta - mean * scale.
        conv(BN(x)) = conv(scale * x + shift) = (W * scale) @ x + W @ shift + bias.
        So W' = W * scale (along input dim), b' = W @ shift + bias.
        """
        import torch
        num_features = bn.num_features if hasattr(bn, 'num_features') else conv.in_channels
        gamma = bn.weight.data if bn.weight is not None else torch.ones(num_features)
        beta = bn.bias.data if bn.bias is not None else torch.zeros(num_features)
        mean = bn.running_mean.data
        var = bn.running_var.data
        eps = bn.eps
        scale = gamma / torch.sqrt(var + eps)
        shift = beta - mean * scale

        # Scale weight along input channel dim (dim=1 for Conv2d weight [out, in, kh, kw])
        w = conv.weight.data * scale.reshape(1, -1, 1, 1)

        # Compute new bias: b' = W @ shift + existing_bias
        # W shape: [out, in, kh, kw], shift shape: [in]
        # W @ shift = sum over in of W[:, :, k, l] * shift[i] for each (out, k, l)
        # For 1x1 conv: W shape [out, in, 1, 1], so b' = W.squeeze() @ shift
        w_reshaped = w.reshape(w.shape[0], w.shape[1], -1)  # [out, in, kh*kw]
        b_new = w_reshaped.sum(dim=2) @ shift  # [out]
        if conv.bias is not None:
            b_new = b_new + conv.bias.data

        conv.weight.data = w
        conv.bias = nn.Parameter(b_new)
        return conv

    ops_stub = types.ModuleType('deepvs.models.ops')
    ops_stub.CreateFromParent = CreateFromParent
    ops_stub.convert_linear2conv1x1 = convert_linear2conv1x1
    ops_stub.reparameterize_bn_conv = reparameterize_bn_conv
    sys.modules['deepvs.models.ops'] = ops_stub
    for name in ['deepvs.models.ops.optimize', 'deepvs.models.ops.optimize.detr',
                 'deepvs.models.ops.optimize.create_from_parent']:
        sys.modules[name] = types.ModuleType(name)
    sys.modules['deepvs.models.ops.optimize.create_from_parent'].CreateFromParent = CreateFromParent

    train_stub = types.ModuleType('deepvs.train')
    class TrainDESC: pass
    train_stub.TrainDESC = TrainDESC
    sys.modules['deepvs.train'] = train_stub


def main(weights, out):
    import sys
    DVM_PATH = os.environ.get("DEEP_VISION_MODELS_PATH", "")
    if DVM_PATH and os.path.isdir(DVM_PATH):
        sys.path.insert(0, DVM_PATH)

    _setup_stubs()

    from deepvs.models.keyword_fastvit import FastViTKeyword
    from deepvs.desc.keyword.desc_hyp_keyword import ModelDesc, HeadDesc
    from deepvs.models.backbone.fastvit import FastViTDesc

    # Build model descriptor
    desc = ModelDesc(
        fastvit=FastViTDesc(
            name="fastvit-sa12",
            num_classes=0,
            pretrained=False,
            reparam_attention=True,
            freeze=0,
        ),
        head=HeadDesc(
            proj_target_dim=507,
            proj_feature_dim=512,
            hidden_dim=None,
            dropout=0.0,
        ),
    )

    model = FastViTKeyword.from_desc(desc, 320)

    # Load checkpoint
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    if hasattr(ckpt, "model"):
        model = ckpt.model
    else:
        if "weights" in ckpt:
            sd_to_load = ckpt["weights"]
        elif "model_state_dict" in ckpt:
            sd_to_load = ckpt["model_state_dict"]
        else:
            sd_to_load = ckpt

        # Align prefix: map "model." -> "_fastViT.model."
        new_sd = {}
        for k, v in sd_to_load.items():
            if k.startswith("model."):
                new_sd["_fastViT." + k] = v
            else:
                new_sd[k] = v
        model.load_state_dict(new_sd, strict=False)
    model.eval()

    # Fuse the model (reparameterize all RepConv/BN)
    model.fuse(False)
    model.to("cpu").eval()

    # Get the fused state dict
    sd = model.state_dict()
    p = Pack(sd)

    # State dict keys are prefixed with "_fastViT.model."
    BB = "_fastViT.model."

    # === Stem (3 layers) ===
    # After fusion: reparam_conv is a single biased Conv2d
    p.conv(f"{BB}stem.0.reparam_conv", "stem0")
    p.conv(f"{BB}stem.1.reparam_conv", "stem1")
    p.conv(f"{BB}stem.2.reparam_conv", "stem2")

    # === Stage 0: 2 RepMixerBlocks ===
    for b in range(2):
        bn = f"{BB}stages.0.blocks.{b}"
        nn = f"s0b{b}"
        p.conv(f"{bn}.token_mixer.reparam_conv", f"{nn}/tm")
        p.conv_bn_fuse(f"{bn}.mlp.conv", f"{bn}.mlp.conv.bn", f"{nn}/mlp_conv/dw")
        p.conv(f"{bn}.mlp.fc1", f"{nn}/mlp_fc1")
        p.conv_fold_scale(f"{bn}.mlp.fc2", f"{bn}.layer_scale", f"{nn}/mlp_fc2")

    # === Stage 1: downsample + 2 RepMixerBlocks ===
    p.conv(f"{BB}stages.1.downsample.proj.0.reparam_conv", "s1_down/down0")
    p.conv(f"{BB}stages.1.downsample.proj.1.reparam_conv", "s1_down/down1")
    for b in range(2):
        bn = f"{BB}stages.1.blocks.{b}"
        nn = f"s1b{b}"
        p.conv(f"{bn}.token_mixer.reparam_conv", f"{nn}/tm")
        p.conv_bn_fuse(f"{bn}.mlp.conv", f"{bn}.mlp.conv.bn", f"{nn}/mlp_conv/dw")
        p.conv(f"{bn}.mlp.fc1", f"{nn}/mlp_fc1")
        p.conv_fold_scale(f"{bn}.mlp.fc2", f"{bn}.layer_scale", f"{nn}/mlp_fc2")

    # === Stage 2: downsample + 6 RepMixerBlocks ===
    p.conv(f"{BB}stages.2.downsample.proj.0.reparam_conv", "s2_down/down0")
    p.conv(f"{BB}stages.2.downsample.proj.1.reparam_conv", "s2_down/down1")
    for b in range(6):
        bn = f"{BB}stages.2.blocks.{b}"
        nn = f"s2b{b}"
        p.conv(f"{bn}.token_mixer.reparam_conv", f"{nn}/tm")
        p.conv_bn_fuse(f"{bn}.mlp.conv", f"{bn}.mlp.conv.bn", f"{nn}/mlp_conv/dw")
        p.conv(f"{bn}.mlp.fc1", f"{nn}/mlp_fc1")
        p.conv_fold_scale(f"{bn}.mlp.fc2", f"{bn}.layer_scale", f"{nn}/mlp_fc2")

    # === Stage 3: downsample + pos_emb + 2 AttentionBlocks ===
    p.conv(f"{BB}stages.3.downsample.proj.0.reparam_conv", "s3_down/down0")
    p.conv(f"{BB}stages.3.downsample.proj.1.reparam_conv", "s3_down/down1")
    p.conv(f"{BB}stages.3.pos_emb.reparam_conv", "s3_posemb")

    for b in range(2):
        bn = f"{BB}stages.3.blocks.{b}"
        nn = f"s3b{b}"
        p.batchnorm(f"{bn}.norm", f"{nn}/attn_norm")
        # qkv carries a bias: AttentionBlockOptimized.fuse folds the pre-qkv
        # BatchNorm (norm) into qkv via reparameterize_bn_conv, so the fused
        # qkv is a plain biased Conv2d (no separate norm layer remains).
        p.conv(f"{bn}.token_mixer.qkv", f"{nn}/qkv")
        p.conv_fold_scale(f"{bn}.token_mixer.proj", f"{bn}.layer_scale_1", f"{nn}/proj")
        p.conv_bn_fuse(f"{bn}.mlp.conv", f"{bn}.mlp.conv.bn", f"{nn}/mlp_conv/dw")
        p.conv(f"{bn}.mlp.fc1", f"{nn}/mlp_fc1")
        p.conv_fold_scale(f"{bn}.mlp.fc2", f"{bn}.layer_scale_2", f"{nn}/mlp_fc2")

    # === Final conv (MobileOneBlock with SE) ===
    p.conv(f"{BB}final_conv.reparam_conv", "final_conv")
    p.conv(f"{BB}final_conv.se.fc1", "final_conv/se/se_fc1")
    p.conv(f"{BB}final_conv.se.fc2", "final_conv/se/se_fc2")

    # === Head (MLP) ===
    # mlp.0: Linear(1024, 1024) -> 1x1 conv
    p.linear_to_conv("_head.mlp.0", "head/mlp_fc1")
    # mlp.1: LayerNorm(1024)
    p.layernorm("_head.mlp.1", "head/mlp_ln")
    # final_layer: Linear(1024, 1019) -> 1x1 conv
    p.linear_to_conv("_head.final_layer", "head/final_fc")

    n = p.write(out)
    print(f"Wrote {n} tensors to nntrainer safetensors: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Convert FastViTKeyword .pth to nntrainer safetensors")
    ap.add_argument("--weights", required=True,
                    help="path to the FastViTKeyword checkpoint (.pth)")
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "res",
                                                  "fastvit_keyword.safetensors"),
                    help="output safetensors path")
    args = ap.parse_args()
    main(args.weights, args.out)
