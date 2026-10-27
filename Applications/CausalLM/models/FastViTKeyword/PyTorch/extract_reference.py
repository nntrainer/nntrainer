#!/usr/bin/env python3
"""
@file extract_reference.py
@brief Extract PyTorch FastViTKeyword reference outputs for nntrainer verification.

Builds the FastViTKeyword model (FastViT-S12 backbone + MLP head), loads a
checkpoint, runs a fixed-seed input through the model, and saves:
  - The input tensor
  - Intermediate feature maps (after each stage)
  - The final backbone output (before global average pool)
  - The pooled features (after global average pool)
  - The head MLP intermediate output (after LayerNorm+GELU)
  - The final logits (507-dim) and feature (512-dim) outputs

All outputs are saved as raw float32 .bin files for bit-exact comparison
with the nntrainer C++ implementation.

Usage:
  python extract_reference.py --weights /path/to/ckpt.pth --out ../res
"""
import argparse
import os
import sys
import types
import torch

# Mock overrides decorator to disable method signature validation
overrides_stub = types.ModuleType('overrides')
def overrides_decorator(method=None, *args, **kwargs):
    if callable(method):
        return method
    return lambda f: f
overrides_stub.overrides = overrides_decorator
sys.modules['overrides'] = overrides_stub

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
detr_modeling_stub.BaseModelOutput = DummyModule
detr_modeling_stub.DetrDecoder = DummyModule
detr_modeling_stub.DetrDecoderLayer = DummyModule
detr_modeling_stub.DetrDecoderOutput = DummyModule
detr_modeling_stub.DetrEncoder = DummyModule
detr_modeling_stub.DetrEncoderLayer = DummyModule
detr_modeling_stub.DetrForObjectDetection = DummyModule
detr_modeling_stub.DetrModel = DummyModule
detr_modeling_stub.DetrModelOutput = DummyModule
detr_modeling_stub.DetrObjectDetectionOutput = DummyModule
sys.modules['transformers.models.detr.modeling_detr'] = detr_modeling_stub
sys.modules['transformers.models.detr.modeling_detr'] = detr_modeling_stub

# Mock other transformers submodules using DummyModule
attn_mask_stub = types.ModuleType('transformers.modeling_attn_mask_utils')
attn_mask_stub._prepare_4d_attention_mask_for_sdpa = DummyModule
attn_mask_stub._prepare_4d_causal_attention_mask_for_sdpa = DummyModule
attn_mask_stub._prepare_4d_attention_mask = DummyModule
sys.modules['transformers.modeling_attn_mask_utils'] = attn_mask_stub

modeling_outputs_stub = types.ModuleType('transformers.modeling_outputs')
modeling_outputs_stub.BaseModelOutputWithPastAndCrossAttentions = DummyModule
modeling_outputs_stub.BaseModelOutputWithPoolingAndCrossAttentions = DummyModule
modeling_outputs_stub.CausalLMOutputWithCrossAttentions = DummyModule
sys.modules['transformers.modeling_outputs'] = modeling_outputs_stub

pytorch_utils_stub = types.ModuleType('transformers.pytorch_utils')
pytorch_utils_stub.apply_chunking_to_forward = DummyModule
sys.modules['transformers.pytorch_utils'] = pytorch_utils_stub

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

# Add deep-vision-models to path if available
DVM_PATH = os.environ.get("DEEP_VISION_MODELS_PATH", "")
if DVM_PATH and os.path.isdir(DVM_PATH):
    sys.path.insert(0, DVM_PATH)


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
        import torch
        num_features = bn.num_features if hasattr(bn, 'num_features') else conv.in_channels
        gamma = bn.weight.data if bn.weight is not None else torch.ones(num_features)
        beta = bn.bias.data if bn.bias is not None else torch.zeros(num_features)
        mean = bn.running_mean.data
        var = bn.running_var.data
        eps = bn.eps
        scale = gamma / torch.sqrt(var + eps)
        shift = beta - mean * scale

        w = conv.weight.data * scale.reshape(1, -1, 1, 1)

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


def save_bin(tensor, path):
    arr = tensor.detach().cpu().float().numpy()
    arr.tofile(path)
    print(f"  saved {path} shape={arr.shape} dtype={arr.dtype}")


def main():
    _setup_stubs()
    parser = argparse.ArgumentParser(
        description="Extract FastViTKeyword PyTorch reference outputs for verification"
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="path to the FastViTKeyword checkpoint (.pth)",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(_HERE, "..", "res"),
        help="output dir for reference .bin files (default: ../res)",
    )
    parser.add_argument(
        "--img-size", type=int, default=320, help="input image size (default: 320)"
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Import from deep-vision-models
    from deepvs.models.keyword_fastvit import FastViTKeyword
    from deepvs.desc.keyword.desc_hyp_keyword import ModelDesc, HeadDesc
    from deepvs.models.backbone.fastvit import FastViTDesc

    # Build model descriptor matching the training config
    desc = ModelDesc(
        fastvit=FastViTDesc(
            name="fastvit-sa12",
            num_classes=0,
            pretrained=False,
            reparam_attention=True,
            freeze=0,  # no freeze for inference
        ),
        head=HeadDesc(
            proj_target_dim=507,
            proj_feature_dim=512,
            hidden_dim=None,  # uses backbone num_features (1024)
            dropout=0.0,  # eval mode, dropout is identity
        ),
    )

    model = FastViTKeyword.from_desc(desc, args.img_size)

    # Load checkpoint
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
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

    # Fixed input
    torch.manual_seed(42)
    x = torch.randn(1, 3, args.img_size, args.img_size)
    save_bin(x, os.path.join(args.out, "input.bin"))
    print(f"Input shape: {x.shape}")

    # Hook to capture intermediate outputs
    intermediates = {}
    hooks = []

    # Capture backbone intermediates (forward_intermediates returns 4 stage outputs)
    # We also capture the final backbone output
    backbone = model._fastViT

    # Hook on stem layers
    for i in range(3):
        def make_hook(name):
            def h(mod, inp, out):
                if isinstance(out, torch.Tensor):
                    intermediates[name] = out.detach().clone()
            return h
        hooks.append(backbone.model.stem[i].register_forward_hook(make_hook(f"stem{i}")))

    # Hook on each stage's downsample and blocks
    for s_idx in range(4):
        stage = backbone.model.stages[s_idx]
        # Hook on downsample
        if hasattr(stage, 'downsample') and stage.downsample is not None:
            hooks.append(stage.downsample.register_forward_hook(
                make_hook(f"stage{s_idx}_down"))
            )
        # Hook on each block
        for b_idx in range(len(stage.blocks)):
            hooks.append(stage.blocks[b_idx].register_forward_hook(
                make_hook(f"stage{s_idx}_block{b_idx}"))
            )

    # Hook on final_conv
    hooks.append(backbone.model.final_conv.register_forward_hook(
        make_hook("final_conv"))
    )

    # Hook on the head MLP components
    head = model._head
    # Hook after mlp (Linear+LayerNorm+GELU+Dropout)
    hooks.append(head.mlp.register_forward_hook(make_hook("head_mlp")))
    # Hook after final_layer
    hooks.append(head.final_layer.register_forward_hook(make_hook("head_final")))

    # Run forward pass
    with torch.no_grad():
        logits, features = model(x)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Save key intermediates
    # 1. Backbone intermediates (4 stage outputs)
    backbone_inters = backbone.model.forward_intermediates(x)
    for i, inter in enumerate(backbone_inters[1]):
        save_bin(inter, os.path.join(args.out, f"ref_stage{i}.bin"))

    # 1b. Per-stage downsample-only outputs (for NNTrainer bisection of the
    # downsample block). Hooked as "stage{s}_down".
    for nm in ["stage1_down", "stage2_down", "stage3_down"]:
        if nm in intermediates:
            save_bin(intermediates[nm], os.path.join(args.out, f"ref_{nm}.bin"))

    # 2. Final backbone output (before global avg pool)
    final_feat = backbone_inters[0]
    save_bin(final_feat, os.path.join(args.out, "ref_backbone_out.bin"))

    # 3. Pooled features (after global average pool)
    pooled = final_feat.mean(dim=[2, 3])
    save_bin(pooled, os.path.join(args.out, "ref_pooled.bin"))

    # 4. Head MLP output
    if "head_mlp" in intermediates:
        save_bin(intermediates["head_mlp"], os.path.join(args.out, "ref_head_mlp.bin"))

    # 5. Final outputs: logits (507) and features (512)
    save_bin(logits, os.path.join(args.out, "ref_logits.bin"))
    save_bin(features, os.path.join(args.out, "ref_features.bin"))

    # 6. Sigmoid of logits (for inference comparison)
    save_bin(logits.sigmoid(), os.path.join(args.out, "ref_sigmoid.bin"))

    # Print summary
    print(f"\n=== Reference output summary ===")
    print(f"Input: {x.shape}")
    print(f"Backbone output: {final_feat.shape}")
    print(f"Pooled features: {pooled.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Features: {features.shape}")
    print(f"\nAll saved to: {args.out}")


if __name__ == "__main__":
    main()
