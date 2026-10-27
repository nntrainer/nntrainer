#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
#
# @file convert_weights_from_onnx.py
# @brief Regenerate the NNTrainer fastvit_keyword safetensors DIRECTLY from the
#        verified ONNX reference model (keyword.onnx), bypassing the buggy
#        PyTorch re-fuse path in convert_weights.py.
#
# Why: the existing fastvit_keyword.safetensors was produced by
# convert_weights.py re-fusing a PyTorch checkpoint, and its conv weights
# diverge element-wise from the ONNX reference (which matches the PyTorch
# golden refs). This produced a ~0.76x magnitude shrink per backbone stage
# and collapsed the keyword logits (idx=455 vs the golden idx=327). The ONNX
# is the fused, reparameterized ground truth (no BN ops, 88 inits), so we
# rebuild the safetensors from it with exact NNTrainer-layer naming.
#
# Weight semantics:
#   - stem / downsample / pos_emb / final_conv / SE convs: ONNX weight used
#     as-is for {layer}/conv:filter. Biases: only where ONNX has a bias init
#     (pos_emb.bias, se.fc1.bias, se.fc2.bias) — copied; otherwise zero bias
#     of shape [1, out_ch, 1, 1] to satisfy NNTrainer's conv bias contract.
#   - RepMixer mlp_fc2: ONNX fc2.weight is PRE layer_scale; the layer_scale
#     gamma is a separate Mul in the ONNX graph. NNTrainer folds gamma into
#     fc2, so filter_out = fc2.weight * gamma (broadcast over out_ch).
#   - RepMixer token_mixer (tm): reparam_conv.weight used as-is.
#   - stage3 attention: qkv weight 1536x512, proj weight 512x512, as-is.
#   - head: mlp.0 (Linear->1x1 conv) weight [1024,1024], mlp.1 LayerNorm
#     gamma/bias [1024], final_layer weight [1019,1024]. No biases on the
#     Gemms.
#
# Usage:
#   python convert_weights_from_onnx.py \
#       --onnx /path/keyword.onnx \
#       --out  /path/fastvit_keyword.safetensors
#
# Output shapes match the existing safetensors (filter [out,in,kh,kw],
# bias [1,out,1,1], ln gamma/beta [C]).

import argparse
import re
import numpy as np
import onnx
from safetensors.numpy import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    model = onnx.load(args.onnx, load_external_data=True)
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}

    T = {}  # nntrainer-name -> np.ndarray

    def set_filter(nn_name, w):
        assert w.ndim == 4, f"{nn_name}: expected 4D, got {w.shape}"
        T[nn_name + "/conv:filter"] = w.astype(np.float32)

    def set_bias(nn_name, out_ch, b=None):
        if b is None:
            b = np.zeros(out_ch, dtype=np.float32)
        else:
            b = b.reshape(-1).astype(np.float32)
        T[nn_name + "/conv:bias"] = b.reshape(1, out_ch, 1, 1)

    def conv(name_onnx, nn, has_bias_onnx=False):
        w = inits[name_onnx]
        out_ch = w.shape[0]
        set_filter(nn, w)
        if has_bias_onnx and (name_onnx.replace('.weight', '.bias') in inits):
            set_bias(nn, out_ch, inits[name_onnx.replace('.weight', '.bias')])
        else:
            set_bias(nn, out_ch, None)

    # ---- Stem ----
    for i in (0, 1, 2):
        conv(f'_fastViT.model.stem.{i}.reparam_conv.weight', f'stem{i}')

    # ---- Stages 0,1,2 (RepMixer) ----
    for s in (0, 1, 2):
        nblocks = {0: 2, 1: 2, 2: 6}[s]
        gamma = inits[f'_fastViT.model.stages.{s}.blocks.0.layer_scale.gamma']  # [C,1,1]
        gamma = gamma.reshape(-1)  # [C]
        # downsample (stages 1,2 only; stage 0 has no downsample)
        if s >= 1:
            conv(f'_fastViT.model.stages.{s}.downsample.proj.0.reparam_conv.weight',
                 f's{s}_down/down0')
            conv(f'_fastViT.model.stages.{s}.downsample.proj.1.reparam_conv.weight',
                 f's{s}_down/down1')
        for b in range(nblocks):
            p = f'_fastViT.model.stages.{s}.blocks.{b}'
            nb = f's{s}b{b}'
            conv(f'{p}.token_mixer.reparam_conv.weight', f'{nb}/tm')
            conv(f'{p}.mlp.conv.conv.weight', f'{nb}/mlp_conv/dw')
            conv(f'{p}.mlp.fc1.weight', f'{nb}/mlp_fc1')
            # fold layer_scale gamma into mlp_fc2
            fc2 = inits[f'{p}.mlp.fc2.weight']  # [C_out, C_in, 1, 1]
            fc2_folded = fc2 * gamma.reshape(-1, 1, 1, 1)
            set_filter(f'{nb}/mlp_fc2', fc2_folded)
            set_bias(f'{nb}/mlp_fc2', fc2.shape[0], None)

    # ---- Stage 3 (downsample + posemb + 2 attention blocks) ----
    conv('_fastViT.model.stages.3.downsample.proj.0.reparam_conv.weight', 's3_down/down0')
    conv('_fastViT.model.stages.3.downsample.proj.1.reparam_conv.weight', 's3_down/down1')
    # posemb: has bias
    posemb_w = inits['_fastViT.model.stages.3.pos_emb.reparam_conv.weight']
    set_filter('s3_posemb', posemb_w)
    posemb_b = inits['_fastViT.model.stages.3.pos_emb.reparam_conv.bias']
    set_bias('s3_posemb', posemb_w.shape[0], posemb_b)
    # attention blocks: layer_scale_1 gamma (shared) folded into proj & mlp_fc2
    gamma1 = inits['_fastViT.model.stages.3.blocks.0.layer_scale_1.gamma'].reshape(-1)
    for b in (0, 1):
        p = f'_fastViT.model.stages.3.blocks.{b}'
        nb = f's3b{b}'
        conv(f'{p}.token_mixer.qkv.weight', f'{nb}/qkv')
        proj = inits[f'{p}.token_mixer.proj.weight']
        set_filter(f'{nb}/proj', proj * gamma1.reshape(-1, 1, 1, 1))
        set_bias(f'{nb}/proj', proj.shape[0], None)
        conv(f'{p}.mlp.conv.conv.weight', f'{nb}/mlp_conv/dw')
        conv(f'{p}.mlp.fc1.weight', f'{nb}/mlp_fc1')
        fc2 = inits[f'{p}.mlp.fc2.weight']
        set_filter(f'{nb}/mlp_fc2', fc2 * gamma1.reshape(-1, 1, 1, 1))
        set_bias(f'{nb}/mlp_fc2', fc2.shape[0], None)

    # ---- Final conv (dw 512->1024, no bias) + SE ----
    fc_w = inits['_fastViT.model.final_conv.reparam_conv.weight']
    set_filter('final_conv', fc_w)
    set_bias('final_conv', fc_w.shape[0], None)
    conv('_fastViT.model.final_conv.se.fc1.weight', 'final_conv/se/se_fc1',
         has_bias_onnx=True)
    conv('_fastViT.model.final_conv.se.fc2.weight', 'final_conv/se/se_fc2',
         has_bias_onnx=True)

    # ---- Head (NCHW) ----
    mlp0_w = inits['_head.mlp.0.weight']  # [1024,1024] Linear
    set_filter('head/mlp_fc1', mlp0_w.reshape(1024, 1024, 1, 1))
    set_bias('head/mlp_fc1', 1024, None)
    ln_gamma = inits['_head.mlp.1.weight'].reshape(-1)
    ln_beta = inits['_head.mlp.1.bias'].reshape(-1)
    T['head/mlp_ln:gamma'] = ln_gamma.astype(np.float32).reshape(1, 1024, 1, 1)
    T['head/mlp_ln:beta'] = ln_beta.astype(np.float32).reshape(1, 1024, 1, 1)
    final_w = inits['_head.final_layer.weight']  # [1019,1024]
    set_filter('head/final_fc', final_w.reshape(1019, 1024, 1, 1))
    set_bias('head/final_fc', 1019, None)

    save_file(T, args.out)
    print(f"wrote {len(T)} tensors to {args.out}")


if __name__ == '__main__':
    main()
