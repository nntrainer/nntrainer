#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# @file   simulate_w8a8.py
# @brief  S0 numeric validation for the W8A8 design (W8A8_DESIGN.md).
#
# Simulates, in the PyTorch reference model, exactly the quantization the
# planned nntrainer W8A8 path would apply -- BEFORE any C++ is written:
#   - conv weights fake-quantized per-32-block Q8_0 style (fp16 scale, int8
#     quants), with CRS zero-padded to a block multiple when not aligned
#     (mirrors the padded-weight quantization of the S4 stage);
#   - activations at every would-be-int8 edge fake-quantized to per-tensor
#     symmetric int8 with an FP32 dynamic scale (amax/127):
#       * post-hook on each int8 conv's output  (producer quantize epilogue)
#       * pre-hook on each int8 conv's input    (models the int8 edge; on a
#         concat output this reproduces the rescale-to-max-scale rounding,
#         and is idempotent on tensors already on the same int8 grid)
#   - SiLU/bias/head all stay FP32 (as in the design); maxpool/upsample
#     preserve the int8 grid exactly so they need no extra modeling.
#
# Stages:
#   --stage s3 : only today's Q8_0-eligible convs (out_ch%32==0 and CRS%32==0)
#                are int8; stem/blocks.1 48-ch convs/head conv stay FP32.
#   --stage s4 : every conv is int8 (padding path), except the head
#                final conv keeps an FP32 OUTPUT (its consumer, rtmcc_head,
#                is a hard FP32 boundary) and the stem keeps an FP32 INPUT
#                (the image tensor stays FP32).
#
# Gate: visible keypoints (score=min(max_x,max_y) >= 0.5) must stay at the
# W8A32 baseline (81/87 on the real weights + input_320.bin). Run:
#
#   python3 simulate_w8a8.py --weights /path/pose_base_v311.pt \
#       --input input_320.bin --stage s3
#   python3 simulate_w8a8.py --weights /path/pose_base_v311.pt \
#       --input input_320.bin --stage s4
#
# and compare "visible" against the FP32 run (--stage fp32).

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_pose.common_v2 import Conv  # noqa: E402
from models_pose.fuse import IFuse  # noqa: E402
from models_pose.pose_ref import YOLOv7Posetiny  # noqa: E402
from weight_converter import _load_checkpoint  # noqa: E402

QK = 32  # Q8_0 block size

NKPT = 87
SIMCC_BINS = 640  # img_size 320 * simcc_split_ratio 2


# When True, the per-tensor activation scale is rounded to fp16 before use,
# exactly reproducing the device epilogue's convRoundScaleFp16(amax/127). The
# per-channel int8 GEMM consumes the activation scale as an FP32 multiplier
# (block d is ignored), so this rounding is a device-only precision loss the
# FP32-scale simulation does not have -- toggle it to test whether the ~50-layer
# accumulation of fp16 scale error is what drops device per-channel 81 -> 80.
ACT_SCALE_FP16 = False
# ORT-style ASYMMETRIC (affine) activation quantization: scale = (max-min)/255
# with a zero point, so the full 256-level grid spans the actual value range.
# SiLU outputs live in [~-0.28, max]; symmetric int8 wastes nearly half its
# levels on negatives that never occur, while affine quantization keeps ~2x the
# resolution on every activation edge -- the margin that keeps ORT's borderline
# keypoints (score ~0.5) from flipping.
ACT_ASYM = False


def fake_quant_act_per_tensor(t: torch.Tensor) -> torch.Tensor:
    """Per-tensor int8 fake-quant with dynamic scale. Symmetric by default;
    ACT_ASYM switches to affine (zero-point) quantization like ORT's u8
    activations. ACT_SCALE_FP16 rounds the scale to fp16 (device per-block)."""
    if ACT_ASYM:
        mn, mx = t.min(), t.max()
        if mx == mn:
            return t
        scale = (mx - mn) / 255.0
        if ACT_SCALE_FP16:
            scale = scale.half().float()
        zp = torch.round(-128.0 - mn / scale)
        q = torch.clamp(torch.round(t / scale) + zp, -128, 127)
        return (q - zp) * scale
    amax = t.abs().max()
    if amax == 0:
        return t
    scale = amax / 127.0
    if ACT_SCALE_FP16:
        scale = scale.half().float()
    return torch.clamp(torch.round(t / scale), -128, 127) * scale


def fake_quant_weight_q8_0(w: torch.Tensor) -> torch.Tensor:
    """Per-32-block Q8_0 fake-quant of a conv weight [out, cin, kh, kw].

    Rows are the flattened [out, CRS] matmul rows the runtime quantizes; CRS
    is zero-padded to a QK multiple first (changes block partitioning exactly
    like the padded-weight path would). Scales go through fp16 like block_q8_0.d.
    """
    out_ch = w.shape[0]
    rows = w.reshape(out_ch, -1)
    crs = rows.shape[1]
    crs_pad = (crs + QK - 1) // QK * QK
    if crs_pad != crs:
        rows = torch.nn.functional.pad(rows, (0, crs_pad - crs))
    blocks = rows.reshape(out_ch, crs_pad // QK, QK)
    amax = blocks.abs().amax(dim=2, keepdim=True)
    d = (amax / 127.0).half().float()  # fp16 scale storage, like block_q8_0.d
    q = torch.where(
        d > 0,
        torch.clamp(torch.round(blocks / torch.where(d > 0, d, torch.ones_like(d))),
                    -128, 127) * d,
        blocks,
    )
    deq = q.reshape(out_ch, crs_pad)[:, :crs]
    return deq.reshape_as(w)


def fake_quant_weight_per_channel(w: torch.Tensor) -> torch.Tensor:
    """Per-output-channel symmetric int8 fake-quant of a conv weight
    [out, cin, kh, kw]: ONE FP32 scale per output channel (row of the
    [out, CRS] matmul). This is the ORT / industry-standard weight scheme --
    coarser than per-32-block Q8_0 but it lets the int8 GEMM accumulate int32
    across the whole reduction and apply the scale once (the speed win). No
    CRS padding is needed (the scale spans the whole row), and scales stay
    FP32 (no fp16 rounding), unlike block_q8_0.d.
    """
    out_ch = w.shape[0]
    rows = w.reshape(out_ch, -1)
    amax = rows.abs().amax(dim=1, keepdim=True)  # [out, 1]
    scale = amax / 127.0
    q = torch.where(
        scale > 0,
        torch.clamp(torch.round(rows / torch.where(scale > 0, scale,
                                                    torch.ones_like(scale))),
                    -128, 127) * scale,
        rows,
    )
    return q.reshape_as(w)


def decode_visible(pose: np.ndarray, thr: float = 0.5) -> int:
    """Replicates main.cpp decodeSimcc: score = min(max_x, max_y), >= thr."""
    visible = 0
    for k in range(NKPT):
        mx = pose[k].max()
        my = pose[NKPT + k].max()
        if min(mx, my) >= thr and pose[k].argmax() >= 0:
            visible += 1
    return visible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="real .pt checkpoint")
    ap.add_argument("--input", default="input_320.bin",
                    help="NCHW FP32 1x3x320x320 input bin (the benchmark one)")
    ap.add_argument("--stage", choices=["fp32", "s3", "s4"], default="s3")
    ap.add_argument(
        "--wquant", choices=["per_block", "per_channel"], default="per_block",
        help="weight fake-quant scheme. per_block = current Q8_0 (scale/32); "
        "per_channel = ORT-style (one FP32 scale per output channel, enables "
        "the int32-accumulate kernel). Compare s4 accuracy across both to "
        "decide whether all-int8 per-channel holds the 81/87 gate.")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--act_fp16", action="store_true",
                    help="round each activation scale to fp16 before use, "
                    "matching the device epilogue's convRoundScaleFp16. Use to "
                    "isolate whether fp16 activation-scale rounding is what "
                    "costs the device its keypoint (per_channel 81 -> 80).")
    ap.add_argument("--quant_stem_input", action="store_true",
                    help="ALSO int8-fake-quant the stem conv's input (the raw "
                    "image), reproducing the device per-channel path before the "
                    "stem exclusion fix. The design keeps the image FP32; use "
                    "this to confirm stem-input quantization costs keypoints.")
    ap.add_argument("--act_asym", action="store_true",
                    help="ORT-style asymmetric (zero-point) activation "
                    "fake-quant: scale=(max-min)/255. ~2x activation "
                    "resolution on SiLU outputs vs symmetric int8; use to "
                    "check whether affine activations recover the borderline "
                    "keypoints (device symmetric per-channel reads 79).")
    args = ap.parse_args()
    global ACT_SCALE_FP16, ACT_ASYM
    ACT_SCALE_FP16 = args.act_fp16
    ACT_ASYM = args.act_asym
    wquant = (fake_quant_weight_per_channel if args.wquant == "per_channel"
              else fake_quant_weight_q8_0)

    model = YOLOv7Posetiny(nkpt=NKPT, img_size=320)
    state = _load_checkpoint(args.weights, torch.device("cpu"))
    state = {k: v for k, v in state.items() if not k.startswith("model.")}
    if not any(".bn." in k for k in state):
        for mod in model.modules():
            if isinstance(mod, IFuse):
                mod.fuse()
    miss, unexp = model.load_state_dict(state, strict=False)
    miss = [k for k in miss if not k.startswith("model.")]
    unexp = [k for k in unexp if not k.startswith("model.")]
    if miss or unexp:
        print("missing:", miss[:8])
        print("unexpected:", unexp[:8])
        raise SystemExit("state_dict mismatch")
    model.eval()
    for mod in model.modules():
        if isinstance(mod, IFuse):
            mod.fuse()

    n_wq = n_hooked = 0
    if args.stage != "fp32":
        for name, m in model.named_modules():
            if not isinstance(m, Conv):
                continue
            c = m.conv
            in_ch, out_ch = c.in_channels, c.out_channels
            k = c.kernel_size[0]
            eligible = out_ch % 32 == 0 and (in_ch * k * k) % 32 == 0
            if args.stage == "s3" and not eligible:
                continue
            is_stem = in_ch == 3
            is_head_final = out_ch == NKPT

            with torch.no_grad():
                c.weight.copy_(wquant(c.weight))
            n_wq += 1

            # input edge int8 (except the stem, whose input -- the image --
            # stays FP32 in the real graph; --quant_stem_input overrides to
            # reproduce the pre-fix device behavior)
            if not is_stem or args.quant_stem_input:
                m.register_forward_pre_hook(
                    lambda mod, inp: (fake_quant_act_per_tensor(inp[0]),)
                    + tuple(inp[1:]))
            # output edge int8 (except head final conv: consumer rtmcc_head
            # is a hard FP32 boundary, so its input tensor stays FP32)
            if not is_head_final:
                m.register_forward_hook(
                    lambda mod, inp, out: fake_quant_act_per_tensor(out))
            n_hooked += 1

    x = torch.from_numpy(
        np.fromfile(args.input, dtype=np.float32).reshape(1, 3, 320, 320))
    with torch.no_grad():
        pose = model(x)
    pose = pose[0].numpy()
    assert pose.shape == (2 * NKPT, SIMCC_BINS), pose.shape

    visible = decode_visible(pose, args.thr)
    print(f"stage={args.stage}  wquant={args.wquant}  "
          f"weight-quantized convs={n_wq}  act-hooked convs={n_hooked}"
          f"{'  act_asym' if args.act_asym else ''}"
          f"{'  act_fp16' if args.act_fp16 else ''}")
    print(f"visible keypoints: {visible}/{NKPT} (thr={args.thr})")
    # Borderline margin report: keypoints whose score sits near the threshold.
    # These are the ones every small numeric change flips; watching their
    # margins tells whether a scheme change genuinely adds headroom.
    border = []
    for k in range(NKPT):
        s = min(pose[k].max(), pose[NKPT + k].max())
        if 0.40 <= s < 0.60:
            border.append((float(s), k))
    for s, k in sorted(border):
        print(f"  borderline kp{k:<3d} score={s:.4f}  "
              f"({'+' if s >= args.thr else '-'}{abs(s - args.thr):.4f})")


if __name__ == "__main__":
    main()
