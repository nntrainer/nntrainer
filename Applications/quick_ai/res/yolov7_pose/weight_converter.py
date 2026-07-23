#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# @file   weight_converter.py
# @date   14 July 2026
# @brief  Convert a YOLOv7ReIDtiny (pose + ReID) PyTorch checkpoint into an
#         nntrainer FP32 safetensors, with weight keys that bind by name to the
#         layers built by yolov7_pose_graph.h.
#
#         Pipeline:
#           1. build YOLOv7ReIDtiny, load_state_dict(strict=True), fuse()
#              (folds Conv+BN so every conv becomes a single biased conv).
#           2. map each fused parameter to its nntrainer weight name and layout.
#           3. write <out>/yolov7_pose.safetensors  (FP32).
#
#         Stage 2 (Q8_0 weights) is produced from this FP32 output by
#         nntr_quantize (see the E2E guide), so this script stays FP32-only.
#
# @author Seungbaek Hong <sb92.hong@samsung.com>

import argparse
import importlib.abc
import importlib.machinery
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn
from safetensors.numpy import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_pose.yolo_backbone import YOLOv7ReIDtiny  # noqa: E402

# Head hyper-parameters (must match yolov7_pose_graph.h).
NKPT = 87
FEATMAP = 10
FLATTEN = FEATMAP * FEATMAP  # 100


class _Placeholder(nn.Module):
    """Permissive nn.Module stand-in for an unavailable training-repo class.

    A full torch.save(model) may pickle instance state that re-binds methods
    (e.g. a fused Conv does ``self.forward = self.forward_fuse``); restoring
    such a bound method calls ``getattr(obj, 'forward_fuse')``. Return a no-op
    callable for any attribute the real class would have provided, so the
    (real) parameter/buffer/submodule dicts still deserialize and state_dict()
    recovers the tensors.
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)  # _parameters/_buffers/_modules
        except AttributeError:
            if name.startswith("__") and name.endswith("__"):
                raise
            return lambda *a, **k: None


def _placeholder_class(module_name, cls_name):
    return type(cls_name, (_Placeholder,), {"__module__": module_name})


class _ShimLoader(importlib.abc.Loader):
    """Materialize a module whose every attribute is an nn.Module placeholder."""

    def create_module(self, spec):
        m = types.ModuleType(spec.name)

        def _getattr(name, _m=m):
            cls = _placeholder_class(_m.__name__, name)
            setattr(_m, name, cls)
            return cls

        m.__getattr__ = _getattr
        return m

    def exec_module(self, module):
        pass


class _ShimFinder(importlib.abc.MetaPathFinder):
    """Serve shim modules for training-repo packages absent in this env."""

    def __init__(self, roots):
        self.roots = set(roots)

    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in self.roots:
            return importlib.machinery.ModuleSpec(name, _ShimLoader())
        return None


def _load_checkpoint(checkpoint_path, device):
    """Load a checkpoint saved either as a state_dict or as a full model object.

    A full ``torch.save(model)`` embeds the training repo's class paths (e.g.
    ``models.*``); if that package is absent we install a permissive shim so
    unpickling reconstructs the module tree as nn.Module placeholders, from
    which the (real) state_dict is recovered. Only the tensors matter here.
    """
    installed = []
    while True:
        try:
            obj = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            break
        except ModuleNotFoundError as e:
            root = (e.name or "").split(".")[0]
            if not root or root in installed:
                raise
            sys.meta_path.insert(0, _ShimFinder({root}))
            installed.append(root)
            print(f"[converter] shimming missing package '{root}' for unpickling")

    if hasattr(obj, "state_dict") and not isinstance(obj, dict):
        return obj.state_dict()
    if isinstance(obj, dict):
        for key in ("model", "state_dict", "ema", "weights"):
            v = obj.get(key)
            if hasattr(v, "state_dict") and not isinstance(v, dict):
                return v.state_dict()
            if isinstance(v, dict):
                return v
    return obj


def build_model(checkpoint_path, device):
    model = YOLOv7ReIDtiny(
        nc=1, img_size=320, embed_dim=128, widen_factor=1.5, nkpt=NKPT
    )
    state = _load_checkpoint(checkpoint_path, device)
    # tolerate a "module." prefix on externally-saved state dicts
    if state and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}

    # A checkpoint with no BatchNorm keys was already fused (Conv+BN folded);
    # fuse the reconstruction first so the key sets line up.
    already_fused = not any(".bn." in k for k in state)
    model.eval()
    if already_fused:
        model.fuse()

    missing, unexpected = model.load_state_dict(state, strict=False)
    # Ignore the aliased "model.*" (nn.Sequential) view: those tensors are
    # shared with backbone/head/head_feat and always resolve via the canonical
    # paths, so their absence from either side is benign.
    missing = [k for k in missing if not k.startswith("model.")]
    unexpected = [k for k in unexpected if not k.startswith("model.")]
    if missing or unexpected:
        print(
            f"[converter] load_state_dict: {len(missing)} missing, "
            f"{len(unexpected)} unexpected keys"
        )
        if missing:
            print("  e.g. missing:", missing[:8])
        if unexpected:
            print("  e.g. unexpected:", unexpected[:8])
        raise SystemExit(
            "checkpoint keys do not match the reconstructed model. Compare with "
            "`--inspect` and align models_pose/ (and the weight_converter "
            "mapping) with your model's module hierarchy."
        )
    model.to(device).float()
    if not already_fused:
        model.fuse()
    return model


def np32(t):
    return t.detach().cpu().float().numpy().astype(np.float32)


def _fuse_conv_bn_np(cw, cb, bn_w, bn_b, bn_rm, bn_rv, eps=1e-5):
    """Fold Conv (+optional bias) and BatchNorm into a single biased conv."""
    import numpy as _np

    std = _np.sqrt(bn_rv + eps)
    scale = bn_w / std
    w = cw * scale.reshape(-1, 1, 1, 1)
    b = (cb if cb is not None else _np.zeros_like(bn_rm)) * scale + (
        bn_b - bn_w * bn_rm / std
    )
    return w.astype(_np.float32), b.astype(_np.float32)


def _prepare_state(state):
    """Return a fused, alias-free {key: np.float32} dict.

    Handles both fused checkpoints (Conv+BN already folded) and unfused ones
    (folds every ``<mp>.conv`` + ``<mp>.bn`` pair), and drops the aliased
    ``model.*`` (nn.Sequential) view and ``num_batches_tracked`` buffers.
    """
    sd = {}
    for k, v in state.items():
        if k.startswith("model.") or k.endswith(".num_batches_tracked"):
            continue
        sd[k] = np32(v)

    bn_prefixes = {
        k[: -len(".bn.weight")] for k in sd if k.endswith(".bn.weight")
    }
    if not bn_prefixes:
        return sd  # already fused

    fused = {}
    for mp in bn_prefixes:
        cw = sd.get(mp + ".conv.weight")
        cb = sd.get(mp + ".conv.bias")
        fused[mp + ".conv.weight"], fused[mp + ".conv.bias"] = _fuse_conv_bn_np(
            cw, cb, sd[mp + ".bn.weight"], sd[mp + ".bn.bias"],
            sd[mp + ".bn.running_mean"], sd[mp + ".bn.running_var"],
        )
    out = {}
    for k, v in sd.items():
        mp = None
        for suf in (".conv.weight", ".conv.bias", ".bn.weight", ".bn.bias",
                    ".bn.running_mean", ".bn.running_var"):
            if k.endswith(suf):
                cand = k[: -len(suf)]
                if cand in bn_prefixes:
                    mp = cand
                    break
        if mp is not None:
            continue  # replaced by the fused entry
        out[k] = v
    out.update(fused)
    return out


def q8_0_roundtrip(w):
    """Quantize a conv filter [out, in, kh, kw] to Q8_0 and back to FP32.

    Mirrors nntrainer's block_q8_0 (32-element blocks, fp16-ish scale =
    amax/127 per block, round-to-nearest) applied per output-channel row over
    CRS = in*kh*kw, matching how the conv filter is block-quantized. Only rows
    where out and CRS are both multiples of 32 are quantized; others pass
    through. Running these dequantized weights at FP32 activation reproduces the
    W8A32 numerics on any host (the real Q8_0 conv kernel is ARM-only).
    """
    out = w.shape[0]
    K = int(np.prod(w.shape[1:]))
    if out % 32 != 0 or K % 32 != 0:
        return w
    flat = w.reshape(out, K // 32, 32).astype(np.float32)
    amax = np.max(np.abs(flat), axis=2, keepdims=True)
    # fp16 round-trip of the scale, as stored in block_q8_0.d
    d = (amax / 127.0).astype(np.float16).astype(np.float32)
    d = np.where(d == 0.0, 1.0, d)
    q = np.clip(np.round(flat / d), -128, 127)
    return (q * d).reshape(w.shape).astype(np.float32)


def apply_sim_q8_conv(tensors):
    """Round-trip every eligible conv filter through Q8_0 (in place)."""
    n = 0
    for name in list(tensors):
        if name.endswith(":filter") and tensors[name].ndim == 4:
            rt = q8_0_roundtrip(tensors[name])
            if rt is not tensors[name]:
                tensors[name] = rt
                n += 1
    print(f"[converter] Q8_0-simulated {n} conv filters (weights dequantized)")
    return tensors


def convert(state):
    """Return {nntrainer_weight_name: np.ndarray(float32)} from a state dict."""
    sd = _prepare_state(state)
    out = {}
    consumed = set()

    def take(k):
        consumed.add(k)
        return sd[k]

    # self.model = nn.Sequential(backbone, head, head_feat) re-registers the
    # same tensors under "model.0/1/2.*"; skip those aliases and use the
    # canonical backbone/head/head_feat paths.
    for k in list(sd.keys()):
        if k.startswith("model."):
            consumed.add(k)
            continue
        if k in consumed:
            continue

        # ---- fused Conv modules: "<mp>.conv.weight" / "<mp>.conv.bias" -----
        if k.endswith(".conv.weight"):
            mp = k[: -len(".conv.weight")]
            out[mp + ":filter"] = (take(k))  # [out,in,kh,kw] (NCHW)
            bk = mp + ".conv.bias"
            if bk in sd:
                out[mp + ":bias"] = (take(bk))
            continue

        # ---- head.final_layer: raw nn.Conv2d (no ".conv" wrapper) ----------
        if k == "head.final_layer.weight":
            out["head.final_layer:filter"] = (take(k))
            if "head.final_layer.bias" in sd:
                out["head.final_layer:bias"] = (take("head.final_layer.bias"))
            continue

        # ---- RTMCC head consolidated into one "rtmcc_head" layer ("head") --
        # mlp.0 ScaleNorm scalar g, mlp.1 Linear, GAU weights, cls_x/cls_y.
        if k == "head.mlp.0.g":
            out["head:mlp_ln_g"] = (take(k)).reshape(1)
            continue
        if k == "head.mlp.1.weight":  # [D, F] -> [F, D]
            out["head:mlp_w"] = (take(k)).T.copy()
            continue
        if k == "head.gau.uv.weight":  # [2e+s, D] -> [D, 2e+s]
            out["head:gau_uv"] = (take(k)).T.copy()
            continue
        if k == "head.gau.o.weight":  # [D, e] -> [e, D]
            out["head:gau_o"] = (take(k)).T.copy()
            continue
        if k == "head.gau.gamma":
            out["head:gau_gamma"] = (take(k))
            continue
        if k == "head.gau.beta":
            out["head:gau_beta"] = (take(k))
            continue
        if k == "head.gau.ln.g":
            out["head:gau_ln_g"] = (take(k)).reshape(1)
            continue
        if k == "head.gau.res_scale.scale":
            out["head:gau_res_scale"] = (take(k))
            continue
        if k == "head.cls_x.weight":  # [bins, D] -> [D, bins]
            out["head:cls_x"] = (take(k)).T.copy()
            continue
        if k == "head.cls_y.weight":  # [bins, D] -> [D, bins]
            out["head:cls_y"] = (take(k)).T.copy()
            continue

        # ---- generic Linear (head_feat.fc for the optional ReID head) ------
        if k.endswith(".weight"):
            mp = k[: -len(".weight")]
            w = (take(k))
            if w.ndim == 2:  # Linear [out,in] -> [in,out]
                out[mp + ":weight"] = w.T.copy()
                bk = mp + ".bias"
                if bk in sd:
                    out[mp + ":bias"] = (take(bk))
                continue

        # ---- unmatched .bias handled alongside its weight above ------------
        if k.endswith(".bias") and k in consumed:
            continue

    unconsumed = [k for k in sd.keys() if k not in consumed]
    if unconsumed:
        raise RuntimeError(
            "unmapped checkpoint keys (update converter):\n  "
            + "\n  ".join(unconsumed)
        )
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Convert YOLOv7ReIDtiny .pt -> nntrainer FP32 safetensors"
    )
    ap.add_argument("--weights", default="pose_merged_v311.pt")
    ap.add_argument("--output", default="yolov7_pose.safetensors")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--dump-names",
        action="store_true",
        help="print produced nntrainer weight names and exit",
    )
    ap.add_argument(
        "--inspect",
        action="store_true",
        help="print the raw checkpoint's parameter keys+shapes and exit "
        "(does not require matching the reconstructed model)",
    )
    ap.add_argument(
        "--sim-q8-conv",
        action="store_true",
        help="dequantized-Q8_0 conv weights (FP32 storage) so W8A32 numerics "
        "can be checked on x86 with the w32a32 preset (real Q8_0 conv is "
        "ARM-only)",
    )
    args = ap.parse_args()

    device = torch.device(args.device)

    if args.inspect:
        state = _load_checkpoint(args.weights, device)
        for k in sorted(state):
            v = state[k]
            shape = tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
            print(f"{k}\t{shape}")
        print(f"# total {len(state)} checkpoint entries")
        return

    state = _load_checkpoint(args.weights, device)
    if state and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    tensors = convert(state)
    if args.sim_q8_conv:
        apply_sim_q8_conv(tensors)

    if args.dump_names:
        for name in sorted(tensors):
            print(f"{name}\t{tuple(tensors[name].shape)}")
        print(f"# total {len(tensors)} tensors")
        return

    save_file(tensors, args.output)
    total = sum(t.nbytes for t in tensors.values())
    print(f"[converter] wrote {args.output}")
    print(f"[converter] {len(tensors)} tensors, {total / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    main()
