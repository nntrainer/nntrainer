#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replace FC/head Q4_0 records in a native-TL2 Gemma4 pack with affine QS4CX_WH.

Recover the existing int4 grid from HF BF16, including channel offsets.
No min/max requantization is performed.
Embedding, experts, routers and norms are copied and hashed byte for byte.
Output is a new directory; the source pack is never changed.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hf_name(name):
    if name == "output_of_causallm:weight":
        return "lm_head.weight"
    match = re.fullmatch(r"layer(\d+)_(w[qkv]|attention_out|ffn_(?:gate|up|down)):weight", name)
    if not match:
        raise ValueError(f"Unknown FC tensor {name}")
    layer, kind = match.groups()
    suffix = (f"self_attn.{kind[1]}_proj" if kind.startswith("w") else
              "self_attn.o_proj" if kind == "attention_out" else
              f"mlp.{kind[4:]}_proj")
    return f"model.layers.{layer}.{suffix}.weight"


def bf16_round(w):
    bits = np.asarray(w, np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) &
            np.uint32(0xffff0000)).view(np.float32)


def recover(w):
    """Recover q, scale, offset; every observed value must round-trip as BF16.

    Source FP32 quantizer scales are absent from this checkpoint. Intersect
    BF16 rounding intervals for all observed integer magnitudes, choosing a
    representative scale inside the intersection, then validate every value.
    """
    if w.ndim != 2 or not np.isfinite(w).all() or not np.array_equal(w, bf16_round(w)):
        raise ValueError("Expected a finite BF16 output-major matrix")
    magnitude = np.abs(w)
    unit = np.where(magnitude > 0, magnitude, np.inf).min(axis=1)
    zero = ~np.isfinite(unit)
    unit[zero] = 1
    raw = np.rint(w / unit[:, None])
    if np.any(np.abs(raw) > 15):
        raise ValueError("No inferred affine int4 grid")
    codes = raw.astype(np.int32)
    lo_code, hi_code = codes.min(axis=1), codes.max(axis=1)
    if np.any(hi_code - lo_code > 15):
        raise ValueError("More than 16 consecutive integer levels")
    bits = magnitude.view(np.uint32) >> 16
    prev = ((np.maximum(bits, 1) - 1) << 16).astype(np.uint32).view(np.float32)
    following = ((bits + 1) << 16).astype(np.uint32).view(np.float32)
    denom = np.maximum(np.abs(codes), 1)
    lower = (prev.astype(np.float64) + magnitude) / (2 * denom)
    upper = (following.astype(np.float64) + magnitude) / (2 * denom)
    low = np.where(codes != 0, lower, -np.inf).max(axis=1)
    high = np.where(codes != 0, upper, np.inf).min(axis=1)
    low[zero], high[zero] = 1, 1
    if np.any(low > high):
        raise ValueError("No scale satisfies all BF16 rounding intervals")
    scale = ((low + high) / 2).astype(np.float32)
    for row in np.flatnonzero(np.any(bf16_round(codes * scale[:, None]) != w, axis=1)):
        candidates = [np.float32(low[row]), np.float32(high[row]), unit[row]]
        candidates += [np.nextafter(x, np.float32(direction))
                       for x in candidates[:] for direction in (-np.inf, np.inf)]
        for candidate in candidates:
            if candidate > 0 and np.array_equal(bf16_round(codes[row] * candidate), w[row]):
                scale[row] = candidate
                break
        else:
            raise ValueError("Cannot reproduce a source BF16 channel")
    offset = np.clip(np.zeros(len(w), np.int32), hi_code - 7, lo_code + 8)
    offset[zero] = 0
    q = (codes - offset[:, None]).astype(np.int8)
    if not np.array_equal(bf16_round((q.astype(np.int32) + offset[:, None]) *
                                    scale[:, None]), w):
        raise ValueError("Affine int4 recovery failed BF16 verification")
    return q, scale.astype("<f4"), offset.astype("<i4")


def pack(q):
    n, k = q.shape
    tiles = (q.T.astype(np.uint8) & 15).reshape(k // 32, 32, n // 32, 32)
    slots = tiles.transpose(0, 2, 1, 3).reshape(k // 32, n // 32, 4, 2, 4, 32)
    pairs = slots.transpose(0, 1, 2, 5, 4, 3).reshape(k // 32, n // 32, 512, 2)
    return np.ascontiguousarray(pairs[..., 0] | (pairs[..., 1] << 4))


def verify_chunk(stream, offset, k, n, c0, q, scale, channel_offset):
    # Independent byte/nibble address formula from htp_wh_layout.h.
    r = np.arange(32)[:, None]
    c = np.arange(32)[None, :]
    byte = (r // 8) * 128 + c * 4 + r % 4
    shift = ((r // 4) % 2) * 4
    cn = len(q)
    sums = np.zeros(cn, dtype=np.int32)
    for kt in range(k // 32):
        stream.seek(offset + (kt * (n // 32) + c0 // 32) * 512)
        tiles = np.frombuffer(stream.read(cn * 16), np.uint8).reshape(cn // 32, 512)
        values = ((tiles[:, byte] >> shift) & 15).astype(np.int8)
        values[values > 7] -= 16
        restored = values.transpose(0, 2, 1).reshape(cn, 32)
        if not np.array_equal(restored, q[:, kt * 32:(kt + 1) * 32]):
            raise ValueError("Serialized WH nibble verification failed")
        sums += restored.sum(axis=1, dtype=np.int32)
    for off, expected in ((k * n // 2 + c0 * 4, scale),
                          (k * n // 2 + (n + c0) * 4, sums.astype("<f4")),
                          (k * n // 2 + (2 * n + c0) * 4, channel_offset)):
        stream.seek(offset + off)
        if stream.read(cn * 4) != expected.tobytes():
            raise ValueError("Serialized WH scale/column sum verification failed")


def convert(source, output, hf_source):
    from safetensors import safe_open
    import torch
    torch.set_num_threads(1)

    if output.exists():
        raise FileExistsError(output)
    report = json.loads((source / "conversion_report.json").read_text())
    config = json.loads((source / "nntr_config.json").read_text())
    if config.get("moe_layer_dtype") != "TL2CX_WH":
        raise ValueError("Expected an already verified native-TL2 pack")
    binary = source / report["model_file"]
    print("Verifying native-TL2 source SHA256", flush=True)
    if binary.stat().st_size != report["bytes"] or sha256(binary) != report["sha256"]:
        raise ValueError("Source pack size/hash mismatch")
    index = json.loads((hf_source / "model.safetensors.index.json").read_text())["weight_map"]
    before = after = 0
    plan = []
    for entry in report["tensors"]:
        if entry["offset"] != before:
            raise ValueError("Non-contiguous source tensor plan")
        item = dict(entry, offset=after)
        if entry["dtype"] == "Q4_0" and entry["name"] != "embedding0:Embedding":
            k, n = entry["shape"]
            if k % 32 or n % 32:
                raise ValueError("WH requires complete tiles")
            name = hf_name(entry["name"])
            if name not in index:
                raise KeyError(name)
            item.update(dtype="QS4CX_WH_AFFINE", bytes=k * n // 2 + 12 * n, hf_name=name)
        elif entry["dtype"] not in ("Q4_0", "TL2CX_WH", "FP32"):
            raise ValueError("Unexpected source dtype")
        before += entry["bytes"]
        after += item["bytes"]
        plan.append(item)
    if before != report["bytes"]:
        raise ValueError("Incomplete source tensor plan")
    if shutil.disk_usage(output.parent).free < after + 1024**3:
        raise OSError("Insufficient space")
    output.mkdir()
    target_name = "nntr_gemma4_ternary_tl2_qs4cx_wh_affine.bin"
    partial = output / (target_name + ".partial")
    copied_hash = hashlib.sha256()
    copied_verify = hashlib.sha256()
    metrics = []
    try:
        with binary.open("rb") as src, partial.open("xb+") as dst:
            dst.truncate(after)
            for entry, item in zip(report["tensors"], plan):
                if item["dtype"] != "QS4CX_WH_AFFINE":
                    src.seek(entry["offset"])
                    dst.seek(item["offset"])
                    left = entry["bytes"]
                    while left:
                        chunk = src.read(min(left, 4 * 1024**2))
                        if not chunk:
                            raise EOFError("Truncated source pack")
                        copied_hash.update(chunk)
                        dst.write(chunk)
                        left -= len(chunk)
                    continue
                k, n = item["shape"]
                maximum = squared = energy = 0.
                with safe_open(hf_source / index[item["hf_name"]], framework="pt") as handle:
                    tensor = handle.get_slice(item["hf_name"])
                    if tensor.get_shape() != [n, k]:
                        raise ValueError(f"HF shape mismatch: {item['hf_name']}")
                    for c0 in range(0, n, 256):
                        w = tensor[c0:c0 + 256].float().numpy()
                        q, scale, offsets = recover(w)
                        tiles = pack(q)
                        for kt, row in enumerate(tiles):
                            dst.seek(item["offset"] + (kt * (n // 32) + c0 // 32) * 512)
                            dst.write(row.tobytes())
                        for at, values in ((k * n // 2 + c0 * 4, scale),
                                           (k * n // 2 + (n + c0) * 4,
                                            q.sum(axis=1, dtype=np.int32).astype("<f4")),
                                           (k * n // 2 + (2 * n + c0) * 4, offsets)):
                            dst.seek(item["offset"] + at)
                            dst.write(values.tobytes())
                        dst.flush()
                        verify_chunk(dst, item["offset"], k, n, c0, q, scale, offsets)
                        error = (q.astype(np.int32) + offsets[:, None]) * scale[:, None] - w
                        maximum = max(maximum, float(np.abs(error).max()))
                        squared += np.square(error.astype(np.float64)).sum()
                        energy += np.square(w.astype(np.float64)).sum()
                metrics.append({"name": item["name"], "max_fp32_grid_vs_bf16_error": maximum,
                                "max_bf16_roundtrip_error": 0,
                                "fp32_grid_vs_bf16_rmse": float(np.sqrt(squared / (k * n))),
                                "fp32_grid_vs_bf16_relative_rmse": float(np.sqrt(squared / energy)) if energy else 0.})
                print(f"WH {len(metrics):3}: {item['name']} [{k},{n}], BF16_roundtrip=exact; FP32_grid_error={maximum:.7g}", flush=True)
            dst.flush()
            for item in plan:
                if item["dtype"] == "QS4CX_WH_AFFINE":
                    continue
                dst.seek(item["offset"])
                left = item["bytes"]
                while left:
                    chunk = dst.read(min(left, 4 * 1024**2))
                    if not chunk:
                        raise EOFError("Truncated output pack")
                    copied_verify.update(chunk)
                    left -= len(chunk)
        if copied_hash.digest() != copied_verify.digest():
            raise ValueError("Copied embedding/TL2/router/norm verification failed")
        config.update(model_file_name=target_name, fc_layer_dtype="QS4CX_WH_AFFINE",
                      lmhead_dtype="QS4CX_WH_AFFINE", lmhead_engine="htp")
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                     "generation_config.json", "chat_template.jinja"):
            if (source / name).is_file():
                shutil.copyfile(source / name, output / name)
        (output / "nntr_config.json").write_text(json.dumps(config, indent=2) + "\n")
        result = dict(report, model_file=target_name, bytes=after, sha256=sha256(partial),
                      source_native_tl2_sha256=report["sha256"],
                      hf_fc_head_source=str(hf_source.resolve()),
                      storage_format="TL2CX_WH/1 + QS4CX_WH_AFFINE/1",
                      requantization=False,
                      fc_head_weight_formula="BF16_round(scale[n]*(signed_int4[k,n]+offset[n]))",
                      scale_provenance="Representative of BF16 rounding interval, original FP32 scale not recorded",
                      non_expert_matrix_conversion="FC/head original affine int4 grid recovered from HF BF16; embedding Q4_0 unchanged",
                      fc_head_quantization="No requantization: every recovered value rounds back to the source BF16",
                      fc_head_tensors_verified=len(metrics), fc_head_errors=metrics,
                      unchanged_records_sha256=copied_hash.hexdigest(),
                      dtype_counts=dict(Counter(e["dtype"] for e in plan)), tensors=plan,
                      verification="Every serialized FC/head WH nibble, scale, sum and offset checked against a BF16-exact recovered grid; embedding, TL2 experts, FP32 records hash-identical to verified native pack")
        (output / "conversion_report.json").write_text(json.dumps(result, indent=2) + "\n")
        partial.rename(output / target_name)
    except BaseException:
        shutil.rmtree(output)
        raise
    print(f"Complete: {output / target_name}; {after:,} bytes; {len(metrics)} WH tensors", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-source", type=Path, required=True)
    args = parser.parse_args()
    convert(args.source, args.output, args.hf_source)
