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
import struct

import numpy as np
from htp_weight_utils import (bf16_round, bf16_rows, pack, recover, sha256,
                              verify_chunk, write_affine_wh)


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


def convert(source, output, hf_source):
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
        elif entry["dtype"] not in ("Q4_0", "TL2CX_WH", "FP32", "QS4CX_WH_AFFINE"):
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
                if "hf_name" not in item:  # copied: not an FC/head record of this pass
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
                metric = write_affine_wh(
                    dst, item, bf16_rows(hf_source / index[item["hf_name"]],
                                        item["hf_name"], [n, k]))
                metrics.append(metric)
                maximum = metric["max_fp32_grid_vs_bf16_error"]
                print(f"WH {len(metrics):3}: {item['name']} [{k},{n}], BF16_roundtrip=exact; FP32_grid_error={maximum:.7g}", flush=True)
            dst.flush()
            for item in plan:
                if "hf_name" in item:
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
