# SPDX-License-Identifier: Apache-2.0
"""Build one Gemma4 file with native TL2 experts and affine WH int4 FC/head.

The private first stage copies non-expert records. The final stage recovers
FC/head int4 grids from HF BF16 with channel offsets and no requantization. Experts become TL2CX_WH records: native TL2 codes, N
fp32 scales, N fp32 column sums. No palette or QS2 companion is stored.
Build tools/convert_tl2_model.cpp with hvx_gemv_u8tl2.c before running.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def convert_native(source, output, tool):
    if output.exists():
        raise FileExistsError(output)
    if not tool.is_file():
        raise FileNotFoundError(tool)
    report = json.loads((source / "conversion_report.json").read_text())
    config = json.loads((source / "nntr_config.json").read_text())
    if any(config.get(k) != "Q4_0" for k in
           ("fc_layer_dtype", "embedding_dtype", "lmhead_dtype")):
        raise ValueError("Expected the checked ARM Q4_0 Gemma4 conversion")
    if config.get("moe_layer_dtype") != "QS2CX_WH":
        raise ValueError("Expected exact pure ternary QS2CX_WH experts")
    filename = Path(report["model_file"])
    if filename.name != str(filename):
        raise ValueError("Model filename must not contain directories")
    binary = source / filename
    print("Verifying the source model size and SHA256", flush=True)
    if binary.stat().st_size != report["bytes"] or sha256(binary) != report["sha256"]:
        raise ValueError("Source model size or SHA256 mismatch")
    plan, experts = [], []
    before, after = 0, 0
    for entry in report["tensors"]:
        if entry["offset"] != before:
            raise ValueError("Non-contiguous source weight plan")
        item = dict(entry, offset=after)
        if entry["dtype"] == "QS2CX_WH":
            k, n = entry["shape"]
            if k % 32 or n % 128 or not 0 < k <= 8192 or not 0 < n <= 8192:
                raise ValueError("Unsupported native TL2 shape")
            if entry["bytes"] != k * n // 4 + 4 + 8 * n:
                raise ValueError("Incorrect source expert record size")
            experts.append(entry)
            item.update(dtype="TL2CX_WH", bytes=((k + 23) // 24) *
                        ((n + 127) // 128) * 640 + 8 * n)
        elif entry["dtype"] not in ("Q4_0", "FP32", "QS4CX_WH_AFFINE"):
            raise ValueError("Unsupported non-expert dtype")
        before += entry["bytes"]
        after += item["bytes"]
        plan.append(item)
    if before != report["bytes"] or len(experts) != report["expert_tensors_verified"]:
        raise ValueError("Source weight plan is incomplete")
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < after + 1024**3:
        raise OSError("Insufficient space for the unified model")
    output.mkdir()
    target_name = "nntr_gemma4_ternary_tl2_q40_arm.bin"
    partial = output / (target_name + ".partial")
    catalog = output / "experts.tsv"
    try:
        catalog.write_text("".join(
            f"{e['offset']}\t{e['shape'][0]}\t{e['shape'][1]}\n" for e in experts))
        subprocess.run([str(tool.resolve()), str(binary.resolve()), str(catalog),
                        str(partial)], check=True)
        if partial.stat().st_size != after:
            raise ValueError("Unexpected unified model size")
        if sha256(binary) != report["sha256"]:
            raise ValueError("Source model changed during conversion")
        config.update(model_file_name=target_name, moe_layer_dtype="TL2CX_WH",
                      gemma4_execution_profile="htp_fused",
                      gemma4_weight_format="htp_fused_v1", moe_cache_experts=16,
                      attn_proj_engine="htp", dense_ffn_engine="htp",
                      attention_engine="htp", moe_engine="htp", lmhead_engine="htp",
                      attention_kv_dtype="q8", tokenizer_file="tokenizer.json")
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                     "generation_config.json", "chat_template.jinja"):
            if (source / name).is_file():
                shutil.copyfile(source / name, output / name)
        (output / "nntr_config.json").write_text(json.dumps(config, indent=2) + "\n")
        result = {"source_conversion": str(source.resolve()),
                  "source_sha256": report["sha256"], "model_file": target_name,
                  "bytes": after, "sha256": sha256(partial),
                  "storage_format": "TL2CX_WH/1", "expert_payload_copies": 1,
                  "expert_tensors_verified": len(experts),
                  "max_expert_weight_error": 0, "expert_scale_compensation": False,
                  "non_expert_matrix_conversion": "Q4_0 ARM records copied verbatim",
                  "norms_router_and_scalar_dtype": "FP32",
                  "verification": "Every serialized TL2 code decoded and compared to QS2; "
                                  "scales copied exactly; column sums checked",
                  "requires_updated_htp_runtime_and_dsp": True,
                  "device_validation": "not performed", "tensors": plan}
        (output / "conversion_report.json").write_text(json.dumps(result, indent=2) + "\n")
        partial.rename(output / target_name)
        catalog.unlink()
    except BaseException:
        shutil.rmtree(output)
        raise
    print(f"Unified model: {output / target_name}\nBytes: {after:,}; "
          f"experts verified: {len(experts)}", flush=True)
    return result


def convert(source, output, tool, hf_source):
    # Keep the Q4_0 intermediate private. The published file contains only
    # native TL2 experts, original-grid affine WH FC/head and Q4_0 embedding.
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools"))
    from convert_gemma4_fc_wh import convert as convert_fc
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gemma4-tl2-", dir=output.parent) as tmp:
        native = Path(tmp) / "native"
        convert_native(source, native, tool)
        convert_fc(native, output, hf_source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Verified hf_npu_converter.py output")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--hf-source", type=Path, required=True,
                        help="Original BF16-dequantized HF checkpoint for exact int4-grid recovery")
    args = parser.parse_args()
    convert(args.source, args.output, args.converter, args.hf_source)
