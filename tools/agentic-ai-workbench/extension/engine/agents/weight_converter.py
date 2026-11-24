# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file weight_converter.py
# @brief Weight Converter Agent - Converts safetensors weights to flat float32 binary.
# @author prachi-t17 <prachi.t@samsung.com>

"""
Weight Converter Agent - Converts safetensors weights to flat float32 binary.
"""
import os
import shutil
from pathlib import Path

from . import cache
from .events import bus


# pylint: disable-next=too-many-locals,too-many-branches,too-many-statements
def run(state: dict) -> dict:
    """Pipeline step: convert the safetensors weights to nntrainer's flat float32 file."""
    bus.agent_status("weight_converter", "running")

    weights_path = state.get("weights_path")
    if not weights_path or not os.path.exists(weights_path):
        bus.log("No weights found -- skipping conversion", "warn")
        bus.agent_status("weight_converter", "error", "no weights")
        return state

    converted_dir = os.path.join(state["out_dir"], "converted")
    os.makedirs(converted_dir, exist_ok=True)
    out_path = os.path.join(converted_dir, "converted_weights.bin")

    model_name = state.get("model_name", "unknown")
    cache_root = cache.cache_root_for(state["out_dir"], model_name)
    cached_path = os.path.join(cache_root, "converted_weights.bin")
    meta = cache.read_meta(cache_root)

    if cache.is_fresh(meta, "converted") and os.path.exists(cached_path):
        age = cache.age_days(meta, "converted")
        shutil.copyfile(cached_path, out_path)
        state["converted_weights_path"] = out_path
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        bus.log(f"Using cached converted weights ({age:.1f} days old, {size_mb:.1f} MB)")
        bus.agent_status("weight_converter", "done", f"cached, {age:.1f}d old")
        return state

    safetensor_files = list(Path(weights_path).glob("*.safetensors"))
    if not safetensor_files:
        bus.log("No .safetensors files found -- skipping weight conversion", "warn")
        bus.agent_status("weight_converter", "error", "no .safetensors found")
        return state

    try:
        # pylint: disable-next=import-outside-toplevel  # optional package
        from safetensors import safe_open
    except ImportError:
        bus.log("safetensors not installed -- skipping weight conversion", "warn")
        bus.agent_status("weight_converter", "error", "safetensors not installed")
        return state

    use_torch = False
    try:
        # pylint: disable-next=import-outside-toplevel  # optional package
        import torch
        use_torch = True
    except ImportError:
        pass

    written, total_params = 0, 0
    try:
        with open(out_path, "wb") as out_f:
            for st_file in sorted(safetensor_files):
                framework = "pt" if use_torch else "np"
                with safe_open(st_file, framework=framework) as handle:
                    for key in handle.keys():
                        try:
                            tensor = handle.get_tensor(key)

                            if use_torch:
                                if tensor.dtype == torch.bfloat16:
                                    tensor = tensor.to(torch.float32)
                                elif tensor.dtype != torch.float32:
                                    tensor = tensor.to(torch.float32)
                                out_f.write(tensor.numpy().tobytes())
                                total_params += tensor.numel()
                            else:
                                if hasattr(tensor.dtype, 'name'):
                                    dtype_str = tensor.dtype.name
                                else:
                                    dtype_str = str(tensor.dtype)

                                if 'bfloat16' in dtype_str.lower():
                                    bf16_bytes = tensor.tobytes()
                                    f32_bytes = bytearray()
                                    for i in range(0, len(bf16_bytes), 2):
                                        f32_bytes.extend(bf16_bytes[i:i+2])
                                        f32_bytes.extend(b'\x00\x00')
                                    out_f.write(bytes(f32_bytes))
                                else:
                                    out_f.write(tensor.astype("float32").tobytes())
                                total_params += tensor.size
                            written += 1
                        # a bad tensor or shard is reported, not fatal
                        except Exception as err:  # pylint: disable=broad-except
                            bus.log(
                                f"Warning converting {key}: {err} -- "
                                f"skipping this tensor",
                                "warn"
                            )
                            continue

        state["converted_weights_path"] = out_path
        size_mb = os.path.getsize(out_path) / (1024 * 1024)

        os.makedirs(cache_root, exist_ok=True)
        shutil.copyfile(out_path, cached_path)
        cache.set_entry(cache_root, "converted", {"tensors": written, "size_mb": round(size_mb, 1)})

        bus.log(
            "NOTE: converted_weights.bin is a raw tensor concatenation in "
            "file/key order, NOT aligned to the generated model's layer order -- "
            "do not load it as-is until graph-order alignment is implemented.",
            "warn",
        )
        try:
            with open(os.path.join(converted_dir, "converted_weights.README.txt"), "w",
                      encoding="utf-8") as note:
                note.write(
                    "converted_weights.bin\n"
                    "=====================\n"
                    "Raw float32 concatenation of the model's safetensors tensors in\n"
                    "filename + key order. This does NOT match the order nntrainer's\n"
                    "load_from_file walks addLayer() in generated_model.cpp, so it is\n"
                    "NOT yet safe to load directly. Aligning tensor write-order to the\n"
                    "graph's emitted-layer order is a known pending task (AUDIT L2).\n"
                )
        except OSError:
            pass

        bus.log(
            f"Converted {written} tensors ({total_params:,} params, {size_mb:.1f} MB)",
            "ok"
        )
        bus.agent_status("weight_converter", "done", f"{written} tensors")
    # a bad tensor or shard is reported, not fatal
    except Exception as exc:  # pylint: disable=broad-except
        bus.log(f"Weight conversion error: {exc}", "error")
        bus.agent_status("weight_converter", "error", str(exc))
        state["converted_weights_path"] = None

    return state
