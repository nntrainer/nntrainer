"""
Weight Converter Agent (no LLM).

Concatenates the downloaded HF weight tensors into a single flat float32
binary. This is a mechanical, LLM-free pass.

IMPORTANT (AUDIT L2): tensors are written in filename + safetensors-key
order, which is NOT the order nntrainer's NeuralNetwork::load_from_file
walks its addLayer() calls in generated_model.cpp. The blob is therefore
a convenience artifact, NOT yet a drop-in nntrainer weight file -- loading
it as-is would misalign weights to layers. Aligning the write order to the
graph's emitted-layer order is a known pending task; until then the agent
emits a loud warning and a sidecar README next to the file. (Doing it right
also needs the graph IR, which the current background-thread placement of
this agent doesn't have -- so it's a deliberate, validated follow-up, not a
silent guess here.)

The converted file is cached alongside the raw weights (same 30-day
window) so re-running the pipeline for the same model doesn't repeat
the conversion pass; it's copied into this run's output directory
either way so it always shows up in Artifacts.
"""
import os
import shutil

from . import cache
from .events import bus


def run(state: dict) -> dict:
    """Convert safetensors weights to a flat float32 binary."""
    bus.agent_status("weight_converter", "running")

    weights_path = state.get("weights_path")
    if not weights_path or not os.path.exists(weights_path):
        bus.log("No weights found -- skipping conversion", "warn")
        bus.agent_status("weight_converter", "error", "no weights")
        return state

    converted_dir = os.path.join(state["out_dir"], "converted")
    os.makedirs(converted_dir, exist_ok=True)
    out_path = os.path.join(converted_dir, "converted_weights.bin")

    safetensor_files = list(Path(weights_path).glob("*.safetensors"))
    if not safetensor_files:
        bus.log("No .safetensors files found -- skipping weight conversion", "warn")
        bus.agent_status("weight_converter", "error", "no .safetensors found")
        return state

    written, total_params = 0, 0
    try:
        with open(out_path, "wb") as out_f:
            for st_file in sorted(safetensor_files):
                with safe_open(st_file, framework="np") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        # Convert to float32, handling special dtypes like bfloat16
                        try:
                            # Standard dtypes (float32, float16, int32, etc.)
                            if hasattr(tensor.dtype, 'name'):
                                dtype_str = tensor.dtype.name
                            else:
                                dtype_str = str(tensor.dtype)
                            
                            # bfloat16 needs special handling (not a numpy native type)
                            if 'bfloat16' in dtype_str.lower():
                                # Convert bfloat16 -> float32 via intermediate float
                                # bfloat16 is essentially float32 with reduced precision
                                import struct
                                # Reinterpret bfloat16 bytes as float32 (add 16 zero bits)
                                bf16_bytes = tensor.tobytes()
                                f32_bytes = b''
                                for i in range(0, len(bf16_bytes), 2):
                                    # Each bfloat16 is 2 bytes; pad with zeros to make float32
                                    f32_bytes += bf16_bytes[i:i+2] + b'\x00\x00'
                                out_f.write(f32_bytes)
                            else:
                                # Standard numpy conversion
                                out_f.write(tensor.astype("float32").tobytes())
                        except Exception as e:
                            bus.log(
                                f"Warning converting {key} ({dtype_str}): {e} -- "
                                f"skipping this tensor",
                                "warn"
                            )
                            continue
                        total_params += tensor.size
                        written += 1
        state["converted_weights_path"] = out_path
        size_mb = os.path.getsize(out_path) / (1024 * 1024)

        # Cache the converted file for reuse on the next run of this model.
        # cache_root may not exist yet for a *local* model path (weight_download
        # never created it), so make it before copying -- otherwise the copy
        # raised FileNotFoundError and the whole (successful) conversion got
        # reported as a failure.
        os.makedirs(cache_root, exist_ok=True)
        shutil.copyfile(out_path, cached_path)
        cache.set_entry(cache_root, "converted", {"tensors": written, "size_mb": round(size_mb, 1)})

        # HONEST SCOPING (see AUDIT L2): this pass concatenates tensors in
        # file/key order, which is NOT nntrainer's addLayer() load order, so the
        # blob is not yet safe to feed to NeuralNetwork::load_from_file. Say so
        # loudly and drop a sidecar note rather than letting the (now-corrected)
        # docstring imply an alignment that isn't there.
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
    except Exception as exc:
        bus.log(f"Weight conversion error: {exc}", "error")
        bus.agent_status("weight_converter", "error", str(exc))
        state["converted_weights_path"] = None

    return state
    bus.agent_status("weight_converter", "running")
    weights_path = state.get("weights_path")
    out_dir = state["out_dir"]
    model_name = state["model_name"]

    if not weights_path:
        bus.log("No weights available to convert -- skipping", "warn")
        bus.agent_status("weight_converter", "error", "no weights_path")
        return state

    converted_dir = os.path.join(out_dir, "converted")
    os.makedirs(converted_dir, exist_ok=True)
    out_path = os.path.join(converted_dir, "converted_weights.bin")

    cache_root = cache.cache_root_for(out_dir, model_name)
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

    try:
        from safetensors import safe_open
    except ImportError:
        bus.log("safetensors not installed -- skipping weight conversion", "warn")
        bus.agent_status("weight_converter", "error", "safetensors not installed")
        return state

    safetensor_files = [
        os.path.join(weights_path, f)
        for f in os.listdir(weights_path)
        if f.endswith(".safetensors")
    ] if os.path.isdir(weights_path) else []

    if not safetensor_files:
        bus.log("No .safetensors files found -- skipping weight conversion", "warn")
        bus.agent_status("weight_converter", "error", "no .safetensors found")
        return state

    written, total_params = 0, 0
    try:
        with open(out_path, "wb") as out_f:
            for st_file in sorted(safetensor_files):
                with safe_open(st_file, framework="np") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        out_f.write(tensor.astype("float32").tobytes())
                        total_params += tensor.size
                        written += 1
        state["converted_weights_path"] = out_path
        size_mb = os.path.getsize(out_path) / (1024 * 1024)

        # Cache the converted file for reuse on the next run of this model.
        # cache_root may not exist yet for a *local* model path (weight_download
        # never created it), so make it before copying -- otherwise the copy
        # raised FileNotFoundError and the whole (successful) conversion got
        # reported as a failure.
        os.makedirs(cache_root, exist_ok=True)
        shutil.copyfile(out_path, cached_path)
        cache.set_entry(cache_root, "converted", {"tensors": written, "size_mb": round(size_mb, 1)})

        # HONEST SCOPING (see AUDIT L2): this pass concatenates tensors in
        # file/key order, which is NOT nntrainer's addLayer() load order, so the
        # blob is not yet safe to feed to NeuralNetwork::load_from_file. Say so
        # loudly and drop a sidecar note rather than letting the (now-corrected)
        # docstring imply an alignment that isn't there.
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

        bus.log(f"Converted {written} tensors ({total_params:,} params, {size_mb:.1f} MB) -> {out_path}")
        bus.agent_status("weight_converter", "done", f"{written} tensors")
    except Exception as exc:
        bus.log(f"Weight conversion failed: {exc}", "warn")
        bus.agent_status("weight_converter", "error", str(exc))
        state.setdefault("errors", []).append(f"weight_converter: {exc}")

    return state
