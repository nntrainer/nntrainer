t"""
Weight Download Agent - Downloads model weights from HuggingFace Hub.
"""
import os

from . import cache
from .events import bus


def run(state: dict) -> dict:
    bus.agent_status("weight_download", "running")
    model_name = state["model_name"]
    out_dir = state["out_dir"]
    custom_weights_path = state.get("custom_weights_path")

    if custom_weights_path and os.path.isdir(custom_weights_path):
        state["weights_path"] = custom_weights_path
        bus.log(f"Using custom optimized weights from: {custom_weights_path}")
        bus.agent_status("weight_download", "done", "custom optimized")
        return state

    if os.path.isdir(model_name):
        state["weights_path"] = model_name
        bus.log(f"Using local model directory: {model_name}")
        bus.agent_status("weight_download", "done", "local path")
        return state

    cache_root = cache.cache_root_for(out_dir, model_name)
    weights_dir = os.path.join(cache_root, "weights")
    meta = cache.read_meta(cache_root)

    if cache.is_fresh(meta, "weights") and os.path.isdir(weights_dir) and os.listdir(weights_dir):
        age = cache.age_days(meta, "weights")
        state["weights_path"] = weights_dir
        size_mb = _dir_size_mb(weights_dir)
        bus.log(f"Using cached weights ({age:.1f} days old, {size_mb:.1f} MB, cache expires after 30 days)")
        bus.agent_status("weight_download", "done", f"cached, {age:.1f}d old")
        return state

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        bus.log("huggingface_hub not installed -- skipping weight download", "warn")
        bus.agent_status("weight_download", "error", "huggingface_hub not installed")
        return state

    os.makedirs(weights_dir, exist_ok=True)

    try:
        path = snapshot_download(
            repo_id=model_name,
            local_dir=weights_dir,
            allow_patterns=["*.safetensors", "*.bin", "*.json", "*.model", "tokenizer*"],
        )
        state["weights_path"] = path
        size_mb = _dir_size_mb(path)
        cache.set_entry(cache_root, "weights", {"path": path, "size_mb": round(size_mb, 1)})
        bus.log(f"Downloaded weights to workspace cache ({size_mb:.1f} MB) -- reused for 30 days")
        bus.agent_status("weight_download", "done", f"{size_mb:.1f} MB")
    except Exception as exc:
        bus.log(f"Weight download failed: {exc}", "warn")
        bus.agent_status("weight_download", "error", str(exc))
        state.setdefault("errors", []).append(f"weight_download: {exc}")

    return state


def _dir_size_mb(path: str) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / (1024 * 1024)
