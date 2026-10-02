"""
Weight Download Agent - Downloads model weights from HuggingFace Hub.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from ..core import cache
from ..events import bus
from ..core.cache import WEIGHTS_CACHE_EXPIRY_DAYS


def run(state: dict) -> dict:
    bus.agent_status("weight_download", "running")
    model_name = state["model_name"]
    out_dir = state["out_dir"]
    custom_weights_path = state.get("custom_weights_path")

    if custom_weights_path and os.path.isdir(custom_weights_path):
        return _use_existing_weights(state, custom_weights_path, "custom optimized")
        return state

    if os.path.isdir(model_name):
        return _use_existing_weights(state, model_name, "local path")

    cache_root = cache.cache_root_for(out_dir, model_name)
    weights_dir = os.path.join(cache_root, "weights")
    meta = cache.read_meta(cache_root)

    valid, details = validate_weights(weights_dir)
    if cache.is_fresh(meta, "weights", WEIGHTS_CACHE_EXPIRY_DAYS) and valid:
        age = cache.age_days(meta, "weights")
        state["weights_path"] = weights_dir
        state["weights_verified"] = True
        state["weight_manifest"] = details
        size_mb = _dir_size_mb(weights_dir)
        bus.log(
            f"Using verified cached weights ({age:.1f} days old, {size_mb:.1f} MB; "
            f"cache expires after {WEIGHTS_CACHE_EXPIRY_DAYS} days)"
        )
        bus.agent_status("weight_download", "done", f"cached, {age:.1f}d old")
        return state

    if os.path.isdir(weights_dir) and os.listdir(weights_dir) and not valid:
        bus.log(f"Cached weights failed integrity validation ({details.get('reason')}); re-downloading", "warn")

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
            # A corrupt local snapshot must not be silently reused. Normal
            # expired entries still use Hugging Face's ETag-aware cache.
            force_download=not valid and os.path.isdir(weights_dir) and bool(os.listdir(weights_dir)),
        )
        valid, details = validate_weights(path)
        if not valid:
            raise RuntimeError(f"downloaded weights did not pass integrity validation: {details.get('reason')}")
        state["weights_path"] = path
        state["weights_verified"] = True
        state["weight_manifest"] = details
        size_mb = _dir_size_mb(path)
        cache.set_entry(cache_root, "weights", {
            "path": path,
            "size_mb": round(size_mb, 1),
            "manifest": details,
        })
        bus.log(
            f"Downloaded and verified weights to workspace cache ({size_mb:.1f} MB) -- "
            f"reused for {WEIGHTS_CACHE_EXPIRY_DAYS} days"
        )
        bus.agent_status("weight_download", "done", f"{size_mb:.1f} MB")
    except Exception as exc:
        state["weights_verified"] = False
        bus.log(f"Weight download failed: {exc}", "warn")
        bus.agent_status("weight_download", "error", str(exc))
        state.setdefault("errors", []).append(f"weight_download: {exc}")

    return state


def _use_existing_weights(state: dict, path: str, source: str) -> dict:
    """Accept supplied/local weights only after the same validation as cache."""
    valid, details = validate_weights(path)
    if not valid:
        state["weights_verified"] = False
        state.setdefault("errors", []).append(f"weight_download: invalid {source} weights: {details.get('reason')}")
        bus.log(f"{source.capitalize()} weights failed integrity validation: {details.get('reason')}", "error")
        bus.agent_status("weight_download", "error", "integrity check failed")
        return state

    state["weights_path"] = path
    state["weights_verified"] = True
    state["weight_manifest"] = details
    bus.log(f"Using verified {source} weights from: {path}")
    bus.agent_status("weight_download", "done", source)
    return state


def validate_weights(path: str) -> Tuple[bool, Dict[str, object]]:
    """Validate checkpoint presence and safetensors headers without loading tensors.

    ``safe_open`` parses the full safetensors header, catching truncation and
    malformed offsets while avoiding the RAM cost of materialising a model.
    For PyTorch ``.bin`` checkpoints we deliberately do not unpickle data;
    their presence, size, and index-shard consistency are checked instead.
    """
    root = Path(path)
    if not root.is_dir():
        return False, {"reason": "weight path is not a directory"}

    safetensors = sorted(root.rglob("*.safetensors"))
    bins = sorted(root.rglob("*.bin"))
    checkpoint_files = safetensors + bins
    if not checkpoint_files:
        return False, {"reason": "no .safetensors or .bin checkpoint files found"}

    for checkpoint in checkpoint_files:
        try:
            if checkpoint.stat().st_size <= 0:
                return False, {"reason": f"empty checkpoint file: {checkpoint.name}"}
        except OSError as exc:
            return False, {"reason": f"cannot stat {checkpoint.name}: {exc}"}

    for index in root.rglob("*.index.json"):
        try:
            with index.open("r", encoding="utf-8") as handle:
                weight_map = json.load(handle).get("weight_map", {})
            missing = sorted({name for name in weight_map.values() if not (index.parent / name).is_file()})
            if missing:
                return False, {"reason": f"missing indexed shard(s): {', '.join(missing[:3])}"}
        except (OSError, ValueError, AttributeError) as exc:
            return False, {"reason": f"invalid checkpoint index {index.name}: {exc}"}

    if safetensors:
        try:
            from safetensors import safe_open
        except ImportError:
            return False, {"reason": "safetensors is required to validate checkpoint headers"}
        try:
            for checkpoint in safetensors:
                with safe_open(str(checkpoint), framework="np") as handle:
                    # Iterating keys forces safetensors to parse/validate its
                    # header, but does not read the model tensors themselves.
                    tuple(handle.keys())
        except Exception as exc:
            return False, {"reason": f"invalid safetensors header: {exc}"}

    return True, {
        "checkpoint_files": len(checkpoint_files),
        "safetensors_files": len(safetensors),
        "bin_files": len(bins),
        "total_bytes": sum(item.stat().st_size for item in checkpoint_files),
    }


def _dir_size_mb(path: str) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / (1024 * 1024)
