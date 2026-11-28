# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file weight_download.py
# @brief Weight Download Agent - Downloads model weights from HuggingFace Hub.
# @author prachi-t17 <prachi.t@samsung.com>

"""
Weight Download Agent - Downloads model weights from HuggingFace Hub.
"""
import os
import hashlib

from . import cache
from .events import bus


def run(state: dict) -> dict:
    """Pipeline step: download the model's weights from the Hugging Face Hub."""
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
    weights_dir = weights_dir_for(cache_root, out_dir, model_name)
    meta = cache.read_meta(cache_root)

    if cache.is_fresh(meta, "weights") and os.path.isdir(weights_dir) and os.listdir(weights_dir):
        age = cache.age_days(meta, "weights")
        state["weights_path"] = weights_dir
        size_mb = _dir_size_mb(weights_dir)
        bus.log(
            f"Using cached weights ({age:.1f} days old, {size_mb:.1f} MB, cache expires after 30"
            " days)"
        )
        bus.agent_status("weight_download", "done", f"cached, {age:.1f}d old")
        return state

    try:
        # pylint: disable-next=import-outside-toplevel  # optional package
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
    # Hub errors vary; the step reports the failure
    except Exception as exc:  # pylint: disable=broad-except
        bus.log(f"Weight download failed: {exc}", "warn")
        bus.agent_status("weight_download", "error", str(exc))
        state.setdefault("errors", []).append(f"weight_download: {exc}")

    return state


# While downloading into local_dir, huggingface_hub writes
# <local_dir>\.cache\huggingface\download\<28-char id>.<64-hex sha>.<8 hex>.incomplete
# -- about 145 characters on top of local_dir.
_HF_STAGING_CHARS = 145
_WIN_MAX_PATH = 259


def _long_paths_enabled() -> bool:
    try:
        # pylint: disable-next=import-outside-toplevel  # Windows only
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem"
        ) as k:
            return winreg.QueryValueEx(k, "LongPathsEnabled")[0] == 1
    except OSError:
        return False


def weights_dir_for(cache_root: str, out_dir: str, model_name: str) -> str:
    """<cache_root>/weights normally. On Windows without long-path support, a
    deep checkout plus a long model id pushed huggingface_hub's temporary file
    past 260 characters ([Errno 2] ... .incomplete), so a short folder in the
    same place is used instead: <workspace>/.w/<8-char hash of the model id>."""
    normal = os.path.join(cache_root, "weights")
    if (
        os.name != "nt"
        or len(os.path.abspath(normal)) + _HF_STAGING_CHARS <= _WIN_MAX_PATH
        or _long_paths_enabled()
    ):
        return normal
    short_id = hashlib.sha1(model_name.encode("utf-8")).hexdigest()[:8]
    return os.path.join(os.path.dirname(os.path.abspath(out_dir)), ".w", short_id)


def _dir_size_mb(path: str) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for file in files:
            try:
                total += os.path.getsize(os.path.join(root, file))
            except OSError:
                pass
    return total / (1024 * 1024)
