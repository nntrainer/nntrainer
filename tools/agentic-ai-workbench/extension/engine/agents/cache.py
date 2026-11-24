# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file cache.py
# @brief Workspace-level cache for the agent pipeline.
# @author prachi-t17 <prachi.t@samsung.com>

"""
Workspace-level cache for the agent pipeline.

Two independent things get cached, each with its own 30-day freshness
window:
  - "graph"   -- the traced Compiler IR + compatibility report for a
                 model. Rebuilding this from scratch is now cheap
                 (see compatibility.py's use of AutoModel.from_config,
                 which needs no weight download), but caching it still
                 saves the model-loading + module-walk work on repeat
                 runs of the same model.
  - "weights" -- the downloaded .safetensors/.bin snapshot for a
                 model. This is the expensive one (can be multiple GB),
                 so reusing it across runs is the main time saver.

The cache lives at <workspace>/.ai_compiler_cache/<sanitized model
name>/, sitting next to (not inside) the per-run output directory, so
it survives across multiple "Run Pipeline" invocations and across
deleting the pipeline's output folder.
"""
import hashlib
import json
import os
import re
import time

MAX_AGE_DAYS = 30


def _sanitize(model_name: str) -> str:
    return model_name.replace("/", "__").replace("\\", "__").replace(":", "_")


def cache_key(model_name: str) -> str:
    """Folder name for a model's cache entry. A Hugging Face id keeps its
    readable form (Qwen__Qwen3-0.6B). A local folder uses its own name plus a
    short hash of the full path (Qwen3-0.6B-1a2b3c4d): naming it after the
    whole path made the cache path too long for Windows."""
    if os.path.isdir(model_name) or os.path.isabs(model_name):
        full = os.path.abspath(model_name)
        base = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(full.rstrip("\\/")) or "model")[
            :40
        ]
        return f"{base}-{hashlib.sha1(os.path.normcase(full).encode('utf-8')).hexdigest()[:8]}"
    return _sanitize(model_name)


def cache_root_for(out_dir: str, model_name: str) -> str:
    """The cache folder for a model, next to the output folder."""
    workspace_root = os.path.dirname(os.path.abspath(out_dir))
    return os.path.join(workspace_root, ".ai_compiler_cache", cache_key(model_name))


def _meta_path(cache_root: str) -> str:
    return os.path.join(cache_root, "cache_meta.json")


def read_meta(cache_root: str) -> dict:
    """The cache's metadata, or {} when it is missing or unreadable."""
    path = _meta_path(cache_root)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def write_meta(cache_root: str, meta: dict):
    """Write the cache's metadata."""
    os.makedirs(cache_root, exist_ok=True)
    with open(_meta_path(cache_root), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def is_fresh(meta: dict, key: str, max_age_days: float = MAX_AGE_DAYS) -> bool:
    """True when the cache entry `key` exists and is younger than `max_age_days`."""
    entry = meta.get(key)
    if not entry or "cached_at" not in entry:
        return False
    age = (time.time() - entry["cached_at"]) / 86400
    return age < max_age_days


def age_days(meta: dict, key: str) -> float:
    """Age of the cache entry `key` in days, or -1 when there is none."""
    entry = meta.get(key)
    if not entry or "cached_at" not in entry:
        return -1
    return (time.time() - entry["cached_at"]) / 86400


def set_entry(cache_root: str, key: str, extra: dict):
    """Record the cache entry `key`, with the `extra` fields, as written now."""
    meta = read_meta(cache_root)
    entry = dict(extra)
    entry["cached_at"] = time.time()
    meta[key] = entry
    write_meta(cache_root, meta)
    return meta
