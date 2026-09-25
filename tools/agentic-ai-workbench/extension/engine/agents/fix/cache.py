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
deleting/regenerating nntrainer_out.
"""
import json
import os
import time

MAX_AGE_DAYS = 30


def _sanitize(model_name: str) -> str:
    return model_name.replace("/", "__").replace("\\", "__").replace(":", "_")


def cache_root_for(out_dir: str, model_name: str) -> str:
    workspace_root = os.path.dirname(os.path.abspath(out_dir))
    return os.path.join(workspace_root, ".ai_compiler_cache", _sanitize(model_name))


def _meta_path(cache_root: str) -> str:
    return os.path.join(cache_root, "cache_meta.json")


def read_meta(cache_root: str) -> dict:
    path = _meta_path(cache_root)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def write_meta(cache_root: str, meta: dict):
    os.makedirs(cache_root, exist_ok=True)
    with open(_meta_path(cache_root), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def is_fresh(meta: dict, key: str, max_age_days: float = MAX_AGE_DAYS) -> bool:
    entry = meta.get(key)
    if not entry or "cached_at" not in entry:
        return False
    age_days = (time.time() - entry["cached_at"]) / 86400
    return age_days < max_age_days


def age_days(meta: dict, key: str) -> float:
    entry = meta.get(key)
    if not entry or "cached_at" not in entry:
        return -1
    return (time.time() - entry["cached_at"]) / 86400


def set_entry(cache_root: str, key: str, extra: dict):
    meta = read_meta(cache_root)
    entry = dict(extra)
    entry["cached_at"] = time.time()
    meta[key] = entry
    write_meta(cache_root, meta)
    return meta
