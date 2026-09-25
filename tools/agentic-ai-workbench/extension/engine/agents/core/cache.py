"""
Cache utilities for compatibility agent.

Provides caching for compatibility check results to avoid rebuilding
the same graph repeatedly for the same model.
"""
import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Generic, cheap metadata cache. Weight files have their own longer lifetime
# below because re-downloading multi-gigabyte checkpoints for each run is both
# slow and unnecessary.
CACHE_EXPIRY_DAYS = 0
WEIGHTS_CACHE_EXPIRY_DAYS = 5
SECONDS_PER_DAY = 86400


def cache_root_for(out_dir: str, model_name: str) -> str:
    """Get the cache root directory for a given model."""
    # Sanitize model name for filesystem use
    safe_name = model_name.replace("/", "__").replace(":", "_")
    cache_root = os.path.join(out_dir, "cache", safe_name)
    os.makedirs(cache_root, exist_ok=True)
    return cache_root


def _meta_path(cache_root: str) -> str:
    """Get path to the meta.json file."""
    return os.path.join(cache_root, "meta.json")


def _graph_path(cache_root: str) -> str:
    """Get path to the cached graph file."""
    return os.path.join(cache_root, "graph.json")


def read_meta(cache_root: str) -> Dict[str, Any]:
    """Read the meta.json file, or return empty dict if not found."""
    meta_path = _meta_path(cache_root)
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def write_meta(cache_root: str, meta: Dict[str, Any]) -> None:
    """Write the meta.json file."""
    meta_path = _meta_path(cache_root)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def is_fresh(meta: Dict[str, Any], key: str,
             expiry_days: Optional[float] = None) -> bool:
    """Check if a cached entry is fresh (not expired)."""
    if key not in meta:
        return False
    entry = meta[key]
    timestamp = entry.get("timestamp", 0)
    age_seconds = time.time() - timestamp
    days = CACHE_EXPIRY_DAYS if expiry_days is None else expiry_days
    return age_seconds < (days * SECONDS_PER_DAY)


def age_days(meta: Dict[str, Any], key: str) -> float:
    """Get the age of a cached entry in days."""
    if key not in meta:
        return float("inf")
    entry = meta[key]
    timestamp = entry.get("timestamp", 0)
    age_seconds = time.time() - timestamp
    return age_seconds / SECONDS_PER_DAY


def save_graph(cache_root: str, graph_ir: Any, report: Dict[str, Any], 
               semantic_ir: Optional[Any] = None, 
               semantic_capabilities: Optional[str] = None) -> None:
    """Save graph and related data to cache."""
    # Note: graph_ir and semantic_ir may not be JSON serializable
    # For now, we only cache the report and metadata
    meta = read_meta(cache_root)
    meta["graph"] = {
        "timestamp": time.time(),
        "schema_version": 2,
        "has_semantic_ir": semantic_ir is not None,
        "semantic_capabilities": semantic_capabilities,
    }
    write_meta(cache_root, meta)


def load_graph(cache_root: str) -> Optional[Tuple]:
    """Load graph from cache. Returns None if not available."""
    # For now, return None as we don't serialize the full graph
    # This is a simplified implementation
    return None


def set_entry(cache_root: str, key: str, data: Dict[str, Any]) -> None:
    """Set a cache entry with timestamp."""
    import time
    meta = read_meta(cache_root)
    meta[key] = {
        "timestamp": time.time(),
        **data
    }
    write_meta(cache_root, meta)
