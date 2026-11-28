# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file weight_preview.py
# @brief Weight Preview Agent (no LLM).
# @author prachi-t17 <prachi.t@samsung.com>

"""
Weight Preview Agent (no LLM).

Runs after the background weight download has been joined -- the graphs
are built long before the weights finish downloading, so their Weights
tabs start out with names, shapes and parameter counts only. This step
reads just the tensors the two graph panes show (embedding, final norm,
LM head and the expanded decoder layers, not every layer), computes
min/max/mean/std plus the first few values, merges them into each pane's
weightInfo blocks and re-publishes both graphs.

Reads safetensors checkpoints only: a .bin checkpoint has to be unpickled
in full to reach a single tensor, so it is skipped with a log line.
Tensors are read in row chunks, so an embedding table of a few GB never
has to sit in memory as float32 all at once.
"""
import json
import math
import os
import time

from . import graph_views
from .events import bus

PREVIEW_VALUES = 5
_CHUNK_ELEMENTS = 4 * 1024 * 1024

_DTYPE_NAMES = {
    "BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64",
    "I8": "int8", "U8": "uint8", "I16": "int16", "I32": "int32", "I64": "int64",
    "BOOL": "bool", "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
}

_VIEWS = (("model", "model_graph_view"), ("nntrainer", "nntrainer_graph_view"))


# pylint: disable-next=too-many-locals
def run(state: dict) -> dict:
    """Pipeline step: add per-tensor weight statistics to the graph views."""
    bus.agent_status("weight_preview", "running")

    views = [state.get(key) for _, key in _VIEWS]
    wanted = {
        n["weightInfo"]["name"]
        for view in views if view
        for n in view.get("nodes", []) if n.get("weightInfo")
    }
    if not wanted:
        return _skip(state, "no weight-bearing nodes in the graphs")

    weights_path = state.get("weights_path")
    if not weights_path or not os.path.isdir(weights_path):
        return _skip(state, "weights were not downloaded")

    try:
        # pylint: disable-next=import-outside-toplevel,unused-import  # optional package
        import torch  # noqa: F401  (safe_open's "pt" framework needs it)
        # pylint: disable-next=import-outside-toplevel  # optional package
        from safetensors import safe_open
    except ImportError:
        return _skip(state, "safetensors/torch not installed")

    key_to_file = _safetensors_keys(weights_path, safe_open)
    if not key_to_file:
        return _skip(
            state, "no .safetensors files in the checkpoint (only .safetensors can be previewed)"
        )

    resolved = _resolve_names(wanted, key_to_file)
    by_file: dict = {}
    for name, key in resolved.items():
        by_file.setdefault(key_to_file[key], []).append((name, key))

    started = time.time()
    previews = {}
    for path, items in by_file.items():
        try:
            with safe_open(path, framework="pt") as handle:
                for name, key in items:
                    stats = _tensor_stats(handle, key)
                    if stats:
                        previews[name] = stats
        # a bad shard must not fail the pipeline
        except Exception as exc:  # pylint: disable=broad-except
            bus.log(f"Weight preview: could not read {os.path.basename(path)}: {exc}", "warn")

    if not previews:
        return _skip(state, "none of the graph's weights were found in the checkpoint")

    updated = sum(graph_views.apply_weight_previews(view, previews) for view in views)
    for target, key in _VIEWS:
        if state.get(key):
            bus.graph(state[key], target=target)

    missing = len(wanted) - len(previews)
    bus.log(
        f"Weight preview: read {len(previews)} tensor(s) in {time.time() - started:.1f}s, "
        f"updated {updated} graph node(s)"
        + (f"; {missing} weight name(s) not found in the checkpoint" if missing else "")
    )
    bus.agent_status("weight_preview", "done", f"{len(previews)} tensors")
    return state


def _skip(state: dict, reason: str) -> dict:
    bus.log(f"Weight preview skipped: {reason}", "info")
    bus.agent_status("weight_preview", "skipped", reason)
    return state


def _safetensors_keys(weights_path: str, safe_open) -> dict:
    """{checkpoint tensor key: absolute .safetensors path}."""
    index_path = os.path.join(weights_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as handle:
                weight_map = json.load(handle).get("weight_map") or {}
            return {key: os.path.join(weights_path, fname) for key, fname in weight_map.items()}
        except (OSError, ValueError) as exc:
            bus.log(
                f"Weight preview: unreadable {os.path.basename(index_path)} ({exc}) -- scanning"
                " shards",
                "warn",
            )

    key_to_file = {}
    for fname in sorted(os.listdir(weights_path)):
        if not fname.endswith(".safetensors"):
            continue
        path = os.path.join(weights_path, fname)
        try:
            with safe_open(path, framework="pt") as handle:  # reads the header only
                for key in handle.keys():
                    key_to_file[key] = path
        # a bad shard must not fail the pipeline
        except Exception as exc:  # pylint: disable=broad-except
            bus.log(f"Weight preview: could not open {fname}: {exc}", "warn")
    return key_to_file


def _resolve_names(wanted: set, key_to_file: dict) -> dict:
    """Map each name the graphs use to a checkpoint key. Exact match first;
    otherwise a unique suffix match, which covers checkpoints that prefix
    the base model's names (e.g. "bert.embeddings..." for "embeddings...")."""
    keys = list(key_to_file)
    resolved = {}
    for name in wanted:
        if name in key_to_file:
            resolved[name] = name
            continue
        candidates = [k for k in keys if k.endswith("." + name)]
        if len(candidates) == 1:
            resolved[name] = candidates[0]
    return resolved


def _finite(value: float):
    # NaN/inf would make the JSON event line unparseable in the extension.
    return value if math.isfinite(value) else None


# pylint: disable-next=too-many-locals
def _tensor_stats(handle, key: str):
    tensor_slice = handle.get_slice(key)
    shape = list(tensor_slice.get_shape())
    dtype = _DTYPE_NAMES.get(tensor_slice.get_dtype(), tensor_slice.get_dtype().lower())

    if not shape:
        chunks = [handle.get_tensor(key).reshape(1)]
    else:
        per_row = math.prod(shape[1:]) if len(shape) > 1 else 1
        step = max(1, _CHUNK_ELEMENTS // max(per_row, 1))
        chunks = (tensor_slice[start:start + step] for start in range(0, shape[0], step))

    count, total, total_sq = 0, 0.0, 0.0
    low, high = math.inf, -math.inf
    preview = []
    for chunk in chunks:
        tensor = chunk.float().flatten()
        if tensor.numel() == 0:
            continue
        if len(preview) < PREVIEW_VALUES:
            preview += [float(v) for v in tensor[:PREVIEW_VALUES - len(preview)]]
        tensor64 = tensor.double()
        count += tensor.numel()
        total += float(tensor64.sum())
        total_sq += float((tensor64 * tensor64).sum())
        low = min(low, float(tensor.min()))
        high = max(high, float(tensor.max()))

    if count == 0:
        return None
    mean = total / count
    var = max(0.0, (total_sq - total * mean) / (count - 1)) if count > 1 else 0.0
    return {
        "dtype": dtype,
        "min": _finite(low), "max": _finite(high),
        "mean": _finite(mean), "std": _finite(math.sqrt(var)),
        "preview": [_finite(v) for v in preview],
    }
