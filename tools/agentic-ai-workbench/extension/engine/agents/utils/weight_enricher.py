"""
Weight Enricher - Enriches graph nodes with weight information from manifest.

Attaches metadata from weight manifest to graph nodes, deferring actual
weight data loading to on-demand (when user clicks a node).
"""
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

from ..events import bus


def enrich_nodes_with_weights(nodes: list, state: dict) -> list:
    """
    Enrich graph nodes with weight information from weight manifest.

    Creates weightInfo with shape/dtype/params from manifest for all nodes.
    Actual weight file data is loaded on-demand when user clicks.

    Args:
        nodes: List of graph node dictionaries
        state: Pipeline state containing weight_manifest

    Returns:
        List of nodes with weightInfo metadata added
    """
    weight_manifest = state.get("weight_manifest") or []

    if not weight_manifest or not isinstance(weight_manifest, list):
        return nodes

    # Build a map from target name to manifest entry
    manifest_map = _build_manifest_map(weight_manifest)

    if manifest_map:
        bus.log(f"Enriching nodes with {len(manifest_map)} weight manifest entries", "info")

    # Enrich nodes
    enriched = []
    nodes_with_weights = 0

    for node in nodes:
        node_copy = dict(node)
        node_id = node.get("id") or node.get("label") or ""
        node_type = node.get("type") or ""

        # Try to find corresponding manifest entry (check multiple keys)
        entry = None
        for key in [node_id, node.get("label"), f"{node_type}_weight"]:
            if key and key in manifest_map:
                entry = manifest_map[key]
                break

        # Also try partial matching for templated names like "layer{layer_id}_..."
        if not entry and node_id:
            for manifest_key in manifest_map.keys():
                if "{layer_id}" in manifest_key:
                    base_key = manifest_key.split("{")[0]
                    if node_id.startswith(base_key):
                        entry = manifest_map[manifest_key]
                        break

        if entry:
            node_copy["weightInfo"] = _weight_info_from_manifest(entry, node_id)
            nodes_with_weights += 1
        else:
            node_copy["weightInfo"] = None

        enriched.append(node_copy)

    if nodes_with_weights > 0:
        bus.log(f"Enriched {nodes_with_weights}/{len(nodes)} nodes with weight information", "info")

    return enriched


def _build_manifest_map(manifest: list) -> Dict[str, Dict[str, Any]]:
    """Build a map from target name to manifest entry."""
    manifest_map = {}

    for entry in manifest:
        if not isinstance(entry, dict):
            continue

        target = entry.get("target", "")
        # Use target name as key (may have {layer_id} template)
        if target:
            manifest_map[target] = entry

    return manifest_map


def _weight_info_from_manifest(entry: dict, node_id: str) -> Dict[str, Any]:
    """Create weight info from manifest entry (without loading file data)."""
    target_shape = entry.get("target_shape") or []
    source_shape = entry.get("source_shape") or target_shape

    # Calculate parameter count
    params = 1
    for dim in target_shape:
        if dim and dim > 0:
            params *= dim

    return {
        "name": entry.get("target", node_id),
        "shape": target_shape,
        "dtype": "float32",  # Default; actual type would come from file
        "params": params,
        "source_shape": source_shape,
        "transform": entry.get("transform", "none"),
        "repeated": entry.get("repeated", False),
        "layer_count": entry.get("layer_count"),
        # Statistics loaded on-demand when user clicks
        "preview": None,
    }


def load_weight_stats(node_id: str, state: dict) -> Optional[Dict[str, Any]]:
    """
    Load actual weight statistics for a node on-demand.

    Called when user clicks on a node in the UI to show weight preview.

    Args:
        node_id: Node identifier
        state: Pipeline state

    Returns:
        Weight info with statistics or None
    """
    weight_manifest = state.get("weight_manifest") or []
    weights_path = state.get("converted_weights_path")

    if not weight_manifest or not weights_path:
        return None

    manifest_map = _build_manifest_map(weight_manifest)
    entry = manifest_map.get(node_id)

    if not entry:
        return None

    # Start with manifest info
    weight_info = _weight_info_from_manifest(entry, node_id)

    # Try to load actual weight file for statistics
    weight_file = _find_weight_file(entry, weights_path)
    if weight_file:
        stats = _compute_weight_stats(weight_file)
        if stats:
            weight_info.update(stats)

    return weight_info


def _find_weight_file(entry: dict, weights_path: str) -> Optional[Path]:
    """Find the actual weight file for this manifest entry."""
    if not weights_path or not Path(weights_path).is_dir():
        return None

    # Try to find file based on target name or source
    target = entry.get("target", "")
    weights_dir = Path(weights_path)

    # Look for .npy, .safetensors, or other weight files
    for pattern in [f"{target}*", "*.npy", "*.safetensors"]:
        matches = list(weights_dir.glob(pattern))
        if matches:
            return matches[0]

    return None


def _compute_weight_stats(weight_file: Path) -> Optional[Dict[str, Any]]:
    """Compute statistics from a weight file."""
    if not weight_file.exists():
        return None

    try:
        import numpy as np

        # Try loading as numpy
        if weight_file.suffix == ".npy":
            data = np.load(weight_file, allow_pickle=False)
        else:
            # Try other formats (safetensors, etc.)
            # For now, return None for unsupported formats
            return None

        params = int(np.prod(data.shape))
        return {
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "preview": [float(v) for v in data.flatten()[:5]],
        }
    except Exception as e:
        bus.log(f"Failed to compute weight stats from {weight_file}: {e}", "warn")
        return None
