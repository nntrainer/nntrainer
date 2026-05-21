"""
Adaptive decomposition pipeline for HuggingFace -> NNTrainer conversion.

Strategy:
  1. FUSED OPS FIRST: Trace with LEAF_MODULES, map to NNTrainer fused layers
     (fully_connected, rms_norm, mha_core, swiglu, etc.)
  2. DECOMPOSE UNKNOWNS: If any modules can't be mapped, re-trace with those
     modules excluded from leaves. Their forward() is automatically decomposed
     into tensor ops by the tracer.
  3. OP DECOMPOSITION: For individual ops without direct NNTrainer layer support,
     resolve using:
     a) Direct Tensor methods (inv_sqrt, abs, neg, erf, pow, sqrt, sin, cos, tan)
     b) LazyTensor chains for consecutive arithmetic ops (add, sub, mul, div)
     c) Layer-level decomposition into supported primitives

NNTrainer Tensor & LazyTensor integration:
  - Tensor class provides: pow, sqrt, abs, neg, inv_sqrt, erf, sin, cos, tan
  - LazyTensor (Tensor::chain()) chains: add_i, subtract_i, multiply_i, divide_i,
    dot, transpose, sum, average
  - Ops like rsqrt map directly to Tensor::inv_sqrt() - no decomposition needed
  - Consecutive arithmetic ops can be fused into a single LazyTensor chain
"""

import torch
from typing import Optional

from nntrainer_layers import (
    NNTrainerLayerDef,
    LAYER_INPUT,
    LAYER_POW, LAYER_SQRT, LAYER_MULTIPLY, LAYER_DIVIDE, LAYER_NEGATIVE,
    LAYER_ADDITION, LAYER_SUBTRACT,
    LAYER_DROPOUT, LAYER_EMBEDDING,
    LAYER_RESHAPE, LAYER_PERMUTE, LAYER_TRANSPOSE, LAYER_CONCAT,
    LAYER_RMS_NORM, LAYER_RESHAPED_RMS_NORM, LAYER_LAYER_NORM,
    LAYER_FC,
    LAZY_TENSOR_OPS, TENSOR_DIRECT_METHODS,
    OP_UNSUPPORTED, OP_NOOP, OP_RESHAPE, OP_TRANSPOSE, OP_PERMUTE, OP_SDPA,
)
from tracer import Tracer, LEAF_MODULES
from node_mapper import NodeMapper
from pattern_detector import detect_patterns


# =============================================================================
# Tensor Method Resolution
# =============================================================================
# Ops that have direct NNTrainer Tensor methods don't need layer-level
# decomposition. Instead, they emit a single Tensor method call in C++.
# This is more efficient than creating intermediate NNTrainer layers.

def _resolve_to_tensor_method(layer):
    """Try to resolve an unsupported op to a direct NNTrainer Tensor method.

    Returns:
        NNTrainerLayerDef with tensor_method metadata if resolvable, else None.
    """
    original_op = layer.properties.get("original_op", "")
    method_info = TENSOR_DIRECT_METHODS.get(original_op)
    if not method_info:
        return None

    method_name, is_inplace = method_info
    # Create a layer that carries the tensor method info for the emitter
    resolved = NNTrainerLayerDef(
        layer_type=f"tensor_op:{original_op}",
        name=layer.name,
        properties={
            "tensor_method": method_name,
            "is_inplace": is_inplace,
            "original_op": original_op,
        },
        input_layers=layer.input_layers,
        hf_module_name=layer.hf_module_name,
    )
    return resolved


# =============================================================================
# Layer-level Decomposition Registry (fallback)
# =============================================================================
# For ops that are neither NNTrainer layers nor direct Tensor methods,
# decompose into sequences of supported layers.

def _decompose_reciprocal(layer):
    """reciprocal(x) -> divide(1, x)"""
    return [NNTrainerLayerDef(
        layer_type=LAYER_DIVIDE,
        name=layer.name,
        properties={"numerator": 1.0},
        input_layers=layer.input_layers,
        hf_module_name=layer.hf_module_name,
    )]


# Registry: original_op -> decomposition function (layer-level fallback)
DECOMPOSITION_REGISTRY = {
    "reciprocal": _decompose_reciprocal,
}

# Ops with no decomposition and no Tensor method - truly unsupported
_NO_DECOMPOSITION_OPS = {"exp", "log", "clamp", "clip", "clamp_", "clamp_min", "clamp_max"}


# =============================================================================
# LazyTensor Chain Detection
# =============================================================================

class LazyTensorChain:
    """Represents a sequence of ops that can be fused into a LazyTensor chain.

    In NNTrainer C++, this becomes:
        Tensor result = input.chain()
            .add_i(tensor_a)
            .multiply_i(scalar)
            .subtract_i(tensor_b)
            .run();

    Instead of creating separate NNTrainer layers for each op.
    """
    def __init__(self, layers, start_idx):
        self.layers = layers    # List of NNTrainerLayerDef in the chain
        self.start_idx = start_idx
        self.end_idx = start_idx + len(layers)

    @property
    def chain_length(self):
        return len(self.layers)

    def to_cpp_chain(self, input_var="input"):
        """Generate C++ LazyTensor chain code.

        Returns a string like:
            input.chain().add_i(a).multiply_i(2.0f).sqrt_i().run()
        """
        # Map layer_type -> LazyTensor method for binary ops (with tensor/scalar)
        _BINARY_OPS = {
            "add": "add_i", "addition": "add_i",
            "subtract": "subtract_i",
            "multiply": "multiply_i",
            "divide": "divide_i",
        }
        # Map layer_type -> LazyTensor method for unary ops (no args)
        _UNARY_OPS = {
            "sqrt": "sqrt_i",
            "negative": "neg",
            "exp": "exp_i",
            "log": "log_i",
            "sin": "sin",
            "cos": "cos",
            "tan": "tan",
        }

        parts = [f"{input_var}.chain()"]
        for layer in self.layers:
            lt = layer.layer_type
            if lt in _BINARY_OPS:
                method = _BINARY_OPS[lt]
                if layer.input_layers:
                    parts.append(f"{method}({layer.input_layers[-1]})")
                else:
                    val = layer.properties.get("value", "0")
                    parts.append(f"{method}({val}f)")
            elif lt == "pow":
                exp = layer.properties.get("exponent", "2.0")
                parts.append(f"pow_i({exp}f)")
            elif lt == "clamp":
                mn = layer.properties.get("min", "-inf")
                mx = layer.properties.get("max", "inf")
                parts.append(f"clamp_i({mn}f, {mx}f)")
            elif lt in _UNARY_OPS:
                parts.append(f"{_UNARY_OPS[lt]}()")
        parts.append("run()")
        return ".".join(parts)

    def __repr__(self):
        ops = [l.layer_type for l in self.layers]
        return f"LazyTensorChain({' -> '.join(ops)})"


def detect_lazy_chains(layers, min_chain_length=2):
    """Find sequences of consecutive LazyTensor-compatible ops.

    Args:
        layers: List of NNTrainerLayerDef.
        min_chain_length: Minimum ops to form a chain (default 2).

    Returns:
        List of LazyTensorChain objects found in the layer sequence.
    """
    chains = []
    current_chain = []
    chain_start = 0

    for i, layer in enumerate(layers):
        if layer.layer_type in LAZY_TENSOR_OPS:
            if not current_chain:
                chain_start = i
            current_chain.append(layer)
        else:
            if len(current_chain) >= min_chain_length:
                chains.append(LazyTensorChain(current_chain, chain_start))
            current_chain = []

    # Handle chain at end of list
    if len(current_chain) >= min_chain_length:
        chains.append(LazyTensorChain(current_chain, chain_start))

    return chains


# =============================================================================
# Op Resolution Pipeline
# =============================================================================

def resolve_unsupported_ops(layers):
    """Resolve unsupported ops using the best available strategy.

    Resolution order (most efficient first):
      1. Direct Tensor method (inv_sqrt, abs, neg, erf, pow, sqrt, sin, cos, tan)
      2. Layer-level decomposition (reciprocal -> divide)
      3. Keep as unsupported with warning

    Args:
        layers: List of NNTrainerLayerDef from NodeMapper.

    Returns:
        New list with unsupported ops resolved.
    """
    result = []
    tensor_method_count = 0
    decomposed_count = 0
    unresolvable = []

    for layer in layers:
        if layer.layer_type != OP_UNSUPPORTED:
            result.append(layer)
            continue

        original_op = layer.properties.get("original_op", "")

        # Strategy 1: Direct Tensor method
        resolved = _resolve_to_tensor_method(layer)
        if resolved:
            result.append(resolved)
            tensor_method_count += 1
            continue

        # Strategy 2: Layer-level decomposition
        decompose_fn = DECOMPOSITION_REGISTRY.get(original_op)
        if decompose_fn:
            decomposed = decompose_fn(layer)
            result.extend(decomposed)
            decomposed_count += len(decomposed)
            continue

        # Strategy 3: No resolution available
        unresolvable.append(layer)
        result.append(layer)

    if tensor_method_count > 0:
        print(f"  [RESOLVE] {tensor_method_count} ops -> direct Tensor methods "
              f"(inv_sqrt, abs, etc.)")
    if decomposed_count > 0:
        print(f"  [RESOLVE] {decomposed_count} ops -> layer decomposition")
    if unresolvable:
        print(f"  [WARNING] {len(unresolvable)} ops have no resolution:")
        for layer in unresolvable:
            op = layer.properties.get("original_op", "?")
            print(f"    - {op} at {layer.hf_module_name} ({layer.name})")

    return result


# Keep backward compatibility
decompose_unsupported_ops = resolve_unsupported_ops


def _remove_passthrough_layers(layers, layer_type, label):
    """Remove layers of a given type and rewire downstream inputs.

    For each removed layer, downstream layers that referenced it are
    rewired to point at the removed layer's own input instead.
    Handles chains (e.g. noop -> noop -> real layer).

    Returns the filtered layer list.
    """
    target_names = {l.name for l in layers if l.layer_type == layer_type}
    if not target_names:
        return layers

    # Build bypass map: removed_name -> its input (or None)
    bypass = {}
    for l in layers:
        if l.layer_type == layer_type:
            bypass[l.name] = l.input_layers[0] if l.input_layers else None

    # Resolve chains
    def _resolve(name):
        visited = set()
        while name in bypass and name not in visited:
            visited.add(name)
            name = bypass[name]
        return name

    # Rewire inputs of surviving layers
    for l in layers:
        if l.layer_type != layer_type and l.input_layers:
            l.input_layers = [
                _resolve(inp) if inp in target_names else inp
                for inp in l.input_layers
            ]
            l.input_layers = [x for x in l.input_layers if x]

    filtered = [l for l in layers if l.layer_type != layer_type]
    print(f"  [CLEANUP] Removed {len(target_names)} {label} layers")
    return filtered


# =============================================================================
# Position ID Chain Removal
# =============================================================================

# Layer types that are part of position ID computation chains.
# These are arithmetic/reshape ops used to compute position indices
# from attention masks before feeding into position embedding.
_POSITION_CHAIN_OPS = frozenset({
    LAYER_ADDITION, LAYER_SUBTRACT, LAYER_MULTIPLY, LAYER_DIVIDE,
    OP_RESHAPE,
})


def _remove_position_id_chains(layers):
    """Remove position ID computation chains (XLM-RoBERTa, etc.).

    In models like XLM-RoBERTa, position IDs are computed from input masks:
        input_ids → ne → int → cumsum → type_as → add → mul → add → Embedding

    After noop removal (ne, int, cumsum, type_as are all noops), the remaining
    arithmetic layers (add, mul, add) only serve to compute position indices.
    NNTrainer handles position IDs internally, so these are redundant.

    This function detects arithmetic layers that exclusively feed into
    embedding layers (directly or through other arithmetic layers) and
    removes them. It uses iterative fixed-point analysis to correctly
    handle chains of arbitrary length.

    Returns the filtered layer list.
    """
    by_name = {l.name: l for l in layers}

    # Build consumer graph: layer_name -> set of consumer layer names
    consumers = {}
    for l in layers:
        for inp in (l.input_layers or []):
            consumers.setdefault(inp, set()).add(l.name)

    embedding_names = {l.name for l in layers if l.layer_type == LAYER_EMBEDDING}

    # Iterative fixed-point: mark arithmetic layers whose ALL consumers
    # are either embedding layers or already-marked removable layers.
    removable = set()
    changed = True
    while changed:
        changed = False
        for l in layers:
            if l.name in removable or l.layer_type not in _POSITION_CHAIN_OPS:
                continue
            layer_consumers = consumers.get(l.name, set())
            if not layer_consumers:
                continue
            if all(c in embedding_names or c in removable
                   for c in layer_consumers):
                removable.add(l.name)
                changed = True

    if not removable:
        return layers

    # Rewire embedding inputs past the removed chain
    for l in layers:
        if l.name not in removable and l.input_layers:
            l.input_layers = [
                inp for inp in l.input_layers if inp not in removable
            ]

    filtered = [l for l in layers if l.name not in removable]
    print(f"  [CLEANUP] Removed {len(removable)} position ID "
          f"computation layers")
    return filtered


# =============================================================================
# Rotary Embedding Chain Removal
# =============================================================================

# Tensor op types that can appear in RoPE computation/application chains.
_ROPE_CHAIN_OPS = frozenset({
    LAYER_MULTIPLY, LAYER_ADDITION, LAYER_SUBTRACT, LAYER_NEGATIVE,
    LAYER_RESHAPE, OP_RESHAPE, LAYER_TRANSPOSE, OP_TRANSPOSE,
    LAYER_PERMUTE, OP_PERMUTE,
    "concat", "slice", "cos", "sin", "matmul",
})

# Subset for forward propagation only.  Excludes "concat" and "slice"
# because these are ambiguous: they appear in rotate_half (RoPE) but
# also in KV-cache concatenation.  rotate_half concat/slice are caught
# by the backward pass (all their consumers are removable), whereas
# KV-cache concats feed into SDPA and must survive.
_ROPE_FORWARD_OPS = _ROPE_CHAIN_OPS - {"concat", "slice"}

# Subset for backward propagation.  Excludes transpose because
# transpose layers are the pre-rotation Q/K tensors — the boundary
# between the main data path and the RoPE chain.  SDPA needs to be
# rewired to these after RoPE removal.
_ROPE_BACKWARD_OPS = _ROPE_CHAIN_OPS - {LAYER_TRANSPOSE, OP_TRANSPOSE}


def _remove_rope_chains(layers):
    """Remove rotary position embedding computation chains.

    NNTrainer's mha_core layer handles RoPE internally via rope_theta config
    (same approach as llama.cpp: discard inv_freq, compute at runtime).

    The FX graph contains two parts:
    1. rotary_emb module: inv_freq → matmul → cat → cos/sin
    2. Per-attention rotate_half: Q/K → split → neg → cat → mul(sin)
                                  Q/K → mul(cos)
                                  → add(cos_part, sin_part) → SDPA

    This function:
    - Identifies rotary_emb scope layers (forward: mark by name)
    - Traces forward to mark layers with rope inputs (mul, reshape, etc.)
    - Traces backward to mark layers whose ALL consumers are removable
      (captures rotate_half: split, neg, cat that only feed rope multiply)
    - Rewires SDPA inputs from rotated Q/K (add) to pre-rotation Q/K

    Returns the filtered layer list.
    """
    by_name = {l.name: l for l in layers}

    # Build consumer graph: layer_name -> set of consumer layer names
    consumers = {}
    for l in layers:
        for inp in (l.input_layers or []):
            consumers.setdefault(inp, set()).add(l.name)

    # Step 1: Seed with rotary_emb scope layers
    removable = set()
    for l in layers:
        if "rotary_emb" in l.name:
            removable.add(l.name)

    if not removable:
        return layers

    # Step 2: Forward pass — mark tensor ops that have ANY input from
    # the removable set (these consume rope cos/sin outputs).
    # Uses _ROPE_FORWARD_OPS (excludes concat/slice) to avoid catching
    # KV-cache concats that merely consume the RoPE output.
    changed = True
    while changed:
        changed = False
        for l in layers:
            if l.name in removable or l.layer_type not in _ROPE_FORWARD_OPS:
                continue
            if l.input_layers and any(inp in removable
                                      for inp in l.input_layers):
                removable.add(l.name)
                changed = True

    # Step 3: Backward pass — mark tensor ops whose ALL consumers are
    # already removable (captures rotate_half: split→neg→cat chains
    # that only feed into the rope multiply layers).
    # Uses _ROPE_BACKWARD_OPS (excludes transpose) so that pre-rotation
    # Q/K transpose layers are preserved as rewire targets for SDPA.
    changed = True
    while changed:
        changed = False
        for l in layers:
            if l.name in removable or l.layer_type not in _ROPE_BACKWARD_OPS:
                continue
            layer_consumers = consumers.get(l.name, set())
            if layer_consumers and all(c in removable
                                       for c in layer_consumers):
                removable.add(l.name)
                changed = True

    # Step 4: Build rewire map for terminal addition layers.
    # Each RoPE-application `add` feeds into SDPA with the rotated Q or K.
    # We need to replace it with the pre-rotation Q/K tensor.
    # Pattern: add ← [mul(Q*cos), mul(rotated*sin)]
    #          mul(Q*cos) ← [Q_pre_rotation, cos_unsqueeze]
    #          The non-rope input to mul is the pre-rotation tensor.
    rewire_map = {}  # removable_layer_name -> replacement_name
    for name in removable:
        layer = by_name[name]
        if layer.layer_type != LAYER_ADDITION:
            continue
        # Find the pre-rotation source via the cos-multiply branch
        for inp_name in layer.input_layers:
            inp_layer = by_name.get(inp_name)
            if inp_layer and inp_layer.layer_type == LAYER_MULTIPLY:
                for mul_inp in inp_layer.input_layers:
                    if mul_inp not in removable:
                        rewire_map[name] = mul_inp
                        break
            if name in rewire_map:
                break

    # Step 5: Rewire downstream layers (SDPA, etc.)
    for l in layers:
        if l.name in removable or not l.input_layers:
            continue
        new_inputs = []
        for inp in l.input_layers:
            if inp in rewire_map:
                new_inputs.append(rewire_map[inp])
            elif inp in removable:
                continue  # drop dead reference
            else:
                new_inputs.append(inp)
        l.input_layers = new_inputs

    filtered = [l for l in layers if l.name not in removable]
    print(f"  [CLEANUP] Removed {len(removable)} rotary embedding layers")
    return filtered


# =============================================================================
# MHA Op Absorption
# =============================================================================

# Layer types that mha_core handles internally.
_MHA_ABSORBABLE = frozenset({
    LAYER_RESHAPE, OP_RESHAPE, LAYER_TRANSPOSE, OP_TRANSPOSE,
    LAYER_PERMUTE, OP_PERMUTE, LAYER_CONCAT, "concat",
})

# Layer types that are attention Q/K norms (boundary: keep these).
_ATTN_NORM_TYPES = frozenset({
    LAYER_RMS_NORM, LAYER_RESHAPED_RMS_NORM, LAYER_LAYER_NORM,
})


def _absorb_mha_ops(layers):
    """Absorb reshape/transpose/concat around SDPA into mha_core.

    NNTrainer's mha_core handles input reshape, transpose, KV-cache
    concat, and output transpose/reshape internally.  The FX graph
    produces explicit layers for these, which must be removed so the
    converted graph matches the C++ symbolic tensor API:

        q_proj -> [q_norm] -> mha_core -> o_proj
        k_proj -> [k_norm] -> mha_core
        v_proj            -> mha_core

    For each SDPA layer:
      - Trace each input backward through absorbable ops to find the
        source (FC projection or attention norm).
      - Trace the output forward through absorbable ops to find the
        sink (o_proj FC).
      - Remove intermediate absorbable layers and rewire connections.

    Returns the filtered layer list.
    """
    by_name = {l.name: l for l in layers}

    # Build consumer map
    consumers = {}
    for l in layers:
        for inp in (l.input_layers or []):
            consumers.setdefault(inp, set()).add(l.name)

    sdpa_layers = [l for l in layers if l.layer_type == OP_SDPA]
    if not sdpa_layers:
        return layers

    absorbed = set()

    for sdpa in sdpa_layers:
        # --- Input side: trace each SDPA input backward ---
        new_inputs = []
        for inp_name in (sdpa.input_layers or []):
            source = _trace_back_through(inp_name, by_name, absorbed,
                                         consumers)
            new_inputs.append(source)

            # If the source is an attention norm, also absorb any reshape
            # between the FC projection and the norm (reshaped_rms_norm
            # handles reshape internally via feature_size).
            source_layer = by_name.get(source)
            if source_layer and source_layer.layer_type in _ATTN_NORM_TYPES:
                norm_source = _trace_back_through(
                    source_layer.input_layers[0] if source_layer.input_layers
                    else source, by_name, absorbed, consumers)
                source_layer.input_layers = [norm_source]

        sdpa.input_layers = new_inputs

        # --- Output side: trace forward from SDPA through absorbable ops ---
        _trace_forward_through(sdpa.name, by_name, absorbed, consumers)

    if absorbed:
        layers = [l for l in layers if l.name not in absorbed]
        print(f"  [CLEANUP] Absorbed {len(absorbed)} reshape/transpose/concat "
              f"layers into mha_core")

    return layers


def _trace_back_through(name, by_name, absorbed, consumers):
    """Trace backward from `name` through absorbable ops to find source.

    Returns the name of the source layer (FC, norm, or non-absorbable).
    Marks intermediate absorbable layers in `absorbed` set.
    """
    visited = set()
    current = name
    chain = []  # layers to potentially absorb

    while current in by_name and current not in visited:
        visited.add(current)
        layer = by_name[current]

        # Stop at non-absorbable layers (FC, norm, etc.)
        if layer.layer_type not in _MHA_ABSORBABLE:
            break

        # Only absorb if this layer has a single consumer path into SDPA
        layer_consumers = consumers.get(current, set())
        non_absorbed = layer_consumers - absorbed
        if len(non_absorbed) > 1:
            # Multiple consumers — don't absorb, use as-is
            break

        chain.append(current)

        # Follow single input backward
        if layer.input_layers and len(layer.input_layers) >= 1:
            current = layer.input_layers[0]
        else:
            break

    # Mark chain as absorbed
    absorbed.update(chain)
    return current


def _trace_forward_through(sdpa_name, by_name, absorbed, consumers):
    """Trace forward from SDPA through absorbable ops on the output side.

    Absorbs transpose/reshape between SDPA and o_proj, rewiring o_proj
    to take input directly from SDPA.
    """
    visited = set()
    current = sdpa_name
    chain = []

    while current not in visited:
        visited.add(current)
        layer_consumers = consumers.get(current, set()) - absorbed
        if len(layer_consumers) != 1:
            break

        next_name = next(iter(layer_consumers))
        next_layer = by_name.get(next_name)
        if next_layer is None:
            break

        if next_layer.layer_type not in _MHA_ABSORBABLE:
            # Reached o_proj or other non-absorbable layer — rewire it
            if chain:
                last_absorbed = chain[-1]
                next_layer.input_layers = [
                    sdpa_name if inp == last_absorbed else inp
                    for inp in (next_layer.input_layers or [])
                ]
            break

        chain.append(next_name)
        current = next_name

    absorbed.update(chain)


# =============================================================================
# Orphan Layer Repair
# =============================================================================

def _repair_orphaned_layers(layers, graph):
    """Reconnect non-input layers that lost all input_layers during cleanup.

    After removal passes (noop, dropout, rope, position ID), some layers may
    end up with empty input_layers if their inputs were removed but the layer
    itself survived. This function tries to recover their inputs from the
    original FX graph connectivity.

    Layers that can't be recovered are left as-is (they'll get phantom input
    layers from _add_input_layers_and_shape_info if they have dangling refs,
    or remain inputless if truly orphaned).

    Args:
        layers: List of NNTrainerLayerDef (post-cleanup).
        graph: FX graph from tracer (original node connectivity).

    Returns:
        Updated layers list (same length, some layers may have restored inputs).
    """
    defined = {l.name for l in layers}
    fx_nodes = {node.name: node for node in graph.nodes}

    # Build fx_node_name -> layer_name mapping for recovery
    fx_to_layer = {}
    for l in layers:
        if l.fx_node_name and l.fx_node_name != l.name:
            fx_to_layer[l.fx_node_name] = l.name
        fx_to_layer[l.name] = l.name

    repaired = 0
    for l in layers:
        if l.layer_type == LAYER_INPUT:
            continue

        # Check for truly orphaned (empty inputs) or all-dangling refs
        is_orphaned = not l.input_layers
        if not is_orphaned:
            valid = [inp for inp in l.input_layers if inp in defined]
            if not valid:
                is_orphaned = True

        if not is_orphaned:
            continue

        # Try to recover from FX graph.
        fx_name = l.fx_node_name or l.name
        fx_node = fx_nodes.get(fx_name)
        if fx_node is None:
            continue

        # Walk FX node args to find a surviving layer
        recovered = []
        for arg in fx_node.args:
            if not hasattr(arg, 'name'):
                continue
            candidate = fx_to_layer.get(arg.name)
            if candidate and candidate in defined and candidate != l.name:
                recovered.append(candidate)

        if recovered:
            l.input_layers = recovered
            repaired += 1

    if repaired:
        print(f"  [CLEANUP] Orphan repair: {repaired} reconnected")

    # Remove dead nodes: orphaned layers (no inputs) that have no consumers
    consumer_inputs = set()
    for l in layers:
        for inp in (l.input_layers or []):
            consumer_inputs.add(inp)

    dead_names = set()
    for l in layers:
        if l.layer_type == LAYER_INPUT:
            continue
        if not l.input_layers and l.name not in consumer_inputs:
            dead_names.add(l.name)

    if dead_names:
        layers = [l for l in layers if l.name not in dead_names]
        print(f"  [CLEANUP] Dead node removal: {len(dead_names)} removed "
              f"({', '.join(sorted(dead_names)[:5])}{'...' if len(dead_names) > 5 else ''})")

    return layers


# =============================================================================
# Adaptive Converter Pipeline
# =============================================================================

class AdaptiveConverter:
    """Adaptive converter: fused ops first, tensor op fallback for unknowns.

    This is the main entry point for converting any HuggingFace model to
    NNTrainer layer definitions.

    Pipeline:
        Pass 1: Trace with full LEAF_MODULES
                -> Map to NNTrainer layers
                -> Identify unknown module types

        Pass 2 (if needed): Re-trace with unknown modules excluded from leaves
                -> Their forward() automatically decomposes into tensor ops
                -> Map tensor ops to NNTrainer layers

        Pass 3: Resolve unsupported ops:
                -> Tensor methods (rsqrt -> inv_sqrt, abs -> abs)
                -> Layer decomposition (reciprocal -> divide)

        Pass 4: Detect LazyTensor chain opportunities
                -> Consecutive add/sub/mul/div -> single chain() call

    Usage:
        converter = AdaptiveConverter(model, config)
        result = converter.convert({"input_ids": input_ids})
        result.summary()
    """

    def __init__(self, model, model_config=None, training=False,
                 plugin_registry=None):
        """
        Args:
            model: The HuggingFace model to convert.
            model_config: HuggingFace model config (optional).
            training: If False (default), dropout layers are removed from
                the output since they are no-ops during inference. If True,
                dropout layers are preserved for training use.
            plugin_registry: Optional PluginRegistry for custom layer mappings.
                If provided, registered custom module types are treated as
                leaf modules (not decomposed) and mapped via the registry.
        """
        self.model = model
        self.config = model_config
        self.training = training
        if plugin_registry is not None:
            from plugin_registry import get_global_registry, _global_registry
            # Merge into global registry so module_mapper can find them
            for matcher, spec in plugin_registry._entries:
                _global_registry.register(matcher, spec)

    def convert(self, input_kwargs, max_passes=3):
        """Run the adaptive conversion pipeline.

        Args:
            input_kwargs: Dict of model inputs (e.g. {"input_ids": tensor}).
            max_passes: Maximum re-trace passes for unknown modules.

        Returns:
            ConversionResult with layers, lazy chains, and diagnostics.
        """
        exclude_types = set()
        all_decomposed_types = set()

        # Iterative passes: trace -> map -> find unknowns -> exclude -> re-trace
        for pass_num in range(1, max_passes + 1):
            print(f"\n  [PASS {pass_num}] Tracing with "
                  f"{len(exclude_types)} excluded module types...")

            tracer = Tracer(self.model, exclude_leaf_types=exclude_types)
            with tracer:
                with torch.no_grad():
                    self.model(**input_kwargs)

            mapper = NodeMapper(self.model, tracer.graph, self.config)
            layers = mapper.map_all()

            # Check for unknown module types
            unknown_types = mapper.get_unknown_module_types(layers)

            if not unknown_types:
                print(f"  [PASS {pass_num}] All modules mapped successfully.")
                break

            # Found unknowns - exclude them and re-trace
            new_types = unknown_types - exclude_types
            if not new_types:
                print(f"  [PASS {pass_num}] No new unknown types to decompose.")
                break

            print(f"  [PASS {pass_num}] Decomposing {len(new_types)} unknown "
                  f"module types: {', '.join(sorted(new_types))}")
            exclude_types |= new_types
            all_decomposed_types |= new_types

        # Pass 3: Resolve unsupported ops (Tensor methods > layer decomposition)
        layers = resolve_unsupported_ops(layers)

        # Pass 3.5: Remove dropout layers for inference mode
        if not self.training:
            layers = _remove_passthrough_layers(
                layers, LAYER_DROPOUT, "dropout")

        # Pass 3.6: Remove noop layers (expand, size, _set_grad_enabled, etc.)
        layers = _remove_passthrough_layers(layers, OP_NOOP, "noop")

        # Pass 3.7: Remove position ID computation chains
        # (arithmetic ops that exclusively feed position embeddings)
        layers = _remove_position_id_chains(layers)

        # Pass 3.7.5: Remove rotary embedding chains
        # (NNTrainer mha_core handles RoPE via rope_theta, like llama.cpp)
        layers = _remove_rope_chains(layers)

        # Pass 3.7.7: Absorb reshape/transpose/concat around SDPA into
        # mha_core (NNTrainer handles these internally)
        layers = _absorb_mha_ops(layers)

        # Pass 3.7.9: Repair orphaned layers (non-input layers that lost
        # all input_layers during cleanup passes)
        layers = _repair_orphaned_layers(layers, tracer.graph)

        # Pass 3.8: Convert intermediate op types to final NNTrainer types
        _OP_TO_LAYER = {
            OP_RESHAPE: LAYER_RESHAPE,
            OP_TRANSPOSE: LAYER_TRANSPOSE,
            OP_PERMUTE: LAYER_PERMUTE,
        }
        for layer in layers:
            if layer.layer_type in _OP_TO_LAYER:
                layer.layer_type = _OP_TO_LAYER[layer.layer_type]

        # Pass 3.9: Add input layers for external inputs and extract
        # reshape/slice parameters from FX graph metadata
        layers = _add_input_layers_and_shape_info(
            layers, tracer.graph, input_kwargs)

        # Pass 4: Detect LazyTensor chain opportunities
        lazy_chains = detect_lazy_chains(layers)
        if lazy_chains:
            total_ops = sum(c.chain_length for c in lazy_chains)
            print(f"  [LAZY] Found {len(lazy_chains)} LazyTensor chain "
                  f"opportunities (total {total_ops} ops)")
            print(f"  [LAZY] Note: These chains are decomposed forms of "
                  f"higher-level ops (e.g. GELU activation, relative "
                  f"position bias) that NNTrainer handles internally via "
                  f"built-in layers (activation, mha_core). No C++ "
                  f"LazyTensor code generation needed for weight conversion.")

        # Pass 5: Detect structural patterns (attention, FFN, blocks)
        model_structure = detect_patterns(layers, self.config)

        # Collect diagnostics
        remaining_unknowns = [l for l in layers
                              if l.layer_type.startswith("unknown")
                              or l.layer_type == OP_UNSUPPORTED]
        tensor_ops = [l for l in layers
                      if l.layer_type.startswith("tensor_op:")]

        return ConversionResult(
            layers=layers,
            decomposed_module_types=all_decomposed_types,
            unsupported_ops=[l for l in remaining_unknowns
                             if l.layer_type == OP_UNSUPPORTED],
            unknown_layers=[l for l in remaining_unknowns
                            if l.layer_type.startswith("unknown")],
            tensor_ops=tensor_ops,
            lazy_chains=lazy_chains,
            graph=tracer.graph,
            model_structure=model_structure,
            training=self.training,
        )


def _add_input_layers_and_shape_info(layers, graph, input_kwargs):
    """Add input layers for external inputs and extract shape metadata.

    1. Detects external inputs (referenced but not defined) and creates
       NNTrainer input layers with proper input_shape.
    2. Extracts target_shape for reshape/view operations from FX graph args.
    3. Extracts slice parameters (start_index, end_index, axis) for
       __getitem__ operations from FX graph args.

    Args:
        layers: List of NNTrainerLayerDef
        graph: FX graph from tracer
        input_kwargs: Dict of model inputs with tensor shapes

    Returns:
        Updated layers list with input layers prepended.
    """
    import torch

    # Build name->node lookup from FX graph
    fx_nodes = {node.name: node for node in graph.nodes}

    # Detect external inputs
    defined = set(l.name for l in layers)
    external_inputs = []
    seen = set()
    for l in layers:
        for inp in l.input_layers:
            if inp not in defined and inp not in seen:
                seen.add(inp)
                external_inputs.append(inp)

    # Create input layers for external inputs
    input_layers = []
    for inp_name in external_inputs:
        tensor = input_kwargs.get(inp_name)
        if tensor is not None and isinstance(tensor, torch.Tensor):
            # Convert PyTorch shape to NNTrainer input_shape (C:H:W).
            # Strip batch dimension (dim 0); pad to 3D if needed.
            dims = list(tensor.shape[1:])  # skip batch
            while len(dims) < 3:
                dims.insert(0, 1)
            input_shape = ":".join(str(d) for d in dims[-3:])
        else:
            input_shape = "1:1:1"  # fallback

        input_layers.append(NNTrainerLayerDef(
            layer_type=LAYER_INPUT,
            name=inp_name,
            properties={"input_shape": input_shape},
        ))

    if input_layers:
        count = len(input_layers)
        names = ", ".join(l.name for l in input_layers)
        print(f"  [CLEANUP] Added {count} input layers: {names}")

    # Extract shape info for reshape/view/unsqueeze/squeeze layers
    for layer in layers:
        if layer.layer_type not in (LAYER_RESHAPE, OP_RESHAPE):
            continue
        # Look up FX node by fx_node_name (preferred) or layer name
        node = fx_nodes.get(layer.fx_node_name) or fx_nodes.get(layer.name)
        if node is None:
            continue

        target = getattr(node, 'target', '')

        if target in ('view', 'reshape'):
            # node.args = (input_node, dim1, dim2, ...) or
            # node.args = (input_node, (dim1, dim2, ...))
            shape_args = node.args[1:]
            if len(shape_args) == 1 and isinstance(shape_args[0], (list, tuple)):
                shape_args = shape_args[0]
            # Strip batch dim (first dim), convert to C:H:W
            dims = [a for a in shape_args if isinstance(a, int)]
            if dims:
                dims = dims[1:]  # skip batch
                while len(dims) < 3:
                    dims.insert(0, 1)
                layer.properties["target_shape"] = \
                    ":".join(str(d) for d in dims[-3:])

        else:
            # For unsqueeze, squeeze, or any other reshape variant:
            # use output_shape captured during tracing
            out_shape = node.meta.get('output_shape')
            if out_shape is not None:
                dims = list(out_shape[1:])  # skip batch
                while len(dims) < 3:
                    dims.insert(0, 1)
                layer.properties["target_shape"] = \
                    ":".join(str(d) for d in dims[-3:])

    # Extract slice parameters for __getitem__ layers
    for layer in layers:
        if layer.layer_type != "slice":
            continue
        node = fx_nodes.get(layer.fx_node_name) or fx_nodes.get(layer.name)
        if node is None or len(node.args) < 2:
            continue

        # Determine input tensor rank for PyTorch→NCHW axis conversion
        input_node = node.args[0] if hasattr(node.args[0], 'name') else None
        input_rank = 4  # default
        if input_node:
            in_shape = input_node.meta.get('output_shape')
            if in_shape:
                input_rank = len(in_shape)

        index_arg = node.args[1]
        # Handle multi-dimensional indexing: tensor[..., idx] or
        # tensor[..., start:end]
        # e.g. span_idx[:, :, 0] → args[1] = (slice(None), slice(None), 0)
        # e.g. conv_out[..., :seqlen] → args[1] = (Ellipsis, slice(None, 8))
        if isinstance(index_arg, (list, tuple)):
            # Resolve Ellipsis to concrete axes
            # Ellipsis fills remaining dims to match input_rank
            expanded = []
            for item in index_arg:
                if item is Ellipsis:
                    # Fill with slice(None) for the missing dims
                    n_explicit = sum(1 for x in index_arg
                                     if x is not Ellipsis)
                    for _ in range(input_rank - n_explicit):
                        expanded.append(slice(None))
                else:
                    expanded.append(item)

            if not expanded:
                expanded = list(index_arg)

            # Find the axis being indexed (first non-trivial element)
            for ax, idx in enumerate(expanded):
                if isinstance(idx, int):
                    # Integer indexing: extract single element
                    nn_axis = ax + (4 - input_rank)
                    nn_axis = max(1, min(3, nn_axis))
                    if idx < 0:
                        in_shape = input_node.meta.get('output_shape') \
                            if input_node else None
                        if in_shape and ax < len(in_shape):
                            idx = in_shape[ax] + idx
                        else:
                            break
                    # NNTrainer slice: 1-based, end is exclusive
                    layer.properties["axis"] = nn_axis
                    layer.properties["start_index"] = idx + 1
                    layer.properties["end_index"] = idx + 2
                    break
                elif isinstance(idx, slice) and idx != slice(None):
                    # Slice object: extract range [start:stop]
                    nn_axis = ax + (4 - input_rank)
                    nn_axis = max(1, min(3, nn_axis))
                    start = idx.start if idx.start is not None else 0
                    stop = idx.stop
                    if stop is None:
                        # Open-ended slice — need input shape
                        in_shape = input_node.meta.get('output_shape') \
                            if input_node else None
                        if in_shape and ax < len(in_shape):
                            stop = in_shape[ax]
                        else:
                            break
                    if start < 0 or stop < 0:
                        in_shape = input_node.meta.get('output_shape') \
                            if input_node else None
                        if in_shape and ax < len(in_shape):
                            if start < 0:
                                start = in_shape[ax] + start
                            if stop < 0:
                                stop = in_shape[ax] + stop
                        else:
                            break
                    # NNTrainer slice: 1-based, end is inclusive
                    layer.properties["axis"] = nn_axis
                    layer.properties["start_index"] = start + 1
                    layer.properties["end_index"] = stop
                    break
        elif isinstance(index_arg, int):
            # Simple integer indexing on first non-batch dim
            nn_axis = 1 + (4 - input_rank)
            nn_axis = max(1, min(3, nn_axis))
            layer.properties["axis"] = nn_axis
            layer.properties["start_index"] = index_arg + 1
            layer.properties["end_index"] = index_arg + 2
        elif isinstance(index_arg, slice) and index_arg != slice(None):
            # Simple slice on first non-batch dim
            nn_axis = 1 + (4 - input_rank)
            nn_axis = max(1, min(3, nn_axis))
            start = index_arg.start if index_arg.start is not None else 0
            stop = index_arg.stop
            if stop is None:
                in_shape = input_node.meta.get('output_shape') \
                    if input_node else None
                if in_shape and len(in_shape) > 1:
                    stop = in_shape[1]
            if stop is not None:
                if start < 0 or stop < 0:
                    in_shape = input_node.meta.get('output_shape') \
                        if input_node else None
                    if in_shape and len(in_shape) > 1:
                        if start < 0:
                            start = in_shape[1] + start
                        if stop < 0:
                            stop = in_shape[1] + stop
                layer.properties["axis"] = nn_axis
                layer.properties["start_index"] = start + 1
                layer.properties["end_index"] = stop

    # Fix gather axis: convert PyTorch dim to NCHW axis (1-3)
    for layer in layers:
        if layer.layer_type != "gather":
            continue
        if "axis" not in layer.properties:
            continue
        node = fx_nodes.get(layer.fx_node_name) or fx_nodes.get(layer.name)
        if node is None:
            continue
        # Gather's first input is the data tensor; get its rank
        data_node = node.args[0] if hasattr(node.args[0], 'name') else None
        if data_node:
            data_shape = data_node.meta.get('output_shape')
            if data_shape:
                data_rank = len(data_shape)
                pytorch_axis = layer.properties["axis"]
                # Convert: nchw_dim = pytorch_dim + (4 - rank) for dim > 0
                nn_axis = pytorch_axis + (4 - data_rank)
                nn_axis = max(1, min(3, nn_axis))
                layer.properties["axis"] = nn_axis

    return input_layers + layers


class ConversionResult:
    """Result of adaptive conversion pipeline."""

    def __init__(self, layers, decomposed_module_types, unsupported_ops,
                 unknown_layers, tensor_ops, lazy_chains, graph,
                 model_structure=None, training=False):
        self.layers = layers
        self.decomposed_module_types = decomposed_module_types
        self.unsupported_ops = unsupported_ops
        self.unknown_layers = unknown_layers
        self.tensor_ops = tensor_ops
        self.lazy_chains = lazy_chains
        self.graph = graph
        self.model_structure = model_structure
        self.training = training

    @property
    def is_fully_mapped(self):
        """True if all ops are mapped (no unknowns or unsupported)."""
        return len(self.unsupported_ops) == 0 and len(self.unknown_layers) == 0

    def summary(self):
        """Print a summary of the conversion result."""
        type_counts = {}
        for layer in self.layers:
            type_counts[layer.layer_type] = type_counts.get(layer.layer_type, 0) + 1

        mode = "training" if self.training else "inference"
        print(f"\n{'='*70}")
        print(f"CONVERSION SUMMARY ({mode} mode)")
        print(f"{'='*70}")
        print(f"Total layers: {len(self.layers)}")
        print(f"Fully mapped: {self.is_fully_mapped}")
        if not self.training:
            print(f"Dropout layers: removed (inference mode)")

        if self.decomposed_module_types:
            print(f"Decomposed module types: "
                  f"{', '.join(sorted(self.decomposed_module_types))}")
        if self.tensor_ops:
            tensor_op_names = set(l.properties.get("original_op", "?")
                                  for l in self.tensor_ops)
            print(f"Tensor method ops: {', '.join(sorted(tensor_op_names))} "
                  f"({len(self.tensor_ops)} total)")
        if self.lazy_chains:
            print(f"LazyTensor chains: {len(self.lazy_chains)} chains "
                  f"({sum(c.chain_length for c in self.lazy_chains)} ops fusible)")
            for i, chain in enumerate(self.lazy_chains):
                print(f"  chain[{i}]: {chain}")
            print(f"  (Info: These are decomposed higher-level ops like GELU "
                  f"and position bias. NNTrainer built-in layers handle "
                  f"them; no LazyTensor C++ code needed.)")
        if self.unsupported_ops:
            print(f"Unsupported ops (no NNTrainer equivalent): "
                  f"{len(self.unsupported_ops)}")
            for op in self.unsupported_ops:
                print(f"  - {op.properties.get('original_op', '?')} "
                      f"at {op.hf_module_name}")
        if self.unknown_layers:
            print(f"Unknown layers: {len(self.unknown_layers)}")
            for l in self.unknown_layers:
                print(f"  - {l.layer_type} at {l.hf_module_name}")

        print(f"\nLayer type breakdown:")
        for lt, count in sorted(type_counts.items()):
            marker = ""
            if lt.startswith("unknown"):
                marker = " <<<< UNKNOWN"
            elif lt == OP_UNSUPPORTED:
                marker = " <<<< UNSUPPORTED"
            elif lt.startswith("tensor_op:"):
                marker = " (Tensor method)"
            print(f"  {lt:30s}: {count}{marker}")
        print(f"{'='*70}")
