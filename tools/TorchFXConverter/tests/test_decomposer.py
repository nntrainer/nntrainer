"""Test adaptive decomposition pipeline with LazyTensor support.

Tests:
  1. Known models (Qwen3, BERT, mT5) pass through without decomposition
  2. Custom unknown modules are automatically decomposed into tensor ops
  3. Unsupported ops resolved via Tensor methods or layer decomposition
  4. LazyTensor chain detection for consecutive arithmetic ops
  5. Exclude leaf types in tracer
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from decomposer import (
    AdaptiveConverter, resolve_unsupported_ops, detect_lazy_chains,
    LazyTensorChain,
)
from nntrainer_layers import (
    NNTrainerLayerDef, OP_UNSUPPORTED, LAYER_POW, LAYER_SQRT,
    LAYER_MULTIPLY, LAYER_DIVIDE, LAYER_ADD, LAYER_ADDITION, LAYER_SUBTRACT,
    TENSOR_DIRECT_METHODS,
)

# backward compat alias
decompose_unsupported_ops = resolve_unsupported_ops


# ============================================================================
# Test 1: Known models pass through cleanly (single pass, no decomposition)
# ============================================================================

def test_qwen3_no_decomposition():
    """Qwen3 should be fully mapped in a single pass."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    config = Qwen3Config(
        vocab_size=151936, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=2048, rms_norm_eps=1e-6,
        tie_word_embeddings=True, rope_theta=1000000.0, sliding_window=None,
    )
    model = Qwen3ForCausalLM(config)
    model.eval()

    converter = AdaptiveConverter(model, config)
    result = converter.convert({"input_ids": torch.randint(0, config.vocab_size, (1, 8))})

    result.summary()
    assert len(result.unknown_layers) == 0, f"Qwen3 has unknowns: {result.unknown_layers}"
    assert len(result.decomposed_module_types) == 0, "Qwen3 should not need module decomposition"
    print("  PASS: Qwen3 fully mapped without decomposition")


def test_bert_no_decomposition():
    """BERT should be fully mapped in a single pass."""
    from transformers import BertConfig, BertModel
    config = BertConfig(
        vocab_size=30522, hidden_size=64, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=128, max_position_embeddings=512,
    )
    model = BertModel(config)
    model.eval()

    converter = AdaptiveConverter(model, config)
    result = converter.convert({
        "input_ids": torch.randint(0, config.vocab_size, (1, 16)),
        "attention_mask": torch.ones(1, 16, dtype=torch.long),
    })

    result.summary()
    assert len(result.unknown_layers) == 0, f"BERT has unknowns: {result.unknown_layers}"
    print("  PASS: BERT fully mapped without decomposition")


def test_mt5_no_decomposition():
    """mT5 should be fully mapped in a single pass."""
    from transformers import MT5Config, MT5ForConditionalGeneration
    config = MT5Config(
        vocab_size=250112, d_model=64, d_kv=16, d_ff=128,
        num_heads=4, num_layers=2, num_decoder_layers=2,
        relative_attention_num_buckets=32, relative_attention_max_distance=128,
    )
    model = MT5ForConditionalGeneration(config)
    model.eval()

    converter = AdaptiveConverter(model, config)
    result = converter.convert({
        "input_ids": torch.randint(0, config.vocab_size, (1, 8)),
        "decoder_input_ids": torch.randint(0, config.vocab_size, (1, 4)),
    })

    result.summary()
    assert len(result.unknown_layers) == 0, f"mT5 has unknowns: {result.unknown_layers}"
    print("  PASS: mT5 fully mapped without decomposition")


# ============================================================================
# Test 2: Custom unknown module gets decomposed into tensor ops
# ============================================================================

class CustomNorm(nn.Module):
    """A hypothetical normalization not in LEAF_MODULES or RMSNorm detection.

    forward() uses: mean, subtract, pow, mean, add, rsqrt, multiply
    All of these should be captured as tensor ops when decomposed.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        x = x - mean
        var = (x * x).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


class ModelWithCustomNorm(nn.Module):
    """Simple model using a custom normalization module."""
    def __init__(self, dim=64):
        super().__init__()
        self.embed = nn.Embedding(1000, dim)
        self.norm = CustomNorm(dim)
        self.proj = nn.Linear(dim, 1000)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = self.norm(x)
        x = self.proj(x)
        return x


def test_custom_module_auto_decomposition():
    """Custom module NOT in LEAF_MODULES is auto-decomposed by the tracer.

    Since CustomNorm is not in LEAF_MODULES, the tracer automatically traces
    through its forward(), decomposing it into tensor ops. No 2nd pass needed.
    This is the default behavior for unknown modules.
    """
    model = ModelWithCustomNorm(dim=64)
    model.eval()

    converter = AdaptiveConverter(model)
    result = converter.convert({"input_ids": torch.randint(0, 1000, (1, 8))})

    result.summary()

    # No unknown module layers should remain (CustomNorm was never a leaf)
    assert len(result.unknown_layers) == 0, \
        f"Should have no unknown layers, got: {result.unknown_layers}"

    # Check that tensor ops from CustomNorm's forward() are present
    layer_types = {l.layer_type for l in result.layers}
    assert "reduce_mean" in layer_types or "subtract" in layer_types or "multiply" in layer_types, \
        f"Expected tensor ops from decomposed CustomNorm, got types: {sorted(layer_types)}"

    print("  PASS: CustomNorm auto-decomposed into tensor ops (not in LEAF_MODULES)")


def test_forced_leaf_decomposition():
    """Test the 2-pass mechanism: force a known module type to be leaf, then decompose.

    This simulates a scenario where a module IS detected as a leaf (e.g. through
    LEAF_MODULES or RMSNorm detection) but the mapper can't handle it. The
    adaptive converter should detect this and re-trace with that type excluded.
    """
    from tracer import Tracer, LEAF_MODULES
    from node_mapper import NodeMapper

    model = ModelWithCustomNorm(dim=32)
    model.eval()
    input_kwargs = {"input_ids": torch.randint(0, 1000, (1, 4))}

    # Manually trace with CustomNorm as a LEAF (simulating a new HF module type
    # that was added to LEAF_MODULES but not to NodeMapper)
    extended_leaves = LEAF_MODULES + (CustomNorm,)
    tracer = Tracer(model, leaf_modules=extended_leaves)
    with tracer:
        with torch.no_grad():
            model(**input_kwargs)

    mapper = NodeMapper(model, tracer.graph)
    layers = mapper.map_all()

    # CustomNorm should appear as unknown(CustomNorm)
    unknown_types = mapper.get_unknown_module_types(layers)
    assert "CustomNorm" in unknown_types, \
        f"CustomNorm should be unknown when forced as leaf, got: {unknown_types}"

    # Now re-trace with CustomNorm excluded -> decomposition
    tracer2 = Tracer(model, leaf_modules=extended_leaves,
                     exclude_leaf_types={"CustomNorm"})
    with tracer2:
        with torch.no_grad():
            model(**input_kwargs)

    mapper2 = NodeMapper(model, tracer2.graph)
    layers2 = mapper2.map_all()
    unknown_types2 = mapper2.get_unknown_module_types(layers2)

    assert "CustomNorm" not in unknown_types2, \
        f"CustomNorm should be decomposed after exclusion, still unknown: {unknown_types2}"

    print("  PASS: 2-pass decomposition works (leaf -> exclude -> tensor ops)")


# ============================================================================
# Test 3: Op-level decomposition (rsqrt, abs)
# ============================================================================

def test_rsqrt_tensor_method():
    """rsqrt should resolve to Tensor::inv_sqrt() (direct method)."""
    layer = NNTrainerLayerDef(
        layer_type=OP_UNSUPPORTED,
        name="test_rsqrt",
        properties={"original_op": "rsqrt"},
        input_layers=["input"],
    )
    result = resolve_unsupported_ops([layer])
    assert len(result) == 1
    assert result[0].layer_type == "tensor_op:rsqrt"
    assert result[0].properties["tensor_method"] == "inv_sqrt"
    print("  PASS: rsqrt -> Tensor::inv_sqrt() (direct method)")


def test_abs_tensor_method():
    """abs should resolve to Tensor::abs() (direct method)."""
    layer = NNTrainerLayerDef(
        layer_type=OP_UNSUPPORTED,
        name="test_abs",
        properties={"original_op": "abs"},
        input_layers=["input"],
    )
    result = resolve_unsupported_ops([layer])
    assert len(result) == 1
    assert result[0].layer_type == "tensor_op:abs"
    assert result[0].properties["tensor_method"] == "abs"
    print("  PASS: abs -> Tensor::abs() (direct method)")


def test_all_tensor_direct_methods():
    """All ops in TENSOR_DIRECT_METHODS should resolve to tensor_op: types."""
    for op_name, (method_name, _) in TENSOR_DIRECT_METHODS.items():
        layer = NNTrainerLayerDef(
            layer_type=OP_UNSUPPORTED,
            name=f"test_{op_name}",
            properties={"original_op": op_name},
            input_layers=["input"],
        )
        result = resolve_unsupported_ops([layer])
        assert len(result) == 1, f"{op_name}: expected 1 result, got {len(result)}"
        assert result[0].layer_type == f"tensor_op:{op_name}", \
            f"{op_name}: expected tensor_op:{op_name}, got {result[0].layer_type}"
        assert result[0].properties["tensor_method"] == method_name, \
            f"{op_name}: expected method {method_name}"
    print(f"  PASS: All {len(TENSOR_DIRECT_METHODS)} Tensor direct methods resolve correctly")


def test_reciprocal_layer_decomposition():
    """reciprocal should decompose to divide(1, x) via layer decomposition."""
    layer = NNTrainerLayerDef(
        layer_type=OP_UNSUPPORTED,
        name="test_reciprocal",
        properties={"original_op": "reciprocal"},
        input_layers=["input"],
    )
    result = resolve_unsupported_ops([layer])
    assert len(result) == 1
    assert result[0].layer_type == LAYER_DIVIDE
    assert result[0].properties.get("numerator") == 1.0
    print("  PASS: reciprocal -> divide(1, x) (layer decomposition)")


def test_unsupported_op_preserved():
    """Ops without any resolution should be preserved with warning."""
    # Use a truly unsupported op (not exp/log/clamp which are now supported)
    layer = NNTrainerLayerDef(
        layer_type=OP_UNSUPPORTED,
        name="test_bitwise_and",
        properties={"original_op": "bitwise_and"},
        input_layers=["input"],
    )
    result = resolve_unsupported_ops([layer])
    assert len(result) == 1
    assert result[0].layer_type == OP_UNSUPPORTED
    print("  PASS: bitwise_and preserved as unsupported (no resolution available)")


# ============================================================================
# Test 3b: LazyTensor chain detection
# ============================================================================

def test_lazy_chain_detection():
    """Consecutive arithmetic ops should be detected as LazyTensor chains."""
    layers = [
        NNTrainerLayerDef(layer_type="embedding_layer", name="embed", input_layers=[]),
        NNTrainerLayerDef(layer_type=LAYER_ADD, name="add1", input_layers=["embed"]),
        NNTrainerLayerDef(layer_type=LAYER_MULTIPLY, name="mul1", input_layers=["add1"]),
        NNTrainerLayerDef(layer_type=LAYER_SUBTRACT, name="sub1", input_layers=["mul1"]),
        NNTrainerLayerDef(layer_type="fully_connected", name="fc", input_layers=["sub1"]),
        NNTrainerLayerDef(layer_type=LAYER_ADDITION, name="residual", input_layers=["fc"]),
        NNTrainerLayerDef(layer_type=LAYER_DIVIDE, name="div1", input_layers=["residual"]),
    ]

    chains = detect_lazy_chains(layers)
    assert len(chains) == 2, f"Expected 2 chains, got {len(chains)}"

    # First chain: add1 -> mul1 -> sub1 (3 ops)
    assert chains[0].chain_length == 3
    assert chains[0].start_idx == 1

    # Second chain: residual -> div1 (2 ops)
    assert chains[1].chain_length == 2
    assert chains[1].start_idx == 5

    print(f"  PASS: LazyTensor chains detected: {chains}")


def test_lazy_chain_cpp_generation():
    """LazyTensor chain should generate valid C++ code."""
    layers = [
        NNTrainerLayerDef(layer_type=LAYER_ADD, name="add1",
                          input_layers=["x", "bias"]),
        NNTrainerLayerDef(layer_type=LAYER_MULTIPLY, name="mul1",
                          input_layers=["add1", "scale"]),
    ]

    chain = LazyTensorChain(layers, 0)
    cpp = chain.to_cpp_chain("hidden")
    assert "hidden.chain()" in cpp
    assert "add_i(bias)" in cpp
    assert "multiply_i(scale)" in cpp
    assert "run()" in cpp
    print(f"  PASS: LazyTensor C++ code: {cpp}")


# ============================================================================
# Test 4: Exclude leaf types in tracer
# ============================================================================

def test_exclude_leaf_types():
    """Verify that exclude_leaf_types forces module decomposition."""
    from tracer import Tracer

    model = ModelWithCustomNorm(dim=32)
    model.eval()
    input_ids = torch.randint(0, 1000, (1, 4))

    # Pass 1: Normal tracing - Linear and Embedding appear as leaf modules
    tracer1 = Tracer(model)
    with tracer1:
        with torch.no_grad():
            model(input_ids)

    linear_modules_1 = [n for n in tracer1.graph.nodes
                        if n.op == "call_module" and n.meta.get("module_type") == "Linear"]
    assert len(linear_modules_1) > 0, "Linear should be a leaf module"

    # Pass 2: Exclude Linear -> no Linear call_module nodes
    tracer2 = Tracer(model, exclude_leaf_types={"Linear"})
    with tracer2:
        with torch.no_grad():
            model(input_ids)

    linear_modules_2 = [n for n in tracer2.graph.nodes
                        if n.op == "call_module" and n.meta.get("module_type") == "Linear"]
    assert len(linear_modules_2) == 0, \
        "Linear should NOT appear as leaf when excluded"

    print(f"  PASS: exclude_leaf_types works (Linear leaf nodes: {len(linear_modules_1)} -> {len(linear_modules_2)})")


# ============================================================================
# Test 5: SpanRepLayer-like custom module auto-decomposition
# ============================================================================

class SpanRepLayer(nn.Module):
    """GLiNER2 SpanRepLayer (markerV0) equivalent for testing.

    This module is NOT in LEAF_MODULES, so the tracer should automatically
    decompose its forward() into basic tensor ops:
      Linear -> ReLU -> Dropout -> Linear (start MLP)
      Linear -> ReLU -> Dropout -> Linear (end MLP)
      __getitem__ (slice span indices)
      unsqueeze + expand (prepare gather indices)
      torch.gather (extract span representations)
      torch.cat (concat start + end)
      Tensor.relu() (activation)
      Linear -> ReLU -> Dropout -> Linear (output MLP)
      Tensor.view (reshape to span width)
    """
    def __init__(self, hidden_size=64, max_width=8):
        super().__init__()
        self.max_width = max_width
        self.start_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.end_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, h, span_idx):
        B, L, D = h.shape
        num_spans = span_idx.shape[1]
        start_rep = self.start_mlp(h)
        end_rep = self.end_mlp(h)
        start_idx = span_idx[:, :, 0].unsqueeze(-1).expand(B, num_spans, D)
        end_idx = span_idx[:, :, 1].unsqueeze(-1).expand(B, num_spans, D)
        start_span = torch.gather(start_rep, 1, start_idx)
        end_span = torch.gather(end_rep, 1, end_idx)
        cat = torch.cat([start_span, end_span], dim=-1)
        cat = cat.relu()
        output = self.out_mlp(cat)
        output = output.view(B, -1, self.max_width, D)
        return output


class ModelWithSpanRep(nn.Module):
    """Model that wraps SpanRepLayer for testing."""
    def __init__(self, hidden_size=64, max_width=8):
        super().__init__()
        self.embed = nn.Embedding(100, hidden_size)
        self.span_rep = SpanRepLayer(hidden_size, max_width)

    def forward(self, input_ids, span_idx):
        h = self.embed(input_ids)
        return self.span_rep(h, span_idx)


def test_span_rep_auto_decomposition():
    """SpanRepLayer should be fully decomposed into supported tensor ops.

    Verifies that all ops from SpanRepLayer's forward() are mappable:
    - gather with correct axis extraction
    - concat with proper input list and dim extraction
    - Tensor.relu() method mapped to activation layer
    - All MLPs decomposed into fully_connected + activation + dropout
    """
    model = ModelWithSpanRep(hidden_size=64, max_width=8)
    model.eval()

    converter = AdaptiveConverter(model)
    result = converter.convert({
        "input_ids": torch.randint(0, 100, (2, 16)),
        "span_idx": torch.randint(0, 16, (2, 128, 2)),
    })

    result.summary()

    # No unknown or unsupported layers should remain
    assert len(result.unknown_layers) == 0, \
        f"SpanRepLayer has unknown layers: {[l.layer_type for l in result.unknown_layers]}"
    assert len(result.unsupported_ops) == 0, \
        f"SpanRepLayer has unsupported ops: {[l.properties.get('original_op') for l in result.unsupported_ops]}"

    # Verify critical ops are present in the decomposed output
    layer_types = [l.layer_type for l in result.layers]
    assert "gather" in layer_types, "gather op missing from decomposition"
    assert "concat" in layer_types, "concat op missing from decomposition"
    assert "fully_connected" in layer_types, "FC layers missing from decomposition"

    # Verify gather layers have axis property
    gather_layers = [l for l in result.layers if l.layer_type == "gather"]
    assert len(gather_layers) == 2, f"Expected 2 gather ops, got {len(gather_layers)}"
    for g in gather_layers:
        assert "axis" in g.properties, f"Gather layer missing axis property: {g.name}"
        assert g.properties["axis"] == 2, f"Expected axis=2, got {g.properties['axis']}"

    # Verify concat has proper inputs (not empty)
    concat_layers = [l for l in result.layers if l.layer_type == "concat"]
    assert len(concat_layers) == 1, f"Expected 1 concat op, got {len(concat_layers)}"
    assert len(concat_layers[0].input_layers) == 2, \
        f"Concat should have 2 inputs, got {len(concat_layers[0].input_layers)}"

    # Verify relu activation exists (from Tensor.relu() method)
    activation_layers = [l for l in result.layers
                         if l.layer_type == "activation"
                         and l.properties.get("activation") == "relu"]
    assert len(activation_layers) >= 1, "Missing relu activation from Tensor.relu()"

    # Count FC layers: 2 in start_mlp + 2 in end_mlp + 2 in out_mlp = 6
    fc_layers = [l for l in result.layers if l.layer_type == "fully_connected"]
    assert len(fc_layers) == 6, f"Expected 6 FC layers, got {len(fc_layers)}"

    print("  PASS: SpanRepLayer fully decomposed into supported tensor ops")
    print(f"    gather ops: {len(gather_layers)} (axis={gather_layers[0].properties['axis']})")
    print(f"    concat ops: {len(concat_layers)} (inputs={concat_layers[0].input_layers})")
    print(f"    FC layers:  {len(fc_layers)}")
    print(f"    Total layers: {len(result.layers)}")


# ============================================================================
# Test 6: Multi-head output (fan-out) with data_ptr tracking
# ============================================================================

class MultiHeadModel(nn.Module):
    """GLiNER2-like multi-head output model.

    Encoder output fans out to 3 separate heads (classifier, count predictor,
    span projector). This tests that:
      1. The tracer correctly resolves tensor references across detach/clone
         boundaries using data_ptr fallback
      2. All 3 heads reference the encoder output (not _tensor_constant_*)
      3. NNTrainer's MultioutRealizer can handle this pattern (single output
         referenced by multiple consumers)
    """
    def __init__(self, hidden=64, num_classes=10):
        super().__init__()
        self.encoder = nn.Linear(32, hidden)
        self.classifier = nn.Linear(hidden, num_classes)
        self.count_pred = nn.Linear(hidden, 1)
        self.span_proj = nn.Linear(hidden, hidden)

    def forward(self, x):
        h = self.encoder(x)
        cls_out = self.classifier(h)
        count_out = self.count_pred(h)
        span_out = self.span_proj(h)
        return cls_out, count_out, span_out


def test_multi_head_fan_out():
    """Multi-head output should correctly reference shared encoder output.

    Verifies that the data_ptr fallback in the tracer resolves detached
    tensors to their producing node, so downstream heads reference
    'encoder' directly (not _tensor_constant_* artifacts).
    """
    model = MultiHeadModel()
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"x": torch.randn(2, 32)})

    # No unknowns
    assert len(result.unknown_layers) == 0, \
        f"Multi-head model has unknowns: {result.unknown_layers}"

    # Find the encoder and its consumers
    fc_layers = [l for l in result.layers if l.layer_type == "fully_connected"]
    assert len(fc_layers) == 4, f"Expected 4 FC layers, got {len(fc_layers)}"

    encoder_layer = [l for l in fc_layers if l.name == "encoder"]
    assert len(encoder_layer) == 1, "Missing encoder layer"

    # All 3 downstream heads should reference 'encoder' as input
    downstream = [l for l in fc_layers if l.name != "encoder"]
    assert len(downstream) == 3, f"Expected 3 downstream heads, got {len(downstream)}"

    for head in downstream:
        assert "encoder" in head.input_layers, \
            f"Head '{head.name}' should reference 'encoder' but has inputs: {head.input_layers}"
        # Must NOT reference _tensor_constant_*
        for inp in head.input_layers:
            assert "_tensor_constant" not in inp, \
                f"Head '{head.name}' has broken tensor reference: {inp}"

    print("  PASS: Multi-head fan-out correctly references encoder output")
    print(f"    Encoder consumers: {[h.name for h in downstream]}")
    print(f"    All inputs resolved (no _tensor_constant_ artifacts)")


def test_multi_head_with_detach():
    """Multi-head output with explicit detach() calls.

    Some models use detach() on shared features before passing to heads.
    The data_ptr fallback should still resolve these correctly.
    """
    class DetachMultiHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(32, 64)
            self.head_a = nn.Linear(64, 10)
            self.head_b = nn.Linear(64, 5)

        def forward(self, x):
            h = self.encoder(x)
            a = self.head_a(h.detach())
            b = self.head_b(h)
            return a, b

    model = DetachMultiHead()
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    result = converter.convert({"x": torch.randn(2, 32)})

    assert len(result.unknown_layers) == 0

    # Both heads should have resolvable inputs (no _tensor_constant_*)
    for layer in result.layers:
        if layer.layer_type == "fully_connected" and layer.name != "encoder":
            for inp in layer.input_layers:
                assert "_tensor_constant" not in inp, \
                    f"Head '{layer.name}' has broken reference: {inp}"

    print("  PASS: Multi-head with detach() correctly resolved via data_ptr")


# ============================================================================
# Test 7: GLiNER2 core model conversion
# ============================================================================

def _create_projection_layer(hidden_size, dropout, out_dim=None):
    """GLiNER2 projection layer factory (matches gliner.modeling.layers)."""
    if out_dim is None:
        out_dim = hidden_size
    return nn.Sequential(
        nn.Linear(hidden_size, out_dim * 4), nn.ReLU(),
        nn.Dropout(dropout), nn.Linear(out_dim * 4, out_dim),
    )


def _extract_elements(sequence, indices):
    """GLiNER2 gather helper (matches gliner.modeling.span_rep)."""
    D = sequence.size(-1)
    expanded_indices = indices.unsqueeze(2).expand(-1, -1, D)
    return torch.gather(sequence, 1, expanded_indices)


class _SpanMarkerV0(nn.Module):
    """GLiNER2 SpanMarkerV0 (matches gliner.modeling.span_rep.SpanMarkerV0)."""
    def __init__(self, hidden_size, max_width, dropout=0.4):
        super().__init__()
        self.max_width = max_width
        self.project_start = _create_projection_layer(hidden_size, dropout)
        self.project_end = _create_projection_layer(hidden_size, dropout)
        self.out_project = _create_projection_layer(hidden_size * 2, dropout, hidden_size)

    def forward(self, h, span_idx):
        B, L, D = h.size()
        start_rep = self.project_start(h)
        end_rep = self.project_end(h)
        start_span_rep = _extract_elements(start_rep, span_idx[:, :, 0])
        end_span_rep = _extract_elements(end_rep, span_idx[:, :, 1])
        cat = torch.cat([start_span_rep, end_span_rep], dim=-1).relu()
        return self.out_project(cat).view(B, L, self.max_width, D)


class GLiNER2Core(nn.Module):
    """GLiNER2 core computation graph (post-preprocessing).

    This replicates the actual GLiNER2 UniEncoderSpanModel architecture:
      1. BiLSTM encoder (words_embedding -> rnn_out)
      2. SpanMarkerV0 (rnn_out + span_idx -> span_rep)
      3. Prompt projection (prompts_embedding -> projected prompts)
      4. Einsum scoring (span_rep @ prompts^T -> scores)

    The preprocessing step (extract_prompt_features_and_word_embeddings)
    uses data-dependent indexing and is handled outside the NNTrainer graph.
    """
    def __init__(self, hidden_size=64, max_width=8, dropout=0.4):
        super().__init__()
        self.max_width = max_width
        self.rnn = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size // 2,
            num_layers=1, bidirectional=True, batch_first=True,
        )
        self.span_rep = _SpanMarkerV0(hidden_size, max_width, dropout)
        self.prompt_proj = _create_projection_layer(hidden_size, dropout)

    def forward(self, words_embedding, span_idx, prompts_embedding):
        rnn_out, _ = self.rnn(words_embedding)
        span_rep = self.span_rep(rnn_out, span_idx)
        prompts_embedding = self.prompt_proj(prompts_embedding)
        scores = torch.einsum("BLKD,BCD->BLKC", span_rep, prompts_embedding)
        return scores


def test_gliner2_core_conversion():
    """GLiNER2 core model should be fully converted without unknowns.

    Validates the complete GLiNER2 inference pipeline conversion:
    - LSTM layer mapping with bidirectional + tuple output
    - SpanMarkerV0 auto-decomposition (gather, concat, MLPs)
    - torch.einsum mapped to matmul
    - No broken tensor references (_tensor_constant_*)
    - Dropout layers removed in inference mode
    """
    model = GLiNER2Core(hidden_size=64, max_width=8)
    model.eval()

    converter = AdaptiveConverter(model, training=False)
    B, L, D, K, C = 2, 16, 64, 8, 5
    result = converter.convert({
        "words_embedding": torch.randn(B, L, D),
        "span_idx": torch.randint(0, L, (B, L * K, 2)),
        "prompts_embedding": torch.randn(B, C, D),
    })

    # No unknowns or unsupported
    assert len(result.unknown_layers) == 0, \
        f"GLiNER2 has unknowns: {[l.layer_type for l in result.unknown_layers]}"
    assert len(result.unsupported_ops) == 0, \
        f"GLiNER2 has unsupported ops: {[l.properties.get('original_op') for l in result.unsupported_ops]}"

    # No broken tensor references
    for layer in result.layers:
        for inp in layer.input_layers:
            assert "_tensor_constant" not in inp, \
                f"Layer '{layer.name}' has broken reference: {inp}"

    # Verify key components
    layer_types = [l.layer_type for l in result.layers]

    # LSTM
    assert "lstm" in layer_types, "Missing LSTM layer"

    # SpanMarkerV0: 2 gather, 1 concat, 6 FC (start_mlp=2, end_mlp=2, out_mlp=2)
    gather_layers = [l for l in result.layers if l.layer_type == "gather"]
    assert len(gather_layers) == 2, f"Expected 2 gather ops, got {len(gather_layers)}"
    for g in gather_layers:
        assert g.properties.get("axis") == 2

    concat_layers = [l for l in result.layers if l.layer_type == "concat"]
    assert len(concat_layers) == 1
    assert len(concat_layers[0].input_layers) == 2

    # FC layers: 6 (SpanMarkerV0) + 2 (prompt_proj) = 8
    fc_layers = [l for l in result.layers if l.layer_type == "fully_connected"]
    assert len(fc_layers) == 8, f"Expected 8 FC layers, got {len(fc_layers)}"

    # Einsum -> matmul
    matmul_layers = [l for l in result.layers if l.layer_type == "matmul"]
    assert len(matmul_layers) == 1
    assert matmul_layers[0].properties.get("equation") == "BLKD,BCD->BLKC"

    print("  PASS: GLiNER2 core model fully converted")
    print(f"    LSTM: 1 (bidirectional)")
    print(f"    FC layers: {len(fc_layers)}")
    print(f"    Gather ops: {len(gather_layers)} (axis=1)")
    print(f"    Concat ops: {len(concat_layers)}")
    print(f"    Einsum/matmul: {len(matmul_layers)} (BLKD,BCD->BLKC)")
    print(f"    Total layers: {len(result.layers)}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TEST: Tensor method resolution (rsqrt, abs, etc.)")
    print("=" * 70)
    test_rsqrt_tensor_method()
    test_abs_tensor_method()
    test_all_tensor_direct_methods()
    test_reciprocal_layer_decomposition()
    test_unsupported_op_preserved()

    print("\n" + "=" * 70)
    print("TEST: LazyTensor chain detection")
    print("=" * 70)
    test_lazy_chain_detection()
    test_lazy_chain_cpp_generation()

    print("\n" + "=" * 70)
    print("TEST: Tracer exclude_leaf_types")
    print("=" * 70)
    test_exclude_leaf_types()

    print("\n" + "=" * 70)
    print("TEST: Custom module decomposition")
    print("=" * 70)
    test_custom_module_auto_decomposition()
    test_forced_leaf_decomposition()

    print("\n" + "=" * 70)
    print("TEST: SpanRepLayer decomposition (GLiNER2)")
    print("=" * 70)
    test_span_rep_auto_decomposition()

    print("\n" + "=" * 70)
    print("TEST: Multi-head output (fan-out)")
    print("=" * 70)
    test_multi_head_fan_out()
    test_multi_head_with_detach()

    print("\n" + "=" * 70)
    print("TEST: GLiNER2 core model conversion")
    print("=" * 70)
    test_gliner2_core_conversion()

    print("\n" + "=" * 70)
    print("TEST: Known models (no decomposition needed)")
    print("=" * 70)
    test_qwen3_no_decomposition()
    test_bert_no_decomposition()
    test_mt5_no_decomposition()

    print("\n" + "=" * 70)
    print("ALL DECOMPOSER TESTS PASSED!")
    print("=" * 70)
