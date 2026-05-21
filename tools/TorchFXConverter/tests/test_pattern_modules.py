"""Tests for the patterns/ sub-package modules.

Unit tests for individual sub-modules (scope, attention, ffn, block, config,
data_types) and backward compatibility of the pattern_detector facade.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from nntrainer_layers import (
    NNTrainerLayerDef,
    LAYER_FC, LAYER_RMS_NORM, LAYER_LAYER_NORM,
    LAYER_ADDITION, LAYER_ACTIVATION, LAYER_MULTIPLY,
    LAYER_EMBEDDING, OP_SDPA,
)


# ============================================================================
# Helper: build a minimal NNTrainerLayerDef
# ============================================================================

def _layer(name, layer_type, hf_module_name="", **props):
    """Create a minimal NNTrainerLayerDef for testing."""
    l = NNTrainerLayerDef(layer_type=layer_type, name=name)
    l.hf_module_name = hf_module_name
    l.properties = dict(props)
    l.input_layers = []
    return l


# ============================================================================
# Test scope.py
# ============================================================================

def test_find_block_scopes_llama_style():
    """find_block_scopes detects model.layers.N scopes."""
    from patterns.scope import find_block_scopes
    layers = [
        _layer("q", LAYER_FC, "model.layers.0.self_attn.q_proj"),
        _layer("k", LAYER_FC, "model.layers.0.self_attn.k_proj"),
        _layer("q1", LAYER_FC, "model.layers.1.self_attn.q_proj"),
    ]
    scopes = find_block_scopes(layers)
    assert scopes == ["model.layers.0", "model.layers.1"]


def test_find_block_scopes_bert_style():
    """find_block_scopes detects encoder.layer.N scopes."""
    from patterns.scope import find_block_scopes
    layers = [
        _layer("q", LAYER_FC, "bert.encoder.layer.0.attention.self.query"),
        _layer("q1", LAYER_FC, "bert.encoder.layer.1.attention.self.query"),
    ]
    scopes = find_block_scopes(layers)
    assert scopes == ["bert.encoder.layer.0", "bert.encoder.layer.1"]


def test_find_block_scopes_gpt2_style():
    """find_block_scopes detects transformer.h.N scopes."""
    from patterns.scope import find_block_scopes
    layers = [
        _layer("q", LAYER_FC, "transformer.h.0.attn.q_proj"),
        _layer("q1", LAYER_FC, "transformer.h.1.attn.q_proj"),
    ]
    scopes = find_block_scopes(layers)
    assert scopes == ["transformer.h.0", "transformer.h.1"]


def test_get_layers_in_scope():
    """get_layers_in_scope filters by scope prefix."""
    from patterns.scope import get_layers_in_scope
    layers = [
        _layer("q0", LAYER_FC, "model.layers.0.self_attn.q_proj"),
        _layer("q1", LAYER_FC, "model.layers.1.self_attn.q_proj"),
        _layer("emb", LAYER_EMBEDDING, "model.embed_tokens"),
    ]
    result = get_layers_in_scope(layers, "model.layers.0")
    assert len(result) == 1
    assert result[0].name == "q0"


def test_find_attention_scope():
    """find_attention_scope returns self_attn scope."""
    from patterns.scope import find_attention_scope
    layers = [
        _layer("q", LAYER_FC, "model.layers.0.self_attn.q_proj"),
        _layer("k", LAYER_FC, "model.layers.0.self_attn.k_proj"),
        _layer("up", LAYER_FC, "model.layers.0.mlp.up_proj"),
    ]
    scope = find_attention_scope("model.layers.0", layers)
    assert scope == "model.layers.0.self_attn"


def test_find_ffn_scope():
    """find_ffn_scope returns mlp scope."""
    from patterns.scope import find_ffn_scope
    layers = [
        _layer("q", LAYER_FC, "model.layers.0.self_attn.q_proj"),
        _layer("up", LAYER_FC, "model.layers.0.mlp.up_proj"),
        _layer("down", LAYER_FC, "model.layers.0.mlp.down_proj"),
    ]
    scope = find_ffn_scope("model.layers.0", layers)
    assert scope == "model.layers.0.mlp"


def test_find_ffn_scope_feed_forward():
    """find_ffn_scope detects feed_forward variant."""
    from patterns.scope import find_ffn_scope
    layers = [
        _layer("up", LAYER_FC, "model.layers.0.feed_forward.up"),
    ]
    scope = find_ffn_scope("model.layers.0", layers)
    assert scope == "model.layers.0.feed_forward"


def test_find_cross_attention_scope():
    """find_cross_attention_scope detects EncDecAttention."""
    from patterns.scope import find_cross_attention_scope
    layers = [
        _layer("q", LAYER_FC,
               "decoder.block.0.layer.1.EncDecAttention.q"),
    ]
    scope = find_cross_attention_scope("decoder.block.0", layers)
    assert scope is not None
    assert "EncDecAttention" in scope


# ============================================================================
# Test attention.py
# ============================================================================

def test_detect_attention_qkvo():
    """detect_attention finds Q/K/V/O projections."""
    from patterns.attention import detect_attention
    layers = [
        _layer("q", LAYER_FC, "model.layers.0.self_attn.q_proj", unit=64),
        _layer("k", LAYER_FC, "model.layers.0.self_attn.k_proj", unit=32),
        _layer("v", LAYER_FC, "model.layers.0.self_attn.v_proj", unit=32),
        _layer("o", LAYER_FC, "model.layers.0.self_attn.o_proj", unit=64),
        _layer("sdpa", OP_SDPA, "model.layers.0.self_attn"),
    ]

    @dataclass
    class FakeConfig:
        num_attention_heads: int = 4
        head_dim: int = 16
        rope_theta: float = 0.0

    attn = detect_attention(0, "model.layers.0.self_attn", layers, layers,
                            FakeConfig())
    assert attn.q_proj == "q"
    assert attn.k_proj == "k"
    assert attn.v_proj == "v"
    assert attn.o_proj == "o"
    assert attn.sdpa == "sdpa"
    assert attn.num_heads == 4
    assert attn.head_dim == 16
    assert attn.num_kv_heads == 2  # 32 / 16
    assert attn.attention_type == "gqa"


def test_detect_attention_qk_norms():
    """detect_attention detects Q/K norms."""
    from patterns.attention import detect_attention
    layers = [
        _layer("q", LAYER_FC, "model.layers.0.self_attn.q_proj", unit=64),
        _layer("k", LAYER_FC, "model.layers.0.self_attn.k_proj", unit=64),
        _layer("v", LAYER_FC, "model.layers.0.self_attn.v_proj", unit=64),
        _layer("o", LAYER_FC, "model.layers.0.self_attn.o_proj", unit=64),
        _layer("qn", LAYER_RMS_NORM, "model.layers.0.self_attn.q_norm"),
        _layer("kn", LAYER_RMS_NORM, "model.layers.0.self_attn.k_norm"),
    ]
    attn = detect_attention(0, "model.layers.0.self_attn", layers, layers,
                            None)
    assert attn.q_norm == "qn"
    assert attn.k_norm == "kn"
    assert attn.has_qk_norm


def test_detect_attention_rope_from_config():
    """detect_attention detects RoPE from config.rope_theta."""
    from patterns.attention import detect_attention

    @dataclass
    class FakeConfig:
        num_attention_heads: int = 4
        head_dim: int = 16
        rope_theta: float = 10000.0

    layers = [
        _layer("q", LAYER_FC, "model.layers.0.self_attn.q_proj", unit=64),
        _layer("k", LAYER_FC, "model.layers.0.self_attn.k_proj", unit=64),
        _layer("v", LAYER_FC, "model.layers.0.self_attn.v_proj", unit=64),
        _layer("o", LAYER_FC, "model.layers.0.self_attn.o_proj", unit=64),
    ]
    attn = detect_attention(0, "model.layers.0.self_attn", layers, layers,
                            FakeConfig())
    assert attn.has_rope


def test_detect_attention_mha():
    """detect_attention classifies MHA when num_heads == num_kv_heads."""
    from patterns.attention import detect_attention

    @dataclass
    class FakeConfig:
        num_attention_heads: int = 4
        head_dim: int = 16
        rope_theta: float = 0.0

    layers = [
        _layer("q", LAYER_FC, "model.layers.0.self_attn.q_proj", unit=64),
        _layer("k", LAYER_FC, "model.layers.0.self_attn.k_proj", unit=64),
        _layer("v", LAYER_FC, "model.layers.0.self_attn.v_proj", unit=64),
        _layer("o", LAYER_FC, "model.layers.0.self_attn.o_proj", unit=64),
    ]
    attn = detect_attention(0, "model.layers.0.self_attn", layers, layers,
                            FakeConfig())
    assert attn.attention_type == "mha"
    assert attn.num_kv_heads == 4


# ============================================================================
# Test ffn.py
# ============================================================================

def test_detect_ffn_swiglu():
    """detect_ffn detects SwiGLU pattern."""
    from patterns.ffn import detect_ffn
    layers = [
        _layer("gate", LAYER_FC, "model.layers.0.mlp.gate_proj", unit=128),
        _layer("up", LAYER_FC, "model.layers.0.mlp.up_proj", unit=128),
        _layer("down", LAYER_FC, "model.layers.0.mlp.down_proj", unit=64),
        _layer("act", LAYER_ACTIVATION, "model.layers.0.mlp",
               activation="silu"),
        _layer("mul", LAYER_MULTIPLY, "model.layers.0.mlp"),
    ]
    by_name = {l.name: l for l in layers}
    ffn = detect_ffn(0, "model.layers.0.mlp", layers, by_name)
    assert ffn.ffn_type == "swiglu"
    assert ffn.gate_proj == "gate"
    assert ffn.up_proj == "up"
    assert ffn.down_proj == "down"
    assert ffn.intermediate_size == 128


def test_detect_ffn_gelu():
    """detect_ffn detects GELU FFN pattern."""
    from patterns.ffn import detect_ffn
    layers = [
        _layer("up", LAYER_FC, "model.layers.0.mlp.intermediate.dense",
               unit=128),
        _layer("act", LAYER_ACTIVATION, "model.layers.0.mlp",
               activation="gelu"),
        _layer("down", LAYER_FC, "model.layers.0.mlp.output.dense", unit=64),
    ]
    by_name = {l.name: l for l in layers}
    ffn = detect_ffn(0, "model.layers.0.mlp", layers, by_name)
    assert ffn.ffn_type == "gelu_ffn"
    assert ffn.up_proj == "up"
    assert ffn.down_proj == "down"


def test_detect_ffn_geglu():
    """detect_ffn detects GeGLU pattern (gated with gelu activation)."""
    from patterns.ffn import detect_ffn
    layers = [
        _layer("gate", LAYER_FC, "model.layers.0.mlp.gate_proj", unit=128),
        _layer("up", LAYER_FC, "model.layers.0.mlp.up_proj", unit=128),
        _layer("down", LAYER_FC, "model.layers.0.mlp.down_proj", unit=64),
        _layer("act", LAYER_ACTIVATION, "model.layers.0.mlp",
               activation="gelu"),
        _layer("mul", LAYER_MULTIPLY, "model.layers.0.mlp"),
    ]
    by_name = {l.name: l for l in layers}
    ffn = detect_ffn(0, "model.layers.0.mlp", layers, by_name)
    assert ffn.ffn_type == "geglu"


# ============================================================================
# Test block.py
# ============================================================================

def test_detect_norms_and_residuals():
    """detect_norms_and_residuals finds pre-norm and residuals."""
    from patterns.block import detect_norms_and_residuals
    from patterns.data_types import TransformerBlockPattern, AttentionPattern

    block = TransformerBlockPattern(block_idx=0)
    block.attention = AttentionPattern(block_idx=0, q_proj="q", k_proj="k")

    layers = [
        _layer("norm1", LAYER_RMS_NORM,
               "model.layers.0.input_layernorm"),
        _layer("q", LAYER_FC, "model.layers.0.self_attn.q_proj"),
        _layer("k", LAYER_FC, "model.layers.0.self_attn.k_proj"),
        _layer("res1", LAYER_ADDITION, "model.layers.0"),
        _layer("norm2", LAYER_RMS_NORM,
               "model.layers.0.post_attention_layernorm"),
        _layer("up", LAYER_FC, "model.layers.0.mlp.up_proj"),
        _layer("res2", LAYER_ADDITION, "model.layers.0"),
    ]
    idx_by_name = {l.name: i for i, l in enumerate(layers)}

    detect_norms_and_residuals(block, "model.layers.0", layers, idx_by_name)

    assert block.pre_attn_norm == "norm1"
    assert block.pre_ffn_norm == "norm2"
    assert block.attn_residual == "res1"
    assert block.ffn_residual == "res2"
    assert block.norm_type == "pre_norm"


def test_detect_norms_post_norm():
    """detect_norms_and_residuals detects post-norm architecture."""
    from patterns.block import detect_norms_and_residuals
    from patterns.data_types import TransformerBlockPattern, AttentionPattern

    block = TransformerBlockPattern(block_idx=0)
    block.attention = AttentionPattern(block_idx=0, q_proj="q")

    # Post-norm: norm comes AFTER the projection
    layers = [
        _layer("q", LAYER_FC, "model.layers.0.self_attn.q_proj"),
        _layer("norm1", LAYER_RMS_NORM,
               "model.layers.0.input_layernorm"),
    ]
    idx_by_name = {l.name: i for i, l in enumerate(layers)}

    detect_norms_and_residuals(block, "model.layers.0", layers, idx_by_name)
    assert block.norm_type == "post_norm"


# ============================================================================
# Test config.py
# ============================================================================

def test_extract_config_metadata():
    """extract_config_metadata extracts model type and sizes."""
    from patterns.config import extract_config_metadata
    from patterns.data_types import ModelStructure

    @dataclass
    class FakeConfig:
        model_type: str = "llama"
        hidden_size: int = 4096
        num_hidden_layers: int = 32
        num_attention_heads: int = 32
        num_key_value_heads: int = 8
        intermediate_size: int = 11008
        vocab_size: int = 32000
        rope_theta: float = 10000.0
        tie_word_embeddings: bool = False

    structure = ModelStructure()
    extract_config_metadata(structure, FakeConfig())

    assert structure.model_type == "llama"
    assert structure.hidden_size == 4096
    assert structure.num_heads == 32
    assert structure.num_kv_heads == 8
    assert structure.intermediate_size == 11008
    assert structure.vocab_size == 32000
    assert structure.rope_theta == 10000.0
    assert structure.tie_word_embeddings is False


def test_extract_config_metadata_none():
    """extract_config_metadata handles None config gracefully."""
    from patterns.config import extract_config_metadata
    from patterns.data_types import ModelStructure

    structure = ModelStructure()
    extract_config_metadata(structure, None)
    assert structure.model_type == ""


def test_detect_embedding_and_head():
    """detect_embedding_and_head finds embedding and LM head."""
    from patterns.config import detect_embedding_and_head
    from patterns.data_types import ModelStructure

    layers = [
        _layer("emb", LAYER_EMBEDDING, "model.embed_tokens"),
        _layer("q", LAYER_FC, "model.layers.0.self_attn.q_proj"),
        _layer("lm_head", LAYER_FC, "lm_head"),
    ]

    structure = ModelStructure()
    detect_embedding_and_head(structure, layers)

    assert structure.embedding == "emb"
    assert structure.lm_head == "lm_head"


# ============================================================================
# Test data_types.py
# ============================================================================

def test_dataclass_defaults():
    """Pattern dataclasses initialize with correct defaults."""
    from patterns.data_types import (
        AttentionPattern, FFNPattern, TransformerBlockPattern, ModelStructure,
    )
    attn = AttentionPattern(block_idx=0)
    assert attn.q_proj == ""
    assert attn.attention_type == "mha"
    assert attn.layer_names == []

    ffn = FFNPattern(block_idx=0)
    assert ffn.ffn_type == "standard"
    assert ffn.gate_proj == ""

    block = TransformerBlockPattern(block_idx=0)
    assert block.attention is None
    assert block.ffn is None
    assert block.norm_type == "pre_norm"

    structure = ModelStructure()
    assert structure.blocks == []
    assert structure.arch_type == ""


# ============================================================================
# Test backward compatibility (pattern_detector.py facade)
# ============================================================================

def test_facade_imports():
    """pattern_detector.py re-exports all public symbols."""
    from pattern_detector import (
        PatternDetector, detect_patterns,
        ModelStructure, TransformerBlockPattern,
        AttentionPattern, FFNPattern,
    )
    # Verify they are the same objects as from the patterns package
    from patterns import (
        PatternDetector as P2, detect_patterns as dp2,
        ModelStructure as M2, TransformerBlockPattern as T2,
        AttentionPattern as A2, FFNPattern as F2,
    )
    assert PatternDetector is P2
    assert detect_patterns is dp2
    assert ModelStructure is M2
    assert TransformerBlockPattern is T2
    assert AttentionPattern is A2
    assert FFNPattern is F2


def test_facade_print_block():
    """pattern_detector._print_block is accessible."""
    from pattern_detector import _print_block
    from patterns import print_block
    assert _print_block is print_block
