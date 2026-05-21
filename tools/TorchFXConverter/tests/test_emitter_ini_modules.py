"""Tests for the emitter_ini/ sub-package and emitter interface standardization.

Tests:
  - emitter_ini/helpers.py: norm type detection, property formatting
  - emitter_ini/flat.py: flat mode emission
  - emitter_ini/structured.py: block-level emission helpers
  - BaseEmitter inheritance: all emitters inherit from BaseEmitter
  - Backward compatibility: existing import paths still work
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nntrainer_layers import NNTrainerLayerDef, LAYER_FC, LAYER_EMBEDDING


# ============================================================================
# Helper
# ============================================================================

def _layer(name, layer_type, hf_module_name="", **props):
    l = NNTrainerLayerDef(layer_type=layer_type, name=name)
    l.hf_module_name = hf_module_name
    l.properties = dict(props)
    l.input_layers = []
    return l


# ============================================================================
# Test helpers.py
# ============================================================================

def test_norm_type_for_model_bert():
    from emitter_ini.helpers import norm_type_for_model
    assert norm_type_for_model("bert") == "layer_normalization"
    assert norm_type_for_model("roberta") == "layer_normalization"
    assert norm_type_for_model("distilbert") == "layer_normalization"
    assert norm_type_for_model("albert") == "layer_normalization"


def test_norm_type_for_model_llama():
    from emitter_ini.helpers import norm_type_for_model
    assert norm_type_for_model("llama") == "rms_norm"
    assert norm_type_for_model("qwen3") == "rms_norm"
    assert norm_type_for_model("gemma") == "rms_norm"


def test_format_property_bool():
    from emitter_ini.helpers import format_property
    assert format_property("disable_bias", True) == "disable_bias = true"
    assert format_property("disable_bias", False) == "disable_bias = false"


def test_format_property_list():
    from emitter_ini.helpers import format_property
    assert format_property("shape", [1, 2, 3]) == "shape = 1,2,3"


def test_format_property_scalar():
    from emitter_ini.helpers import format_property
    assert format_property("unit", 128) == "unit = 128"
    assert format_property("name", "test") == "name = test"


# ============================================================================
# Test flat.py
# ============================================================================

def test_flat_emission():
    from emitter_ini.flat import emit_flat
    layers = [
        _layer("emb", "embedding_layer", unit=100),
        _layer("fc1", "fully_connected", unit=64),
    ]
    layers[0].input_layers = []
    layers[1].input_layers = ["emb"]

    output = emit_flat(layers, batch_size=2)
    assert "batch_size = 2" in output
    assert "[emb]" in output
    assert "[fc1]" in output
    assert "input_layers = emb" in output
    assert "flat mode" in output


# ============================================================================
# Test structured.py block helpers
# ============================================================================

def test_emit_attention_layers():
    from emitter_ini.structured import _emit_attention_layers
    from patterns.data_types import (
        TransformerBlockPattern, AttentionPattern, ModelStructure,
    )

    s = ModelStructure()
    s.num_heads = 4
    s.num_kv_heads = 2
    s.head_dim = 16
    s.hidden_size = 64
    s.norm_eps = 1e-6
    s.rope_theta = 10000.0

    b0 = TransformerBlockPattern(block_idx=0)
    b0.attention = AttentionPattern(
        block_idx=0, has_qk_norm=True, has_rope=True)

    lines = []
    result = _emit_attention_layers(lines, b0, s, "layer0",
                                    "layer0_attention_norm", "rms_norm")
    output = "\n".join(lines)

    assert "[layer0_wq]" in output
    assert "[layer0_wk]" in output
    assert "[layer0_wv]" in output
    assert "[layer0_q_norm]" in output
    assert "[layer0_k_norm]" in output
    assert "[layer0_attention]" in output
    assert "num_heads = 4" in output
    assert "num_heads_kv = 2" in output
    assert "rope_theta = 10000" in output
    assert "[layer0_attention_out]" in output
    assert result == "layer0_attention_out"


def test_emit_ffn_layers_swiglu():
    from emitter_ini.structured import _emit_ffn_layers
    from patterns.data_types import (
        TransformerBlockPattern, FFNPattern, ModelStructure,
    )

    s = ModelStructure()
    s.intermediate_size = 128
    s.hidden_size = 64

    b0 = TransformerBlockPattern(block_idx=0)
    b0.ffn = FFNPattern(block_idx=0, ffn_type="swiglu")

    lines = []
    _emit_ffn_layers(lines, b0, s, "layer0", "layer0_ffn_norm")
    output = "\n".join(lines)

    assert "[layer0_ffn_up]" in output
    assert "[layer0_ffn_gate]" in output
    assert "[layer0_ffn_swiglu]" in output
    assert "Type = swiglu" in output
    assert "[layer0_ffn_down]" in output
    assert "unit = 128" in output


def test_emit_ffn_layers_gelu():
    from emitter_ini.structured import _emit_ffn_layers
    from patterns.data_types import (
        TransformerBlockPattern, FFNPattern, ModelStructure,
    )

    s = ModelStructure()
    s.intermediate_size = 128
    s.hidden_size = 64

    b0 = TransformerBlockPattern(block_idx=0)
    b0.ffn = FFNPattern(block_idx=0, ffn_type="gelu_ffn")

    lines = []
    _emit_ffn_layers(lines, b0, s, "layer0", "layer0_ffn_norm")
    output = "\n".join(lines)

    assert "[layer0_ffn_fc1]" in output
    assert "[layer0_ffn_act]" in output
    assert "Activation = gelu" in output
    assert "[layer0_ffn_down]" in output


# ============================================================================
# Test BaseEmitter inheritance
# ============================================================================

def test_ini_emitter_inherits_base():
    from emitter_ini import IniEmitter
    from emitter_base import BaseEmitter
    assert issubclass(IniEmitter, BaseEmitter)


def test_json_emitter_inherits_base():
    from emitter_json import JsonEmitter
    from emitter_base import BaseEmitter
    assert issubclass(JsonEmitter, BaseEmitter)


def test_cpp_emitter_inherits_base():
    from emitter_cpp import CppEmitter
    from emitter_base import BaseEmitter
    assert issubclass(CppEmitter, BaseEmitter)


def test_base_emitter_common_init():
    """All emitters get _by_name dict from BaseEmitter.__init__."""
    from emitter_ini import IniEmitter
    from emitter_json import JsonEmitter
    from patterns.data_types import ModelStructure

    layers = [_layer("fc1", LAYER_FC, unit=64)]
    structure = ModelStructure()

    ini = IniEmitter(layers, structure)
    assert "fc1" in ini._by_name

    json_e = JsonEmitter(layers, structure)
    assert "fc1" in json_e._by_name


# ============================================================================
# Test backward compatibility
# ============================================================================

def test_backward_compat_imports():
    """Existing imports from emitter_ini still work."""
    from emitter_ini import emit_ini, IniEmitter
    assert callable(emit_ini)
    assert callable(IniEmitter)


def test_backward_compat_json_imports():
    """Existing imports from emitter_json still work."""
    from emitter_json import emit_json, emit_json_string, JsonEmitter
    assert callable(emit_json)
    assert callable(emit_json_string)
    assert callable(JsonEmitter)


def test_backward_compat_ini_output():
    """IniEmitter produces valid INI in flat mode."""
    from emitter_ini import IniEmitter
    from patterns.data_types import ModelStructure

    layers = [_layer("fc1", LAYER_FC, unit=64)]
    layers[0].input_layers = []
    structure = ModelStructure()

    emitter = IniEmitter(layers, structure)
    output = emitter.emit(mode="flat")
    assert "[Model]" in output
    assert "[fc1]" in output
    assert "unit = 64" in output
