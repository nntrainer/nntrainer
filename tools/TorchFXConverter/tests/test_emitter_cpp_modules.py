"""Tests for the refactored emitter_cpp sub-modules.

Validates that each sub-module works correctly in isolation and that
backward compatibility with the original emitter_cpp API is preserved.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from decomposer import AdaptiveConverter


# ============================================================================
# Fixtures
# ============================================================================

def _convert_qwen3():
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
    result = converter.convert(
        {"input_ids": torch.randint(0, config.vocab_size, (1, 8))})
    return result.layers, result.model_structure, config, model


# ============================================================================
# Test: helpers module
# ============================================================================

def test_helpers_cpp_layer():
    """Test _cpp_layer generates correct createLayer() calls."""
    from emitter_cpp.helpers import _cpp_layer

    lines = _cpp_layer("fully_connected", [
        'withKey("name", "layer0_wq")',
        'withKey("unit", 64)',
    ])
    joined = "\n".join(lines)
    assert 'createLayer("fully_connected"' in joined
    assert 'withKey("name", "layer0_wq")' in joined
    assert 'withKey("unit", 64)' in joined
    assert joined.strip().endswith("}));")


def test_helpers_class_name():
    """Test class name generation for various model types."""
    from emitter_cpp.helpers import _class_name

    assert _class_name("qwen3", "decoder_only") == "Qwen3CausalLM"
    assert _class_name("bert", "encoder_only") == "BertModel"
    assert _class_name("mt5", "encoder_decoder") == "MT5Model"
    assert _class_name("qwen2", "embedding") == "Qwen2EmbeddingModel"
    assert _class_name("gemma", "embedding") == "GemmaEmbeddingModel"
    # Unknown type falls back to capitalized + Model
    assert _class_name("custom_model", "decoder_only") == "Custom_modelModel"


def test_helpers_file_basename():
    """Test file basename generation."""
    from emitter_cpp.helpers import _file_basename

    assert _file_basename("Qwen3CausalLM") == "qwen3_causal_lm"
    assert _file_basename("BertModel") == "bert_model"
    assert _file_basename("MT5Model") == "mt5_model"
    assert _file_basename("GemmaEmbeddingModel") == "gemma_embedding_model"


def test_helpers_sanitize_model_name():
    """Test model name sanitization."""
    from emitter_cpp.helpers import _sanitize_model_name

    assert _sanitize_model_name("KaLM-embedding-v2.5") == "kalm_embedding_v2_5"
    assert _sanitize_model_name("Qwen/Qwen3-0.6B") == "qwen3_0_6b"
    assert _sanitize_model_name("./gliner2-multi-v1/") == "gliner2_multi_v1"


def test_helpers_get_output_filenames():
    """Test output filename generation."""
    from emitter_cpp.helpers import get_output_filenames

    fnames = get_output_filenames("qwen3", "decoder_only")
    assert fnames["header"] == "qwen3_causal_lm.h"
    assert fnames["source"] == "qwen3_causal_lm.cpp"
    assert fnames["ini"] == "qwen3_causal_lm.ini"
    assert fnames["json"] == "qwen3_causal_lm.json"

    fnames_named = get_output_filenames("qwen3", "decoder_only",
                                        model_name="MyModel-v1")
    assert fnames_named["header"] == "mymodel_v1.h"


def test_helpers_get_norm_type():
    """Test norm type selection."""
    from emitter_cpp.helpers import get_norm_type

    assert get_norm_type("qwen3") == "rms_norm"
    assert get_norm_type("llama") == "rms_norm"
    assert get_norm_type("bert") == "layer_normalization"
    assert get_norm_type("roberta") == "layer_normalization"


# ============================================================================
# Test: header module
# ============================================================================

def test_header_flat():
    """Test flat header generation."""
    from emitter_cpp.header import emit_flat_header
    from pattern_detector import ModelStructure

    s = ModelStructure()
    s.model_type = "custom"
    s.arch_type = "encoder_only"

    header = emit_flat_header(s)
    assert "class CustomModel" in header
    assert "#ifndef" in header
    assert "#define" in header
    assert "virtual void constructModel();" in header
    assert "ModelHandle model;" in header


def test_header_structured():
    """Test structured header generation with real model."""
    layers, structure, _, _ = _convert_qwen3()

    from emitter_cpp.header import emit_structured_header

    blocks_info = {
        "attn_block": structure.blocks[0] if structure.blocks else None,
        "is_hybrid": False,
        "op_types": ["attention"],
    }
    header = emit_structured_header(structure, blocks_info)

    assert "class Qwen3CausalLM" in header
    assert "createTransformerDecoderBlock" in header
    assert "createAttention" in header
    assert "createMlp" in header
    assert "registerCustomLayers" in header
    assert "NUM_VOCAB = 151936" in header
    assert "DIM = 64" in header
    assert "NUM_HEADS = 4" in header
    assert "reshaped_rms_norm.h" in header  # QK norm include


# ============================================================================
# Test: source_custom module
# ============================================================================

def test_custom_layer_collection():
    """Test custom layer class collection."""
    layers, structure, _, _ = _convert_qwen3()

    from emitter_cpp.source_custom import collect_custom_layer_classes

    attn_block = None
    for b in structure.blocks:
        if b.attention is not None:
            attn_block = b
            break

    classes = collect_custom_layer_classes(structure, "rms_norm", attn_block)
    assert "EmbeddingLayer" in classes
    assert "TieWordEmbedding" in classes
    assert "RMSNormLayer" in classes
    assert "MHACoreLayer" in classes
    assert "ReshapedRMSNormLayer" in classes  # Qwen3 has QK norm
    assert "SwiGLULayer" in classes


def test_register_custom_layers_output():
    """Test registerCustomLayers() C++ code generation."""
    from emitter_cpp.source_custom import emit_register_custom_layers

    code = emit_register_custom_layers("TestModel", ["MHACoreLayer", "RMSNormLayer"])
    assert "TestModel::registerCustomLayers()" in code
    assert "nntrainer::createLayer<causallm::MHACoreLayer>" in code
    assert "nntrainer::createLayer<causallm::RMSNormLayer>" in code
    assert "try {" in code


# ============================================================================
# Test: backward compatibility
# ============================================================================

def test_backward_compat_imports():
    """Test that old import patterns still work."""
    # These are the imports used by converter.py and test files
    from emitter_cpp import emit_cpp, emit_cpp_header, emit_cpp_source
    from emitter_cpp import get_output_filenames
    from emitter_cpp import CppEmitter

    assert callable(emit_cpp)
    assert callable(emit_cpp_header)
    assert callable(emit_cpp_source)
    assert callable(get_output_filenames)
    assert callable(CppEmitter)


def test_backward_compat_output():
    """Test that CppEmitter produces the same key content as before."""
    layers, structure, _, _ = _convert_qwen3()

    from emitter_cpp import CppEmitter

    emitter = CppEmitter(layers, structure)
    cpp_code = emitter.emit()

    # Same assertions as test_cpp_emitter_qwen3 in test_emitters.py
    assert "createLayer" in cpp_code
    assert "mha_core" in cpp_code
    assert "fully_connected" in cpp_code
    assert "rms_norm" in cpp_code
    assert "swiglu" in cpp_code
    assert ".add(" in cpp_code
    assert "tie_word_embeddings" in cpp_code
    assert "reshaped_rms_norm" in cpp_code
    assert "class Qwen3CausalLM" in cpp_code
    assert "Qwen3CausalLM::createAttention" in cpp_code
    assert "Qwen3CausalLM::createMlp" in cpp_code
    assert "Qwen3CausalLM::createTransformerDecoderBlock" in cpp_code
    assert "Qwen3CausalLM::constructModel" in cpp_code


def test_backward_compat_internal_methods():
    """Test that internal methods still work for any external callers."""
    layers, structure, _, _ = _convert_qwen3()

    from emitter_cpp import CppEmitter

    emitter = CppEmitter(layers, structure)
    assert emitter._get_norm_type() == "rms_norm"
    assert emitter._file_base() == "qwen3_causal_lm"
    assert len(emitter._collect_custom_layer_classes()) > 0
    assert emitter._CUSTOM_LAYER_CLASS["mha_core"] == "MHACoreLayer"


# ============================================================================
# Test: emitter_base interface
# ============================================================================

def test_base_emitter_interface():
    """Test that BaseEmitter cannot be instantiated directly."""
    from emitter_base import BaseEmitter
    import pytest

    with pytest.raises(TypeError):
        BaseEmitter([], None)


if __name__ == "__main__":
    test_helpers_cpp_layer()
    test_helpers_class_name()
    test_helpers_file_basename()
    test_helpers_sanitize_model_name()
    test_helpers_get_output_filenames()
    test_helpers_get_norm_type()
    test_header_flat()
    test_header_structured()
    test_custom_layer_collection()
    test_register_custom_layers_output()
    test_backward_compat_imports()
    test_backward_compat_output()
    test_backward_compat_internal_methods()
    test_base_emitter_interface()
    print("\nALL EMITTER CPP MODULE TESTS PASSED!")
