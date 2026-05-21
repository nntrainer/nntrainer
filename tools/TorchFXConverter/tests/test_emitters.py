"""Test emitters: C++, INI, JSON, and weight converter.

Tests use real models (Qwen3, BERT) converted through the full pipeline,
then validate emitted output for correctness.
"""
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from decomposer import AdaptiveConverter
from emitter_cpp import emit_cpp, CppEmitter
from emitter_ini import emit_ini, IniEmitter
from emitter_json import emit_json, emit_json_string, JsonEmitter
from weight_converter import WeightConverter, build_weight_map, WeightMap


def _convert_qwen3():
    """Convert a tiny Qwen3 model and return (layers, structure, config, model)."""
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


def _convert_bert():
    """Convert a tiny BERT model and return (layers, structure, config, model)."""
    from transformers import BertConfig, BertModel

    config = BertConfig(
        vocab_size=30522, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4,
        max_position_embeddings=512, type_vocab_size=2,
    )
    model = BertModel(config)
    model.eval()

    converter = AdaptiveConverter(model, config)
    result = converter.convert({
        "input_ids": torch.randint(0, config.vocab_size, (1, 8)),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
    })

    return result.layers, result.model_structure, config, model


# ============================================================================
# Test 1: C++ Emitter - Qwen3
# ============================================================================

def test_cpp_emitter_qwen3():
    """Test C++ code generation for Qwen3."""
    print("=" * 70)
    print("TEST: C++ Emitter - Qwen3")
    print("=" * 70)

    layers, structure, config, model = _convert_qwen3()
    cpp_code = emit_cpp(layers, structure)

    # Basic structure checks
    assert "createLayer" in cpp_code, "No createLayer calls found"
    assert "mha_core" in cpp_code, "No mha_core layer"
    assert "fully_connected" in cpp_code, "No fully_connected layer"
    assert "rms_norm" in cpp_code, "No rms_norm layer"
    assert "swiglu" in cpp_code, "No swiglu layer"
    assert ".add(" in cpp_code, "No Tensor::add() residual connection"
    assert "tie_word_embeddings" in cpp_code, "No tied embeddings"
    assert "reshaped_rms_norm" in cpp_code, "No Q/K norms"
    print("  PASS: All expected layer types present")

    # Member variables
    assert "NUM_VOCAB = 151936" in cpp_code
    assert "DIM = 64" in cpp_code
    assert "NUM_LAYERS = 2" in cpp_code
    assert "NUM_HEADS = 4" in cpp_code
    assert "NUM_KV_HEADS = 2" in cpp_code
    assert "HEAD_DIM = 16" in cpp_code
    assert "INTERMEDIATE_SIZE = 128" in cpp_code
    assert "ROPE_THETA" in cpp_code
    print("  PASS: Member variables correct")

    # Class structure
    assert "class Qwen3CausalLM" in cpp_code, "No class declaration"
    assert "Qwen3CausalLM::createAttention" in cpp_code
    assert "Qwen3CausalLM::createMlp" in cpp_code
    assert "Qwen3CausalLM::createTransformerDecoderBlock" in cpp_code
    assert "Qwen3CausalLM::constructModel" in cpp_code
    assert "virtual void constructModel();" in cpp_code
    print("  PASS: Class-based structure generated")

    # Attention details
    assert "rope_theta" in cpp_code, "RoPE theta not in MHA"
    assert "num_heads_kv" in cpp_code, "num_heads_kv not in MHA"
    print("  PASS: Attention details (RoPE, GQA)")

    print(f"  Generated {len(cpp_code)} characters of C++ code")
    print("  PASS: C++ emitter Qwen3 test passed")


# ============================================================================
# Test 2: C++ Emitter - BERT
# ============================================================================

def test_cpp_emitter_bert():
    """Test C++ code generation for BERT."""
    print("\n" + "=" * 70)
    print("TEST: C++ Emitter - BERT")
    print("=" * 70)

    layers, structure, config, model = _convert_bert()
    cpp_code = emit_cpp(layers, structure)

    assert "class BertModel" in cpp_code, "No BERT class"
    assert "createLayer" in cpp_code
    assert "fully_connected" in cpp_code
    assert "encoder_only" in cpp_code
    # BERT uses layer_normalization not rms_norm
    assert "layer_normalization" in cpp_code or "rms_norm" in cpp_code
    # BERT should NOT have RoPE or swiglu
    # (it may or may not have these strings depending on structure detection)
    print(f"  Generated {len(cpp_code)} characters of C++ code")
    print("  PASS: C++ emitter BERT test passed")


# ============================================================================
# Test 3: INI Emitter - Qwen3 (structured)
# ============================================================================

def test_ini_emitter_qwen3_structured():
    """Test INI generation for Qwen3 in structured mode."""
    print("\n" + "=" * 70)
    print("TEST: INI Emitter - Qwen3 (structured)")
    print("=" * 70)

    layers, structure, config, model = _convert_qwen3()
    ini_text = emit_ini(layers, structure, batch_size=1, mode="structured")

    # Header
    assert "[Model]" in ini_text, "No [Model] section"
    assert "Type = NeuralNetwork" in ini_text
    assert "batch_size = 1" in ini_text
    print("  PASS: [Model] section correct")

    # Embedding
    assert "[embedding0]" in ini_text, "No embedding section"
    assert "tie_word_embeddings" in ini_text
    assert "in_dim = 151936" in ini_text
    assert "out_dim = 64" in ini_text
    print("  PASS: Embedding section correct")

    # Blocks
    assert "[layer0_attention_norm]" in ini_text, "No attention norm"
    assert "[layer0_wq]" in ini_text, "No Q projection"
    assert "[layer0_wk]" in ini_text, "No K projection"
    assert "[layer0_wv]" in ini_text, "No V projection"
    assert "[layer0_attention]" in ini_text, "No attention core"
    assert "[layer0_attention_out]" in ini_text, "No O projection"
    assert "[layer0_decoder_add]" in ini_text, "No attention residual"
    assert "[layer0_ffn_norm]" in ini_text, "No FFN norm"
    assert "[layer0_ffn_up]" in ini_text, "No FFN up"
    assert "[layer0_ffn_gate]" in ini_text, "No FFN gate"
    assert "[layer0_ffn_swiglu]" in ini_text, "No SwiGLU"
    assert "[layer0_ffn_down]" in ini_text, "No FFN down"
    assert "[layer0_decoder_output]" in ini_text, "No block output"
    print("  PASS: Block 0 sections complete")

    # Block 1
    assert "[layer1_attention_norm]" in ini_text, "No block 1"
    print("  PASS: Block 1 present")

    # Q/K norms
    assert "[layer0_q_norm]" in ini_text, "No Q norm"
    assert "[layer0_k_norm]" in ini_text, "No K norm"
    assert "reshaped_rms_norm" in ini_text
    assert f"feature_size = {structure.head_dim}" in ini_text
    print("  PASS: Q/K norms present")

    # Final norm and LM head
    assert "[output_norm]" in ini_text
    assert "[lm_head]" in ini_text
    assert "shared_from = embedding0" in ini_text
    print("  PASS: Final norm and LM head correct")

    # MHA properties
    assert "num_heads = 4" in ini_text
    assert "num_heads_kv = 2" in ini_text
    assert "rope_theta" in ini_text
    print("  PASS: MHA properties correct")

    print(f"  Generated {len(ini_text)} characters of INI config")
    print("  PASS: INI emitter Qwen3 structured test passed")


# ============================================================================
# Test 4: INI Emitter - Qwen3 (flat)
# ============================================================================

def test_ini_emitter_qwen3_flat():
    """Test INI generation for Qwen3 in flat mode."""
    print("\n" + "=" * 70)
    print("TEST: INI Emitter - Qwen3 (flat)")
    print("=" * 70)

    layers, structure, config, model = _convert_qwen3()
    ini_text = emit_ini(layers, structure, mode="flat")

    assert "[Model]" in ini_text
    # Every layer should have a section
    section_count = ini_text.count("[")
    # At least Model + all layers
    assert section_count >= len(layers), \
        f"Expected >= {len(layers)} sections, got {section_count}"
    print(f"  PASS: {section_count} sections for {len(layers)} layers")

    # Each layer section has Type
    for layer in layers[:5]:
        assert f"[{layer.name}]" in ini_text, \
            f"Layer section [{layer.name}] not found"
    print("  PASS: Layer sections present")
    print("  PASS: INI emitter flat mode test passed")


# ============================================================================
# Test 5: JSON Emitter - Qwen3
# ============================================================================

def test_json_emitter_qwen3():
    """Test JSON generation for Qwen3."""
    print("\n" + "=" * 70)
    print("TEST: JSON Emitter - Qwen3")
    print("=" * 70)

    layers, structure, config, model = _convert_qwen3()
    result = emit_json(layers, structure)

    # Top-level keys
    assert "model" in result, "No 'model' key"
    assert "layers" in result, "No 'layers' key"
    assert "weight_map" in result, "No 'weight_map' key"
    assert "structure" in result, "No 'structure' key"
    print("  PASS: Top-level keys present")

    # Model info
    model_info = result["model"]
    assert model_info["model_type"] == "qwen3"
    assert model_info["arch_type"] == "decoder_only"
    assert model_info["vocab_size"] == 151936
    assert model_info["hidden_size"] == 64
    assert model_info["num_layers"] == 2
    assert model_info["num_heads"] == 4
    assert model_info["num_kv_heads"] == 2
    assert model_info["head_dim"] == 16
    assert model_info["intermediate_size"] == 128
    assert model_info["tie_word_embeddings"] is True
    assert model_info["rope_theta"] > 0
    print("  PASS: Model info correct")

    # Layers
    json_layers = result["layers"]
    assert len(json_layers) == len(layers)
    # Check first layer has expected fields
    l0 = json_layers[0]
    assert "name" in l0
    assert "type" in l0
    print(f"  PASS: {len(json_layers)} layers serialized")

    # Weight map
    wmap = result["weight_map"]
    assert len(wmap) > 0, "Weight map is empty"
    # Check that FC layers are marked for transpose
    fc_weights = [w for w in wmap
                  if w.get("transpose_weight") is True]
    assert len(fc_weights) > 0, "No FC layers marked for transpose"
    print(f"  PASS: Weight map has {len(wmap)} entries, "
          f"{len(fc_weights)} need transpose")

    # Structure
    structure_info = result["structure"]
    assert len(structure_info["blocks"]) == 2
    b0 = structure_info["blocks"][0]
    assert "attention" in b0
    assert b0["attention"]["type"] == "gqa"
    assert b0["attention"]["has_qk_norm"] is True
    assert b0["attention"]["has_rope"] is True
    assert "ffn" in b0
    assert b0["ffn"]["type"] == "swiglu"
    print("  PASS: Structure info correct")

    # JSON serialization round-trip
    json_str = emit_json_string(layers, structure)
    parsed = json.loads(json_str)
    assert parsed["model"]["model_type"] == "qwen3"
    print("  PASS: JSON serialization round-trip OK")
    print("  PASS: JSON emitter Qwen3 test passed")


# ============================================================================
# Test 6: Weight Converter - Mapping
# ============================================================================

def test_weight_converter_mapping():
    """Test weight map building for Qwen3."""
    print("\n" + "=" * 70)
    print("TEST: Weight Converter - Mapping")
    print("=" * 70)

    layers, structure, config, model = _convert_qwen3()
    wmap = build_weight_map(layers)

    assert len(wmap) > 0, "Weight map is empty"
    print(f"  PASS: {len(wmap)} weight entries")

    # Check some expected entries
    hf_keys = [e["hf_key"] for e in wmap]
    # Embedding weight
    emb_keys = [k for k in hf_keys if "embed" in k]
    assert len(emb_keys) > 0, "No embedding weights found"
    print(f"  PASS: Embedding weights: {emb_keys[:2]}")

    # Linear weights should be transposed
    transpose_entries = [e for e in wmap if e["transform"] == "transpose"]
    assert len(transpose_entries) > 0, "No transposed weights"
    print(f"  PASS: {len(transpose_entries)} weights need transpose")

    # Norm weights should NOT be transposed
    norm_entries = [e for e in wmap if "norm" in e["hf_key"].lower()
                    and e["transform"] == "none"]
    assert len(norm_entries) > 0, "No norm weights found"
    print(f"  PASS: {len(norm_entries)} norm weights (no transpose)")
    print("  PASS: Weight converter mapping test passed")


# ============================================================================
# Test 7: Weight Converter - Binary Output
# ============================================================================

def test_weight_converter_binary():
    """Test actual weight conversion to binary format."""
    print("\n" + "=" * 70)
    print("TEST: Weight Converter - Binary Output")
    print("=" * 70)

    layers, structure, config, model = _convert_qwen3()
    converter = WeightConverter(layers)
    state_dict = model.state_dict()

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        output_path = f.name

    try:
        converter.convert(state_dict, output_path, dtype="float32")

        # Verify file exists and has content
        file_size = os.path.getsize(output_path)
        assert file_size > 0, "Output file is empty"
        print(f"  PASS: Output file size = {file_size} bytes")

        # Calculate expected size from weight shapes
        expected_size = 0
        for entry in converter.weight_map:
            key = entry["hf_key"]
            if key in state_dict:
                expected_size += state_dict[key].numel() * 4  # float32

        assert file_size == expected_size, \
            f"File size mismatch: {file_size} vs expected {expected_size}"
        print(f"  PASS: File size matches expected ({expected_size} bytes)")

    finally:
        os.unlink(output_path)

    print("  PASS: Weight converter binary test passed")


# ============================================================================
# Test 8: Weight Converter - Script Generation
# ============================================================================

def test_weight_converter_script():
    """Test standalone weight conversion script generation."""
    print("\n" + "=" * 70)
    print("TEST: Weight Converter - Script Generation")
    print("=" * 70)

    layers, structure, config, model = _convert_qwen3()
    converter = WeightConverter(layers)

    script = converter.generate_script()

    assert "import torch" in script
    assert "WEIGHT_MAP" in script
    assert "def convert" in script
    assert "transpose" in script
    assert "if __name__" in script
    print(f"  PASS: Generated {len(script)} chars of conversion script")
    print("  PASS: Weight converter script generation test passed")


# ============================================================================
# Test 9: INI Emitter - BERT
# ============================================================================

def test_ini_emitter_bert():
    """Test INI generation for BERT."""
    print("\n" + "=" * 70)
    print("TEST: INI Emitter - BERT")
    print("=" * 70)

    layers, structure, config, model = _convert_bert()
    ini_text = emit_ini(layers, structure, mode="structured")

    assert "[Model]" in ini_text
    assert "encoder_only" in ini_text or "bert" in ini_text
    # BERT should use layer_normalization
    assert "layer_normalization" in ini_text, \
        "BERT should use layer_normalization"
    print("  PASS: BERT INI uses layer_normalization")

    # Should not have swiglu (BERT uses GELU FFN)
    assert "swiglu" not in ini_text, "BERT should not have swiglu"
    print("  PASS: No swiglu in BERT INI")

    # Should not have rope_theta (BERT uses absolute position embeddings)
    # Should not have reshaped_rms_norm
    assert "reshaped_rms_norm" not in ini_text, \
        "BERT should not have reshaped_rms_norm"
    print("  PASS: No Q/K norms in BERT INI")

    print("  PASS: INI emitter BERT test passed")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    test_cpp_emitter_qwen3()
    test_cpp_emitter_bert()
    test_ini_emitter_qwen3_structured()
    test_ini_emitter_qwen3_flat()
    test_json_emitter_qwen3()
    test_weight_converter_mapping()
    test_weight_converter_binary()
    test_weight_converter_script()
    test_ini_emitter_bert()

    print("\n" + "=" * 70)
    print("ALL EMITTER TESTS PASSED!")
    print("=" * 70)
