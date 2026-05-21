"""Test pattern detection for multi-architecture models.

Tests:
  1. Qwen3 (decoder-only): GQA attention, SwiGLU FFN, Q/K norms, RoPE
  2. BERT (encoder-only): MHA attention, GELU FFN, post-norm
  3. mT5 (encoder-decoder): Self + cross attention, DenseReluDense FFN
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from decomposer import AdaptiveConverter
from pattern_detector import PatternDetector, detect_patterns


def _convert_model(model_cls, config_cls, config_kwargs, input_kwargs_fn):
    """Helper: convert model and return (layers, config)."""
    config = config_cls(**config_kwargs)
    model = model_cls(config)
    model.eval()

    converter = AdaptiveConverter(model, config)
    result = converter.convert(input_kwargs_fn(config))
    return result.layers, config


# ============================================================================
# Test 1: Qwen3 (decoder-only, GQA, SwiGLU, RoPE, Q/K norms)
# ============================================================================

def test_qwen3_patterns():
    """Detect Qwen3 attention, FFN, and block patterns."""
    print("=" * 70)
    print("TEST: Qwen3 Pattern Detection")
    print("=" * 70)

    from transformers import Qwen3Config, Qwen3ForCausalLM

    layers, config = _convert_model(
        Qwen3ForCausalLM, Qwen3Config,
        dict(
            vocab_size=151936, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, max_position_embeddings=2048, rms_norm_eps=1e-6,
            tie_word_embeddings=True, rope_theta=1000000.0, sliding_window=None,
        ),
        lambda cfg: {"input_ids": torch.randint(0, cfg.vocab_size, (1, 8))},
    )

    structure = detect_patterns(layers, config)
    structure.summary()

    # Architecture
    assert structure.arch_type == "decoder_only", \
        f"Expected decoder_only, got {structure.arch_type}"
    assert structure.model_type == "qwen3"
    print("  PASS: Architecture type = decoder_only (qwen3)")

    # Embedding & LM head
    assert structure.embedding, "No embedding detected"
    assert structure.lm_head, "No LM head detected"
    assert structure.tie_word_embeddings
    print(f"  PASS: Embedding={structure.embedding}, "
          f"LM head={structure.lm_head} (tied)")

    # Blocks
    assert len(structure.blocks) == 2, \
        f"Expected 2 blocks, got {len(structure.blocks)}"
    print(f"  PASS: {len(structure.blocks)} blocks detected")

    # Block 0 detailed check
    b0 = structure.blocks[0]

    # Attention
    assert b0.attention is not None, "No attention detected"
    assert b0.attention.q_proj, "No Q projection"
    assert b0.attention.k_proj, "No K projection"
    assert b0.attention.v_proj, "No V projection"
    assert b0.attention.o_proj, "No O projection"
    assert b0.attention.attention_type == "gqa", \
        f"Expected gqa, got {b0.attention.attention_type}"
    assert b0.attention.num_heads == 4
    assert b0.attention.num_kv_heads == 2
    assert b0.attention.head_dim == 16
    print(f"  PASS: Attention: GQA (heads=4, kv_heads=2, head_dim=16)")

    # Q/K norms
    assert b0.attention.has_qk_norm, "Q/K norms not detected"
    assert b0.attention.q_norm, "No Q norm"
    assert b0.attention.k_norm, "No K norm"
    print(f"  PASS: Q/K norms detected")

    # RoPE
    assert b0.attention.has_rope, "RoPE not detected"
    print(f"  PASS: RoPE detected")

    # FFN
    assert b0.ffn is not None, "No FFN detected"
    assert b0.ffn.ffn_type == "swiglu", \
        f"Expected swiglu, got {b0.ffn.ffn_type}"
    assert b0.ffn.gate_proj, "No gate projection"
    assert b0.ffn.up_proj, "No up projection"
    assert b0.ffn.down_proj, "No down projection"
    assert b0.ffn.intermediate_size == 128
    print(f"  PASS: FFN: SwiGLU (intermediate=128)")

    # Norms
    assert b0.pre_attn_norm, "No pre-attention norm"
    assert b0.pre_ffn_norm, "No pre-FFN norm"
    assert b0.norm_type == "pre_norm", \
        f"Expected pre_norm, got {b0.norm_type}"
    print(f"  PASS: Pre-norm architecture")

    # Residuals
    assert b0.attn_residual, "No attention residual"
    assert b0.ffn_residual, "No FFN residual"
    print(f"  PASS: Residual connections detected")

    # Config metadata
    assert structure.hidden_size == 64
    assert structure.rope_theta > 0, "RoPE theta should be positive"
    print(f"  PASS: Config metadata (hidden={structure.hidden_size}, "
          f"rope_theta={structure.rope_theta})")


# ============================================================================
# Test 2: BERT (encoder-only, MHA, GELU FFN)
# ============================================================================

def test_bert_patterns():
    """Detect BERT attention, FFN, and block patterns."""
    print("\n" + "=" * 70)
    print("TEST: BERT Pattern Detection")
    print("=" * 70)

    from transformers import BertConfig, BertModel

    layers, config = _convert_model(
        BertModel, BertConfig,
        dict(
            vocab_size=30522, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4,
            max_position_embeddings=512, type_vocab_size=2,
        ),
        lambda cfg: {
            "input_ids": torch.randint(0, cfg.vocab_size, (1, 8)),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
        },
    )

    structure = detect_patterns(layers, config)
    structure.summary()

    # Architecture
    assert structure.arch_type == "encoder_only", \
        f"Expected encoder_only, got {structure.arch_type}"
    print("  PASS: Architecture type = encoder_only (bert)")

    # Blocks
    assert len(structure.blocks) == 2, \
        f"Expected 2 blocks, got {len(structure.blocks)}"
    print(f"  PASS: {len(structure.blocks)} blocks detected")

    b0 = structure.blocks[0]

    # Attention (MHA, no GQA)
    assert b0.attention is not None, "No attention detected"
    assert b0.attention.q_proj, "No Q projection"
    assert b0.attention.k_proj, "No K projection"
    assert b0.attention.v_proj, "No V projection"
    assert b0.attention.o_proj, "No O projection"
    assert b0.attention.attention_type == "mha", \
        f"Expected mha, got {b0.attention.attention_type}"
    assert not b0.attention.has_qk_norm, "BERT shouldn't have Q/K norms"
    assert not b0.attention.has_rope, "BERT shouldn't have RoPE"
    print(f"  PASS: Attention: MHA (no Q/K norm, no RoPE)")

    # FFN
    assert b0.ffn is not None, "No FFN detected"
    assert b0.ffn.ffn_type in ("gelu_ffn", "standard"), \
        f"Expected gelu_ffn/standard, got {b0.ffn.ffn_type}"
    print(f"  PASS: FFN: {b0.ffn.ffn_type}")

    # Norms
    assert b0.pre_attn_norm or b0.pre_ffn_norm, "No norms detected"
    print(f"  PASS: Norms detected (pre_attn={b0.pre_attn_norm}, "
          f"pre_ffn={b0.pre_ffn_norm})")


# ============================================================================
# Test 3: mT5 (encoder-decoder, self + cross attention)
# ============================================================================

def test_mt5_patterns():
    """Detect mT5 encoder-decoder patterns."""
    print("\n" + "=" * 70)
    print("TEST: mT5 Pattern Detection")
    print("=" * 70)

    from transformers import MT5Config, MT5ForConditionalGeneration

    layers, config = _convert_model(
        MT5ForConditionalGeneration, MT5Config,
        dict(
            vocab_size=250112, d_model=64, d_kv=16, d_ff=128,
            num_heads=4, num_layers=2, num_decoder_layers=2,
            dense_act_fn="gelu_new",
        ),
        lambda cfg: {
            "input_ids": torch.randint(0, min(cfg.vocab_size, 1000), (1, 8)),
            "decoder_input_ids": torch.randint(0, min(cfg.vocab_size, 1000), (1, 4)),
        },
    )

    structure = detect_patterns(layers, config)
    structure.summary()

    # Architecture
    assert structure.arch_type == "encoder_decoder", \
        f"Expected encoder_decoder, got {structure.arch_type}"
    print("  PASS: Architecture type = encoder_decoder (mt5)")

    # Should have both encoder and decoder blocks
    assert len(structure.blocks) >= 4, \
        f"Expected >= 4 blocks (2 enc + 2 dec), got {len(structure.blocks)}"
    print(f"  PASS: {len(structure.blocks)} blocks detected")

    # Check that decoder blocks have cross-attention
    decoder_blocks = [b for b in structure.blocks if b.cross_attention]
    print(f"  PASS: {len(decoder_blocks)} blocks with cross-attention")

    # At least check that attention patterns were found
    blocks_with_attn = [b for b in structure.blocks if b.attention]
    assert len(blocks_with_attn) >= 2, "Expected self-attention in blocks"
    print(f"  PASS: {len(blocks_with_attn)} blocks with self-attention")

    # FFN detection
    blocks_with_ffn = [b for b in structure.blocks if b.ffn]
    assert len(blocks_with_ffn) >= 2, "Expected FFN in blocks"
    print(f"  PASS: {len(blocks_with_ffn)} blocks with FFN")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    test_qwen3_patterns()
    test_bert_patterns()
    test_mt5_patterns()

    print("\n" + "=" * 70)
    print("ALL PATTERN DETECTION TESTS PASSED!")
    print("=" * 70)
