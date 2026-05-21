"""Test tracer and node mapper with multiple architectures:
  1. Encoder-only: BERT
  2. Encoder-decoder: mT5
  3. Sentence transformer / KaLM-style: Qwen2 (base of KaLM-Embedding)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tracer import Tracer
from node_mapper import NodeMapper


def test_bert():
    """Test encoder-only: BERT"""
    from transformers import BertConfig, BertModel

    print("=" * 70)
    print("TEST: BERT (Encoder-only)")
    print("=" * 70)

    config = BertConfig(
        vocab_size=30522,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=512,
        type_vocab_size=2,
    )
    model = BertModel(config)
    model.eval()

    # Show architecture
    print("\n--- Model Architecture ---")
    for name, module in model.named_modules():
        if name.count(".") <= 2:
            print(f"  {name}: {type(module).__name__}")

    # Trace
    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    attention_mask = torch.ones(1, 16, dtype=torch.long)

    tracer = Tracer(model)
    with tracer:
        with torch.no_grad():
            out = model(input_ids, attention_mask=attention_mask)

    tracer.print_graph_summary()

    # Map
    mapper = NodeMapper(model, tracer.graph, config)
    layers = mapper.map_all()

    type_counts = {}
    for layer in layers:
        type_counts[layer.layer_type] = type_counts.get(layer.layer_type, 0) + 1

    print(f"\nTotal mapped layers: {len(layers)}")
    print(f"Layer type counts: {type_counts}")

    # Print leaf module layers
    print("\n--- Key Leaf Module Layers ---")
    for layer in layers:
        if layer.hf_module_name:
            props_str = ", ".join(f"{k}={v}" for k, v in layer.properties.items())
            print(f"  {layer.layer_type:25s} {layer.name:50s} {{{props_str}}}")

    # Verify BERT-specific structures
    assert "embedding_layer" in type_counts, "BERT should have embeddings"
    assert "fully_connected" in type_counts, "BERT should have FC layers (attention projections)"
    assert "layer_normalization" in type_counts, "BERT should have LayerNorm (not RMSNorm)"

    # BERT has LayerNorm, not RMSNorm
    assert "rms_norm" not in type_counts, "BERT should NOT have RMSNorm"

    print("\nBERT: PASSED!\n")
    return layers


def test_mt5():
    """Test encoder-decoder: mT5"""
    from transformers import MT5Config, MT5ForConditionalGeneration

    print("=" * 70)
    print("TEST: mT5 (Encoder-Decoder)")
    print("=" * 70)

    config = MT5Config(
        vocab_size=250112,
        d_model=64,
        d_kv=16,
        d_ff=128,
        num_heads=4,
        num_layers=2,
        num_decoder_layers=2,
        is_encoder_decoder=True,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        tie_word_embeddings=False,
    )
    model = MT5ForConditionalGeneration(config)
    model.eval()

    # Show architecture
    print("\n--- Model Architecture (top-level) ---")
    for name, module in model.named_modules():
        if name.count(".") <= 2:
            print(f"  {name}: {type(module).__name__}")

    # Trace - encoder-decoder needs both input_ids and decoder_input_ids
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    decoder_input_ids = torch.randint(0, config.vocab_size, (1, 4))

    tracer = Tracer(model)
    with tracer:
        with torch.no_grad():
            out = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

    tracer.print_graph_summary()

    # Map
    mapper = NodeMapper(model, tracer.graph, config)
    layers = mapper.map_all()

    type_counts = {}
    for layer in layers:
        type_counts[layer.layer_type] = type_counts.get(layer.layer_type, 0) + 1

    print(f"\nTotal mapped layers: {len(layers)}")
    print(f"Layer type counts: {type_counts}")

    # Print encoder vs decoder layers
    print("\n--- Key Layers (Encoder) ---")
    for layer in layers:
        if layer.hf_module_name and "encoder" in layer.hf_module_name:
            if layer.layer_type in ("fully_connected", "rms_norm", "layer_normalization", "embedding_layer"):
                print(f"  {layer.layer_type:25s} {layer.hf_module_name}")

    print("\n--- Key Layers (Decoder) ---")
    for layer in layers:
        if layer.hf_module_name and "decoder" in layer.hf_module_name:
            if layer.layer_type in ("fully_connected", "rms_norm", "layer_normalization", "embedding_layer"):
                print(f"  {layer.layer_type:25s} {layer.hf_module_name}")

    # Verify encoder-decoder structure
    assert "fully_connected" in type_counts, "mT5 should have FC layers"
    assert "embedding_layer" in type_counts, "mT5 should have embeddings"

    # mT5 uses RMSNorm (T5LayerNorm is actually RMSNorm)
    # Check for either rms_norm or layer_normalization
    has_norm = "rms_norm" in type_counts or "layer_normalization" in type_counts
    assert has_norm, "mT5 should have normalization layers"

    # Encoder-decoder should have cross-attention layers
    encoder_fc = [l for l in layers if l.hf_module_name and "encoder" in l.hf_module_name and l.layer_type == "fully_connected"]
    decoder_fc = [l for l in layers if l.hf_module_name and "decoder" in l.hf_module_name and l.layer_type == "fully_connected"]
    print(f"\nEncoder FC layers: {len(encoder_fc)}")
    print(f"Decoder FC layers: {len(decoder_fc)}")

    # Decoder should have more FC layers than encoder (self-attn + cross-attn + FFN)
    assert len(decoder_fc) > len(encoder_fc), \
        f"Decoder ({len(decoder_fc)} FC) should have more FC than encoder ({len(encoder_fc)} FC) due to cross-attention"

    print("\nmT5: PASSED!\n")
    return layers


def test_qwen2_embedding():
    """Test sentence transformer / KaLM-style: Qwen2 base model.

    KaLM-Embedding is based on Qwen2 with bidirectional attention.
    We test Qwen2Model (base, not ForCausalLM) to simulate this.
    """
    from transformers import Qwen2Config, Qwen2Model

    print("=" * 70)
    print("TEST: Qwen2 base model (KaLM-Embedding style)")
    print("=" * 70)

    config = Qwen2Config(
        vocab_size=151936,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        rope_theta=1000000.0,
        sliding_window=None,
    )
    model = Qwen2Model(config)
    model.eval()

    print("\n--- Model Architecture ---")
    for name, module in model.named_modules():
        if name.count(".") <= 2:
            print(f"  {name}: {type(module).__name__}")

    # Trace
    input_ids = torch.randint(0, config.vocab_size, (1, 8))

    tracer = Tracer(model)
    with tracer:
        with torch.no_grad():
            out = model(input_ids)

    tracer.print_graph_summary()

    # Map
    mapper = NodeMapper(model, tracer.graph, config)
    layers = mapper.map_all()

    type_counts = {}
    for layer in layers:
        type_counts[layer.layer_type] = type_counts.get(layer.layer_type, 0) + 1

    print(f"\nTotal mapped layers: {len(layers)}")
    print(f"Layer type counts: {type_counts}")

    print("\n--- Key Leaf Module Layers ---")
    for layer in layers:
        if layer.hf_module_name:
            props_str = ", ".join(f"{k}={v}" for k, v in layer.properties.items())
            print(f"  {layer.layer_type:25s} {layer.hf_module_name:50s} {{{props_str}}}")

    # Verify Qwen2 structure (similar to Qwen3 but no Q/K norms)
    assert "embedding_layer" in type_counts
    assert "fully_connected" in type_counts
    assert "rms_norm" in type_counts
    # Qwen2 has RMSNorm, not LayerNorm
    assert "layer_normalization" not in type_counts

    print("\nQwen2 (KaLM-style): PASSED!\n")
    return layers


def test_gemma_embedding():
    """Test Gemma base model (embedding-gemma style).

    Tests: GemmaModel with GeGLU activation, MQA, RoPE, and no LM head.
    """
    from transformers import GemmaConfig, GemmaModel
    from decomposer import AdaptiveConverter
    from pattern_detector import detect_patterns

    print("=" * 70)
    print("TEST: Gemma base model (embedding-gemma style)")
    print("=" * 70)

    config = GemmaConfig(
        vocab_size=256128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=8192,
    )
    config.architectures = ["GemmaModel"]
    model = GemmaModel(config)
    model.eval()

    print("\n--- Model Architecture ---")
    for name, module in model.named_modules():
        if name.count(".") <= 2:
            print(f"  {name}: {type(module).__name__}")

    # Trace via AdaptiveConverter (full pipeline)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    converter = AdaptiveConverter(model, config)
    result = converter.convert({"input_ids": input_ids})

    layers = result.layers
    structure = result.model_structure

    type_counts = {}
    for layer in layers:
        type_counts[layer.layer_type] = type_counts.get(layer.layer_type, 0) + 1

    print(f"\nTotal mapped layers: {len(layers)}")
    print(f"Layer type counts: {type_counts}")

    # Verify Gemma structure
    assert "embedding_layer" in type_counts, "Gemma should have embeddings"
    assert "fully_connected" in type_counts, "Gemma should have FC layers"
    assert "rms_norm" in type_counts, "Gemma should have RMSNorm"
    assert "activation" in type_counts, "Gemma should have activation layers (GELU)"
    assert result.is_fully_mapped, "All ops should be mapped"

    # Verify model structure detection
    assert structure is not None, "Structure should be detected"
    assert structure.arch_type == "embedding", \
        f"Expected 'embedding' arch, got '{structure.arch_type}'"
    assert structure.model_type == "gemma"
    assert structure.num_layers == 2, \
        f"Expected 2 blocks, got {structure.num_layers}"

    # Verify block detection
    for block in structure.blocks:
        assert block.attention is not None, "Block should have attention"
        assert block.attention.attention_type == "mqa", \
            f"Gemma should use MQA, got {block.attention.attention_type}"
        assert block.attention.has_rope, "Gemma should have RoPE"
        assert block.ffn is not None, "Block should have FFN"
        assert block.ffn.ffn_type == "geglu", \
            f"Gemma should use GeGLU, got {block.ffn.ffn_type}"
        assert block.pre_attn_norm, "Block should have pre-attention norm"
        assert block.pre_ffn_norm, "Block should have pre-FFN norm"

    # Verify GELU activation is properly detected (not decomposed)
    gelu_layers = [l for l in layers if l.layer_type == "activation"
                   and l.properties.get("activation") == "gelu"]
    assert len(gelu_layers) == 2, \
        f"Expected 2 GELU activations (one per block), got {len(gelu_layers)}"

    # Verify no LM head detected
    assert not structure.lm_head, "Embedding model should not have LM head"

    print("\nGemma (embedding): PASSED!\n")


def test_gemma3_text_embedding():
    """Test Gemma3-based embedding model (embeddinggemma-300m style).

    Tests: Gemma3TextModel with Q/K norms, 4 norms per block, scaled embedding,
    GeGLU activation, GQA with RoPE.
    """
    from transformers import Gemma3TextConfig, Gemma3TextModel
    from decomposer import AdaptiveConverter

    print("=" * 70)
    print("TEST: Gemma3 text model (embeddinggemma-300m style)")
    print("=" * 70)

    config = Gemma3TextConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=1024,
    )
    config.architectures = ["Gemma3TextModel"]
    model = Gemma3TextModel(config)
    model.eval()

    print("\n--- Model Architecture ---")
    for name, module in model.named_modules():
        if name.count(".") <= 2:
            print(f"  {name}: {type(module).__name__}")

    # Trace via AdaptiveConverter
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    converter = AdaptiveConverter(model, config)
    result = converter.convert({"input_ids": input_ids})

    layers = result.layers
    structure = result.model_structure

    type_counts = {}
    for layer in layers:
        type_counts[layer.layer_type] = type_counts.get(layer.layer_type, 0) + 1

    print(f"\nTotal mapped layers: {len(layers)}")
    print(f"Layer type counts: {type_counts}")

    # Verify basic structure
    assert result.is_fully_mapped, "All ops should be mapped"
    assert structure is not None
    assert structure.arch_type == "embedding"
    assert structure.model_type == "gemma3_text"
    assert structure.num_layers == 2

    # Verify Gemma3-specific features
    for block in structure.blocks:
        # Q/K norms
        assert block.attention.has_qk_norm, "Gemma3 should have Q/K norms"
        assert block.attention.q_norm, "Q norm should be detected"
        assert block.attention.k_norm, "K norm should be detected"

        # GQA
        assert block.attention.attention_type == "gqa"
        assert block.attention.num_kv_heads == 2

        # GeGLU
        assert block.ffn.ffn_type == "geglu"

        # 4 norms per block (Gemma3 specific)
        assert block.pre_attn_norm, "Should have pre-attention norm"
        assert block.post_attn_norm, "Should have post-attention norm"
        assert block.pre_ffn_norm, "Should have pre-FFN norm"
        assert block.post_ffn_norm, "Should have post-FFN norm"

    # Verify scaled embedding
    emb_layers = [l for l in layers if l.layer_type == "embedding_layer"]
    assert len(emb_layers) == 1
    assert emb_layers[0].properties.get("embed_scale") == 8.0, \
        "Gemma3 scaled embedding should capture embed_scale"

    # Verify GELU activation detected as leaf
    gelu_layers = [l for l in layers if l.layer_type == "activation"
                   and l.properties.get("activation") == "gelu"]
    assert len(gelu_layers) == 2, \
        f"Expected 2 GELU activations, got {len(gelu_layers)}"

    print("\nGemma3 (embeddinggemma): PASSED!\n")


def test_functiongemma_causal():
    """Test FunctionGemma (Gemma3-based CausalLM for function calling).

    FunctionGemma-270m-it is architecturally identical to Gemma3 270M
    but fine-tuned for function calling.  It uses Gemma3ForCausalLM with
    model_type="gemma3_text".

    Tests: Gemma3ForCausalLM with Q/K norms, 4 norms per block, scaled
    embedding, GeGLU activation, GQA with RoPE, and LM head.
    """
    from transformers import Gemma3TextConfig, Gemma3ForCausalLM
    from decomposer import AdaptiveConverter

    print("=" * 70)
    print("TEST: FunctionGemma (Gemma3 CausalLM for function calling)")
    print("=" * 70)

    # Tiny config matching FunctionGemma-270m-it architecture
    config = Gemma3TextConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=1024,
    )
    config.architectures = ["Gemma3ForCausalLM"]
    model = Gemma3ForCausalLM(config)
    model.eval()

    print("\n--- Model Architecture ---")
    for name, module in model.named_modules():
        if name.count(".") <= 2:
            print(f"  {name}: {type(module).__name__}")

    # Trace via AdaptiveConverter
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    converter = AdaptiveConverter(model, config)
    result = converter.convert({"input_ids": input_ids})

    layers = result.layers
    structure = result.model_structure

    type_counts = {}
    for layer in layers:
        type_counts[layer.layer_type] = type_counts.get(layer.layer_type, 0) + 1

    print(f"\nTotal mapped layers: {len(layers)}")
    print(f"Layer type counts: {type_counts}")

    # Verify basic structure
    assert result.is_fully_mapped, "All ops should be mapped"
    assert structure is not None
    assert structure.arch_type == "decoder_only"
    assert structure.model_type == "gemma3_text"
    assert structure.num_layers == 2

    # Verify Gemma3-specific features
    for block in structure.blocks:
        # Q/K norms
        assert block.attention.has_qk_norm, "Gemma3 should have Q/K norms"
        assert block.attention.q_norm, "Q norm should be detected"
        assert block.attention.k_norm, "K norm should be detected"

        # GQA
        assert block.attention.attention_type == "gqa"
        assert block.attention.num_kv_heads == 2

        # GeGLU
        assert block.ffn.ffn_type == "geglu"

        # 4 norms per block (Gemma3 specific)
        assert block.pre_attn_norm, "Should have pre-attention norm"
        assert block.post_attn_norm, "Should have post-attention norm"
        assert block.pre_ffn_norm, "Should have pre-FFN norm"
        assert block.post_ffn_norm, "Should have post-FFN norm"

    # Verify scaled embedding
    emb_layers = [l for l in layers if l.layer_type == "embedding_layer"]
    assert len(emb_layers) == 1
    assert emb_layers[0].properties.get("embed_scale") == 8.0, \
        "Gemma3 scaled embedding should capture embed_scale"

    # Verify LM head (CausalLM should have one)
    assert structure.lm_head, "CausalLM should have LM head"

    # Verify GELU activations
    gelu_layers = [l for l in layers if l.layer_type == "activation"
                   and l.properties.get("activation") == "gelu"]
    assert len(gelu_layers) == 2, \
        f"Expected 2 GELU activations, got {len(gelu_layers)}"

    print("\nFunctionGemma (Gemma3 CausalLM): PASSED!\n")


def test_granite4_causal():
    """Test Granite 4.0 (IBM dense transformer, GraniteMoeHybrid with no MoE).

    Granite-4.0-350M uses GraniteMoeHybridForCausalLM with num_experts=0,
    making it a pure dense transformer. Architecture: GQA, SwiGLU, RMSNorm,
    RoPE, shared embeddings, with scaling multipliers.
    """
    from transformers import GraniteMoeHybridConfig, GraniteMoeHybridForCausalLM
    from decomposer import AdaptiveConverter

    print("=" * 70)
    print("TEST: Granite 4.0 (dense transformer, GraniteMoeHybrid)")
    print("=" * 70)

    num_layers = 2
    config = GraniteMoeHybridConfig(
        vocab_size=100544,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32768,
        rms_norm_eps=1e-5,
        hidden_act='silu',
        tie_word_embeddings=True,
        embedding_multiplier=12.0,
        logits_scaling=0.08838834764831845,
        residual_multiplier=0.22360679774997896,
        attention_multiplier=0.08838834764831845,
        num_local_experts=0,
        num_experts_per_tok=0,
        position_embedding_type='rope',
        layer_types=['attention'] * num_layers,
    )
    config.architectures = ["GraniteMoeHybridForCausalLM"]
    model = GraniteMoeHybridForCausalLM(config)
    model.eval()

    print("\n--- Model Architecture ---")
    for name, module in model.named_modules():
        if name.count(".") <= 2:
            print(f"  {name}: {type(module).__name__}")

    # Trace via AdaptiveConverter
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    converter = AdaptiveConverter(model, config)
    result = converter.convert({"input_ids": input_ids})

    layers = result.layers
    structure = result.model_structure

    type_counts = {}
    for layer in layers:
        type_counts[layer.layer_type] = type_counts.get(layer.layer_type, 0) + 1

    print(f"\nTotal mapped layers: {len(layers)}")
    print(f"Layer type counts: {type_counts}")

    # Verify basic structure
    assert result.is_fully_mapped, \
        f"All ops should be mapped, unknowns: {result.unknown_layers}, unsupported: {result.unsupported_ops}"
    assert structure is not None
    assert structure.arch_type == "decoder_only"
    assert structure.num_layers == num_layers

    # Verify Granite-specific features
    assert "rms_norm" in type_counts, "Granite should have RMSNorm"
    assert "fully_connected" in type_counts, "Granite should have FC layers"
    assert "embedding_layer" in type_counts, "Granite should have embeddings"
    assert "activation" in type_counts, "Granite should have activation layers (SiLU)"

    # Verify SwiGLU activation (SiLU gate)
    silu_layers = [l for l in layers if l.layer_type == "activation"
                   and l.properties.get("activation") == "swish"]
    assert len(silu_layers) == num_layers, \
        f"Expected {num_layers} SiLU activations (one per block), got {len(silu_layers)}"

    # Verify block structure
    for block in structure.blocks:
        assert block.attention is not None, "Block should have attention"
        assert block.attention.attention_type == "gqa", \
            f"Granite should use GQA, got {block.attention.attention_type}"
        assert block.attention.has_rope, "Granite should have RoPE"
        assert block.ffn is not None, "Block should have FFN"

    print("\nGranite 4.0: PASSED!\n")


def test_resnet18():
    """Test CNN: ResNet-18.

    Tests: Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d,
    residual (addition), Linear, flatten.
    """
    from torchvision.models import resnet18
    from decomposer import AdaptiveConverter

    print("=" * 70)
    print("TEST: ResNet-18 (CNN)")
    print("=" * 70)

    model = resnet18(weights=None)
    model.eval()

    print("\n--- Model Architecture ---")
    for name, module in model.named_modules():
        if name.count(".") <= 1:
            print(f"  {name}: {type(module).__name__}")

    # Trace via AdaptiveConverter
    input_tensor = torch.randn(1, 3, 224, 224)
    converter = AdaptiveConverter(model, None)
    result = converter.convert({"x": input_tensor})

    layers = result.layers

    type_counts = {}
    for layer in layers:
        type_counts[layer.layer_type] = type_counts.get(layer.layer_type, 0) + 1

    print(f"\nTotal mapped layers: {len(layers)}")
    print(f"Layer type counts: {type_counts}")

    # Verify basic structure
    assert result.is_fully_mapped, \
        f"All ops should be mapped, unknowns: {result.unknown_layers}, unsupported: {result.unsupported_ops}"

    # Verify ResNet components
    assert type_counts.get("conv2d", 0) == 20, \
        f"ResNet-18 should have 20 conv2d layers, got {type_counts.get('conv2d', 0)}"
    assert type_counts.get("batch_normalization", 0) == 20, \
        f"ResNet-18 should have 20 batchnorm layers, got {type_counts.get('batch_normalization', 0)}"
    assert type_counts.get("activation", 0) == 17, \
        f"ResNet-18 should have 17 activation (ReLU) layers, got {type_counts.get('activation', 0)}"
    assert type_counts.get("pooling2d", 0) == 2, \
        f"ResNet-18 should have 2 pooling layers, got {type_counts.get('pooling2d', 0)}"
    assert type_counts.get("addition", 0) == 8, \
        f"ResNet-18 should have 8 residual additions, got {type_counts.get('addition', 0)}"
    assert type_counts.get("fully_connected", 0) == 1, \
        f"ResNet-18 should have 1 FC layer, got {type_counts.get('fully_connected', 0)}"
    assert type_counts.get("flatten", 0) == 1, \
        f"ResNet-18 should have 1 flatten, got {type_counts.get('flatten', 0)}"

    print("\nResNet-18: PASSED!\n")


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# Multi-Architecture Tracer + Node Mapper Validation")
    print("#" * 70 + "\n")

    test_bert()
    test_mt5()
    test_qwen2_embedding()
    test_gemma_embedding()
    test_gemma3_text_embedding()
    test_functiongemma_causal()
    test_granite4_causal()
    test_resnet18()

    print("=" * 70)
    print("ALL ARCHITECTURE TESTS PASSED!")
    print("=" * 70)
