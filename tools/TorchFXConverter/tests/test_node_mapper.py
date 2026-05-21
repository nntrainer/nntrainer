"""Test the node mapper with Qwen3 tiny config."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from tracer import Tracer
from node_mapper import NodeMapper


def create_tiny_qwen3():
    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        rope_theta=1000000.0,
        sliding_window=None,
    )
    model = Qwen3ForCausalLM(config)
    model.eval()
    return model, config


def test_node_mapper_qwen3():
    print("=" * 70)
    print("TEST: Node Mapper with Qwen3 (tiny config)")
    print("=" * 70)

    model, config = create_tiny_qwen3()

    # Trace
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    tracer = Tracer(model)
    with tracer:
        with torch.no_grad():
            model(input_ids)

    # Map
    mapper = NodeMapper(model, tracer.graph, config)
    layers = mapper.map_all()

    # Print mapped layers
    print(f"\nTotal mapped layers: {len(layers)}")
    print("\n--- Mapped Layers ---")

    type_counts = {}
    for layer in layers:
        type_counts[layer.layer_type] = type_counts.get(layer.layer_type, 0) + 1
        marker = ""
        if layer.has_weight:
            marker += " [W]"
        if layer.has_bias:
            marker += " [B]"
        if layer.transpose_weight:
            marker += " [T]"
        props_str = ", ".join(f"{k}={v}" for k, v in layer.properties.items())
        input_str = " <- " + ",".join(layer.input_layers) if layer.input_layers else ""
        print(f"  {layer.layer_type:25s} {layer.name:50s} {{{props_str}}}{input_str}{marker}")

    print(f"\n--- Layer Type Counts ---")
    for lt, count in sorted(type_counts.items()):
        print(f"  {lt:25s}: {count}")

    # Verify key layer types present
    assert "fully_connected" in type_counts, "Missing fully_connected layers"
    assert "embedding_layer" in type_counts, "Missing embedding_layer"
    assert "rms_norm" in type_counts, "Missing rms_norm layers"
    assert "addition" in type_counts, "Missing addition (residual) layers"

    # Verify layer counts for 2-layer Qwen3
    assert type_counts["embedding_layer"] == 1, f"Expected 1 embedding, got {type_counts['embedding_layer']}"
    assert type_counts["fully_connected"] == 15, f"Expected 15 FC, got {type_counts['fully_connected']}"
    assert type_counts["rms_norm"] == 9, f"Expected 9 RMSNorm, got {type_counts['rms_norm']}"

    # Verify weight info
    fc_layers = [l for l in layers if l.layer_type == "fully_connected"]
    for fc in fc_layers:
        assert fc.has_weight, f"FC layer {fc.name} should have weight"
        assert fc.transpose_weight, f"FC layer {fc.name} should transpose weight"
        assert fc.weight_hf_key, f"FC layer {fc.name} should have weight key"

    emb_layers = [l for l in layers if l.layer_type == "embedding_layer"]
    for emb in emb_layers:
        assert emb.has_weight, f"Embedding {emb.name} should have weight"
        assert not emb.transpose_weight, f"Embedding {emb.name} should NOT transpose"

    print("\n--- C++ Code Preview (first 5 layers) ---")
    for layer in layers[:5]:
        if layer.layer_type not in ("reshape_op", "transpose_op", "permute_op"):
            print(f"  {layer.to_cpp_call()}")

    print("\nPASSED!")


if __name__ == "__main__":
    test_node_mapper_qwen3()
