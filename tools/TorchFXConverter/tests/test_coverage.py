"""Verify that every FX graph node is mapped to SOME NNTrainer layer type.
No node should be silently dropped (returning None from mapper)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tracer import Tracer
from node_mapper import NodeMapper


def check_coverage(model, config, input_kwargs, model_name):
    """Check that all non-trivial nodes are mapped."""
    tracer = Tracer(model)
    with tracer:
        with torch.no_grad():
            model(**input_kwargs)

    mapper = NodeMapper(model, tracer.graph, config)
    layers = mapper.map_all()

    # Count by type
    type_counts = {}
    unknown_layers = []
    for layer in layers:
        type_counts[layer.layer_type] = type_counts.get(layer.layer_type, 0) + 1
        if layer.layer_type.startswith("unknown"):
            unknown_layers.append(layer)

    print(f"\n{'='*70}")
    print(f"COVERAGE: {model_name}")
    print(f"{'='*70}")
    print(f"Total mapped layers: {len(layers)}")
    print(f"\nLayer type counts:")
    for lt, count in sorted(type_counts.items()):
        marker = " <<<< UNKNOWN" if lt.startswith("unknown") else ""
        print(f"  {lt:30s}: {count}{marker}")

    if unknown_layers:
        print(f"\n*** {len(unknown_layers)} UNKNOWN LAYERS ***")
        for layer in unknown_layers:
            print(f"  {layer.layer_type:40s} name={layer.name} scope={layer.hf_module_name}")
    else:
        print(f"\n  All operations mapped! No unknowns.")

    # Verify: separate NNTrainer-native types from intermediate/noop types
    nntr_types = [l for l in layers if not l.layer_type.startswith("unknown")
                  and l.layer_type not in ("noop", "reshape_op", "transpose_op", "permute_op", "sdpa")]
    intermediate_types = [l for l in layers if l.layer_type in ("noop", "reshape_op", "transpose_op", "permute_op", "sdpa")]

    print(f"\n  NNTrainer-native layers: {len(nntr_types)}")
    print(f"  Intermediate (will be collapsed by pattern_detector): {len(intermediate_types)}")

    nntr_counts = {}
    for l in nntr_types:
        nntr_counts[l.layer_type] = nntr_counts.get(l.layer_type, 0) + 1
    print(f"\n  NNTrainer-native breakdown:")
    for lt, count in sorted(nntr_counts.items()):
        print(f"    {lt:30s}: {count}")

    return unknown_layers


def test_all():
    all_unknowns = []

    # Qwen3
    from transformers import Qwen3Config, Qwen3ForCausalLM
    config = Qwen3Config(
        vocab_size=151936, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=2048, rms_norm_eps=1e-6,
        tie_word_embeddings=True, rope_theta=1000000.0, sliding_window=None,
    )
    model = Qwen3ForCausalLM(config)
    model.eval()
    unknowns = check_coverage(model, config,
        {"input_ids": torch.randint(0, config.vocab_size, (1, 8))}, "Qwen3")
    all_unknowns.extend(unknowns)

    # BERT
    from transformers import BertConfig, BertModel
    config = BertConfig(
        vocab_size=30522, hidden_size=64, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=128, max_position_embeddings=512,
    )
    model = BertModel(config)
    model.eval()
    unknowns = check_coverage(model, config,
        {"input_ids": torch.randint(0, config.vocab_size, (1, 16)),
         "attention_mask": torch.ones(1, 16, dtype=torch.long)}, "BERT")
    all_unknowns.extend(unknowns)

    # mT5
    from transformers import MT5Config, MT5ForConditionalGeneration
    config = MT5Config(
        vocab_size=250112, d_model=64, d_kv=16, d_ff=128,
        num_heads=4, num_layers=2, num_decoder_layers=2,
        relative_attention_num_buckets=32, relative_attention_max_distance=128,
    )
    model = MT5ForConditionalGeneration(config)
    model.eval()
    unknowns = check_coverage(model, config,
        {"input_ids": torch.randint(0, config.vocab_size, (1, 8)),
         "decoder_input_ids": torch.randint(0, config.vocab_size, (1, 4))}, "mT5")
    all_unknowns.extend(unknowns)

    print(f"\n{'='*70}")
    if all_unknowns:
        print(f"TOTAL UNKNOWNS across all models: {len(all_unknowns)}")
        for u in all_unknowns:
            print(f"  {u.layer_type:40s} {u.hf_module_name}")
        print("SOME OPS ARE NOT MAPPED!")
    else:
        print("ALL OPERATIONS MAPPED ACROSS ALL ARCHITECTURES!")
    print(f"{'='*70}")

    assert len(all_unknowns) == 0, f"{len(all_unknowns)} unknown operations found"


if __name__ == "__main__":
    test_all()
