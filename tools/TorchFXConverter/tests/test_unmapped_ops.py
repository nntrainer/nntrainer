"""Check what FX graph nodes are NOT mapped to NNTrainer layers.
Identifies gaps where we need tensor op fallbacks."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tracer import Tracer
from node_mapper import NodeMapper


def collect_unmapped(model, config, input_kwargs, model_name):
    """Trace & map a model, report all unmapped nodes."""
    tracer = Tracer(model)
    with tracer:
        with torch.no_grad():
            model(**input_kwargs)

    mapper = NodeMapper(model, tracer.graph, config)
    layers = mapper.map_all()
    mapped_names = {l.name for l in layers}

    unmapped = []
    for node in tracer.graph.nodes:
        if node.op in ("placeholder", "output", "get_attr"):
            continue
        # Check if this node was mapped
        # For call_module, the mapper uses sanitized name
        # For others, it uses node.name or scope-based name
        was_mapped = False
        for layer in layers:
            if layer.name == node.name or layer.name == node.name.replace(".", "_"):
                was_mapped = True
                break
        # Also check by matching the node itself
        if node.op == "call_module":
            sanitized = node.target.replace(".", "_")
            was_mapped = was_mapped or any(l.name == sanitized for l in layers)

        if not was_mapped and node.op in ("call_function", "call_method"):
            target = node.target if isinstance(node.target, str) else getattr(node.target, "__name__", str(node.target))
            scope = node.meta.get("scope", "")
            unmapped.append((node.op, target, scope, node.name))

    return unmapped, layers


def test_unmapped_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    config = Qwen3Config(
        vocab_size=151936, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=2048, rms_norm_eps=1e-6,
        tie_word_embeddings=True, rope_theta=1000000.0, sliding_window=None,
    )
    model = Qwen3ForCausalLM(config)
    model.eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 8))

    unmapped, layers = collect_unmapped(model, config, {"input_ids": input_ids}, "Qwen3")

    print("=" * 70)
    print("UNMAPPED OPS: Qwen3")
    print("=" * 70)
    seen = set()
    for op, target, scope, name in unmapped:
        key = (op, target)
        if key not in seen:
            count = sum(1 for o, t, _, _ in unmapped if (o, t) == key)
            seen.add(key)
            print(f"  {op:15s} {target:35s} x{count:3d}  (example scope: {scope})")

    return unmapped


def test_unmapped_bert():
    from transformers import BertConfig, BertModel
    config = BertConfig(
        vocab_size=30522, hidden_size=64, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=128, max_position_embeddings=512,
    )
    model = BertModel(config)
    model.eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    attention_mask = torch.ones(1, 16, dtype=torch.long)

    unmapped, layers = collect_unmapped(model, config, {"input_ids": input_ids, "attention_mask": attention_mask}, "BERT")

    print("\n" + "=" * 70)
    print("UNMAPPED OPS: BERT")
    print("=" * 70)
    seen = set()
    for op, target, scope, name in unmapped:
        key = (op, target)
        if key not in seen:
            count = sum(1 for o, t, _, _ in unmapped if (o, t) == key)
            seen.add(key)
            print(f"  {op:15s} {target:35s} x{count:3d}  (example scope: {scope})")

    return unmapped


def test_unmapped_mt5():
    from transformers import MT5Config, MT5ForConditionalGeneration
    config = MT5Config(
        vocab_size=250112, d_model=64, d_kv=16, d_ff=128,
        num_heads=4, num_layers=2, num_decoder_layers=2,
        relative_attention_num_buckets=32, relative_attention_max_distance=128,
    )
    model = MT5ForConditionalGeneration(config)
    model.eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    decoder_input_ids = torch.randint(0, config.vocab_size, (1, 4))

    unmapped, layers = collect_unmapped(model, config,
        {"input_ids": input_ids, "decoder_input_ids": decoder_input_ids}, "mT5")

    print("\n" + "=" * 70)
    print("UNMAPPED OPS: mT5")
    print("=" * 70)
    seen = set()
    for op, target, scope, name in unmapped:
        key = (op, target)
        if key not in seen:
            count = sum(1 for o, t, _, _ in unmapped if (o, t) == key)
            seen.add(key)
            print(f"  {op:15s} {target:35s} x{count:3d}  (example scope: {scope})")

    return unmapped


if __name__ == "__main__":
    all_unmapped = {}

    qwen3_unmapped = test_unmapped_qwen3()
    bert_unmapped = test_unmapped_bert()
    mt5_unmapped = test_unmapped_mt5()

    # Collect all unique unmapped ops across all models
    print("\n" + "=" * 70)
    print("SUMMARY: All unique unmapped operations across all models")
    print("=" * 70)
    all_ops = set()
    for unmapped_list in [qwen3_unmapped, bert_unmapped, mt5_unmapped]:
        for op, target, scope, name in unmapped_list:
            all_ops.add((op, target))

    for op, target in sorted(all_ops, key=lambda x: (x[0], x[1])):
        models = []
        if any(t == target for _, t, _, _ in qwen3_unmapped):
            models.append("Qwen3")
        if any(t == target for _, t, _, _ in bert_unmapped):
            models.append("BERT")
        if any(t == target for _, t, _, _ in mt5_unmapped):
            models.append("mT5")
        print(f"  {op:15s} {target:35s} models: {', '.join(models)}")
