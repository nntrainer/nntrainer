"""Test the tracer with Qwen3 model (tiny config for fast testing)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from tracer import Tracer


def create_tiny_qwen3():
    """Create a tiny Qwen3 model for testing (2 layers instead of 28).

    Uses Qwen3Config directly to avoid network access.
    Mimics Qwen3-0.6B architecture but with reduced dimensions.
    """
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


def test_trace_qwen3():
    print("=" * 70)
    print("TEST: Trace Qwen3 (tiny config: 2 layers, hidden=64)")
    print("=" * 70)

    model, config = create_tiny_qwen3()

    # Show model architecture
    print("\n--- Model Architecture ---")
    for name, module in model.named_modules():
        if name.count(".") <= 2:  # Only show top 2 levels
            print(f"  {name}: {type(module).__name__}")

    # Create sample input
    input_ids = torch.randint(0, config.vocab_size, (1, 8))

    # Trace the model
    tracer = Tracer(model)
    with tracer:
        with torch.no_grad():
            out = model(input_ids)

    # Print graph summary
    tracer.print_graph_summary()

    # Print all call_module nodes (these are the leaf modules we care about)
    print("\n--- Leaf Module Nodes ---")
    for node in tracer.graph.nodes:
        if node.op == "call_module":
            meta_info = []
            if node.meta.get("module_type"):
                meta_info.append(f"type={node.meta['module_type']}")
            if node.meta.get("in_features"):
                meta_info.append(f"in={node.meta['in_features']}")
            if node.meta.get("out_features"):
                meta_info.append(f"out={node.meta['out_features']}")
            if node.meta.get("is_rmsnorm"):
                meta_info.append("RMSNorm")
            if node.meta.get("eps"):
                meta_info.append(f"eps={node.meta['eps']}")
            if node.meta.get("num_embeddings"):
                meta_info.append(f"vocab={node.meta['num_embeddings']}")
            if node.meta.get("embedding_dim"):
                meta_info.append(f"dim={node.meta['embedding_dim']}")
            print(f"  {node.target:50s} [{', '.join(meta_info)}]")

    # Print function/method nodes (for residual add, reshape, etc.)
    print("\n--- Function/Method Nodes ---")
    for node in tracer.graph.nodes:
        if node.op in ("call_function", "call_method"):
            target_name = node.target if isinstance(node.target, str) else getattr(node.target, "__name__", str(node.target))
            scope = node.meta.get("scope", "")
            if target_name not in ("_set_grad_enabled", "detach", "clone"):
                print(f"  {node.op:15s} {target_name:30s} scope={scope}")

    # Save full graph to file for reference
    output_file = os.path.join(os.path.dirname(__file__), "qwen3_tiny_graph.txt")
    with open(output_file, "w") as f:
        f.write("Qwen3 Tiny Config Graph Dump\n")
        f.write(f"Config: hidden={config.hidden_size}, layers={config.num_hidden_layers}, ")
        f.write(f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}\n")
        f.write(f"head_dim={config.head_dim}, intermediate={config.intermediate_size}\n\n")

        f.write("=" * 100 + "\n")
        f.write("LEAF MODULE NODES:\n")
        f.write("=" * 100 + "\n")
        for node in tracer.graph.nodes:
            if node.op == "call_module":
                f.write(f"  {node.target:50s} type={node.meta.get('module_type', '?')}\n")
                f.write(f"    args: {[str(a) for a in node.args if hasattr(a, 'name')]}\n")
                for k, v in node.meta.items():
                    if k not in ("scope", "output_type", "leaf_module", "module_type", "module_class"):
                        f.write(f"    meta.{k} = {v}\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write("ALL NODES:\n")
        f.write("=" * 100 + "\n")
        for node in tracer.graph.nodes:
            target_name = node.target if isinstance(node.target, str) else getattr(node.target, "__name__", str(node.target))
            args_str = ", ".join(str(a) for a in node.args)
            f.write(f"  [{node.op:15s}] {node.name:40s} target={target_name:40s} args=({args_str})\n")

    print(f"\nFull graph saved to: {output_file}")

    # Count expected structures for 2-layer Qwen3
    leaf_modules = tracer.get_leaf_modules()
    module_types = {}
    for name, mod in leaf_modules.items():
        type_name = type(mod).__name__
        module_types[type_name] = module_types.get(type_name, 0) + 1

    print(f"\nModule type counts: {module_types}")

    # Verify expected structure:
    # - 1 embedding
    # - 2 layers x (4 Linear for attn + 3 Linear for MLP) = 14 Linear
    # - 2 layers x (2 RMSNorm per block) = 4 RMSNorm + 1 final RMSNorm = 5 RMSNorm
    # - 2 layers x (2 RMSNorm for Q/K norm) = 4 reshaped RMSNorm (Qwen3-specific)
    # - 1 lm_head (Linear)
    print("\nExpected for 2-layer Qwen3:")
    print("  1 Embedding")
    print("  15 Linear (2*(4 attn + 3 FFN) + 1 lm_head)")
    print("  5 RMSNorm (2*2 per block + 1 final)")
    print("  4 RMSNorm for Q/K norms (Qwen3-specific, 2*2)")
    print("DONE!")


if __name__ == "__main__":
    test_trace_qwen3()
