# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

## @package generate_gemma4_moe_reference
## @brief Generate Hugging Face reference fixtures for a tiny Gemma4 MoE model.
## @author Jungwon-Lee <jungone.lee@samsung.com>

"""Generate Hugging Face reference fixtures for a tiny Gemma4 MoE model.

The NNTrainer binary is written through the production Gemma4 MoE converter so
the fixture also verifies PLE-disabled, K-equals-V, router, norm, and fused
expert ordering.
"""

import argparse
import importlib.util
import json
import pathlib
import types

import numpy as np
import torch
import transformers
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

from generate_gemma4_reference import INPUT_IDS, N_GEN, TINY_TOKENIZER


THIS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[4]
DEFAULT_OUT = THIS_DIR.parent / "gemma4_moe_tiny"
CONVERTER_PATH = (
    REPO_ROOT / "Applications/CausalLM/res/gemma4/gemma4_moe_weight_converter.py"
)

TINY_MOE_TEXT_CONFIG = dict(
    hidden_size=64,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=8,
    global_head_dim=8,
    num_global_key_value_heads=4,
    hidden_size_per_layer_input=0,
    vocab_size_per_layer_input=32,
    vocab_size=32,
    max_position_embeddings=8,
    rms_norm_eps=1e-6,
    rope_theta=1000000,
    sliding_window=4,
    layer_types=["sliding_attention", "full_attention"],
    tie_word_embeddings=True,
    hidden_activation="gelu_pytorch_tanh",
    attention_dropout=0.0,
    pad_token_id=0,
    num_kv_shared_layers=0,
    use_double_wide_mlp=False,
    attention_k_eq_v=True,
    enable_moe_block=True,
    num_experts=4,
    top_k_experts=2,
    moe_intermediate_size=32,
)


def load_production_converter():
    """Load the production converter directly from the repository."""
    spec = importlib.util.spec_from_file_location(
        "gemma4_moe_weight_converter", CONVERTER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(seed):
    """Build a deterministic tiny Gemma4 MoE text model."""
    torch.manual_seed(seed)
    config = Gemma4TextConfig(**TINY_MOE_TEXT_CONFIG)
    model = Gemma4TextModel(config)
    model.eval()
    return model


def convert_weights(model, output_path):
    """Write weights using the same ordered walker as production conversion."""
    converter = load_production_converter()
    wrapper_config = types.SimpleNamespace(text_config=model.config)
    with open(output_path, "wb") as output_file:
        converter.save_gemma4_moe_bin(
            model.state_dict(), wrapper_config, "float32", output_file, True
        )


def run_forward(model, input_ids):
    """Return tied-embedding logits for the final prompt token."""
    ids = torch.tensor([input_ids], dtype=torch.long)
    with torch.no_grad():
        hidden = model(ids, use_cache=False).last_hidden_state[0, -1, :]
        logits = hidden @ model.embed_tokens.weight.T
    return logits.float().tolist()


def run_greedy_with_margin(model, input_ids, count):
    """Generate greedy tokens and return the minimum top-two logit margin."""
    ids = list(input_ids)
    generated = []
    minimum_margin = float("inf")
    with torch.no_grad():
        for _ in range(count):
            inputs = torch.tensor([ids], dtype=torch.long)
            hidden = model(inputs, use_cache=False).last_hidden_state[0, -1, :]
            logits = hidden @ model.embed_tokens.weight.T
            top2 = torch.topk(logits.float(), k=2).values
            minimum_margin = min(
                minimum_margin, float((top2[0] - top2[1]).item())
            )
            token = int(logits.argmax().item())
            generated.append(token)
            ids.append(token)
    return generated, minimum_margin


def find_stable_seed(count, generated_tokens):
    """Select the seed with the largest minimum greedy logit margin."""
    candidates = []
    for seed in range(count):
        model = build_model(seed)
        _, margin = run_greedy_with_margin(
            model, INPUT_IDS, generated_tokens
        )
        candidates.append((margin, seed))
    candidates.sort(reverse=True)
    print(f"[search] best seeds={candidates[:10]}")
    return candidates[0][1]


def write_configs(output_dir, binary_name, tokenizer_path):
    """Write the three configs consumed by the CausalLM differential tests."""
    config_json = {
        "architectures": ["Gemma4ForCausalLM"],
        "bos_token_id": 0,
        "eos_token_id": [31],
        "num_hidden_layers": 2,
        "text_config": {
            "attention_k_eq_v": True,
            "enable_moe_block": True,
            "global_head_dim": 8,
            "head_dim": 8,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 64,
            "hidden_size_per_layer_input": 0,
            "intermediate_size": 64,
            "layer_types": ["sliding_attention", "full_attention"],
            "max_position_embeddings": 8,
            "moe_intermediate_size": 32,
            "num_attention_heads": 8,
            "num_experts": 4,
            "num_global_key_value_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 4,
            "num_kv_shared_layers": 0,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000,
                },
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": 1000000,
                    "partial_rotary_factor": 0.25,
                },
            },
            "rope_theta": 1000000,
            "sliding_window": 4,
            "tie_word_embeddings": True,
            "top_k_experts": 2,
            "use_double_wide_mlp": False,
            "vocab_size": 32,
            "vocab_size_per_layer_input": 32,
        },
    }
    generation_json = {
        "bos_token_id": 0,
        "eos_token_id": 31,
        "do_sample": False,
        "top_k": 1,
        "top_p": 1.0,
        "temperature": 1.0,
    }
    nntrainer_json = {
        "bad_word_ids": [],
        "batch_size": 1,
        "embedding_dtype": "FP32",
        "fc_layer_dtype": "FP32",
        "init_seq_len": 4,
        "lmhead_dtype": "FP32",
        "max_seq_len": 8,
        "model_file_name": binary_name,
        "model_tensor_type": "FP32-FP32",
        "model_type": "CausalLM",
        "num_to_generate": 1,
        "tokenizer_file": pathlib.Path(tokenizer_path).name,
    }
    for filename, payload in (
        ("config.json", config_json),
        ("generation_config.json", generation_json),
        ("nntr_config.json", nntrainer_json),
    ):
        with open(output_dir / filename, "w") as output_file:
            json.dump(payload, output_file, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Generate tiny Gemma4 MoE Hugging Face fixtures"
    )
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    # Seed 17 was selected from [0, 100) for a 0.102 minimum greedy margin.
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--n", type=int, default=N_GEN)
    parser.add_argument(
        "--search-seeds",
        type=int,
        default=0,
        help="Search seeds [0, N) and use the one with the widest greedy margin",
    )
    parser.add_argument("--transformers-commit", default="unknown")
    args = parser.parse_args()

    if args.search_seeds > 0:
        args.seed = find_stable_seed(args.search_seeds, args.n)

    output_dir = args.out.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(args.seed)
    print(f"[generate] parameters={sum(p.numel() for p in model.parameters()):,}")

    binary_name = "nntr_gemma4_moe_tiny_fp32.bin"
    convert_weights(model, output_dir / binary_name)

    tokenizer_path = output_dir / "tokenizer.json"
    with open(tokenizer_path, "w") as output_file:
        json.dump(TINY_TOKENIZER, output_file, indent=2)
    write_configs(output_dir, binary_name, tokenizer_path)

    logits = run_forward(model, INPUT_IDS)
    tokens, greedy_margin = run_greedy_with_margin(model, INPUT_IDS, args.n)
    for filename, payload in (
        ("input_ids.json", INPUT_IDS),
        ("reference_logits.json", logits),
        ("reference_tokens.json", tokens),
    ):
        with open(output_dir / filename, "w") as output_file:
            json.dump(payload, output_file)

    sorted_logits = np.sort(np.asarray(logits, dtype=np.float32))
    top2_margin = float(sorted_logits[-1] - sorted_logits[-2])
    meta = {
        "seed": args.seed,
        "n_gen": args.n,
        "input_ids": INPUT_IDS,
        "logits_atol_fp32": 1e-2,
        "logits_atol_q40": 5.0,
        "prefix_match_min": 2,
        "top2_logit_margin": top2_margin,
        "minimum_greedy_margin": greedy_margin,
        "transformers_version": transformers.__version__,
        "transformers_commit": args.transformers_commit,
        "torch_version": torch.__version__,
    }
    with open(output_dir / "meta.json", "w") as output_file:
        json.dump(meta, output_file, indent=2)

    print(
        f"[generate] seed={args.seed}, argmax={int(np.argmax(logits))}, "
        f"margin={top2_margin}, greedy_margin={greedy_margin}"
    )
    print(f"[generate] tokens={tokens}")
    print(f"[generate] output={output_dir}")


if __name__ == "__main__":
    main()
