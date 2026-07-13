# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

## @file generate_gemma4_reference.py
## @brief Generate golden fixtures for Gemma4 differential tests.
##
## Creates a tiny Gemma4 model (matching the C++ test adapter dimensions)
## with a fixed random seed, converts its weights to NNTrainer binary format,
## runs a HuggingFace forward pass and greedy generation, then saves results
## as JSON fixtures.
##
## Weight save order MUST match NNTrainer's topological weight-load order
## (= the order layers request weights during incremental_inference), which is
## also the order used by the production converter res/gemma4/weight_converter.py:
##
##   1. embedding0
##   Per decoder layer i:
##   2. layerI_attention_norm
##   3. layerI_wq, layerI_q_norm        ← q_norm immediately after wq
##   4. layerI_wk, layerI_k_norm        ← k_norm immediately after wk
##   5. layerI_wv
##   6. layerI_attention_out
##   7. layerI_post_attention_norm
##   8. layerI_pre_ffn_norm
##   9. layerI_ffn_gate, layerI_ffn_up, layerI_ffn_down
##  10. layerI_post_ffn_norm
##  11. layerI_per_layer_input_gate
##      (layer 0 ONLY, here:) per_layer_input_embedding,
##                            per_layer_input_projection, per_layer_projection_norm
##  12. layerI_per_layer_input_proj
##  13. layerI_post_per_layer_input_norm
##  14. layerI_layer_scalar
##   N. output_norm
##   N+1. output_of_quick_ai
##
## NOTE: the global per-layer-input weights load INSIDE layer 0 (between the
## per_layer_input_gate and per_layer_input_proj), NOT before the decoder loop.
## Writing them too early misaligns every subsequent weight (RMSNorm gammas read
## FC bytes ~0, zeroing activations) and the differential test diverges badly.
##
## Usage:
##   python3 generate_gemma4_reference.py [--out <dir>] [--seed <int>] [--n <int>]

import argparse
import json
import pathlib
import struct

import numpy as np
import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

THIS_DIR = pathlib.Path(__file__).resolve().parent
# generators -> quick_ai_reference -> models -> unittest -> test -> repo root
REPO_ROOT = THIS_DIR.parents[4]
# Fixtures live one level up, next to this generators/ directory.
DEFAULT_OUT = THIS_DIR.parent / "gemma4_tiny"

TINY_TEXT_CONFIG = dict(
    hidden_size=64,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=8,
    global_head_dim=8,
    hidden_size_per_layer_input=32,
    vocab_size_per_layer_input=32,
    vocab_size=32,
    max_position_embeddings=8,
    rms_norm_eps=1e-6,
    rope_theta=1000000,
    sliding_window=4,
    layer_types=["sliding_attention", "full_attention"],
    tie_word_embeddings=True,
    hidden_act="gelu_pytorch_tanh",
    attention_dropout=0.0,
    pad_token_id=0,
    num_kv_shared_layers=0,
    use_double_wide_mlp=False,
)

INPUT_IDS = [1, 4, 2, 3]
N_GEN = 4

TINY_TOKENIZER = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [
        {
            "id": 31,
            "content": "<eos>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
    ],
    "normalizer": None,
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": None,
    "decoder": None,
    "model": {
        "type": "WordLevel",
        "vocab": {
            "<unk>": 0,
            "hello": 1,
            "world": 2,
            **{f"tok{i}": i for i in range(3, 31)},
            "<eos>": 31,
        },
        "unk_token": "<unk>",
    },
}


def save_fp32(f, tensor: torch.Tensor, name: str) -> None:
    arr = tensor.float().detach().numpy().astype(np.float32)
    f.write(arr.tobytes())
    print(f"  {name:55s} shape={list(tensor.shape)}  bytes={arr.nbytes}")


def convert_weights(model: Gemma4TextModel, bin_path: pathlib.Path) -> None:
    """Save weights in NNTrainer topological sort order (= insertion order)."""
    sd = model.state_dict()
    n_layers = model.config.num_hidden_layers

    total_bytes = 0

    def save(tensor: torch.Tensor, name: str) -> None:
        nonlocal total_bytes
        arr = tensor.float().detach().numpy().astype(np.float32)
        f.write(arr.tobytes())
        total_bytes += arr.nbytes
        print(f"  {name:55s} shape={list(tensor.shape)}")

    with open(bin_path, "wb") as f:
        # 1. embedding0  (tie_word_embeddings shares with output_of_quick_ai)
        save(sd["embed_tokens.weight"], "embedding0")

        # Decoder layers 0..n_layers-1
        for i in range(n_layers):
            pfx = f"layers.{i}."

            # attention_norm
            save(sd[f"{pfx}input_layernorm.weight"],
                 f"layer{i}_attention_norm")

            # Attention: NNTrainer loads wq, q_norm, wk, k_norm, wv (the norm of
            # each projection is consumed right after that projection).
            # NNTrainer FC stores [in, out]; HF Linear.weight is [out, in].
            save(sd[f"{pfx}self_attn.q_proj.weight"].T.contiguous(),
                 f"layer{i}_wq")
            save(sd[f"{pfx}self_attn.q_norm.weight"], f"layer{i}_q_norm")
            save(sd[f"{pfx}self_attn.k_proj.weight"].T.contiguous(),
                 f"layer{i}_wk")
            save(sd[f"{pfx}self_attn.k_norm.weight"], f"layer{i}_k_norm")
            save(sd[f"{pfx}self_attn.v_proj.weight"].T.contiguous(),
                 f"layer{i}_wv")

            # attention_out
            save(sd[f"{pfx}self_attn.o_proj.weight"].T.contiguous(),
                 f"layer{i}_attention_out")

            # post_attention_norm, pre_ffn_norm
            save(sd[f"{pfx}post_attention_layernorm.weight"],
                 f"layer{i}_post_attention_norm")
            save(sd[f"{pfx}pre_feedforward_layernorm.weight"],
                 f"layer{i}_pre_ffn_norm")

            # ffn_gate, ffn_up, ffn_down
            save(sd[f"{pfx}mlp.gate_proj.weight"].T.contiguous(),
                 f"layer{i}_ffn_gate")
            save(sd[f"{pfx}mlp.up_proj.weight"].T.contiguous(),
                 f"layer{i}_ffn_up")
            save(sd[f"{pfx}mlp.down_proj.weight"].T.contiguous(),
                 f"layer{i}_ffn_down")

            # post_ffn_norm
            save(sd[f"{pfx}post_feedforward_layernorm.weight"],
                 f"layer{i}_post_ffn_norm")

            # per_layer_input_gate: FC [in=hidden, out=ple_dim]
            save(sd[f"{pfx}per_layer_input_gate.weight"].T.contiguous(),
                 f"layer{i}_per_layer_input_gate")

            # Global per-layer-input weights are created before the decoder loop
            # in C++, but they are first *consumed* inside layer 0's per-layer
            # path, so NNTrainer loads them here (only once, during layer 0).
            if i == 0:
                save(sd["embed_tokens_per_layer.weight"],
                     "per_layer_input_embedding")
                # HF [out=n_layers*ple_dim, in=hidden] → NNTrainer FC [in, out]
                save(sd["per_layer_model_projection.weight"].T.contiguous(),
                     "per_layer_input_projection")
                save(sd["per_layer_projection_norm.weight"],
                     "per_layer_projection_norm")

            # per_layer_input_proj: FC [in=ple_dim, out=hidden]
            save(sd[f"{pfx}per_layer_projection.weight"].T.contiguous(),
                 f"layer{i}_per_layer_input_proj")

            # post_per_layer_input_norm
            save(sd[f"{pfx}post_per_layer_input_norm.weight"],
                 f"layer{i}_post_per_layer_input_norm")

            # layer_scalar (buffer, shape [1])
            save(sd[f"{pfx}layer_scalar"], f"layer{i}_layer_scalar")

        # output_norm
        save(sd["norm.weight"], "output_norm")

        # output_of_quick_ai (tie_word_embeddings: same weight as embedding0)
        save(sd["embed_tokens.weight"], "output_of_quick_ai")

    size_kb = bin_path.stat().st_size / 1024
    print(f"[converter] saved {bin_path} ({size_kb:.1f} KB, {total_bytes} bytes)")


def build_tiny_config() -> Gemma4TextConfig:
    return Gemma4TextConfig(**TINY_TEXT_CONFIG)


def build_model(config: Gemma4TextConfig, seed: int) -> Gemma4TextModel:
    torch.manual_seed(seed)
    model = Gemma4TextModel(config)
    model.eval()
    return model


def run_forward(model: Gemma4TextModel, input_ids: list) -> list:
    ids = torch.tensor([input_ids], dtype=torch.long)
    with torch.no_grad():
        out = model(ids, use_cache=False)
    hidden = out.last_hidden_state[0, -1, :]
    logits = (hidden @ model.embed_tokens.weight.T).float().tolist()
    return logits


def run_greedy(model: Gemma4TextModel, input_ids: list, n: int) -> list:
    ids = list(input_ids)
    generated = []
    with torch.no_grad():
        for _ in range(n):
            inp = torch.tensor([ids], dtype=torch.long)
            out = model(inp, use_cache=False)
            hidden = out.last_hidden_state[0, -1, :]
            logits = hidden @ model.embed_tokens.weight.T
            next_tok = int(logits.argmax().item())
            generated.append(next_tok)
            ids.append(next_tok)
    return generated


def write_nntr_configs(out_dir: pathlib.Path, bin_name: str,
                       tokenizer_path: str) -> None:
    config_json = {
        "architectures": ["Gemma4ForCausalLM"],
        "bos_token_id": 0,
        "eos_token_id": [31],
        "num_hidden_layers": 2,
        "text_config": {
            "head_dim": 8,
            "global_head_dim": 8,
            "hidden_size": 64,
            "hidden_size_per_layer_input": 32,
            "intermediate_size": 64,
            "layer_types": ["sliding_attention", "full_attention"],
            "max_position_embeddings": 8,
            "num_attention_heads": 8,
            "num_hidden_layers": 2,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000,
            # Per-attention-type RoPE (mirrors HF Gemma4 config.rope_scaling).
            # nntrainer reads this under the "rope_parameters" key; omitting it
            # makes the sliding layer use theta=1e6 instead of 10000 (and the
            # full layer skip partial_rotary_factor), which diverges from HF.
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
            "sliding_window": 4,
            "tie_word_embeddings": True,
            "vocab_size": 32,
            "vocab_size_per_layer_input": 32,
        },
    }
    generation_config_json = {
        "bos_token_id": 0,
        "eos_token_id": 31,
        "do_sample": False,
        "top_k": 1,
        "top_p": 1.0,
        "temperature": 1.0,
    }
    nntr_config_json = {
        "bad_word_ids": [],
        "batch_size": 1,
        "embedding_dtype": "FP32",
        "fc_layer_dtype": "FP32",
        "init_seq_len": 4,
        "lmhead_dtype": "FP32",
        "max_seq_len": 8,
        "model_file_name": bin_name,
        "model_tensor_type": "FP32-FP32",
        "model_type": "quick_ai",
        "num_to_generate": 1,
        "tokenizer_file": tokenizer_path,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_json, f, indent=2)
    with open(out_dir / "generation_config.json", "w") as f:
        json.dump(generation_config_json, f, indent=2)
    with open(out_dir / "nntr_config.json", "w") as f:
        json.dump(nntr_config_json, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Gemma4 tiny reference fixtures")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    # seed 18: the tiny model's top logits are far enough apart that Q4_0
    # quantization noise does not flip the argmax (greedy stays stable).
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--n", type=int, default=N_GEN)
    args = parser.parse_args()

    out_dir: pathlib.Path = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[generate] output dir: {out_dir}")
    print(f"[generate] seed={args.seed}, n_gen={args.n}, input_ids={INPUT_IDS}")

    config = build_tiny_config()
    model = build_model(config, args.seed)
    print(f"[generate] model params: {sum(p.numel() for p in model.parameters()):,}")

    bin_name = "nntr_gemma4_tiny_fp32.bin"
    bin_path = out_dir / bin_name
    print("[generate] weight save order:")
    convert_weights(model, bin_path)

    tokenizer_path = out_dir / "tokenizer.json"
    with open(tokenizer_path, "w") as f:
        json.dump(TINY_TOKENIZER, f, indent=2)

    write_nntr_configs(out_dir, bin_name, str(tokenizer_path))

    ref_logits = run_forward(model, INPUT_IDS)
    with open(out_dir / "reference_logits.json", "w") as f:
        json.dump(ref_logits, f)
    argmax_tok = int(np.argmax(ref_logits))
    print(f"[generate] reference logits: argmax={argmax_tok}")

    ref_tokens = run_greedy(model, INPUT_IDS, args.n)
    with open(out_dir / "reference_tokens.json", "w") as f:
        json.dump(ref_tokens, f)
    print(f"[generate] reference tokens: {ref_tokens}")

    with open(out_dir / "input_ids.json", "w") as f:
        json.dump(INPUT_IDS, f)

    import transformers
    meta = {
        "seed": args.seed,
        "n_gen": args.n,
        "input_ids": INPUT_IDS,
        "logits_atol_fp32": 1e-2,
        "logits_atol_q40": 5.0,
        "prefix_match_min": 2,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[generate] done. transformers={transformers.__version__}")


if __name__ == "__main__":
    main()
