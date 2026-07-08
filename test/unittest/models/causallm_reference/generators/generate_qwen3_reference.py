# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

## @file generate_qwen3_reference.py
## @brief Generate golden fixtures for Qwen3 differential tests.
##
## This script creates a tiny Qwen3 model (matching the C++ test adapter
## dimensions) with a fixed random seed, converts its weights via the
## existing weight_converter, runs a HuggingFace forward pass and greedy
## generation, then saves the results as JSON fixtures.
##
## Usage:
##   python3 generate_qwen3_reference.py [--out <dir>] [--seed <int>] [--n <int>]
##
## The default output directory is:
##   test/unittest/models/causallm_reference/qwen3_tiny/
## (relative to the nntrainer repo root, resolved from this file's location).
##
## Requirements: torch >= 2.0, transformers >= 4.51 (Qwen3 support)

import argparse
import json
import os
import sys
import pathlib

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

# ---------------------------------------------------------------------------
# Repo layout helpers
# ---------------------------------------------------------------------------

THIS_DIR = pathlib.Path(__file__).resolve().parent
# generators -> causallm_reference -> models -> unittest -> test -> repo root
REPO_ROOT = THIS_DIR.parents[4]
CONVERTER_DIR = (REPO_ROOT / "Applications" / "CausalLM" / "res" / "qwen3"
                 / "qwen3-0.6b")
# Fixtures live one level up, next to this generators/ directory.
DEFAULT_OUT = THIS_DIR.parent / "qwen3_tiny"

sys.path.insert(0, str(CONVERTER_DIR))
from weight_converter import save_qwen3_for_nntrainer  # noqa: E402

# ---------------------------------------------------------------------------
# Tiny model dimensions — must match makeTinyQwen3Config() in C++
# ---------------------------------------------------------------------------

TINY_CONFIG = dict(
    hidden_size=64,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=8,
    vocab_size=32,
    max_position_embeddings=8,
    rope_theta=10000.0,
    rms_norm_eps=1e-5,
    tie_word_embeddings=True,
    # Disable dropout/cache features that change numerical results
    attention_dropout=0.0,
    hidden_act="silu",
)

# Fixed input token IDs — all < vocab_size (32), length == init_seq_len (4)
INPUT_IDS = [1, 4, 2, 3]

# 32-token input for flash attention tests (step_size >= FLASH_MIN_PREFILL=32)
# Cycles through tokens 1-30, avoids eos (31) to prevent early generation stop
FLASH_INPUT_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    1, 2,
]

# Number of greedy tokens to generate
N_GEN = 4

# ---------------------------------------------------------------------------
# Tiny tokenizer JSON — matches writeTinyTokenizer() in C++
# ---------------------------------------------------------------------------

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


def build_tiny_config(max_position_embeddings: int = 8) -> Qwen3Config:
    cfg = dict(**TINY_CONFIG)
    cfg["max_position_embeddings"] = max_position_embeddings
    return Qwen3Config(**cfg)


def build_model(config: Qwen3Config, seed: int) -> Qwen3ForCausalLM:
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(config)
    model.eval()
    return model


def convert_weights(model: Qwen3ForCausalLM, config: Qwen3Config, bin_path: pathlib.Path) -> None:
    """Write weights in nntrainer binary format using the existing converter."""
    params = model.state_dict()
    tie = getattr(config, "tie_word_embeddings", True)
    with open(bin_path, "wb") as f:
        save_qwen3_for_nntrainer(params, config.num_hidden_layers, "float32", f, tie)
    print(f"[converter] saved {bin_path} ({bin_path.stat().st_size / 1024:.1f} KB)")


def run_forward(model: Qwen3ForCausalLM, input_ids: list[int]) -> list[float]:
    """Return last-token logits from a forward pass (no KV-cache tricks)."""
    ids = torch.tensor([input_ids], dtype=torch.long)
    with torch.no_grad():
        out = model(ids, use_cache=False)
    # logits: [1, seq_len, vocab_size] — take the last token
    logits = out.logits[0, -1, :].float().tolist()
    return logits


def run_greedy(model: Qwen3ForCausalLM, input_ids: list[int], n: int) -> list[int]:
    """Return n greedy tokens generated after the prompt."""
    ids = torch.tensor([input_ids], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=n,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.0,
            use_cache=True,
            eos_token_id=None,  # disable early stopping so we always get n tokens
        )
    # out: [1, init_len + n]
    tokens = out[0, len(input_ids):].tolist()
    return tokens[:n]


def write_nntr_configs(
    out_dir: pathlib.Path,
    bin_name: str,
    tokenizer_path: str,
    max_position_embeddings: int = 8,
    init_seq_len: int = 4,
    max_seq_len: int = 8,
    use_flash_attention: bool = False,
) -> None:
    config_json = {
        "architectures": ["Qwen3ForCausalLM"],
        "bos_token_id": 0,
        "eos_token_id": [31],
        "head_dim": 8,
        "hidden_size": 64,
        "intermediate_size": 64,
        "is_causal": True,
        "max_position_embeddings": max_position_embeddings,
        "num_attention_heads": 8,
        "num_hidden_layers": 1,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000,
        "tie_word_embeddings": True,
        "vocab_size": 32,
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
        "init_seq_len": init_seq_len,
        "lmhead_dtype": "FP32",
        "max_seq_len": max_seq_len,
        "model_file_name": bin_name,
        "model_tensor_type": "FP32-FP32",
        "model_type": "CausalLM",
        "num_to_generate": 1,
        "tokenizer_file": tokenizer_path,
    }
    if use_flash_attention:
        nntr_config_json["use_flash_attention"] = True

    with open(out_dir / "config.json", "w") as f:
        json.dump(config_json, f, indent=2)
    with open(out_dir / "generation_config.json", "w") as f:
        json.dump(generation_config_json, f, indent=2)
    with open(out_dir / "nntr_config.json", "w") as f:
        json.dump(nntr_config_json, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Qwen3 tiny reference fixtures")
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help="Output directory (default: qwen3_tiny or qwen3_flash_tiny)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n", type=int, default=N_GEN,
                        help="Number of greedy tokens to generate")
    parser.add_argument("--flash", action="store_true",
                        help="Generate 32-token flash-attention fixture (qwen3_flash_tiny)")
    parser.add_argument("--max-pos-emb", type=int, default=None,
                        help="Override max_position_embeddings (default: 8, or 64 for --flash)")
    args = parser.parse_args()

    # Select parameters based on --flash flag
    if args.flash:
        input_ids = FLASH_INPUT_IDS
        max_pos_emb = args.max_pos_emb if args.max_pos_emb is not None else 64
        init_seq_len = 32
        max_seq_len = 64
        bin_name = "nntr_qwen3_flash_tiny_fp32.bin"
        default_out = THIS_DIR.parent / "qwen3_flash_tiny"
    else:
        input_ids = INPUT_IDS
        max_pos_emb = args.max_pos_emb if args.max_pos_emb is not None else 8
        init_seq_len = 4
        max_seq_len = 8
        bin_name = "nntr_qwen3_tiny_fp32.bin"
        default_out = DEFAULT_OUT

    out_dir: pathlib.Path = (args.out if args.out is not None else default_out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[generate] output dir: {out_dir}")
    print(f"[generate] flash={args.flash}, seed={args.seed}, n_gen={args.n}")
    print(f"[generate] input_ids={input_ids} (len={len(input_ids)})")
    print(f"[generate] max_pos_emb={max_pos_emb}, init_seq_len={init_seq_len}, max_seq_len={max_seq_len}")

    # --- Build model ---
    config = build_tiny_config(max_position_embeddings=max_pos_emb)
    model = build_model(config, args.seed)
    print(f"[generate] model params: {sum(p.numel() for p in model.parameters()):,}")

    # --- Convert weights ---
    bin_path = out_dir / bin_name
    convert_weights(model, config, bin_path)

    # --- Write tokenizer ---
    tokenizer_path = out_dir / "tokenizer.json"
    with open(tokenizer_path, "w") as f:
        json.dump(TINY_TOKENIZER, f, indent=2)
    print(f"[generate] tokenizer: {tokenizer_path}")

    # --- Write nntrainer configs ---
    write_nntr_configs(
        out_dir, bin_name, str(tokenizer_path),
        max_position_embeddings=max_pos_emb,
        init_seq_len=init_seq_len,
        max_seq_len=max_seq_len,
        use_flash_attention=args.flash,
    )
    print("[generate] config.json / generation_config.json / nntr_config.json written")

    # --- HF forward: reference logits ---
    ref_logits = run_forward(model, input_ids)
    with open(out_dir / "reference_logits.json", "w") as f:
        json.dump(ref_logits, f)
    argmax_tok = int(np.argmax(ref_logits))
    print(f"[generate] reference logits: {len(ref_logits)} values, argmax={argmax_tok}")

    # --- HF greedy: reference tokens ---
    ref_tokens = run_greedy(model, input_ids, args.n)
    with open(out_dir / "reference_tokens.json", "w") as f:
        json.dump(ref_tokens, f)
    print(f"[generate] reference tokens: {ref_tokens}")

    # --- Fixed input_ids ---
    with open(out_dir / "input_ids.json", "w") as f:
        json.dump(input_ids, f)

    # --- Meta ---
    import transformers
    meta = {
        "seed": args.seed,
        "n_gen": args.n,
        "input_ids": input_ids,
        "flash": args.flash,
        "init_seq_len": init_seq_len,
        "max_seq_len": max_seq_len,
        "logits_atol_fp32": 1e-2,
        "logits_atol_q40": 5.0,
        # The Q4_0 greedy prefix check requires this many matching tokens, so it
        # cannot exceed the number of generated reference tokens (n_gen). The
        # flash fixture uses n_gen=1, so clamp to args.n.
        "prefix_match_min": min(2, args.n),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[generate] meta.json written (transformers={transformers.__version__})")

    print("\n[generate] done. Commit the following directory:")
    print(f"  {out_dir}")
    print("\nTo regenerate fixtures:")
    regen_flag = " --flash" if args.flash else ""
    print(f"  python3 {pathlib.Path(__file__).relative_to(REPO_ROOT)}{regen_flag}")


if __name__ == "__main__":
    main()
