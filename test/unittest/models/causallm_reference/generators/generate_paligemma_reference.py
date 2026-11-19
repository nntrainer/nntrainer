# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

## @file generate_paligemma_reference.py
## @brief Generate golden fixtures for the PaliGemma differential tests.
##
## Creates a tiny PaliGemma model (matching the C++ test dimensions) with a
## fixed seed, converts its weights via the existing weight_converter, runs a
## HuggingFace forward pass and greedy generation, then saves the results as
## JSON fixtures.
##
## The fixture prompt carries NO image tokens and token_type_ids all zero, so
## the HF attention mask is one bidirectional block — the same mask the
## nntrainer is_causal=false graph runs for the same token ids. Generation
## steps are causal either way (the last row attends the whole cache).
##
## Usage:
##   python3 generate_paligemma_reference.py [--out <dir>] [--seed <int>] [--n <int>]
##
## The default output directory is:
##   test/unittest/models/causallm_reference/paligemma_tiny/
##
## Requirements: torch >= 2.0, transformers >= 4.45 (PaliGemma support)

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from transformers import PaliGemmaConfig, PaliGemmaForConditionalGeneration

THIS_DIR = pathlib.Path(__file__).resolve().parent
# generators -> causallm_reference -> models -> unittest -> test -> repo root
REPO_ROOT = THIS_DIR.parents[4]
# weight_converter.py lives in res/paligemma
CONVERTER_DIR = REPO_ROOT / "Applications" / "CausalLM" / "res" / "paligemma"
# Fixtures live one level up, next to this generators/ directory.
DEFAULT_OUT = THIS_DIR.parent / "paligemma_tiny"

sys.path.insert(0, str(CONVERTER_DIR))
from weight_converter import save_paligemma_for_nntrainer  # noqa: E402

# Tiny model dimensions — must match makeTinyPaliGemmaConfig() in C++
TINY_TEXT_CONFIG = dict(
    hidden_size=64,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=8,
    vocab_size=32,
    max_position_embeddings=8,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    tie_word_embeddings=True,
    attention_dropout=0.0,
    hidden_activation="gelu_pytorch_tanh",
)

# The vision tower is never exercised by the text-only fixture, but the HF
# model class builds it, so it needs internally consistent tiny dimensions.
TINY_VISION_CONFIG = dict(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    image_size=8,
    patch_size=2,
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


def build_model(seed: int):
    torch.manual_seed(seed)
    num_patches = (TINY_VISION_CONFIG["image_size"] //
                   TINY_VISION_CONFIG["patch_size"]) ** 2
    config = PaliGemmaConfig(
        text_config=TINY_TEXT_CONFIG,
        vision_config=TINY_VISION_CONFIG,
        image_token_index=31,
        num_image_tokens=num_patches,
        projection_dim=TINY_TEXT_CONFIG["hidden_size"],
        vocab_size=TINY_TEXT_CONFIG["vocab_size"],
        bos_token_id=0,
        eos_token_id=31,
        pad_token_id=0,
    )
    model = PaliGemmaForConditionalGeneration(config)
    model.eval()
    return model, config


def convert_weights(model, bin_path: pathlib.Path) -> None:
    params = model.state_dict()
    # Tied lm_head: the C++ tie_word_embeddings layer shares embedding0,
    # so only the embedding table is saved (the converter never writes
    # lm_head for PaliGemma).
    with open(bin_path, "wb") as f:
        save_paligemma_for_nntrainer(params, model.config.text_config, f)
    print(f"[converter] saved {bin_path} ({bin_path.stat().st_size / 1024:.1f} KB)")


def run_forward(model, input_ids: list[int]) -> list[float]:
    ids = torch.tensor([input_ids], dtype=torch.long)
    token_type_ids = torch.zeros_like(ids)  # every row is the prefix block
    with torch.no_grad():
        out = model(input_ids=ids, token_type_ids=token_type_ids,
                    use_cache=False)
    return out.logits[0, -1, :].float().tolist()


def run_greedy(model, input_ids: list[int], n: int) -> list[int]:
    ids = torch.tensor([input_ids], dtype=torch.long)
    token_type_ids = torch.zeros_like(ids)
    attention_mask = torch.ones_like(ids)
    with torch.no_grad():
        out = model.generate(
            input_ids=ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            max_new_tokens=n,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.0,
            use_cache=True,
            eos_token_id=None,
        )
    return out[0, len(input_ids):].tolist()[:n]


def write_nntr_configs(out_dir: pathlib.Path, bin_name: str,
                       tokenizer_path: str) -> None:
    config_json = {
        "architectures": ["PaliGemmaForConditionalGeneration"],
        "bos_token_id": 0,
        "eos_token_id": [31],
        "head_dim": 8,
        "hidden_size": 64,
        "image_token_index": 31,
        "intermediate_size": 64,
        "max_position_embeddings": 8,
        "num_attention_heads": 8,
        "num_hidden_layers": 2,
        "num_key_value_heads": 4,
        "projection_dim": 64,
        "rms_norm_eps": 1e-6,
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
        "init_seq_len": 4,
        "lmhead_dtype": "FP32",
        "max_seq_len": 8,
        "model_file_name": bin_name,
        "model_tensor_type": "FP32-FP32",
        "model_type": "CausalLM",
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
        description="Generate PaliGemma tiny reference fixtures")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=N_GEN)
    args = parser.parse_args()

    out_dir: pathlib.Path = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[generate] output dir: {out_dir}")

    model, config = build_model(args.seed)
    print(f"[generate] model params: {sum(p.numel() for p in model.parameters()):,}")

    bin_name = "nntr_paligemma_tiny_fp32.bin"
    convert_weights(model, out_dir / bin_name)

    tokenizer_path = out_dir / "tokenizer.json"
    with open(tokenizer_path, "w") as f:
        json.dump(TINY_TOKENIZER, f, indent=2)

    write_nntr_configs(out_dir, bin_name, str(tokenizer_path))

    ref_logits = run_forward(model, INPUT_IDS)
    with open(out_dir / "reference_logits.json", "w") as f:
        json.dump(ref_logits, f)
    print(f"[generate] reference logits: {len(ref_logits)} values, "
          f"argmax={int(np.argmax(ref_logits))}, "
          f"max|logit|={max(abs(v) for v in ref_logits):.4f}")

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
        # Q4_0 tolerance: tune by running the differential test once and
        # reading the reported max deviation; set to observed x safety factor.
        "logits_atol_q40": 5.0,
        "prefix_match_min": 2,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[generate] done. Commit: {out_dir}")


if __name__ == "__main__":
    main()
