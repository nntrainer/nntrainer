# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
## @file generate_gemma4_htp_moe_reference.py
## @brief Golden fixtures for the Gemma-4 MoE (enable_moe_block) differential
##        test: gemma-4-26B-A4B's text_config shape made tiny.
##
## Same recipe as generate_gemma4_reference.py (random tiny HF model, weights
## saved in nntrainer's load order, HF logits and greedy tokens as JSON), with
## what the 26B adds (docs/htp_attention/55_gemma4_moe_htp_task.md):
##   - a router + experts beside the dense MLP in every layer,
##   - attention_k_eq_v: the full-attention layer has no v_proj (V is K),
##   - hidden_size_per_layer_input = 0: no per-layer input path,
##   - use_bidirectional_attention = "vision" (text stays causal).
##
## Weight save order per layer (unittest_causallm_gemma4_moe's
## WeightBearingLayerOrderMatchesConverter holds the same list):
##   attention_norm, wq, q_norm, wk, k_norm, [wv: sliding only],
##   attention_out, post_attention_norm, pre_ffn_norm,
##   ffn_gate, ffn_up, ffn_down,
##   post_ffn_norm_1, pre_ffn_norm_2, router_norm (= router.scale * H^-0.5),
##   sparse_moe: router.proj.T, per_expert_scale, per expert gate_up[e].T,
##               down[e].T,
##   post_ffn_norm_2, post_ffn_norm, layer_scalar
##
## Usage:
##   python3 generate_gemma4_htp_moe_reference.py [--out <dir>] [--seed <int>]

import argparse
import json
import pathlib

import numpy as np
import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

THIS_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_OUT = THIS_DIR.parent / "gemma4_htp_moe_tiny"

HIDDEN = 64
NUM_LAYERS = 2
LAYER_TYPES = ["sliding_attention", "full_attention"]
NUM_EXPERTS = 4
TOP_K = 2
MOE_INTER = 64

TINY_TEXT_CONFIG = dict(
    hidden_size=HIDDEN,
    intermediate_size=64,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=8,
    global_head_dim=8,
    num_global_key_value_heads=2,
    attention_k_eq_v=True,
    enable_moe_block=True,
    num_experts=NUM_EXPERTS,
    top_k_experts=TOP_K,
    moe_intermediate_size=MOE_INTER,
    hidden_size_per_layer_input=0,
    vocab_size_per_layer_input=32,
    vocab_size=32,
    max_position_embeddings=8,
    rms_norm_eps=1e-6,
    rope_theta=1000000,
    sliding_window=4,
    layer_types=LAYER_TYPES,
    tie_word_embeddings=True,
    hidden_activation="gelu_pytorch_tanh",
    attention_dropout=0.0,
    pad_token_id=0,
    num_kv_shared_layers=0,
    use_double_wide_mlp=False,
    use_bidirectional_attention="vision",
)

K_EQ_V = True
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


def convert_weights(model: Gemma4TextModel, bin_path: pathlib.Path) -> None:
    sd = model.state_dict()
    total = 0

    with open(bin_path, "wb") as f:

        def save(tensor: torch.Tensor, name: str) -> None:
            nonlocal total
            arr = tensor.float().detach().contiguous().numpy().astype(np.float32)
            f.write(arr.tobytes())
            total += arr.nbytes
            print(f"  {name:40s} shape={list(tensor.shape)}")

        save(sd["embed_tokens.weight"], "embedding0")
        for i in range(NUM_LAYERS):
            p = f"layers.{i}."
            full = LAYER_TYPES[i] == "full_attention"
            save(sd[f"{p}input_layernorm.weight"], f"layer{i}_attention_norm")
            save(sd[f"{p}self_attn.q_proj.weight"].T, f"layer{i}_wq")
            save(sd[f"{p}self_attn.q_norm.weight"], f"layer{i}_q_norm")
            save(sd[f"{p}self_attn.k_proj.weight"].T, f"layer{i}_wk")
            save(sd[f"{p}self_attn.k_norm.weight"], f"layer{i}_k_norm")
            if not (K_EQ_V and full):  # attention_k_eq_v: no v_proj on full
                save(sd[f"{p}self_attn.v_proj.weight"].T, f"layer{i}_wv")
            save(sd[f"{p}self_attn.o_proj.weight"].T, f"layer{i}_attention_out")
            save(sd[f"{p}post_attention_layernorm.weight"],
                 f"layer{i}_post_attention_norm")
            save(sd[f"{p}pre_feedforward_layernorm.weight"],
                 f"layer{i}_pre_ffn_norm")
            save(sd[f"{p}mlp.gate_proj.weight"].T, f"layer{i}_ffn_gate")
            save(sd[f"{p}mlp.up_proj.weight"].T, f"layer{i}_ffn_up")
            save(sd[f"{p}mlp.down_proj.weight"].T, f"layer{i}_ffn_down")
            # the MoE half, as res/gemma4/htp_weight_converter.py writes it
            save(sd[f"{p}post_feedforward_layernorm_1.weight"],
                 f"layer{i}_post_ffn_norm_1")
            save(sd[f"{p}pre_feedforward_layernorm_2.weight"],
                 f"layer{i}_pre_ffn_norm_2")
            save(sd[f"{p}router.scale"] * HIDDEN ** -0.5,
                 f"layer{i}_router_norm")
            save(sd[f"{p}router.proj.weight"].T, f"layer{i}_sparse_moe:gate")
            save(sd[f"{p}router.per_expert_scale"],
                 f"layer{i}_sparse_moe:expert_bias")
            gate_up = sd[f"{p}experts.gate_up_proj"]  # [E, 2I, H]
            down = sd[f"{p}experts.down_proj"]  # [E, H, I]
            for e in range(NUM_EXPERTS):
                save(gate_up[e].T, f"layer{i}_sparse_moe:expert_gate_up_{e}")
                save(down[e].T, f"layer{i}_sparse_moe:expert_down_{e}")
            save(sd[f"{p}post_feedforward_layernorm_2.weight"],
                 f"layer{i}_post_ffn_norm_2")
            save(sd[f"{p}post_feedforward_layernorm.weight"],
                 f"layer{i}_post_ffn_norm")
            save(sd[f"{p}layer_scalar"], f"layer{i}_layer_scalar")
        save(sd["norm.weight"], "output_norm")
        save(sd["embed_tokens.weight"], "output_of_causallm")

    print(f"[converter] saved {bin_path} ({total} bytes)")


def run_forward(model, input_ids):
    ids = torch.tensor([input_ids], dtype=torch.long)
    with torch.no_grad():
        out = model(ids, use_cache=False)
    hidden = out.last_hidden_state[0, -1, :]
    return (hidden @ model.embed_tokens.weight.T).float().tolist()


def run_greedy(model, input_ids, n):
    ids = list(input_ids)
    generated = []
    with torch.no_grad():
        for _ in range(n):
            out = model(torch.tensor([ids], dtype=torch.long), use_cache=False)
            logits = out.last_hidden_state[0, -1, :] @ model.embed_tokens.weight.T
            tok = int(logits.argmax().item())
            generated.append(tok)
            ids.append(tok)
    return generated


def write_nntr_configs(out_dir, bin_name, tokenizer_path):
    text = dict(TINY_TEXT_CONFIG)
    text.pop("hidden_activation")
    text.pop("attention_dropout")
    text.pop("pad_token_id")
    text["rope_parameters"] = {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000},
        "full_attention": {"rope_type": "proportional", "rope_theta": 1000000,
                           "partial_rotary_factor": 0.25},
    }
    config_json = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "bos_token_id": 0,
        "eos_token_id": [31],
        "text_config": text,
    }
    generation_config_json = {
        "bos_token_id": 0, "eos_token_id": 31, "do_sample": False,
        "top_k": 1, "top_p": 1.0, "temperature": 1.0,
    }
    nntr_config_json = {
        "gemma4_execution_profile": "htp_fused",
        "gemma4_weight_format": "htp_fused_v1",
        "bad_word_ids": [], "batch_size": 1,
        "embedding_dtype": "FP32", "fc_layer_dtype": "FP32",
        "init_seq_len": 4, "lmhead_dtype": "FP32", "max_seq_len": 8,
        "model_file_name": bin_name, "model_tensor_type": "FP32-FP32",
        "model_type": "CausalLM", "num_to_generate": 1,
        "tokenizer_file": tokenizer_path,
    }
    json.dump(config_json, open(out_dir / "config.json", "w"), indent=2)
    json.dump(generation_config_json,
              open(out_dir / "generation_config.json", "w"), indent=2)
    json.dump(nntr_config_json, open(out_dir / "nntr_config.json", "w"),
              indent=2)


def make_reference_model(seed=18):
    config = Gemma4TextConfig(**TINY_TEXT_CONFIG)
    torch.manual_seed(seed)
    model = Gemma4TextModel(config)
    # Random experts and router: Gemma4's init leaves per_expert_scale and
    # router.scale at ones and layer_scalar at one; the experts get the
    # module's normal init. Spread the scales so a wrong wiring shows.
    with torch.no_grad():
        for layer in model.layers:
            layer.router.per_expert_scale.copy_(
                0.5 + torch.rand(NUM_EXPERTS))
            layer.router.scale.copy_(0.5 + torch.rand(HIDDEN))
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=18)
    ap.add_argument("--n", type=int, default=N_GEN)
    ap.add_argument("--global-kv", type=int, default=2,
                    help="debug variant: num_global_key_value_heads")
    ap.add_argument("--no-k-eq-v", action="store_true",
                    help="debug variant: v_proj on every layer, 4 global kv heads")
    args = ap.parse_args()
    TINY_TEXT_CONFIG["num_global_key_value_heads"] = args.global_kv
    if args.no_k_eq_v:
        TINY_TEXT_CONFIG["attention_k_eq_v"] = False
        TINY_TEXT_CONFIG["num_global_key_value_heads"] = 4
        global K_EQ_V
        K_EQ_V = False
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = make_reference_model(args.seed)
    print(f"[generate] params: {sum(p.numel() for p in model.parameters()):,}")

    bin_name = "nntr_gemma4_moe_tiny_fp32.bin"
    convert_weights(model, out_dir / bin_name)
    tok_path = out_dir / "tokenizer.json"
    json.dump(TINY_TOKENIZER, open(tok_path, "w"), indent=2)
    write_nntr_configs(out_dir, bin_name, "tokenizer.json")

    ref_logits = run_forward(model, INPUT_IDS)
    json.dump(ref_logits, open(out_dir / "reference_logits.json", "w"))
    ref_tokens = run_greedy(model, INPUT_IDS, args.n)
    json.dump(ref_tokens, open(out_dir / "reference_tokens.json", "w"))
    json.dump(INPUT_IDS, open(out_dir / "input_ids.json", "w"))
    import transformers
    json.dump({
        "seed": args.seed, "n_gen": args.n, "input_ids": INPUT_IDS,
        "logits_atol_fp32": 1e-2, "logits_atol_q40": 5.0,
        "prefix_match_min": 2,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
    }, open(out_dir / "meta.json", "w"), indent=2)
    print(f"[generate] argmax={int(np.argmax(ref_logits))} tokens={ref_tokens}")


if __name__ == "__main__":
    main()
