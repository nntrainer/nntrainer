# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

## @file generate_lfm2_moe_reference.py
## @brief Generate golden fixtures for LFM2-MoE tiny differential tests.
##
## Creates a tiny LFM2-MoE model (matching the tiny C++ config) with a fixed
## seed, runs a pure-PyTorch forward pass that mirrors nntrainer's LFM2-MoE
## computation (hybrid conv/attention backbone + sigmoid-routed MoE with an
## expert bias used only for top-k selection), then saves weights and reference
## outputs as JSON + binary fixtures.
##
## Usage:
##   python3 generate_lfm2_moe_reference.py [--out <dir>] [--seed <int>] [--n <int>]
##
## Default output: test/unittest/models/causallm_reference/lfm2_moe_tiny/
##
## Binary layout (USE_EMBEDDING=false, tie_word_embeddings=true):
##   [embed_tokens.T] [per-layer weights] [embedding_norm]
##   Dense layers (id < num_dense_layers) use the dense SwiGLU FFN; MoE layers
##   emit: router gate.T, expert_bias, then per expert
##         cat(gate, up).T, down.T
##   (matching Lfm2MoELayer::finalize weight request order).
##
## Requirements: torch >= 2.0, numpy

import argparse
import json
import pathlib

import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_OUT = THIS_DIR.parent / "lfm2_moe_tiny"

# ---------- Config (must match the tiny C++ Lfm2Moe config) ----------------

DIM          = 64
INTERMEDIATE = 64        # dense FFN intermediate size
N_LAYERS     = 3
LAYER_TYPES  = ["conv", "attention", "conv"]
N_HEADS      = 8
N_KV_HEADS   = 4
HEAD_DIM     = 8
GQA_SIZE     = N_HEADS // N_KV_HEADS   # = 2
VOCAB        = 32
MAX_POS      = 8
ROPE_THETA   = 10000.0
RMS_EPS      = 1e-6
CONV_DIM     = 64
CONV_DIM_OUT = 64
CONV_K       = 3        # conv_L_cache
TIE_EMBED    = True

# MoE-specific config
NUM_EXPERTS          = 4
NUM_EXPERTS_PER_TOK  = 2
MOE_INTERMEDIATE     = 32       # divisible by 32 (Q4_0-friendly)
NUM_DENSE_LAYERS     = 1        # layers [0, NUM_DENSE_LAYERS) are dense
USE_EXPERT_BIAS      = True
NORM_TOPK_PROB       = True
ROUTED_SCALING       = 1.0

INPUT_IDS = [1, 4, 2, 3]
N_GEN     = 4
SEED      = 42


def is_moe_layer(i: int) -> bool:
    return i >= NUM_DENSE_LAYERS

# ---------- Tiny tokenizer (matches C++ writeTinyTokenizer) ----------------

TINY_TOKENIZER = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [
        {"id": 31, "content": "<eos>", "single_word": False, "lstrip": False,
         "rstrip": False, "normalized": False, "special": True},
    ],
    "normalizer": None,
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": None,
    "decoder": None,
    "model": {
        "type": "WordLevel",
        "vocab": {
            "<unk>": 0, "hello": 1, "world": 2,
            **{f"tok{i}": i for i in range(3, 31)},
            "<eos>": 31,
        },
        "unk_token": "<unk>",
    },
}

# ---------- PyTorch primitives matching nntrainer --------------------------

def rms_norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """RMS norm. x: [B, T, D], w: [D]."""
    rms = x.float().pow(2).mean(-1, keepdim=True).add(RMS_EPS).rsqrt()
    return (x.float() * rms * w.float()).to(x.dtype)


def per_head_rms_norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Reshaped RMS norm (per-head). x: [B, T, n_heads*head_dim], w: [head_dim]."""
    B, T, D = x.shape
    n_heads = D // HEAD_DIM
    xf = x.float().view(B, T, n_heads, HEAD_DIM)
    rms = xf.pow(2).mean(-1, keepdim=True).add(RMS_EPS).rsqrt()
    return (xf * rms * w.float()).view(B, T, D).to(x.dtype)


def apply_rope(q: torch.Tensor, k: torch.Tensor, T: int):
    """Standard rotary position embedding. q/k: [B, n, T, head_dim]."""
    device = q.device
    half = HEAD_DIM // 2
    inv_freq = 1.0 / (ROPE_THETA ** (
        torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(T, device=device).float()
    freqs = torch.outer(pos, inv_freq)              # [T, half]
    emb   = torch.cat([freqs, freqs], dim=-1)       # [T, head_dim]
    cos   = emb.cos()[None, None]                   # [1, 1, T, hd]
    sin   = emb.sin()[None, None]

    def rot_half(v):
        h = v.shape[-1] // 2
        return torch.cat([-v[..., h:], v[..., :h]], dim=-1)

    return ((q * cos + rot_half(q) * sin).to(q.dtype),
            (k * cos + rot_half(k) * sin).to(k.dtype))


def gqa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Grouped-query causal attention. q:[B,nq,T,hd] k/v:[B,nkv,T,hd]."""
    B, nq, T, hd = q.shape
    nkv = k.shape[1]
    g   = nq // nkv
    k = k.unsqueeze(2).expand(B, nkv, g, T, hd).reshape(B, nq, T, hd)
    v = v.unsqueeze(2).expand(B, nkv, g, T, hd).reshape(B, nq, T, hd)
    scores = torch.einsum("bnid,bnjd->bnij", q, k) * (hd ** -0.5)
    mask   = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
    scores.masked_fill_(mask[None, None], float("-inf"))
    attn = F.softmax(scores.float(), dim=-1).to(q.dtype)
    out  = torch.einsum("bnij,bnjd->bnid", attn, v)
    return out.transpose(1, 2).reshape(B, T, nq * hd)


def causal_conv1d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Depthwise causal conv1d matching CausalConv1DLayer.
    x: [B, T, F].  w: [K, F] in nntrainer format (row 0=cur, 1=t-1, 2=t-2).
    """
    out = x * w[0]
    if w.shape[0] > 1 and x.shape[1] > 1:
        out = out.clone()
        out[:, 1:] = out[:, 1:] + x[:, :-1] * w[1]
    if w.shape[0] > 2 and x.shape[1] > 2:
        out[:, 2:] = out[:, 2:] + x[:, :-2] * w[2]
    return out


def fc(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Linear with w in PyTorch [out, in] convention. y = x @ w.T."""
    return x @ w.T


def swiglu_ffn(normed: torch.Tensor, w_up, w_gate, w_down) -> torch.Tensor:
    """SwiGLU FFN: silu(gate) * up -> down."""
    up   = fc(normed, w_up)
    gate = fc(normed, w_gate)
    return fc(F.silu(gate) * up, w_down)


def moe_ffn(normed: torch.Tensor, gate_w, expert_bias, experts) -> torch.Tensor:
    """LFM2-MoE routing matching Lfm2MoELayer.

    normed: [B, T, DIM]. gate_w: [num_experts, DIM]. expert_bias: [num_experts].
    experts: list of (w_up, w_gate, w_down) per expert.
      - scores = sigmoid(normed @ gate_w.T)               (bias-free)
      - top-k selection uses scores + expert_bias
      - routing weights are gathered from the bias-free scores, normalised,
        then scaled by ROUTED_SCALING.
    """
    B, T, _ = normed.shape
    router_logits = fc(normed, gate_w)                      # [B, T, E]
    scores = torch.sigmoid(router_logits)                  # bias-free scores
    scores_for_routing = scores + expert_bias if USE_EXPERT_BIAS else scores

    topk_val, topk_idx = torch.topk(scores_for_routing, NUM_EXPERTS_PER_TOK,
                                    dim=-1)                # [B, T, K]
    weights = torch.gather(scores, -1, topk_idx)           # bias-free weights
    if NORM_TOPK_PROB:
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-6)
    weights = weights * ROUTED_SCALING

    out = torch.zeros_like(normed)
    for b in range(B):
        for t in range(T):
            for k in range(NUM_EXPERTS_PER_TOK):
                e = int(topk_idx[b, t, k].item())
                w = weights[b, t, k]
                w_up, w_gate, w_down = experts[e]
                tok = normed[b, t]                          # [DIM]
                h = F.silu(tok @ w_gate.T) * (tok @ w_up.T)  # [MOE_INTER]
                out[b, t] += w * (h @ w_down.T)             # [DIM]
    return out

# ---------- Weight generation ----------------------------------------------

def build_weights(seed: int) -> dict:
    """Generate weight tensors keyed by HF-style names.

    FC weights follow PyTorch convention [out, in].
    Conv weight is generated directly in nntrainer [K, F] format.
    Norm gammas are ones (identity scaling); FC/expert weights use small
    Gaussian initialisation; the expert bias uses a small non-zero init so it
    influences top-k selection.
    """
    torch.manual_seed(seed)
    rng  = lambda *s: torch.randn(*s) * 0.02
    ones = lambda *s: torch.ones(*s)

    W: dict = {}
    for i, lt in enumerate(LAYER_TYPES):
        W[f"layers.{i}.operator_norm.weight"] = ones(DIM)
        if lt in ("attention", "full_attention"):
            W[f"layers.{i}.self_attn.q_proj.weight"]       = rng(DIM, DIM)
            W[f"layers.{i}.self_attn.q_layernorm.weight"]  = ones(HEAD_DIM)
            W[f"layers.{i}.self_attn.k_proj.weight"]       = rng(DIM // GQA_SIZE, DIM)
            W[f"layers.{i}.self_attn.k_layernorm.weight"]  = ones(HEAD_DIM)
            W[f"layers.{i}.self_attn.v_proj.weight"]       = rng(DIM // GQA_SIZE, DIM)
            W[f"layers.{i}.self_attn.out_proj.weight"]     = rng(DIM, DIM)
        elif lt == "conv":
            W[f"layers.{i}.conv.in_proj.weight"]           = rng(3 * CONV_DIM, DIM)
            W[f"layers.{i}.conv.conv.weight"]              = rng(CONV_K, CONV_DIM)
            W[f"layers.{i}.conv.out_proj.weight"]          = rng(DIM, CONV_DIM)

        W[f"layers.{i}.ffn_norm.weight"] = ones(DIM)
        if is_moe_layer(i):
            # Router gate + per-expert bias + per-expert SwiGLU experts.
            W[f"layers.{i}.moe.gate.weight"]   = rng(NUM_EXPERTS, DIM)
            W[f"layers.{i}.moe.expert_bias"]   = torch.randn(NUM_EXPERTS) * 0.1
            for e in range(NUM_EXPERTS):
                W[f"layers.{i}.moe.experts.{e}.w3"] = rng(MOE_INTERMEDIATE, DIM)  # up
                W[f"layers.{i}.moe.experts.{e}.w1"] = rng(MOE_INTERMEDIATE, DIM)  # gate
                W[f"layers.{i}.moe.experts.{e}.w2"] = rng(DIM, MOE_INTERMEDIATE)  # down
        else:
            W[f"layers.{i}.feed_forward.w3.weight"] = rng(INTERMEDIATE, DIM)  # up
            W[f"layers.{i}.feed_forward.w1.weight"] = rng(INTERMEDIATE, DIM)  # gate
            W[f"layers.{i}.feed_forward.w2.weight"] = rng(DIM, INTERMEDIATE)  # down

    W["embedding_norm.weight"] = ones(DIM)
    W["embed_tokens.weight"]   = rng(VOCAB, DIM)  # [vocab, dim]
    return W

# ---------- Binary save (nntrainer format) ---------------------------------

def save_nntrainer_bin(W: dict, path: pathlib.Path) -> None:
    """Save weights in the order expected by nntrainer's LFM2-MoE weight loader.

    FC weights are transposed on save (nntrainer stores [in, out], tensors here
    are [out, in]). The conv weight is already [K, F]. The embedding is stored
    FIRST because embedding0 is the first weighted layer in the graph.
    MoE layers save: gate.T, expert_bias, then per expert
    cat(gate, up).T, down.T.
    """
    def write(f, t: torch.Tensor) -> None:
        f.write(t.float().contiguous().cpu().numpy().tobytes())

    with open(path, "wb") as f:
        # Embedding first — embedding0 is the first weighted layer in the graph.
        write(f, W["embed_tokens.weight"])     # [vocab, dim] as-is

        for i, lt in enumerate(LAYER_TYPES):
            write(f, W[f"layers.{i}.operator_norm.weight"])
            if lt in ("attention", "full_attention"):
                write(f, W[f"layers.{i}.self_attn.q_proj.weight"].T)
                write(f, W[f"layers.{i}.self_attn.q_layernorm.weight"])
                write(f, W[f"layers.{i}.self_attn.k_proj.weight"].T)
                write(f, W[f"layers.{i}.self_attn.k_layernorm.weight"])
                write(f, W[f"layers.{i}.self_attn.v_proj.weight"].T)
                write(f, W[f"layers.{i}.self_attn.out_proj.weight"].T)
            elif lt == "conv":
                write(f, W[f"layers.{i}.conv.in_proj.weight"].T)
                write(f, W[f"layers.{i}.conv.conv.weight"])            # [K, F] as-is
                write(f, W[f"layers.{i}.conv.out_proj.weight"].T)

            write(f, W[f"layers.{i}.ffn_norm.weight"])
            if is_moe_layer(i):
                write(f, W[f"layers.{i}.moe.gate.weight"].T)          # [DIM, E]
                write(f, W[f"layers.{i}.moe.expert_bias"])            # [E]
                for e in range(NUM_EXPERTS):
                    gate_up = torch.cat(
                        [W[f"layers.{i}.moe.experts.{e}.w1"],
                         W[f"layers.{i}.moe.experts.{e}.w3"]], dim=0)
                    write(f, gate_up.T)                                # gate | up
                    write(f, W[f"layers.{i}.moe.experts.{e}.w2"].T)   # down
            else:
                write(f, W[f"layers.{i}.feed_forward.w3.weight"].T)   # up
                write(f, W[f"layers.{i}.feed_forward.w1.weight"].T)   # gate
                write(f, W[f"layers.{i}.feed_forward.w2.weight"].T)   # down

        # Output norm last (before the tied lm_head which saves nothing)
        write(f, W["embedding_norm.weight"])

    sz = path.stat().st_size
    print(f"[save] {path.name}  ({sz / 1024:.1f} KB,  {sz // 4:,} floats)")

# ---------- Forward pass ---------------------------------------------------

def forward(W: dict, input_ids: list) -> torch.Tensor:
    """LFM2-MoE forward pass matching nntrainer's computation.
    Returns last-token logits: [vocab_size].
    """
    W_embed = W["embed_tokens.weight"]              # [vocab, dim]
    ids = torch.tensor(input_ids, dtype=torch.long)
    B, T = 1, len(ids)

    x = W_embed[ids].unsqueeze(0).float()           # [1, T, DIM]

    for i, lt in enumerate(LAYER_TYPES):
        if lt in ("attention", "full_attention"):
            normed = rms_norm(x, W[f"layers.{i}.operator_norm.weight"])
            q = fc(normed, W[f"layers.{i}.self_attn.q_proj.weight"])
            k = fc(normed, W[f"layers.{i}.self_attn.k_proj.weight"])
            v = fc(normed, W[f"layers.{i}.self_attn.v_proj.weight"])
            q = per_head_rms_norm(q, W[f"layers.{i}.self_attn.q_layernorm.weight"])
            k = per_head_rms_norm(k, W[f"layers.{i}.self_attn.k_layernorm.weight"])
            q = q.view(B, T, N_HEADS,    HEAD_DIM).transpose(1, 2)
            k = k.view(B, T, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
            v = v.view(B, T, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
            q, k = apply_rope(q, k, T)
            attn_out = gqa_attention(q, k, v)       # [B, T, DIM]
            attn_out = fc(attn_out, W[f"layers.{i}.self_attn.out_proj.weight"])
            x = x + attn_out                         # residual

        elif lt == "conv":
            normed = rms_norm(x, W[f"layers.{i}.operator_norm.weight"])
            proj = fc(normed, W[f"layers.{i}.conv.in_proj.weight"])
            a, b, c = proj.split(CONV_DIM, dim=-1)
            gated = a * c                            # gate_a ⊙ gate_c
            conv_out = causal_conv1d(gated, W[f"layers.{i}.conv.conv.weight"])
            gated_out = b * conv_out                 # gate_b ⊙ conv_out
            proj_back = fc(gated_out, W[f"layers.{i}.conv.out_proj.weight"])
            x = x + proj_back                        # conv residual

        # FFN (dense SwiGLU or MoE)
        normed = rms_norm(x, W[f"layers.{i}.ffn_norm.weight"])
        if is_moe_layer(i):
            experts = [
                (W[f"layers.{i}.moe.experts.{e}.w3"],
                 W[f"layers.{i}.moe.experts.{e}.w1"],
                 W[f"layers.{i}.moe.experts.{e}.w2"])
                for e in range(NUM_EXPERTS)
            ]
            x = x + moe_ffn(normed, W[f"layers.{i}.moe.gate.weight"],
                            W[f"layers.{i}.moe.expert_bias"], experts)
        else:
            x = x + swiglu_ffn(normed,
                               W[f"layers.{i}.feed_forward.w3.weight"],
                               W[f"layers.{i}.feed_forward.w1.weight"],
                               W[f"layers.{i}.feed_forward.w2.weight"])

    # Output norm + tied LM head
    x = rms_norm(x, W["embedding_norm.weight"])     # [1, T, DIM]
    logits = x[0, -1] @ W_embed.T                  # [vocab]
    return logits


def greedy_generate(W: dict, input_ids: list, n: int) -> list:
    """n steps of greedy decoding (no repetition penalty, no sampling)."""
    ids = list(input_ids)
    for _ in range(n):
        logits = forward(W, ids)
        next_tok = int(logits.argmax().item())
        ids.append(next_tok)
        if next_tok == 31:  # eos_token_id
            break
    return ids[len(input_ids):][:n]

# ---------- Config JSON files ----------------------------------------------

def write_configs(out_dir: pathlib.Path, bin_name: str,
                  tokenizer_path: str) -> None:
    config = {
        "architectures":         ["Lfm2MoeForCausalLM"],
        "bos_token_id":          0,
        "conv_L_cache":          CONV_K,
        "conv_bias":             False,
        "conv_dim":              CONV_DIM,
        "conv_dim_out":          CONV_DIM_OUT,
        "eos_token_id":          [31],
        "head_dim":              HEAD_DIM,
        "hidden_size":           DIM,
        "intermediate_size":     INTERMEDIATE,
        "is_causal":             True,
        "layer_types":           LAYER_TYPES,
        "max_position_embeddings": MAX_POS,
        "num_attention_heads":   N_HEADS,
        "num_hidden_layers":     N_LAYERS,
        "num_key_value_heads":   N_KV_HEADS,
        "num_experts":           NUM_EXPERTS,
        "num_experts_per_tok":   NUM_EXPERTS_PER_TOK,
        "moe_intermediate_size": MOE_INTERMEDIATE,
        "num_dense_layers":      NUM_DENSE_LAYERS,
        "norm_topk_prob":        NORM_TOPK_PROB,
        "use_expert_bias":       USE_EXPERT_BIAS,
        "routed_scaling_factor": ROUTED_SCALING,
        "rms_norm_eps":          RMS_EPS,
        "rope_theta":            ROPE_THETA,
        "tie_word_embeddings":   TIE_EMBED,
        "vocab_size":            VOCAB,
    }
    gen_config = {
        "bos_token_id": 0,
        "eos_token_id": 31,
        "do_sample": False,
        "top_k": 1,
        "top_p": 1.0,
        "temperature": 1.0,
    }
    nntr_config = {
        "bad_word_ids":      [],
        "batch_size":        1,
        "embedding_dtype":   "FP32",
        "fc_layer_dtype":    "FP32",
        "init_seq_len":      4,
        "lmhead_dtype":      "FP32",
        "max_seq_len":       8,
        "model_file_name":   bin_name,
        "model_tensor_type": "FP32-FP32",
        "model_type":        "CausalLM",
        "num_to_generate":   1,
        "tokenizer_file":    tokenizer_path,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "generation_config.json").write_text(json.dumps(gen_config, indent=2))
    (out_dir / "nntr_config.json").write_text(json.dumps(nntr_config, indent=2))

# ---------- Main -----------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LFM2-MoE tiny reference fixtures (pure-PyTorch)")
    parser.add_argument("--out",  type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n",    type=int, default=N_GEN)
    args = parser.parse_args()

    out_dir: pathlib.Path = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[generate] output dir: {out_dir}")

    W = build_weights(args.seed)
    print(f"[generate] weight tensors: {len(W)}, "
          f"total params: {sum(t.numel() for t in W.values()):,}")

    bin_name = "nntr_lfm2_moe_tiny_fp32.bin"
    save_nntrainer_bin(W, out_dir / bin_name)

    tokenizer_path = out_dir / "tokenizer.json"
    tokenizer_path.write_text(json.dumps(TINY_TOKENIZER, indent=2))

    write_configs(out_dir, bin_name, str(tokenizer_path))

    ref_logits = forward(W, INPUT_IDS).tolist()
    (out_dir / "reference_logits.json").write_text(json.dumps(ref_logits))
    argmax = int(np.argmax(ref_logits))
    print(f"[generate] reference logits: argmax={argmax}, "
          f"max|logit|={max(abs(v) for v in ref_logits):.4f}")

    ref_tokens = greedy_generate(W, INPUT_IDS, args.n)
    (out_dir / "reference_tokens.json").write_text(json.dumps(ref_tokens))
    print(f"[generate] reference tokens: {ref_tokens}")

    (out_dir / "input_ids.json").write_text(json.dumps(INPUT_IDS))

    meta = {
        "seed":             args.seed,
        "n_gen":            args.n,
        "input_ids":        INPUT_IDS,
        "logits_atol_fp32": 1e-2,
        "logits_atol_q40":  5.0,
        "prefix_match_min": 2,
        "torch_version":    torch.__version__,
        "note": (
            "Pure-PyTorch reference; no HuggingFace transformers dependency. "
            "Forward pass mirrors nntrainer's LFM2-MoE computation "
            "(sigmoid router + expert bias for top-k selection)."
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n[generate] done. Commit directory: {out_dir}")


if __name__ == "__main__":
    main()
