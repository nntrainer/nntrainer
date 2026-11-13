## SPDX-License-Identifier: Apache-2.0
## Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
##
## @file weight_converter.py
## @brief weight conversion script for the PaliGemma text decoder (Gemma1)
## @author SeungBaek Hong <sb92.hong@samsung.com>
##
## Reads the HF PaliGemma checkpoint (safetensors shards via
## PaliGemmaForConditionalGeneration, or any state_dict) and writes
## nntrainer-format binaries:
##   - nntr_paligemma_3b_fp32.bin      text decoder weights
##   - nntr_paligemma_3b_emb_fp32.bin  raw embedding table (use_embedding
##                                     bypass / lookupEmbedding cache)
## The save order must match PaliGemmaCausalLM::constructModel()'s graph:
##   embedding0 -> per layer {attention_norm(+1), q, k, v, o,
##   post_attention_norm(+1), gate, up, down} -> output_norm(+1)
## Connected layout ties the head to its embedding record. The bypass layout
## appends a head record, except when cpu_external_head reuses the CPU table.
##
## Key layouts: recent transformers builds expose the model as
##   model.language_model.* / model.vision_tower.* / model.multi_modal_projector.*
## while the google/paligemma-3b-pt-224 checkpoint files (and older
## transformers) use
##   language_model.model.* / vision_tower.vision_model.* / multi_modal_projector.*
## Every lookup resolves both spellings.

import argparse
from pathlib import Path

import numpy as np
import torch

TEXT_PREFIXES = ["model.language_model.", "language_model.model."]


def save_weight(weight, file, is_rms=False, transpose=False, dtype=np.float32):
    """Save one tensor in nntrainer order.

    is_rms: Gemma bakes the +1.0 of its weight(1+w) RMSNorm into the stored
    gamma (the nntrainer rms_norm layer multiplies the plain gamma).
    transpose: PyTorch Linear weights are [out, in]; nntrainer FC wants
    [in, out].
    """
    array = weight.detach().cpu().float().numpy()
    if is_rms:
        array = array + 1.0
    if transpose:
        array = array.T
    np.array(array, dtype=dtype).tofile(file)


def resolve_text_key(params, suffix):
    """Return the tensor for a text-decoder weight under either key layout."""
    missing = []
    for prefix in TEXT_PREFIXES:
        key = prefix + suffix
        if key in params:
            return params[key]
        missing.append(key)
    raise KeyError(f"none of {missing} found in checkpoint")


def save_paligemma_for_nntrainer(params, config, file, bypass_layout=False, dtype=np.float32, weight_transposed=False, cpu_external_head=False):
    """Convert and save the PaliGemma text decoder as nntrainer format.

    bypass_layout: use_embedding=true accepts precomputed multimodal embeddings.
    Its final head normally owns a checkpoint record because the token embedding
    node is disconnected. cpu_external_head omits that duplicate record and binds
    the standalone FP32 CPU embedding table instead. Connected token-ID models
    store the embedding first and share it with their tied head.

    weight_transposed: store all dense projections, including a bypass head,
    in [output,input] order. The matching config must set weight_transposed=true.
    """
    if cpu_external_head and (not bypass_layout or np.dtype(dtype) != np.dtype(np.float32)):
        raise ValueError("cpu_external_head requires the FP32 embedding bypass layout")
    def write_weight(weight, file, **kwargs):
        if weight_transposed and kwargs.get("transpose"):
            kwargs["transpose"] = False
        save_weight(weight, file, dtype=dtype, **kwargs)

    n_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    num_kv_heads = config.num_key_value_heads
    head_dim = (
        config.head_dim
        if hasattr(config, "head_dim")
        else hidden_size // config.num_attention_heads
    )
    kv_dim = num_kv_heads * head_dim

    # 1. Token embedding (tied lm_head shares it on the connected layout)
    embed = resolve_text_key(params, "embed_tokens.weight")
    if not bypass_layout:
        write_weight(embed, file)

    # 2. Decoder layers, in graph order
    for i in range(n_layers):
        suffix = f"layers.{i}."
        # attention_norm
        write_weight(
            resolve_text_key(params, suffix + "input_layernorm.weight"),
            file,
            is_rms=True,
        )
        # Q/K/V/O projections (MQA: k/v are [kv_dim, hidden])
        write_weight(resolve_text_key(params, suffix + "self_attn.q_proj.weight"), file, transpose=True)
        write_weight(resolve_text_key(params, suffix + "self_attn.k_proj.weight"), file, transpose=True)
        write_weight(resolve_text_key(params, suffix + "self_attn.v_proj.weight"), file, transpose=True)
        write_weight(resolve_text_key(params, suffix + "self_attn.o_proj.weight"), file, transpose=True)
        # post_attention_layernorm
        write_weight(
            resolve_text_key(params, suffix + "post_attention_layernorm.weight"),
            file,
            is_rms=True,
        )
        # MLP: gate -> up -> down (graph order)
        write_weight(resolve_text_key(params, suffix + "mlp.gate_proj.weight"), file, transpose=True)
        write_weight(resolve_text_key(params, suffix + "mlp.up_proj.weight"), file, transpose=True)
        write_weight(resolve_text_key(params, suffix + "mlp.down_proj.weight"), file, transpose=True)
        print(f"  layer {i + 1}/{n_layers} done")

    # 3. Final norm
    write_weight(resolve_text_key(params, "norm.weight"), file, is_rms=True)

    # 4. (bypass layout only) the lm head's own weight record, last —
    # fully_connected layout, i.e. the TRANSPOSED table.
    if bypass_layout and not cpu_external_head:
        write_weight(embed, file, transpose=True)

    print(
        f"Saved text decoder ({n_layers} layers, hidden {hidden_size}, "
        f"kv {kv_dim}, layout={'bypass' if bypass_layout else 'connected'})"
    )


def save_embedding_bin(params, file):
    """Save the raw token-embedding table for lookupEmbedding() (the
    use_embedding=true bypass caches it separately from the model bin)."""
    save_weight(resolve_text_key(params, "embed_tokens.weight"), file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./paligemma-3b-pt-224")
    parser.add_argument(
        "--output_name", type=str, default="./nntr_paligemma_3b_fp32.bin"
    )
    parser.add_argument("--connected", action="store_true",
                        help="token-id layout; default is the multimodal embedding bypass")
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--weight-transposed", action="store_true",
                        help="store dense weights as [output,input]; set nntr_config weight_transposed=true")
    parser.add_argument("--cpu-external-head", action="store_true",
                        help="omit tied CPU head; set nntr_config cpu_external_head=true")
    args = parser.parse_args()
    if args.cpu_external_head and (args.connected or args.dtype != "fp32"):
        parser.error("--cpu-external-head requires the FP32 embedding bypass layout")

    from transformers import AutoConfig, PaliGemmaForConditionalGeneration

    config = AutoConfig.from_pretrained(args.model_path)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float32
    )
    model.eval()
    print(model)

    with open(args.output_name, "wb") as f_model:
        save_paligemma_for_nntrainer(model.state_dict(), config.text_config, f_model,
                                    bypass_layout=not args.connected,
                                    dtype=np.float16 if args.dtype == "fp16" else np.float32,
                                    weight_transposed=args.weight_transposed,
                                    cpu_external_head=args.cpu_external_head)

    emb_name = str(Path(args.output_name).with_name("nntr_paligemma_3b_emb_fp32.bin"))
    with open(emb_name, "wb") as f_emb:
        save_embedding_bin(model.state_dict(), f_emb)
    print(f"Saved embedding table to {emb_name}")
