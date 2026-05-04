## SPDX-License-Identifier: Apache-2.0
## Copyright (C) 2025 Seungbaek Hong <sb92.hong@samsung.com>
##
## @file weight_converter.py
## @brief weight conversion script for gemma3 model
## @author SeungBaek Hong <sb92.hong@samsung.com>

import argparse
import torch
import numpy as np
import math
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

total_size = 0
def save_gemma3_for_nntrainer(params, config, dtype, file, save_lm_head=True):
    """Convert and save weights as nntrainer format for multi-head attention model"""
    n_layers = config.num_hidden_layers
    hidden_size = config.hidden_size

    def save_weight(weight, is_rms=False):
        if is_rms:
            weight = weight + 1.0
        np.array(weight, dtype=dtype).tofile(file)

    def save_projection(layer_name, proj_name):
        """Save projection layer weights (with LoRA support)"""
        lora_key = f"{layer_name}{proj_name}.lora_A.default.weight"
        if lora_key in params:
            save_weight(params[f"{layer_name}{proj_name}.base_layer.weight"].permute(1, 0))
            save_weight(params[f"{layer_name}{proj_name}.lora_A.default.weight"].permute(1, 0))
            save_weight(params[f"{layer_name}{proj_name}.lora_B.default.weight"].permute(1, 0))
        else:
            save_weight(params[f"{layer_name}{proj_name}.weight"].permute(1, 0))

    def save_attention(layer_name):
        """Save attention layer weights"""
        save_weight(params[f"{layer_name}input_layernorm.weight"], is_rms=True)

        # Save in NNTrainer graph order:
        # attention_norm -> Q -> K -> V -> q_norm -> k_norm -> O
        save_projection(layer_name, "self_attn.q_proj")
        save_projection(layer_name, "self_attn.k_proj")
        save_projection(layer_name, "self_attn.v_proj")
        if f"{layer_name}self_attn.q_norm.weight" in params:
            save_weight(params[f"{layer_name}self_attn.q_norm.weight"], is_rms=True)
        if f"{layer_name}self_attn.k_norm.weight" in params:
            save_weight(params[f"{layer_name}self_attn.k_norm.weight"], is_rms=True)
        save_projection(layer_name, "self_attn.o_proj")

    def save_feed_forward(layer_name):
        """Save feed forward layer weights"""
        save_weight(params[f"{layer_name}post_attention_layernorm.weight"], is_rms=True)
        save_weight(params[f"{layer_name}pre_feedforward_layernorm.weight"], is_rms=True)
        # Save in NNTrainer graph order:
        # post_attention_norm -> pre_ffn_norm -> gate -> up -> down -> post_ffn_norm
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            save_projection(layer_name, f"mlp.{proj}")
        save_weight(params[f"{layer_name}post_feedforward_layernorm.weight"], is_rms=True)

    save_weight(params["model.embed_tokens.weight"])

    for layer_idx in range(n_layers):
        layer_prefix = f"model.layers.{layer_idx}."
        save_attention(layer_prefix)
        save_feed_forward(layer_prefix)

    save_weight(params["model.norm.weight"], is_rms=True)
    if save_lm_head:
        save_weight(params["lm_head.weight"].permute(1, 0))


def save_embedding_gemma_for_nntrainer(model, dtype, file):
    params = model.state_dict()
    gemma_params = {
        key.removeprefix("0.auto_model.").removeprefix("0."): value
        for key, value in params.items()
        if key.startswith("0.")
    }

    save_gemma3_for_nntrainer(
        gemma_params, model[0].auto_model.config, dtype, file, save_lm_head=False
    )

    def save_weight(weight):
        np.array(weight, dtype=dtype).tofile(file)

    for module_name, module in model._modules.items():
        component = module.__class__.__name__
        if component in ["Transformer", "Pooling", "Normalize"]:
            continue
        if component != "Dense":
            raise NotImplementedError(
                f"Unsupported SentenceTransformer module: {module_name} ({component})"
            )

        save_weight(params[f"{module_name}.linear.weight"].permute(1, 0))
        bias_key = f"{module_name}.linear.bias"
        if bias_key in params:
            save_weight(params[bias_key])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["causal", "embedding"], default="causal")
    parser.add_argument("--model_path", type=str, default="./270m")
    parser.add_argument("--output_name", type=str, default="./nntr_gemma3_270m_fp32.bin")
    parser.add_argument("--data_type", choices=["float32", "float16"], default="float32")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    data_dtype = args.data_type
    model_path = args.model_path
    output_name = args.output_name

    if args.model_type == "causal":
        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=args.trust_remote_code
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32, trust_remote_code=args.trust_remote_code
        )
        model.eval()

        print(model)

        with open(output_name, "wb") as f_model :
            save_gemma3_for_nntrainer(model.state_dict(), config, data_dtype, f_model)
    else:
        from sentence_transformers import SentenceTransformer

        torch_dtype = torch.float16 if data_dtype == "float16" else torch.float32
        model = SentenceTransformer(
            model_path,
            trust_remote_code=args.trust_remote_code,
            model_kwargs={"torch_dtype": torch_dtype},
        )
        model.eval()

        with open(output_name, "wb") as f_model :
            save_embedding_gemma_for_nntrainer(model, data_dtype, f_model)
