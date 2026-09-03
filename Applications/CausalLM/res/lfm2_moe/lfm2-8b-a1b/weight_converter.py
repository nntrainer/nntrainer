# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

"""Convert LFM2-8B-A1B Hugging Face weights to NNTrainer format.

The NNTrainer LFM2-MoE layer stores each expert as a fused gate/up projection
followed by its down projection.  Both the current Hugging Face fused expert
layout and the older per-expert w1/w3/w2 layout are supported.
"""

import argparse

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM


def save_lfm2_moe_for_nntrainer(params, config, dtype, output):
    """Write weights in the order requested by the NNTrainer LFM2-MoE graph."""

    output_dtype = np.dtype(dtype)
    if output_dtype != np.dtype("float32"):
        raise ValueError(
            "LFM2-MoE conversion currently requires float32 because norms, "
            "router weights, expert biases, and convolution weights are "
            "stored as FP32 in the NNTrainer graph"
        )

    num_layers = config.num_hidden_layers
    num_experts = config.num_experts
    num_dense_layers = getattr(config, "num_dense_layers", 0)
    layer_types = config.layer_types

    if len(layer_types) != num_layers:
        raise ValueError("layer_types size must match num_hidden_layers")

    def find_weight(*names):
        for name in names:
            if name in params:
                return name, params[name]
        raise KeyError(f"none of the weight names exist: {', '.join(names)}")

    def write_tensor(name, tensor, transpose=False):
        if transpose:
            tensor = tensor.transpose(-2, -1)
        tensor = tensor.detach().to(device="cpu", dtype=torch.float32)
        array = tensor.contiguous().numpy().astype(output_dtype, copy=False)
        print(f"{name}: {tuple(tensor.shape)}")
        array.tofile(output)

    def write_weight(*names, transpose=False):
        name, tensor = find_weight(*names)
        write_tensor(name, tensor, transpose)

    def write_attention(layer_prefix):
        write_weight(f"{layer_prefix}self_attn.q_proj.weight", transpose=True)
        write_weight(
            f"{layer_prefix}self_attn.q_layernorm.weight",
            f"{layer_prefix}self_attn.q_norm.weight",
        )
        write_weight(f"{layer_prefix}self_attn.k_proj.weight", transpose=True)
        write_weight(
            f"{layer_prefix}self_attn.k_layernorm.weight",
            f"{layer_prefix}self_attn.k_norm.weight",
        )
        write_weight(f"{layer_prefix}self_attn.v_proj.weight", transpose=True)
        write_weight(
            f"{layer_prefix}self_attn.out_proj.weight",
            f"{layer_prefix}self_attn.o_proj.weight",
            transpose=True,
        )

    def write_conv(layer_prefix):
        write_weight(f"{layer_prefix}conv.in_proj.weight", transpose=True)

        conv_name, conv_weight = find_weight(
            f"{layer_prefix}conv.conv.weight"
        )
        expected_shape = (
            config.hidden_size,
            1,
            config.conv_L_cache,
        )
        if tuple(conv_weight.shape) != expected_shape:
            raise ValueError(
                f"{conv_name} must have shape {expected_shape}, got "
                f"{tuple(conv_weight.shape)}"
            )

        # PyTorch Conv1d cross-correlation stores the current-token
        # coefficient at the last kernel index. NNTrainer causal_conv1d uses
        # row 0 for the current token, followed by t-1, t-2, ... .
        conv_weight = conv_weight[:, 0, :].flip(-1).transpose(0, 1)
        write_tensor(conv_name, conv_weight)

        write_weight(f"{layer_prefix}conv.out_proj.weight", transpose=True)

    def write_dense_ffn(layer_prefix):
        # NNTrainer keeps the historical up, gate, down order.
        write_weight(f"{layer_prefix}feed_forward.w3.weight", transpose=True)
        write_weight(f"{layer_prefix}feed_forward.w1.weight", transpose=True)
        write_weight(f"{layer_prefix}feed_forward.w2.weight", transpose=True)

    def write_expert_bias(layer_prefix):
        bias_name = f"{layer_prefix}feed_forward.expert_bias"
        if bias_name in params:
            write_tensor(bias_name, params[bias_name])
            return

        # The NNTrainer layer always requests this tensor. A checkpoint with
        # expert bias disabled therefore needs a zero-filled placeholder.
        write_tensor(
            f"{bias_name} (zeros)", torch.zeros(num_experts, dtype=torch.float32)
        )

    def write_fused_experts(layer_prefix):
        expert_prefix = f"{layer_prefix}feed_forward.experts."
        gate_up_name = f"{expert_prefix}gate_up_proj"
        down_name = f"{expert_prefix}down_proj"

        if gate_up_name in params and down_name in params:
            gate_up = params[gate_up_name]
            down = params[down_name]
            if gate_up.shape[0] != num_experts or down.shape[0] != num_experts:
                raise ValueError("fused expert count does not match num_experts")

            for expert_id in range(num_experts):
                # HF: [expert, 2 * intermediate, hidden]
                # NNTrainer: [hidden, gate | up]
                write_tensor(
                    f"{gate_up_name}[{expert_id}]",
                    gate_up[expert_id],
                    transpose=True,
                )
                write_tensor(
                    f"{down_name}[{expert_id}]",
                    down[expert_id],
                    transpose=True,
                )
            return

        # Legacy checkpoints store gate (w1), up (w3), and down (w2) in
        # separate Linear modules for each expert.
        for expert_id in range(num_experts):
            expert_name = f"{expert_prefix}{expert_id}."
            gate_name, gate = find_weight(
                f"{expert_name}w1.weight", f"{expert_name}w1"
            )
            up_name, up = find_weight(
                f"{expert_name}w3.weight", f"{expert_name}w3"
            )
            down_weight_name, down = find_weight(
                f"{expert_name}w2.weight", f"{expert_name}w2"
            )
            gate_up = torch.cat((gate, up), dim=0)
            write_tensor(
                f"cat({gate_name}, {up_name})", gate_up, transpose=True
            )
            write_tensor(down_weight_name, down, transpose=True)

    def write_moe_ffn(layer_prefix):
        write_weight(f"{layer_prefix}feed_forward.gate.weight", transpose=True)
        write_expert_bias(layer_prefix)
        write_fused_experts(layer_prefix)

    # embedding0 is the first weighted layer in the graph.
    write_weight("model.embed_tokens.weight")

    for layer_id, layer_type in enumerate(layer_types):
        layer_prefix = f"model.layers.{layer_id}."
        write_weight(f"{layer_prefix}operator_norm.weight")

        if layer_type in ("attention", "full_attention"):
            write_attention(layer_prefix)
        elif layer_type == "conv":
            write_conv(layer_prefix)
        else:
            raise ValueError(f"unsupported LFM2 layer type: {layer_type}")

        write_weight(f"{layer_prefix}ffn_norm.weight")
        if layer_id < num_dense_layers:
            write_dense_ffn(layer_prefix)
        else:
            write_moe_ffn(layer_prefix)

    write_weight("model.embedding_norm.weight", "model.norm.weight")
    if not getattr(config, "tie_word_embeddings", False):
        write_weight("lm_head.weight", transpose=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="LiquidAI/LFM2-8B-A1B"
    )
    parser.add_argument(
        "--output_name", type=str, default="./nntr_lfm2_8b_a1b_fp32.bin"
    )
    parser.add_argument("--data_type", choices=("float32",), default="float32")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    with open(args.output_name, "wb") as output:
        save_lfm2_moe_for_nntrainer(
            model.state_dict(), config, args.data_type, output
        )


if __name__ == "__main__":
    main()
