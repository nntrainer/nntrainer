# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

## @package gemma4_moe_weight_converter
## @brief Convert HuggingFace Gemma4 MoE weights to the nntrainer format.
## @author Jungwon-Lee <jungone.lee@samsung.com>

"""Convert HuggingFace Gemma4 MoE weights to the nntrainer format."""

import importlib.util
from pathlib import Path


_BASE_CONVERTER_PATH = Path(__file__).with_name("weight_converter.py")
_base_spec = importlib.util.spec_from_file_location(
    "gemma4_weight_converter", _BASE_CONVERTER_PATH
)
base = importlib.util.module_from_spec(_base_spec)
_base_spec.loader.exec_module(base)


class TensorSlice:
    """A lazily materialized tensor slice with explicit shape metadata."""

    def __init__(self, tensor, index, shape):
        self._tensor = tensor
        self._index = index
        self._shape = tuple(shape)

    @property
    def shape(self):
        return self._shape

    def materialize(self):
        if isinstance(self._tensor, base.LazyTensor):
            return self._tensor._handle.get_slice(self._tensor._key)[self._index]
        if hasattr(self._tensor, "materialize"):
            return self._tensor.materialize()[self._index]
        return self._tensor[self._index]


def iter_gemma4_moe_weight_specs(params, config):
    """Yield Gemma4 MoE weights in the nntrainer model-layer order."""
    text_config = config.text_config
    num_experts = int(text_config.num_experts)
    expert_intermediate = int(text_config.moe_intermediate_size)
    hidden_size = int(text_config.hidden_size)

    def moe_ffn_specs(layer_idx, layer_prefix, resolve):
        yield (f"layer{layer_idx}_post_ffn_norm_1", base.SUFFIX_GAMMA,
               resolve(f"{layer_prefix}post_feedforward_layernorm_1.weight"),
               False)
        yield (f"layer{layer_idx}_pre_ffn_norm_2", base.SUFFIX_GAMMA,
               resolve(f"{layer_prefix}pre_feedforward_layernorm_2.weight"),
               False)

        sparse_name = f"layer{layer_idx}_sparse_moe"
        yield (sparse_name, "router",
               resolve(f"{layer_prefix}router.proj.weight"), True)
        yield (sparse_name, "router_scale",
               resolve(f"{layer_prefix}router.scale"), False)
        yield (sparse_name, "router_per_expert_scale",
               resolve(f"{layer_prefix}router.per_expert_scale"), False)

        gate_up = resolve(f"{layer_prefix}experts.gate_up_proj")
        down = resolve(f"{layer_prefix}experts.down_proj")
        for expert in range(num_experts):
            gate = TensorSlice(
                gate_up,
                (expert, slice(0, expert_intermediate), slice(None)),
                (expert_intermediate, hidden_size),
            )
            up = TensorSlice(
                gate_up,
                (expert, slice(expert_intermediate, None), slice(None)),
                (expert_intermediate, hidden_size),
            )
            down_expert = TensorSlice(
                down,
                (expert, slice(None), slice(None)),
                (hidden_size, expert_intermediate),
            )
            yield (sparse_name, f"expert_gate_{expert}", gate, True)
            yield (sparse_name, f"expert_up_{expert}", up, True)
            yield (sparse_name, f"expert_down_{expert}", down_expert, True)

        yield (f"layer{layer_idx}_post_ffn_norm_2", base.SUFFIX_GAMMA,
               resolve(f"{layer_prefix}post_feedforward_layernorm_2.weight"),
               False)

    yield from base.iter_gemma4_weight_specs(
        params, config, additional_ffn_specs=moe_ffn_specs
    )


def save_gemma4_moe_bin(params, config, dtype, file, tie_word_embeddings):
    """Write Gemma4 MoE weights as nntrainer's binary layout."""
    return base.save_gemma4_bin(
        params, config, dtype, file, tie_word_embeddings,
        iter_gemma4_moe_weight_specs,
    )


def save_gemma4_moe_safetensors(params, config, dtype, output_path,
                                tie_word_embeddings):
    """Write Gemma4 MoE weights as an nntrainer safetensors file."""
    return base.save_gemma4_safetensors(
        params, config, dtype, output_path, tie_word_embeddings,
        iter_gemma4_moe_weight_specs,
    )


def main():
    """Convert a Gemma4 MoE HuggingFace model to nntrainer weights."""
    args = base.parse_args()
    config = base.AutoConfig.from_pretrained(args.model_path,
                                             trust_remote_code=True)
    params, source = base.load_model_state(args.model_path)
    text_config = config.text_config
    tie_word_embeddings = base.get_tie_word_embeddings(text_config, config)

    print(f"Loading Gemma4 MoE model from: {args.model_path}")
    print(f"Weight source: {source}")
    if args.safetensors:
        output_name = base.get_safetensors_output_name(args.output_name)
        save_gemma4_moe_safetensors(
            params, config, args.data_type, output_name, tie_word_embeddings
        )
        return

    with open(args.output_name, "wb") as output_file:
        save_gemma4_moe_bin(
            params, config, args.data_type, output_file, tie_word_embeddings
        )
    print(f"Saved binary: {args.output_name}")


if __name__ == "__main__":
    main()
