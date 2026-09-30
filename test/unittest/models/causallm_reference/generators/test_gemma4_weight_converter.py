# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

## @package test_gemma4_weight_converter
## @brief Unit tests for Gemma4 MoE weight-spec conversion.
## @author Jungwon-Lee <jungone.lee@samsung.com>

"""Unit tests for Gemma4 MoE weight-spec conversion."""

import importlib.util
import pathlib
import sys
import types
import unittest

numpy_stub = types.ModuleType("numpy")
numpy_stub.float32 = "float32"
sys.modules.setdefault("numpy", numpy_stub)
torch_stub = types.ModuleType("torch")
torch_stub.float32 = "float32"
sys.modules.setdefault("torch", torch_stub)
transformers_stub = types.ModuleType("transformers")
transformers_stub.AutoConfig = object
transformers_stub.AutoModelForCausalLM = object
sys.modules.setdefault("transformers", transformers_stub)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]
CONVERTER_PATH = (
    REPO_ROOT / "Applications/CausalLM/res/gemma4/gemma4_moe_weight_converter.py"
)


def load_converter():
    """Load the converter without requiring Transformers for spec-only tests."""
    spec = importlib.util.spec_from_file_location(
        "gemma4_moe_weight_converter", CONVERTER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONVERTER = load_converter()


class ShapeTensor:
    """Tensor metadata sufficient for the weight-spec walker."""

    def __init__(self, *shape):
        self.shape = tuple(shape)


class SequenceTensor(ShapeTensor):
    """Small 3-D tensor supporting the expert slices used by the converter."""

    def __getitem__(self, index):
        expert, rows, columns = index
        row_range = range(self.shape[1])[rows]
        column_range = range(self.shape[2])[columns]
        if isinstance(row_range, int):
            row_range = [row_range]
        if isinstance(column_range, int):
            column_range = [column_range]
        expert_offset = expert * self.shape[1] * self.shape[2]
        return [
            [expert_offset + row * self.shape[2] + column
             for column in column_range]
            for row in row_range
        ]


class Config:
    """Minimal attribute-based config wrapper used by the converter."""

    def __init__(self, **values):
        self.__dict__.update(values)


def make_moe_state():
    """Create a complete two-layer MoE state with recognizable expert rows."""
    hidden = 8
    intermediate = 6
    experts = 3
    expert_intermediate = 4
    state = {"embed_tokens.weight": ShapeTensor(16, hidden)}

    for layer in range(2):
        prefix = f"layers.{layer}."
        state.update(
            {
                f"{prefix}input_layernorm.weight": ShapeTensor(hidden),
                f"{prefix}self_attn.q_proj.weight": ShapeTensor(hidden, hidden),
                f"{prefix}self_attn.q_norm.weight": ShapeTensor(hidden),
                f"{prefix}self_attn.k_proj.weight": ShapeTensor(hidden, hidden),
                f"{prefix}self_attn.k_norm.weight": ShapeTensor(hidden),
                f"{prefix}self_attn.o_proj.weight": ShapeTensor(hidden, hidden),
                f"{prefix}post_attention_layernorm.weight": ShapeTensor(hidden),
                f"{prefix}pre_feedforward_layernorm.weight": ShapeTensor(hidden),
                f"{prefix}mlp.gate_proj.weight": ShapeTensor(intermediate, hidden),
                f"{prefix}mlp.up_proj.weight": ShapeTensor(intermediate, hidden),
                f"{prefix}mlp.down_proj.weight": ShapeTensor(hidden, intermediate),
                f"{prefix}post_feedforward_layernorm_1.weight": ShapeTensor(hidden),
                f"{prefix}pre_feedforward_layernorm_2.weight": ShapeTensor(hidden),
                f"{prefix}router.proj.weight": ShapeTensor(experts, hidden),
                f"{prefix}router.scale": ShapeTensor(hidden),
                f"{prefix}router.per_expert_scale": ShapeTensor(experts),
                f"{prefix}post_feedforward_layernorm_2.weight": ShapeTensor(hidden),
                f"{prefix}post_feedforward_layernorm.weight": ShapeTensor(hidden),
                f"{prefix}layer_scalar": ShapeTensor(1),
            }
        )
        if layer == 0:
            state[f"{prefix}self_attn.v_proj.weight"] = ShapeTensor(
                hidden, hidden
            )

        gate_up = SequenceTensor(experts, 2 * expert_intermediate, hidden)
        state[f"{prefix}experts.gate_up_proj"] = gate_up
        state[f"{prefix}experts.down_proj"] = SequenceTensor(
            experts, hidden, expert_intermediate
        )

    state["norm.weight"] = ShapeTensor(hidden)
    return state


def make_moe_config():
    text_config = Config(
        num_hidden_layers=2,
        num_kv_shared_layers=0,
        layer_types=["sliding_attention", "full_attention"],
        attention_k_eq_v=True,
        hidden_size_per_layer_input=0,
        enable_moe_block=True,
        num_experts=3,
        top_k_experts=2,
        moe_intermediate_size=4,
        hidden_size=8,
    )
    return Config(text_config=text_config)


class Gemma4WeightConverterTest(unittest.TestCase):
    """Verify MoE ordering, K=V omission, and fused expert splitting."""

    def test_moe_specs_match_graph_and_split_fused_experts(self):
        specs = list(
            CONVERTER.iter_gemma4_moe_weight_specs(
                make_moe_state(), make_moe_config()
            )
        )
        keys = [f"{name}:{suffix}" for name, suffix, _, _ in specs]

        self.assertIn("layer0_wv:weight", keys)
        self.assertNotIn("layer1_wv:weight", keys)
        self.assertNotIn("per_layer_input_embedding:Embedding", keys)

        router = keys.index("layer0_sparse_moe:router")
        self.assertEqual(
            keys[router : router + 6],
            [
                "layer0_sparse_moe:router",
                "layer0_sparse_moe:router_scale",
                "layer0_sparse_moe:router_per_expert_scale",
                "layer0_sparse_moe:expert_gate_0",
                "layer0_sparse_moe:expert_up_0",
                "layer0_sparse_moe:expert_down_0",
            ],
        )

        by_key = {
            f"{name}:{suffix}": (tensor, transpose)
            for name, suffix, tensor, transpose in specs
        }
        gate, gate_transpose = by_key["layer0_sparse_moe:expert_gate_1"]
        up, up_transpose = by_key["layer0_sparse_moe:expert_up_1"]
        fused = make_moe_state()["layers.0.experts.gate_up_proj"]

        self.assertTrue(gate_transpose)
        self.assertTrue(up_transpose)
        self.assertEqual(gate.materialize(), fused[1, :4, :])
        self.assertEqual(up.materialize(), fused[1, 4:, :])

    def test_lazy_expert_slice_does_not_materialize_full_tensor(self):
        source = SequenceTensor(2, 6, 4)

        class SliceHandle:
            def get_slice(self, key):
                self.key = key
                return self

            def get_shape(self):
                return source.shape

            def __getitem__(self, index):
                self.index = index
                return source[index]

            def get_tensor(self, key):
                raise AssertionError("full tensor must not be loaded")

        handle = SliceHandle()
        lazy = CONVERTER.base.LazyTensor(handle, "experts.gate_up_proj")
        view = CONVERTER.TensorSlice(
            lazy, (1, slice(0, 3), slice(None)), (3, 4)
        )

        self.assertEqual(view.materialize(), source[1, :3, :])
        self.assertEqual(handle.key, "experts.gate_up_proj")


if __name__ == "__main__":
    unittest.main()
