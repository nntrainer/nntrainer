# SPDX-License-Identifier: Apache-2.0
## @file test_gemma4_weight_converter_stream.py
## @brief Unit tests for the bounded-memory Gemma4 FP32 converter

"""Unit tests for the bounded-memory Gemma4 FP32 converter."""

import importlib.util
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock


torch_stub = types.ModuleType("torch")
torch_stub.float32 = "float32"
safetensors_stub = types.ModuleType("safetensors")
safetensors_stub.safe_open = None

sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("safetensors", safetensors_stub)

SCRIPT = (
    pathlib.Path(__file__).parents[5]
    / "Applications"
    / "CausalLM"
    / "res"
    / "gemma4"
    / "weight_converter_stream.py"
)
SPEC = importlib.util.spec_from_file_location(
    "gemma4_weight_converter_stream", SCRIPT
)
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)


class MatrixSlice:
    """Small safetensors slice stand-in for layout tests."""

    def __init__(self, rows, columns):
        self.values = [
            [row * columns + column for column in range(columns)]
            for row in range(rows)
        ]

    def get_shape(self):
        return (len(self.values), len(self.values[0]))

    def __getitem__(self, index):
        rows, columns = index
        return [
            [self.values[row][column] for column in range(len(self.values[0]))[
                columns
            ]]
            for row in range(len(self.values))[rows]
        ]


class SingleTensorSource:
    def __init__(self, tensor):
        self.tensor = tensor

    def get_slice(self, _key):
        return self.tensor


class ListWriter(CONVERTER.Fp32Writer):
    """Fp32Writer variant that writes integer lists instead of torch data."""

    def _write(self, tensor, transpose):
        if transpose:
            tensor = [list(row) for row in zip(*tensor)]
        values = [value for row in tensor for value in row]
        self.output.extend(values)
        self.elements_written += len(values)


class RecordingSlice:
    def __init__(self, shape):
        self.shape = shape
        self.requests = []

    def get_shape(self):
        return self.shape

    def __getitem__(self, index):
        self.requests.append(index)
        return index


class IndexWriter(CONVERTER.Fp32Writer):
    def _write(self, _tensor, _transpose):
        return


class Gemma4WeightConverterStreamTest(unittest.TestCase):
    def test_streamed_transpose_preserves_output_layout(self):
        source = SingleTensorSource(MatrixSlice(3, 5))
        output = []
        writer = ListWriter(
            source, output, 3 * 2 * CONVERTER.FP32_BYTES
        )

        writer.write_tensor("matrix", (3, 5), transpose=True)

        self.assertEqual(
            output,
            [0, 5, 10, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14],
        )
        self.assertEqual(writer.elements_written, 15)

    def test_expert_view_composes_chunk_with_fused_slice(self):
        tensor = RecordingSlice((2, 6, 4))
        writer = IndexWriter(
            SingleTensorSource(tensor), [], 3 * 2 * CONVERTER.FP32_BYTES
        )

        writer.write_view(
            "experts.gate_up_proj",
            (2, 6, 4),
            (1, slice(3, None), slice(None)),
            (3, 4),
            transpose=True,
        )

        self.assertEqual(
            tensor.requests,
            [
                (1, slice(3, 6, 1), slice(0, 2, 1)),
                (1, slice(3, 6, 1), slice(2, 4, 1)),
            ],
        )

    def test_source_keeps_only_active_shard_open(self):
        events = []

        class Handle:
            def __init__(self, path):
                self.path = pathlib.Path(path).name

            def get_slice(self, key):
                events.append(("slice", self.path, key))
                return key

        class Context:
            def __init__(self, path):
                self.path = pathlib.Path(path).name

            def __enter__(self):
                events.append(("open", self.path))
                return Handle(self.path)

            def __exit__(self, _exc_type, _exc_value, _traceback):
                events.append(("close", self.path))

        with tempfile.TemporaryDirectory() as directory:
            model_path = pathlib.Path(directory)
            for shard in ("a.safetensors", "b.safetensors"):
                (model_path / shard).touch()

            source = CONVERTER.SafetensorsSource.__new__(
                CONVERTER.SafetensorsSource
            )
            source.model_path = model_path
            source.weight_map = {
                "model.weight_a": "a.safetensors",
                "model.weight_b": "b.safetensors",
            }
            source._active_context = None
            source._active_handle = None
            source._active_shard = None

            with mock.patch.object(
                CONVERTER,
                "safe_open",
                side_effect=lambda path, framework, device: Context(path),
            ):
                self.assertEqual(source.get_slice("weight_a"), "model.weight_a")
                self.assertEqual(source.get_slice("weight_a"), "model.weight_a")
                self.assertEqual(source.get_slice("weight_b"), "model.weight_b")
                source.close()

        self.assertEqual(
            events,
            [
                ("open", "a.safetensors"),
                ("slice", "a.safetensors", "model.weight_a"),
                ("slice", "a.safetensors", "model.weight_a"),
                ("close", "a.safetensors"),
                ("open", "b.safetensors"),
                ("slice", "b.safetensors", "model.weight_b"),
                ("close", "b.safetensors"),
            ],
        )

    def test_nested_config_builds_moe_plan(self):
        config = {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "text_config": {
                "hidden_size": 64,
                "vocab_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "global_head_dim": 16,
                "num_global_key_value_heads": 1,
                "intermediate_size": 96,
                "moe_intermediate_size": 24,
                "num_experts": 4,
                "enable_moe_block": True,
                "layer_types": ["sliding_attention", "full_attention"],
                "attention_k_eq_v": True,
                "hidden_size_per_layer_input": 0,
                "tie_word_embeddings": True,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            plan = CONVERTER.load_model_plan(directory)

        self.assertEqual(plan["global_head_dim"], 16)
        self.assertEqual(plan["moe_intermediate_size"], 24)
        self.assertEqual(plan["num_experts"], 4)
        self.assertTrue(plan["attention_k_eq_v"])
        self.assertTrue(plan["tied_embeddings"])

    def test_weight_order_omits_full_attention_v_and_splits_experts(self):
        calls = []

        class RecordingWriter:
            def __init__(self, _source, _output, _chunk_bytes):
                self.elements_written = 0

            def write_tensor(self, key, shape, transpose=False):
                calls.append(("tensor", key, shape, transpose))

            def write_view(self, key, source_shape, index, shape,
                           transpose=False):
                calls.append(
                    ("view", key, source_shape, index, shape, transpose)
                )

        model = {
            "hidden_size": 64,
            "vocab_size": 32,
            "num_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "global_head_dim": 16,
            "num_global_key_value_heads": 1,
            "intermediate_size": 96,
            "moe_intermediate_size": 24,
            "num_experts": 2,
            "num_kv_shared_layers": 0,
            "layer_types": ["sliding_attention", "full_attention"],
            "attention_k_eq_v": True,
            "enable_moe": True,
            "ple_size": 0,
            "ple_vocab_size": 32,
            "tied_embeddings": True,
        }

        with mock.patch.object(CONVERTER, "Fp32Writer", RecordingWriter):
            CONVERTER.write_gemma4(None, None, model, 64)

        tensor_keys = [call[1] for call in calls if call[0] == "tensor"]
        self.assertIn("layers.0.self_attn.v_proj.weight", tensor_keys)
        self.assertNotIn("layers.1.self_attn.v_proj.weight", tensor_keys)
        self.assertEqual(tensor_keys[-2:], ["norm.weight", "embed_tokens.weight"])

        expert_calls = [call for call in calls if call[0] == "view"]
        self.assertEqual(len(expert_calls), 2 * 2 * 3)
        self.assertEqual(
            expert_calls[0][3],
            (0, slice(0, 24), slice(None)),
        )
        self.assertEqual(
            expert_calls[1][3],
            (0, slice(24, None), slice(None)),
        )
        self.assertEqual(expert_calls[2][1], "layers.0.experts.down_proj")


if __name__ == "__main__":
    unittest.main()
