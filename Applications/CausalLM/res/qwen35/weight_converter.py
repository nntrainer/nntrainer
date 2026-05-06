#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert Qwen3.5 text weights to nntrainer CausalLM format."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from safetensors import safe_open


class SafeTensorSet:
    def __init__(self, files):
        self.files = files
        self.tensor_to_file = {}
        for file_path in files:
            with safe_open(str(file_path), framework="pt", device="cpu") as reader:
                for key in reader.keys():
                    if key in self.tensor_to_file:
                        raise RuntimeError(f"Duplicate tensor key found: {key}")
                    self.tensor_to_file[key] = file_path

    def get_tensor(self, name):
        file_path = self.tensor_to_file.get(name)
        if file_path is None:
            raise KeyError(f"Tensor not found in safetensors: {name}")
        with safe_open(str(file_path), framework="pt", device="cpu") as reader:
            return reader.get_tensor(name)


def tensor_to_numpy(
    tensor_set,
    name,
    dtype,
    *,
    transpose=False,
    add_one=False,
    squeeze_conv=False,
    neg_exp=False,
):
    tensor = tensor_set.get_tensor(name).float()
    if squeeze_conv:
        tensor = tensor.squeeze(1)
    if transpose:
        tensor = tensor.t().contiguous()
    array = tensor.cpu().numpy()
    if add_one:
        array = array + 1.0
    if neg_exp:
        array = -np.exp(array)
    return np.asarray(array, dtype=dtype)


def write_tensor(file, array):
    array.tofile(file)


def get_text_config(config):
    return dict(config.get("text_config", config))


def save_qwen35_for_nntrainer(model_path, output_file, dtype):
    model_path = Path(model_path)
    with open(model_path / "config.json", "r", encoding="utf-8") as fp:
        config = json.load(fp)
    text_config = get_text_config(config)
    num_layers = int(text_config["num_hidden_layers"])
    layer_types = text_config["layer_types"]

    safetensor_files = sorted(model_path.glob("model*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors file found in {model_path}")

    prefix = "model.language_model."
    tensor_set = SafeTensorSet(safetensor_files)

    with open(output_file, "wb") as fout:
        write_tensor(fout, tensor_to_numpy(tensor_set, prefix + "embed_tokens.weight", dtype))

        for layer_idx in range(num_layers):
            layer = f"{prefix}layers.{layer_idx}."
            write_tensor(
                fout,
                tensor_to_numpy(tensor_set, layer + "input_layernorm.weight", dtype, add_one=True),
            )

            if layer_types[layer_idx] == "linear_attention":
                linear = layer + "linear_attn."
                write_tensor(
                    fout,
                    tensor_to_numpy(tensor_set, linear + "in_proj_qkv.weight", dtype, transpose=True),
                )
                write_tensor(
                    fout,
                    tensor_to_numpy(tensor_set, linear + "conv1d.weight", dtype, squeeze_conv=True),
                )
                write_tensor(
                    fout,
                    tensor_to_numpy(tensor_set, linear + "in_proj_z.weight", dtype, transpose=True),
                )
                write_tensor(
                    fout,
                    tensor_to_numpy(tensor_set, linear + "in_proj_b.weight", dtype, transpose=True),
                )
                write_tensor(
                    fout,
                    tensor_to_numpy(tensor_set, linear + "in_proj_a.weight", dtype, transpose=True),
                )
                write_tensor(fout, tensor_to_numpy(tensor_set, linear + "dt_bias", dtype))
                write_tensor(fout, tensor_to_numpy(tensor_set, linear + "A_log", dtype, neg_exp=True))
                write_tensor(fout, tensor_to_numpy(tensor_set, linear + "norm.weight", dtype))
                write_tensor(
                    fout,
                    tensor_to_numpy(tensor_set, linear + "out_proj.weight", dtype, transpose=True),
                )
            elif layer_types[layer_idx] == "full_attention":
                attn = layer + "self_attn."
                write_tensor(
                    fout,
                    tensor_to_numpy(tensor_set, attn + "q_proj.weight", dtype, transpose=True),
                )
                write_tensor(
                    fout,
                    tensor_to_numpy(tensor_set, attn + "k_proj.weight", dtype, transpose=True),
                )
                write_tensor(
                    fout,
                    tensor_to_numpy(tensor_set, attn + "v_proj.weight", dtype, transpose=True),
                )
                write_tensor(
                    fout,
                    tensor_to_numpy(tensor_set, attn + "q_norm.weight", dtype, add_one=True),
                )
                write_tensor(
                    fout,
                    tensor_to_numpy(tensor_set, attn + "k_norm.weight", dtype, add_one=True),
                )
                write_tensor(
                    fout,
                    tensor_to_numpy(tensor_set, attn + "o_proj.weight", dtype, transpose=True),
                )
            else:
                raise RuntimeError(f"Unsupported layer type: {layer_types[layer_idx]}")

            write_tensor(
                fout,
                tensor_to_numpy(tensor_set, layer + "post_attention_layernorm.weight", dtype, add_one=True),
            )
            write_tensor(
                fout,
                tensor_to_numpy(tensor_set, layer + "mlp.up_proj.weight", dtype, transpose=True),
            )
            write_tensor(
                fout,
                tensor_to_numpy(tensor_set, layer + "mlp.gate_proj.weight", dtype, transpose=True),
            )
            write_tensor(
                fout,
                tensor_to_numpy(tensor_set, layer + "mlp.down_proj.weight", dtype, transpose=True),
            )

        write_tensor(fout, tensor_to_numpy(tensor_set, prefix + "norm.weight", dtype, add_one=True))


def write_runtime_files(model_path, output_dir, output_name, args):
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(model_path / "config.json", "r", encoding="utf-8") as fp:
        original_config = json.load(fp)
    text_config = get_text_config(original_config)
    text_config["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    text_config["model_type"] = "qwen3_5_text"
    text_config["tie_word_embeddings"] = True
    text_config.setdefault("bos_token_id", text_config.get("eos_token_id"))

    with open(output_dir / "config.json", "w", encoding="utf-8") as fp:
        json.dump(text_config, fp, indent=2)

    generation_config = {
        "bos_token_id": text_config.get("bos_token_id"),
        "eos_token_id": text_config.get("eos_token_id"),
        "do_sample": False,
        "temperature": 1.0,
        "top_k": 20,
        "top_p": 0.95,
    }
    with open(output_dir / "generation_config.json", "w", encoding="utf-8") as fp:
        json.dump(generation_config, fp, indent=2)

    tokenizer_src = model_path / "tokenizer.json"
    tokenizer_dst = output_dir / "tokenizer.json"
    if tokenizer_src.resolve() != tokenizer_dst.resolve():
        shutil.copy2(tokenizer_src, tokenizer_dst)

    nntr_config = {
        "model_type": "CausalLM",
        "model_tensor_type": "FP32-FP32",
        "model_file_name": output_name,
        "fc_layer_dtype": "FP32",
        "embedding_dtype": "FP32",
        "lmhead_dtype": "FP32",
        "lora_rank": 0,
        "lora_alpha": 0,
        "lora_target": [],
        "bad_word_ids": [],
        "fsu": False,
        "fsu_lookahead": 2,
        "num_to_generate": args.num_to_generate,
        "init_seq_len": args.init_seq_len,
        "max_seq_len": args.max_seq_len,
        "batch_size": 1,
        "tokenizer_file": str(tokenizer_dst),
        "sample_input": args.sample_input,
    }

    if args.disable_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        nntr_config["sample_input_ids"] = tokenizer.encode(
            args.sample_input, add_special_tokens=False
        )
        nntr_config["disable_tokenizer"] = True

    with open(output_dir / "nntr_config.json", "w", encoding="utf-8") as fp:
        json.dump(nntr_config, fp, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_name", type=str, default="nntr_qwen35_fp32.bin")
    parser.add_argument("--data_type", type=str, default="float32")
    parser.add_argument("--init_seq_len", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--num_to_generate", type=int, default=1)
    parser.add_argument(
        "--disable_tokenizer",
        action="store_true",
        help="Write sample_input_ids and force the runtime to bypass tokenizer.",
    )
    parser.add_argument(
        "--sample_input",
        type=str,
        default="<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.model_path) / "nntrainer"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / args.output_name

    dtype = np.dtype(args.data_type)
    write_runtime_files(args.model_path, output_dir, args.output_name, args)
    save_qwen35_for_nntrainer(args.model_path, output_file, dtype)
    print(output_file)


if __name__ == "__main__":
    main()
