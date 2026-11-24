# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file verify_fixture.py
# @brief Tiny Hugging Face reference fixture for one generated model (used by Verify).
# @author prachi-t17 <prachi.t@samsung.com>

"""
Tiny Hugging Face reference fixture for one generated model (used by Verify).

A generic version of upstream's
test/unittest/models/causallm_reference/generators/generate_<model>_reference.py:
shrink the model's own config.json to a tiny model of the same architecture,
build it with random weights, convert the weights with the model's *generated*
weight_converter.py, and record Hugging Face's prefill logits and greedy
tokens. The files have exactly the layout of an upstream fixture
(config/generation_config/nntr_config/tokenizer/input_ids/meta/
reference_logits/reference_tokens + the .bin), so upstream's own
runFp32DifferentialChecks compares nntrainer against them.

One deliberate difference from upstream: weights are drawn with
initializer_range 0.2 and every 1-D parameter (norm weights, biases) gets
N(0, 0.1) noise. With the default 0.02 init and zero biases, attention is
almost uniform and biases vanish, so a wrong RoPE, a missing bias or an
ignored norm weight would still match within 0.01. The pass rule itself is
upstream's: fp32 atol 0.01 on the logits, and the greedy tokens.

Runs as its own process (`python -I verify_fixture.py ...`), because it
executes the generated converter. The last stdout line is a JSON summary.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import pathlib
import sys
from typing import Any, Dict, List

HIDDEN, HEADS, INTER, MOE_INTER, VOCAB = 64, 4, 64, 32, 32
INIT_LEN, N_GEN = 4, 4
MAX_SEQ = INIT_LEN + N_GEN
INPUT_IDS = [1, 4, 2, 3]
INIT_STD, VECTOR_NOISE = 0.2, 0.1

# The tiny tokenizer upstream fixtures use (matches writeTinyTokenizer() in C++).
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

_DROP = {
    "auto_map",
    "quantization_config",
    "_name_or_path",
    "torch_dtype",
    "dtype",
    "transformers_version",
    "pad_token_id",
}


class FixtureError(RuntimeError):
    """The reference fixture could not be built."""


def _diverse(values: List[Any], count: int) -> List[Any]:
    """n entries of a per-layer list, covering each distinct value once first."""
    picked, seen = [], set()
    for value in values:
        key = json.dumps(value, sort_keys=True)
        if key not in seen:
            seen.add(key)
            picked.append(value)
    picked = picked[:count]
    for value in values:
        if len(picked) >= count:
            break
        picked.append(value)
    return picked[:count]


# pylint: disable-next=too-many-branches
def shrink_config(cfg: Dict[str, Any], layers: int = 2) -> Dict[str, Any]:
    """The same architecture with tiny dimensions; every other key is kept."""
    if "auto_map" in cfg:
        raise FixtureError(
            "the model uses custom Hugging Face code (auto_map); verify supports built-in "
            "transformers architectures only"
        )
    if "text_config" in cfg:
        raise FixtureError("multimodal configs (text_config) are not supported by verify yet")
    for key in ("hidden_size", "num_attention_heads", "num_hidden_layers", "vocab_size"):
        if not isinstance(cfg.get(key), int):
            raise FixtureError(f"config.json has no integer {key}")
    o_hidden, o_heads, o_layers = (
        cfg["hidden_size"],
        cfg["num_attention_heads"],
        cfg["num_hidden_layers"],
    )
    o_kv = cfg.get("num_key_value_heads") or o_heads
    o_head_dim = cfg.get("head_dim") or o_hidden // o_heads
    group = max(1, o_heads // max(1, o_kv))
    kv_heads = HEADS if group == 1 else max(1, HEADS // group)        # GQA/MQA stays GQA/MQA
    explicit = cfg.get("head_dim") and cfg["head_dim"] != o_hidden // o_heads
    head_dim = (
        32 if explicit else HIDDEN // HEADS
    )  # keep "head_dim != hidden/heads" if the model has it
    layers = max(1, min(layers, o_layers))

    tiny_cfg = {k: v for k, v in cfg.items() if k not in _DROP}
    tiny_cfg.update(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_key_value_heads=kv_heads,
        num_hidden_layers=layers,
        vocab_size=VOCAB,
        initializer_range=INIT_STD,
        bos_token_id=0,
        eos_token_id=[31] if isinstance(cfg.get("eos_token_id"), list) else 31,
    )
    if "head_dim" in cfg:
        tiny_cfg["head_dim"] = head_dim
    for key, val in cfg.items():
        if not isinstance(val, int) or isinstance(val, bool):
            continue
        if key == "intermediate_size":
            tiny_cfg[key] = INTER
        elif "intermediate_size" in key:  # moe_/shared_expert_intermediate_size
            tiny_cfg[key] = MOE_INTER
        elif key in ("n_embd", "d_model", "hidden_dim", "embed_dim") and val == o_hidden:
            tiny_cfg[key] = HIDDEN
        elif key == "query_pre_attn_scalar" and val == o_head_dim:
            tiny_cfg[key] = head_dim
        elif key in ("num_experts", "num_local_experts", "n_routed_experts"):
            tiny_cfg[key] = min(val, 4)
        elif key == "num_experts_per_tok":
            tiny_cfg[key] = min(val, 2)
        elif key in ("max_window_layers", "first_k_dense_replace"):
            tiny_cfg[key] = min(val, layers)
        elif key == "decoder_sparse_step":
            tiny_cfg[key] = 1
    if cfg.get("sliding_window") and cfg.get(
        "use_sliding_window", True
    ):  # only a window the model uses
        tiny_cfg["sliding_window"] = min(cfg["sliding_window"], INIT_LEN)
    if not (
        cfg.get("rope_scaling")
        or cfg.get("rope_parameters", {}).get("original_max_position_embeddings")
    ):
        tiny_cfg["max_position_embeddings"] = MAX_SEQ
    for key, val in cfg.items():  # per-layer lists / layer index lists
        if (
            isinstance(val, list)
            and len(val) == o_layers
            and key not in ("eos_token_id", "architectures")
        ):
            tiny_cfg[key] = _diverse(val, layers)
        elif key in ("mlp_only_layers",) and isinstance(val, list):
            tiny_cfg[key] = [i for i in val if isinstance(i, int) and i < layers]
    return tiny_cfg


def _load_converter(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("workbench_weight_converter", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod


# pylint: disable-next=too-many-locals,too-many-arguments,too-many-branches
def convert(converter: pathlib.Path, model, hf_config, n_layers: int, bin_path: pathlib.Path,
            params_dict: Dict[str, Any] = None) -> str:
    """Call the converter's save_<x>_for_nntrainer(params, n_layers | config, dtype, file)
    with `model`'s state dict, or with `params_dict` (real weights read from safetensors)."""
    mod = _load_converter(converter)
    fns = [
        (n, f)
        for n, f in vars(mod).items()
        if callable(f)
        and n.startswith("save_")
        and n.endswith("_for_nntrainer")
        and getattr(f, "__module__", "") == mod.__name__
    ]
    fns.sort(key=lambda nf: ("embedding" in nf[0], nf[0]))
    if not fns:
        raise FixtureError(
            f"{converter.name} has no save_<model>_for_nntrainer(params, n_layers, dtype, file)"
        )
    name, func = fns[0]
    params = list(inspect.signature(func).parameters.values())
    args: List[Any] = []
    for i, param in enumerate(params):
        pname = param.name.lower()
        if i == 0:
            if pname == "model":
                if model is None:
                    raise FixtureError(f"{name} takes a model object, not a state dict")
                args.append(model)
            else:
                args.append(params_dict if params_dict is not None else model.state_dict())
        elif pname in ("n_layers", "num_layers", "layers", "n_layer", "num_hidden_layers"):
            args.append(n_layers)
        elif pname in ("config", "cfg", "hf_config", "model_config"):
            args.append(hf_config)
        elif pname in ("dtype", "data_type"):
            args.append("float32")
        elif pname in ("file", "f", "out", "fp", "output"):
            args.append(None)                                   # filled below
        elif param.default is not inspect.Parameter.empty:
            break
        else:
            raise FixtureError(f"{name}: cannot tell what to pass for parameter {param.name!r}")
    kwargs = {}
    if any(p.name == "tie_word_embeddings" for p in params):
        kwargs["tie_word_embeddings"] = bool(getattr(hf_config, "tie_word_embeddings", False))
    with open(bin_path, "wb") as handle:
        args = [handle if a is None else a for a in args]
        func(*args, **kwargs)
    if bin_path.stat().st_size == 0:
        raise FixtureError(f"{name} wrote no weights")
    return name


# pylint: disable-next=too-many-locals,too-many-arguments
def build(config_path: pathlib.Path, converter: pathlib.Path, nntr_config_path: pathlib.Path,
          out_dir: pathlib.Path, seed: int = 42, layers: int = 2) -> Dict[str, Any]:
    """Build the tiny Hugging Face reference fixture for a generated model."""
    # pylint: disable-next=import-outside-toplevel,import-error  # optional package
    import numpy as np
    # pylint: disable-next=import-outside-toplevel,import-error  # optional package
    import torch
    # pylint: disable-next=import-outside-toplevel,import-error  # optional package
    import transformers
    # pylint: disable-next=import-outside-toplevel,import-error  # optional package
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    tiny = shrink_config(cfg, layers)
    if "model_type" not in tiny:
        arch = (tiny.get("architectures") or [""])[0]
        cls = getattr(transformers, arch, None)
        if cls is None:
            raise FixtureError(f"config.json has no model_type and transformers has no {arch!r}")
        tiny["model_type"] = cls.config_class.model_type
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(tiny, indent=2), encoding="utf-8")
    hf_config = AutoConfig.from_pretrained(str(out_dir))

    torch.manual_seed(seed)
    try:
        model = AutoModelForCausalLM.from_config(
            hf_config, dtype=torch.float32, attn_implementation="eager"
        )
    except TypeError:
        model = AutoModelForCausalLM.from_config(
            hf_config, torch_dtype=torch.float32, attn_implementation="eager"
        )
    model.eval()
    gen = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for _name, path in model.named_parameters():
            if path.dim() == 1:
                path.add_(torch.randn(path.shape, generator=gen, dtype=path.dtype) * VECTOR_NOISE)

    bin_name = f"nntr_{out_dir.name}_fp32.bin"
    func = convert(converter, model, hf_config, tiny["num_hidden_layers"], out_dir / bin_name)

    tok_path = out_dir / "tokenizer.json"
    tok_path.write_text(json.dumps(TINY_TOKENIZER, indent=2), encoding="utf-8")
    nntr = (
        json.loads(nntr_config_path.read_text(encoding="utf-8"))
        if nntr_config_path.is_file()
        else {}
    )
    nntr.update(
        {
            "bad_word_ids": [],
            "batch_size": 1,
            "embedding_dtype": "FP32",
            "fc_layer_dtype": "FP32",
            "lmhead_dtype": "FP32",
            "init_seq_len": INIT_LEN,
            "max_seq_len": MAX_SEQ,
            "model_file_name": bin_name,
            "model_tensor_type": "FP32-FP32",
            "model_type": "CausalLM",
            "num_to_generate": 1,
            "tokenizer_file": str(tok_path),
            "fsu": False,
        }
    )
    (out_dir / "nntr_config.json").write_text(json.dumps(nntr, indent=2), encoding="utf-8")
    (out_dir / "generation_config.json").write_text(
        json.dumps(
            {
                "bos_token_id": 0,
                "eos_token_id": 31,
                "do_sample": False,
                "top_k": 1,
                "top_p": 1.0,
                "temperature": 1.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    ids = torch.tensor([INPUT_IDS], dtype=torch.long)
    with torch.no_grad():
        logits = model(ids, use_cache=False).logits[0, -1, :].float().tolist()
        out = model.generate(
            ids,
            max_new_tokens=N_GEN,
            do_sample=False,
            use_cache=True,
            eos_token_id=None,
            pad_token_id=0,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    tokens = out[0, len(INPUT_IDS):].tolist()[:N_GEN]
    (out_dir / "reference_logits.json").write_text(json.dumps(logits), encoding="utf-8")
    (out_dir / "reference_tokens.json").write_text(json.dumps(tokens), encoding="utf-8")
    (out_dir / "input_ids.json").write_text(json.dumps(INPUT_IDS), encoding="utf-8")
    meta = {
        "seed": seed,
        "n_gen": N_GEN,
        "input_ids": INPUT_IDS,
        "logits_atol_fp32": 1e-2,
        "logits_atol_q40": 5.0,
        "prefix_match_min": 2,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "workbench": {
            "source_config": str(config_path),
            "converter": str(converter),
            "converter_fn": func,
            "initializer_range": INIT_STD,
            "vector_noise": VECTOR_NOISE,
            "layers": tiny["num_hidden_layers"],
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "ok": True,
        "fixture": str(out_dir),
        "converter_fn": func,
        "argmax": int(np.argmax(logits)),
        "max_abs_logit": max(abs(v) for v in logits),
        "tokens": tokens,
        "layers": tiny["num_hidden_layers"],
        "params": sum(p.numel() for p in model.parameters()),
    }


def convert_real(
    model_dir: pathlib.Path, converter: pathlib.Path, bin_path: pathlib.Path
) -> Dict[str, Any]:
    """The model's real weights (every *.safetensors shard in `model_dir`) through the
    generated converter."""
    # pylint: disable-next=import-outside-toplevel,import-error  # optional package
    from safetensors.torch import load_file
    # pylint: disable-next=import-outside-toplevel,import-error  # optional package
    from transformers import AutoConfig

    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FixtureError(f"no *.safetensors in {model_dir}")
    params: Dict[str, Any] = {}
    for shard in shards:
        # float32, as upstream converters get them (their main() loads the
        # model in float32); numpy has no bfloat16.
        params.update(
            {
                k: (t.float() if t.is_floating_point() else t)
                for k, t in load_file(str(shard)).items()
            }
        )
    hf_config = AutoConfig.from_pretrained(str(model_dir))
    cfg = getattr(hf_config, "text_config", None) or hf_config
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = bin_path.with_suffix(".part")
    func = convert(converter, None, hf_config, cfg.num_hidden_layers, tmp, params_dict=params)
    tmp.replace(bin_path)
    return {
        "ok": True,
        "bin": str(bin_path),
        "converter_fn": func,
        "bytes": bin_path.stat().st_size,
        "tensors": len(params),
        "shards": len(shards),
    }


def main(argv=None) -> int:
    """Command line: build a reference fixture, or convert the real weights."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--config", type=pathlib.Path)
    parser.add_argument("--converter", required=True, type=pathlib.Path)
    parser.add_argument("--nntr-config", type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument(
        "--real-weights",
        type=pathlib.Path,
        help=(
            "convert this model folder's real weights to --out (a .bin) instead of making a fixture"
        ),
    )
    args = parser.parse_args(argv)
    try:
        if args.real_weights:
            summary = convert_real(
                args.real_weights.resolve(), args.converter.resolve(), args.out.resolve()
            )
        else:
            if not (args.config and args.nntr_config):
                raise FixtureError("--config and --nntr-config are needed for a fixture")
            summary = build(
                args.config.resolve(),
                args.converter.resolve(),
                args.nntr_config.resolve(),
                args.out.resolve(),
                args.seed,
                args.layers,
            )
    except FixtureError as exc:
        summary = {"ok": False, "error": str(exc)}
    except Exception as exc:  # pylint: disable=broad-except  # converter / transformers failures
        summary = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(summary))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
