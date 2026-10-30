#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The rotated ternary Gemma4-26B-A4B checkpoint -> the HTP pack, in one run.

Input: the HF directory of the pre-rotated, fake-quantized BF16 checkpoint
(experts exact ternary per output channel, FC/head on an int4 grid, and a
rotation_meta.json naming the Hadamard rotation and its sign seed).

Stages:
  0. here: HF -> positional QS2CX_WH pack (experts as exact ternary codes with
     their BF16 scale and column sum, embedding Q4_0, norms/router FP32, and
     FC/head placeholders of Q4_0 size)
  1. tools/convert_tl2_model (built here): QS2CX_WH experts -> native TL2CX_WH
  2. res/gemma4/convert_gemma4_tl2.py -> tools/convert_gemma4_fc_wh.py: FC/head
     as affine QS4CX_WH, the int4 grid recovered from the BF16
  3. here: rotation_signs.json from rotation_meta.json's seed, the nntr_config
     rotation keys, config.json's global attention keys

The experts of --affine-layers (default 28,29: the recipe's 4-bit expert
layers) are written as QS4CX_WH_AFFINE, the int4 grid recovered from the BF16
like FC/head; every other MoE layer must be exact ternary and becomes TL2CX_WH.
A layer whose values are not ternary is written affine whatever the option.

Usage:
  python3 weight_converter.py --source <HF dir> --output <new dir> [--affine-layers 28,29]
Needs numpy; gcc-13 and g++ for stage 1.
The output directory must not exist: nothing is overwritten.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools"))
from convert_gemma4_fc_wh import pack as pack_wh, recover as recover_wh  # noqa: E402
MODEL_FILE = "nntr_gemma4_26b_a4b_tl2_rotated.bin"
ACT_ENCODINGS = "p0-linear8-mse-asym-safe16-boundaries.encodings.json"
ACT_TABLE_FILE = "act_scales.json"
STAGE0_FILE = "nntr_gemma4_26b_a4b_qs2_stage0.bin"
PALETTE = bytes((0xff, 0x00, 0x01, 0x01))  # QS2CX_WH codes 0, 1, 2 -> -1, 0, +1
COPIED = ("config.json", "tokenizer.json", "tokenizer_config.json",
          "generation_config.json", "chat_template.jinja")
# Runtime keys of the pack; convert_gemma4_tl2.py and convert_gemma4_fc_wh.py
# then switch the dtypes and engines to the HTP values.
STAGE0_CONFIG = {
    "attn_proj_engine": "htp", "bad_word_ids": [], "batch_size": 1,
    "dense_ffn_engine": "htp", "embedding_dtype": "Q4_0",
    "fc_layer_dtype": "Q4_0", "init_seq_len": 2048, "lmhead_dtype": "Q4_0",
    "max_seq_len": 2304, "model_file_name": STAGE0_FILE,
    "model_tensor_type": "Q4_0-FP32", "model_type": "CausalLM",
    "moe_cache_experts": 16, "moe_decode_prefetch": True, "moe_engine": "htp",
    "moe_layer_dtype": "QS2CX_WH", "num_to_generate": 128,
    "sample_input": "<bos><|turn>user\nHello<turn|>\n<|turn>model\n",
    "skip_prefill": False, "tokenizer_file": "tokenizer.json"}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def act_scale_table(source):
    """Static 8-bit linear-input scales: one (scale, offset) per site, the
    '<module>/in' sites of the prefill graph (the generation graph repeats them),
    sorted by name. None when the checkpoint has no encodings file."""
    path = source / ACT_ENCODINGS
    if not path.is_file():
        return None
    graph = json.loads(path.read_text())["graphs"]["prefill"]
    sites = []
    for name in sorted(graph):
        if not name.endswith("/in"):
            continue
        enc = graph[name]["encodings"]
        if len(enc) != 1 or enc[0]["bitwidth"] != 8 or enc[0]["is_symmetric"] != "False":
            continue  # 16-bit boundaries stay out of the table
        sites.append({"name": name, "scale": float(enc[0]["scale"]),
                      "offset": int(enc[0]["offset"])})
    return sites


class Checkpoint:
    """BF16 safetensors read as float32 without torch."""

    def __init__(self, root):
        self.root = root
        self.meta = {}
        index = json.loads((root / "model.safetensors.index.json").read_text())
        for name in sorted(set(index["weight_map"].values())):
            with (root / name).open("rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                header = json.loads(stream.read(size))
            for tensor, entry in header.items():
                if tensor != "__metadata__":
                    self.meta[tensor] = (name, 8 + size, entry)

    def shape(self, name):
        return list(self.meta[name][2]["shape"])

    def tensor(self, name, rows=None):
        file, base, entry = self.meta[name]
        if entry["dtype"] != "BF16":
            raise ValueError(f"{name}: expected BF16, got {entry['dtype']}")
        start, end = entry["data_offsets"]
        shape = list(entry["shape"])
        if rows is not None:
            per_row = int(np.prod(shape[1:])) * 2
            start, end = start + rows[0] * per_row, start + rows[1] * per_row
            shape[0] = rows[1] - rows[0]
        with (self.root / file).open("rb") as stream:
            stream.seek(base + start)
            raw = np.fromfile(stream, np.uint16, (end - start) // 2)
        return (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape)


def is_ternary(w):
    scale = np.abs(w).max(axis=-1, keepdims=True)
    return bool(np.array_equal(np.rint(w / np.where(scale > 0, scale, 1)) * scale, w))


def build_plan(config, ckpt, affine_layers):
    """The pack's record order; shapes are [K, N] (HF stores [N, K])."""
    hidden, experts = config["hidden_size"], config["num_experts"]
    plan = []

    def add(name, dtype, shape, hf):
        plan.append({"name": name, "dtype": dtype, "shape": shape, "hf": hf})

    def fc(name, hf):
        n, k = ckpt.shape(hf)
        add(name, "Q4_0", [k, n], hf)  # placeholder; stage 2 writes the record

    def f32(name, hf, shape=None):
        add(name, "FP32", shape or ckpt.shape(hf), hf)

    add("embedding0:Embedding", "Q4_0", ckpt.shape("model.embed_tokens.weight"),
        "model.embed_tokens.weight")
    for layer in range(config["num_hidden_layers"]):
        h, p = f"model.layers.{layer}.", f"layer{layer}_"
        f32(p + "attention_norm:gamma", h + "input_layernorm.weight")
        fc(p + "wq:weight", h + "self_attn.q_proj.weight")
        f32(p + "q_norm:gamma", h + "self_attn.q_norm.weight")
        fc(p + "wk:weight", h + "self_attn.k_proj.weight")
        f32(p + "k_norm:gamma", h + "self_attn.k_norm.weight")
        if h + "self_attn.v_proj.weight" in ckpt.meta:  # absent where K == V
            fc(p + "wv:weight", h + "self_attn.v_proj.weight")
        fc(p + "attention_out:weight", h + "self_attn.o_proj.weight")
        f32(p + "post_attention_norm:gamma", h + "post_attention_layernorm.weight")
        f32(p + "pre_ffn_norm:gamma", h + "pre_feedforward_layernorm.weight")
        fc(p + "ffn_gate:weight", h + "mlp.gate_proj.weight")
        fc(p + "ffn_up:weight", h + "mlp.up_proj.weight")
        fc(p + "ffn_down:weight", h + "mlp.down_proj.weight")
        f32(p + "post_ffn_norm_1:gamma", h + "post_feedforward_layernorm_1.weight")
        f32(p + "pre_ffn_norm_2:gamma", h + "pre_feedforward_layernorm_2.weight")
        f32(p + "router_norm:gamma", h + "router.scale")
        f32(p + "sparse_moe:gate", h + "router.proj.weight", [hidden, experts])
        f32(p + "sparse_moe:expert_bias", h + "router.per_expert_scale")
        gate_up = ckpt.shape(h + "experts.gate_up_proj")  # [E, N, K]
        down = ckpt.shape(h + "experts.down_proj")
        if gate_up[0] != experts or down[0] != experts:
            raise ValueError(f"layer {layer}: expert count mismatch")
        affine = layer in affine_layers or not all(  # expert 0 decides; stage 0 checks the rest
            is_ternary(ckpt.tensor(h + name, (0, 1))) for name in ("experts.gate_up_proj",
                                                                   "experts.down_proj"))
        if affine and layer not in affine_layers:
            print(f"layer {layer}: experts are not ternary, written affine", flush=True)
        dtype = "QS4CX_WH_AFFINE" if affine else "QS2CX_WH"
        for e in range(experts):
            add(f"{p}sparse_moe:expert_gate_up_{e}", dtype,
                [gate_up[2], gate_up[1]], (h + "experts.gate_up_proj", e))
            add(f"{p}sparse_moe:expert_down_{e}", dtype,
                [down[2], down[1]], (h + "experts.down_proj", e))
        f32(p + "post_ffn_norm_2:gamma", h + "post_feedforward_layernorm_2.weight")
        f32(p + "post_ffn_norm:gamma", h + "post_feedforward_layernorm.weight")
        f32(p + "layer_scalar:scalar_multiplier", h + "layer_scalar")
    f32("output_norm:gamma", "model.norm.weight")
    fc("output_of_causallm:weight", "lm_head.weight")

    offset = 0
    for entry in plan:
        k, n = (entry["shape"] + [1])[:2]
        if entry["dtype"] == "FP32":
            size = 4 * int(np.prod(entry["shape"]))
        elif entry["dtype"] == "Q4_0":
            if k % 32:
                raise ValueError(f"{entry['name']}: Q4_0 needs K % 32 == 0")
            size = k * n // 32 * 18
        elif entry["dtype"] == "QS4CX_WH_AFFINE":
            if k % 32 or n % 32:
                raise ValueError(f"{entry['name']}: WH needs complete 32x32 tiles")
            size = k * n // 2 + 12 * n
        else:
            if k % 32 or n % 128:
                raise ValueError(f"{entry['name']}: TL2 needs K % 32 and N % 128 == 0")
            size = k * n // 4 + 4 + 8 * n
        entry["offset"], entry["bytes"] = offset, size
        offset += size
    return plan, offset


def q4_0(x):
    """ggml quantize_row_q4_0_ref on [rows, cols]: f16 d = max/-8, nibbles +8."""
    rows, cols = x.shape
    blocks = x.reshape(rows, cols // 32, 32)
    at = np.abs(blocks).argmax(axis=2)
    extreme = np.take_along_axis(blocks, at[:, :, None], 2)[:, :, 0]
    d = (extreme / -8).astype(np.float16)
    inverse = np.where(d != 0, 1 / d.astype(np.float32), 0).astype(np.float32)
    q = np.minimum(15, (blocks * inverse[:, :, None] + 8.5).astype(np.int8)).astype(np.uint8)
    out = np.empty((rows, cols // 32, 18), np.uint8)
    out[:, :, :2] = d.astype("<f2").view(np.uint8).reshape(rows, cols // 32, 2)
    out[:, :, 2:] = q[:, :, :16] | (q[:, :, 16:] << 4)
    return out.reshape(rows, -1)


def qs2_record(w):
    """QS2CX_WH of an exact ternary [N, K] matrix: codes, palette, scales, sums.

    Byte of (k, n): ((k//32)*(N//32) + n//32)*256 + (r//16)*128 + n%32*4 + r%4,
    shift 2*((r//4)%4), r = k % 32 (tools/convert_tl2_model.cpp).
    """
    n, k = w.shape
    scale = np.abs(w).max(axis=1)
    q = np.rint(w / np.where(scale > 0, scale, 1)[:, None]).astype(np.int8)
    if not np.array_equal(q * scale[:, None], w):
        raise ValueError("not an exact ternary matrix")
    tile = (q.T + 1).astype(np.uint8).reshape(k // 32, 2, 4, 4, n // 32, 32)
    tile = tile.transpose(0, 4, 1, 5, 3, 2)  # [kt, nt, r_hi, c, r_lo, r_mid]
    packed = np.left_shift(tile, 2 * np.arange(4, dtype=np.uint8)).sum(axis=5, dtype=np.uint8)
    return (packed.tobytes() + PALETTE + scale.astype("<f4").tobytes() +
            q.sum(axis=1, dtype=np.int32).astype("<f4").tobytes()), q


def affine_record(w):
    """QS4CX_WH_AFFINE of an [N, K] matrix on an int4 grid: WH nibble tiles, N
    f32 scales, N f32 column sums, N i32 channel offsets (convert_gemma4_fc_wh)."""
    q, scale, offsets = recover_wh(w)
    return (pack_wh(q).tobytes() + scale.tobytes() +
            q.sum(axis=1, dtype=np.int32).astype("<f4").tobytes() + offsets.tobytes()), q


class WhDecoder:
    """Independent read-back of the WH nibble tiles (htp_wh_layout.h formula)."""

    def __init__(self, k, n):
        r, c = np.arange(32)[:, None], np.arange(32)[None, :]
        self.byte = (r // 8) * 128 + c * 4 + r % 4
        self.shift = ((r // 4) % 2) * 4
        self.k, self.n = k, n

    def __call__(self, record):
        k, n = self.k, self.n
        tiles = np.frombuffer(record, np.uint8, k * n // 2).reshape(k // 32, n // 32, 512)
        values = ((tiles[:, :, self.byte] >> self.shift) & 15).astype(np.int8)
        values[values > 7] -= 16
        return values.transpose(1, 3, 0, 2).reshape(n, k)  # [N, K]


class Qs2Decoder:
    """Independent read-back of qs2_record through the address formula."""

    def __init__(self, k, n):
        kk, nn = np.arange(k)[:, None], np.arange(n)[None, :]
        r = kk % 32
        self.at = ((kk // 32) * (n // 32) + nn // 32) * 256 + (r // 16) * 128 + (nn % 32) * 4 + r % 4
        self.shift = (2 * ((r // 4) % 4)).astype(np.uint8)
        self.codes = k * n // 4

    def __call__(self, record):
        body = np.frombuffer(record, np.uint8, self.codes)
        return (((body[self.at] >> self.shift) & 3).astype(np.int8) - 1).T  # [N, K]


def stage0(source, qs2, config, ckpt, plan, total):
    qs2.mkdir()
    binary = qs2 / STAGE0_FILE
    decoders, verified, started, affine_layers = {}, 0, time.time(), set()
    rsqrt_hidden = np.float32(1 / np.sqrt(config["hidden_size"]))
    with binary.open("xb") as dst:
        dst.truncate(total)  # FC/head placeholders stay zero; stage 2 writes them
        for entry in plan:
            name, dtype, hf = entry["name"], entry["dtype"], entry["hf"]
            if dtype == "Q4_0" and name != "embedding0:Embedding":
                continue
            dst.seek(entry["offset"])
            if dtype == "Q4_0":
                rows = ckpt.shape(hf)[0]
                for r0 in range(0, rows, 8192):
                    dst.write(q4_0(ckpt.tensor(hf, (r0, min(rows, r0 + 8192)))).tobytes())
            elif dtype == "FP32":
                w = ckpt.tensor(hf)
                if name.endswith("router_norm:gamma"):
                    w = w * rsqrt_hidden  # the router's 1/sqrt(hidden) folded in
                elif name.endswith("sparse_moe:gate"):
                    w = w.T  # [experts, hidden] -> [hidden, experts]
                if list(w.shape) != entry["shape"]:
                    raise ValueError(f"{name}: shape {list(w.shape)} != {entry['shape']}")
                dst.write(np.ascontiguousarray(w, dtype="<f4").tobytes())
            else:
                w = ckpt.tensor(hf[0], (hf[1], hf[1] + 1))[0]
                k, n = entry["shape"]
                if dtype == "QS2CX_WH":
                    record, q = qs2_record(w)
                    decoder = decoders.setdefault((dtype, k, n), Qs2Decoder(k, n))
                    verified += 1
                else:
                    record, q = affine_record(w)
                    decoder = decoders.setdefault((dtype, k, n), WhDecoder(k, n))
                    affine_layers.add(int(name[5:name.index("_")]))
                if len(record) != entry["bytes"]:
                    raise ValueError(f"{name}: record size")
                if not np.array_equal(decoder(record), q):
                    raise ValueError(f"{name}: {dtype} code layout read-back failed")
                dst.write(record)
                if hf[1] == 0 and name.endswith("gate_up_0"):
                    print(f"stage 0: {name.split('_')[0]} experts ... "
                          f"({time.time() - started:.0f} s)", flush=True)
    for name in COPIED:
        if (source / name).is_file():
            shutil.copyfile(source / name, qs2 / name)
    config_out = dict(STAGE0_CONFIG)
    if affine_layers:
        config_out["moe_layer_dtype_per_layer"] = {
            str(layer): "QS4CX_WH_AFFINE" for layer in sorted(affine_layers)}
    (qs2 / "nntr_config.json").write_text(json.dumps(config_out, indent=2) + "\n")
    report = {"source": str(source.resolve()), "model_file": STAGE0_FILE, "bytes": total,
              "sha256": sha256(binary), "expert_tensors_verified": verified,
              "tensors": [{k: v for k, v in e.items() if k != "hf"} for e in plan]}
    (qs2 / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_tl2_tool(repo, build_dir):
    """tools/convert_tl2_model with the scalar TL2 kernel (gemma4-tl2-affine-handoff.md)."""
    tool = build_dir / "convert_tl2_model"
    if tool.is_file():
        return tool
    build_dir.mkdir(parents=True, exist_ok=True)
    obj = build_dir / "tl2_scalar.o"
    hvx = repo / "nntrainer/tensor/htp_backend/hvx"
    subprocess.run(["gcc-13", "-O2", "-I" + str(hvx),
                    "-c", str(hvx / "hvx_gemv_u8tl2.c"), "-o", str(obj)], check=True)
    subprocess.run(["g++", "-std=c++17", "-O2", "-pthread",
                    f"-I{repo}/nntrainer/tensor", f"-I{repo}/nntrainer/tensor/htp_backend",
                    f"-I{hvx}", str(repo / "tools/convert_tl2_model.cpp"), str(obj),
                    "-o", str(tool)], check=True)
    return tool


def rotation_signs(meta, config):
    """The checkpoint's +-1 per input dimension: RandomState(seed).randint(0, 2),
    hidden then moe_intermediate from one stream, sign = 1 - 2 * bit."""
    if meta.get("mode") != "hadamard" or meta.get("rotation_mode") != "both":
        raise ValueError(f"unsupported rotation_meta: {meta}")
    hidden, inter = config["hidden_size"], config["moe_intermediate_size"]
    if (meta.get("hidden"), meta.get("moe_intermediate"), meta.get("n_moe")) != (
            hidden, inter, config["num_hidden_layers"]):
        raise ValueError("rotation_meta.json does not describe this config")
    rng = np.random.RandomState(meta["seed"])
    signs = [(1 - 2 * rng.randint(0, 2, size)).tolist() for size in (hidden, inter)]
    return {"format": "nntr_hadamard_signs_v1",
            "hadamard": "H44xHp, x <- (x H) * s per input dimension",
            "source": f"numpy RandomState({meta['seed']}).randint(0, 2): hidden then "
                      "moe_intermediate, s = 1 - 2 * bit (rotation_meta.json)",
            "in": signs[0], "down": signs[1]}


def global_attention_keys(config, ckpt):
    """global_head_dim / num_global_key_value_heads, read off the global layers."""
    local = config["num_attention_heads"] * config["head_dim"]
    for layer in range(config["num_hidden_layers"]):
        q_out = ckpt.shape(f"model.layers.{layer}.self_attn.q_proj.weight")[0]
        if q_out != local:
            head_dim = q_out // config["num_attention_heads"]
            kv = ckpt.shape(f"model.layers.{layer}.self_attn.k_proj.weight")[0] // head_dim
            return {"global_head_dim": head_dim, "num_global_key_value_heads": kv}
    return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="HF checkpoint dir")
    parser.add_argument("--output", type=Path, required=True, help="new dir (must not exist)")
    parser.add_argument("--repo", type=Path, default=REPO, help="nntrainer checkout")
    parser.add_argument("--build-dir", type=Path, default=Path("/tmp/gemma4_convert_build"))
    parser.add_argument("--affine-layers", default="28,29",
                        help="MoE layers whose experts are written QS4CX_WH_AFFINE "
                             "(comma separated, '' for none)")
    parser.add_argument("--keep-stage0", action="store_true",
                        help="keep the QS2CX_WH intermediate next to the output")
    args = parser.parse_args()
    source, output, repo = args.source.resolve(), args.output.resolve(), args.repo.resolve()
    qs2 = output.with_name(output.name + ".qs2")
    for path in (output, qs2):
        if path.exists():
            sys.exit(f"refusing to overwrite existing {path}")
    config = json.loads((source / "config.json").read_text())
    meta = json.loads((source / "rotation_meta.json").read_text())
    if config.get("architectures") != ["Gemma4ForCausalLM"]:
        sys.exit(f"not a Gemma4ForCausalLM checkpoint: {config.get('architectures')}")
    signs = rotation_signs(meta, config)
    ckpt = Checkpoint(source)
    affine_layers = {int(x) for x in args.affine_layers.split(",") if x.strip()}
    plan, total = build_plan(config, ckpt, affine_layers)
    if shutil.disk_usage(output.parent).free < 3 * total + 1024**3:
        sys.exit("need room for the stage-0 pack, the native pack and the output")
    tool = build_tl2_tool(repo, args.build_dir)
    sys.path.insert(0, str(repo / "Applications/CausalLM/res/gemma4"))
    from convert_gemma4_tl2 import convert  # noqa: E402  (stages 1 and 2)

    print(f"stage 0: {len(plan)} records, {total:,} bytes -> {qs2}", flush=True)
    stage0_report = stage0(source, qs2, config, ckpt, plan, total)
    print(f"stage 0: done, sha256 {stage0_report['sha256']}", flush=True)
    convert(qs2, output, tool, source)

    report = json.loads((output / "conversion_report.json").read_text())
    binary = output / report["model_file"]
    binary.rename(output / MODEL_FILE)
    (output / "rotation_signs.json").write_text(json.dumps(signs) + "\n")
    nntr_config = json.loads((output / "nntr_config.json").read_text())
    nntr_config.update(model_file_name=MODEL_FILE, moe_hadamard=meta["rotation_mode"],
                       moe_hadamard_scope="all", moe_hadamard_signs="rotation_signs.json")
    table = act_scale_table(source)
    if table:
        # Tail of the binary: float32 scale, int32 offset per site. The loader
        # reads the weights ahead of it and ignores the tail; the HTP runtime
        # reads it only with htp_act_quant "static" (default "dynamic").
        size = (output / MODEL_FILE).stat().st_size
        with (output / MODEL_FILE).open("ab") as stream:
            for site in table:
                stream.write(struct.pack("<fi", site["scale"], site["offset"]))
        (output / ACT_TABLE_FILE).write_text(json.dumps(
            {"format": "nntr_act_scales_v1", "source": ACT_ENCODINGS,
             "record": "<float32 scale, int32 offset>", "sites": table}, indent=1) + "\n")
        nntr_config.update(act_scale_offset=size, act_scale_bytes=8 * len(table),
                           act_scale_sites=len(table), act_scale_json=ACT_TABLE_FILE)
    (output / "nntr_config.json").write_text(json.dumps(nntr_config, indent=2) + "\n")
    extra = global_attention_keys(config, ckpt)
    if extra.keys() - config.keys():
        (output / "config.json").write_text(json.dumps({**config, **extra}, indent=2) + "\n")
    report.update(model_file=MODEL_FILE, hf_source=str(source), rotation_meta=meta,
                  rotation_signs="rotation_signs.json (regenerated from the seed)",
                  stage0_qs2_sha256=stage0_report["sha256"],
                  config_json_added=sorted(extra.keys() - config.keys()),
                  experts="TL2CX_WH (exact ternary, BF16 scales); QS4CX_WH_AFFINE for "
                          "layers " + ",".join(
                              sorted(nntr_config.get("moe_layer_dtype_per_layer", {}),
                                     key=int)))
    (output / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not args.keep_stage0:
        shutil.rmtree(qs2)
    print(f"done: {output / MODEL_FILE} ({(output / MODEL_FILE).stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
