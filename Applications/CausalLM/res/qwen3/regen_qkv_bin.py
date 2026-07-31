#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

"""Regenerate a Qwen3 nntr_*.bin with Q/K/V weights stored contiguously.

Background (docs/backend_guide/HEXAGON_NPU_OBSERVATION_LOG.md, "QKV batching
implementation" entries): QKVLayer batches Q/K/V into one graph node, moving
q_norm/k_norm to *after* all three projections in node order. nntrainer's
weight loader (NeuralNetwork::load in nntrainer/models/neuralnet.cpp)
assigns file offsets by walking node order and summing tensor sizes
sequentially, assuming that order matches the file's byte layout. A .bin
written for the original (3-separate-FC) topology has q_norm between Q and
K, and k_norm between K and V - incompatible with the batched topology,
where the loader assumes Q, K, V are contiguous. Symptom: Q loads correctly
(first weight, position unaffected), K is corrupted by exactly q_norm's byte
size, V by q_norm+k_norm's combined size, and the corruption compounds
through every later layer via the residual stream - producing degenerate
repeated-token output.

This script re-reads the original HF checkpoint directly - no torch,
transformers, or safetensors library needed. The format is simple:
[8-byte header length][JSON header][raw tensor bytes], and BF16 -> FP32 is
just a left-shift into the top 16 bits of a uint32. Supports both
single-file (model.safetensors) and sharded (model.safetensors.index.json +
model-NNNNN-of-NNNNN.safetensors) checkpoints.

up_proj/gate_proj are NOT reordered: the original file already has them
contiguous (no norm interleaved between them), so GateUpLayer's batching was
never affected by this bug - only QKVLayer needed a matching .bin.

Usage:
  python3 regen_qkv_bin.py --checkpoint /path/to/hf/checkpoint --output /path/to/nntr_qwen3_fp32.bin
"""

import argparse
import json
import os
import struct

import numpy as np


class SafetensorsReader:
    """Reads tensors directly from a single-file or sharded safetensors
    checkpoint, with BF16 -> FP32 conversion. No safetensors/torch dependency."""

    def __init__(self, checkpoint_dir):
        single_path = os.path.join(checkpoint_dir, "model.safetensors")
        index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")

        self.checkpoint_dir = checkpoint_dir
        self._headers = {}  # shard_path -> (header, data_start)

        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            self.tensor_to_shard = {
                name: os.path.join(checkpoint_dir, shard)
                for name, shard in index["weight_map"].items()
            }
        elif os.path.exists(single_path):
            header, data_start = self._read_header(single_path)
            self._headers[single_path] = (header, data_start)
            self.tensor_to_shard = {name: single_path for name in header
                                    if name != "__metadata__"}
        else:
            raise FileNotFoundError(
                f"No model.safetensors or model.safetensors.index.json found "
                f"in {checkpoint_dir}")

    @staticmethod
    def _read_header(path):
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(hlen))
            data_start = 8 + hlen
        return header, data_start

    def has(self, name):
        return name in self.tensor_to_shard

    def read(self, name, transpose=False):
        shard_path = self.tensor_to_shard[name]
        if shard_path not in self._headers:
            self._headers[shard_path] = self._read_header(shard_path)
        header, data_start = self._headers[shard_path]

        meta = header[name]
        assert meta["dtype"] == "BF16", (
            f"unexpected dtype for {name}: {meta['dtype']} "
            "(only BF16 checkpoints are handled)")
        start, end = meta["data_offsets"]
        shape = meta["shape"]

        with open(shard_path, "rb") as f:
            f.seek(data_start + start)
            raw = f.read(end - start)

        u16 = np.frombuffer(raw, dtype="<u2").reshape(shape)
        # bfloat16 -> float32: bf16 is the top 16 bits of an IEEE-754 float32.
        u32 = u16.astype(np.uint32) << 16
        arr = u32.view(np.float32).reshape(shape)

        if transpose:
            arr = np.ascontiguousarray(arr.T)
        return arr


def convert(checkpoint_dir, output_path):
    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        config = json.load(f)

    n_layers = config["num_hidden_layers"]
    tie_word_embeddings = config.get("tie_word_embeddings", True)

    reader = SafetensorsReader(checkpoint_dir)

    with open(output_path, "wb") as out:
        def w(name, transpose=False):
            reader.read(name, transpose=transpose).tofile(out)

        w("model.embed_tokens.weight")

        for i in range(n_layers):
            p = f"model.layers.{i}."
            print(f"layer {i}/{n_layers}...")

            w(f"{p}input_layernorm.weight")

            # Q, K, V contiguous (the fix), THEN q_norm, k_norm, THEN o_proj -
            # matches QKVLayer's batched node order, not the original
            # interleaved order (q_proj, q_norm, k_proj, k_norm, v_proj).
            w(f"{p}self_attn.q_proj.weight", transpose=True)
            w(f"{p}self_attn.k_proj.weight", transpose=True)
            w(f"{p}self_attn.v_proj.weight", transpose=True)
            if reader.has(f"{p}self_attn.q_norm.weight"):
                w(f"{p}self_attn.q_norm.weight")
            if reader.has(f"{p}self_attn.k_norm.weight"):
                w(f"{p}self_attn.k_norm.weight")
            w(f"{p}self_attn.o_proj.weight", transpose=True)

            w(f"{p}post_attention_layernorm.weight")

            # up, gate, down - unchanged order (already contiguous in the
            # source checkpoint; GateUpLayer's batching was never affected).
            w(f"{p}mlp.up_proj.weight", transpose=True)
            w(f"{p}mlp.gate_proj.weight", transpose=True)
            w(f"{p}mlp.down_proj.weight", transpose=True)

        w("model.norm.weight")

        if not tie_word_embeddings:
            w("lm_head.weight", transpose=True)

    print(f"wrote {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="Path to HF checkpoint dir (config.json + "
                         "model.safetensors or model.safetensors.index.json)")
    ap.add_argument("--output", required=True, help="Output .bin path")
    args = ap.parse_args()
    convert(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
