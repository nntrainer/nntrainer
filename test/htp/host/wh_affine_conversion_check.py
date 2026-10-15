# SPDX-License-Identifier: Apache-2.0
"""Synthetic HF BF16 -> serialized affine WH without requantization.

Run with numpy, torch and safetensors installed (converter dependencies).
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("wh_converter", ROOT / "tools/convert_gemma4_fc_wh.py")
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, hf, output = root / "source", root / "hf", root / "output"
        source.mkdir()
        hf.mkdir()
        k, n = 64, 32
        # Channels require positive, negative and zero offsets. All-zero
        # channels and unequal per-channel scales are covered as well.
        codes = np.tile(np.arange(-8, 8, dtype=np.int32), (n, k // 16))
        codes += (np.arange(n, dtype=np.int32) % 7 - 3)[:, None]
        w = converter.bf16_round(codes * (np.arange(n, dtype=np.float32) + 1)[:, None] * .0013)
        w[0] = 0
        header = {"model.layers.0.self_attn.q_proj.weight":
                  {"dtype": "BF16", "shape": [n, k], "data_offsets": [0, n * k * 2]}}
        encoded = json.dumps(header).encode()
        encoded += b" " * (-len(encoded) % 8)
        payload = (w.view(np.uint32) >> 16).astype("<u2").tobytes()
        (hf / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
        (hf / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {next(iter(header)): "model.safetensors"}}))
        entries = [
            {"name": "embedding0:Embedding", "shape": [64, 32], "dtype": "Q4_0", "bytes": 1152},
            {"name": "layer0_wq:weight", "shape": [k, n], "dtype": "Q4_0", "bytes": 1152},
            {"name": "layer0_sparse_moe:gate", "shape": [k, 4], "dtype": "FP32", "bytes": k * 4 * 4},
            {"name": "expert", "shape": [96, 128], "dtype": "TL2CX_WH", "bytes": 3584}]
        raw = bytearray()
        for e in entries:
            e["offset"] = len(raw)
            raw.extend(bytes((i * 7 + len(raw)) % 256 for i in range(e["bytes"])))
        (source / "model.bin").write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        (source / "conversion_report.json").write_text(json.dumps(
            {"model_file": "model.bin", "bytes": len(raw), "sha256": digest, "tensors": entries}))
        (source / "nntr_config.json").write_text(json.dumps({"moe_layer_dtype": "TL2CX_WH"}))
        converter.convert(source, output, hf)
        report = json.loads((output / "conversion_report.json").read_text())
        result = (output / report["model_file"]).read_bytes()
        for old, new in zip(entries, report["tensors"]):
            if new["dtype"] != "QS4CX_WH_AFFINE":
                assert result[new["offset"]:new["offset"] + new["bytes"]] == raw[old["offset"]:old["offset"] + old["bytes"]]
                continue
            at = new["offset"]
            scale = np.frombuffer(result, "<f4", n, at + k * n // 2)
            sums = np.frombuffer(result, "<f4", n, at + k * n // 2 + 4 * n)
            offsets = np.frombuffer(result, "<i4", n, at + k * n // 2 + 8 * n)
            restored_q = np.empty((n, k), np.int32)
            for c in range(n):
                for r in range(k):
                    byte = ((r // 32) * (n // 32) + c // 32) * 512
                    byte += (r % 32 // 8) * 128 + c % 32 * 4 + r % 4
                    q = (result[at + byte] >> (4 * (r // 4 % 2))) & 15
                    restored_q[c, r] = q - 16 if q > 7 else q
            assert np.array_equal(restored_q.sum(axis=1), sums)
            assert np.count_nonzero(offsets) > 0
            assert np.array_equal(converter.bf16_round(
                (restored_q + offsets[:, None]) * scale[:, None]), w)
        assert report["requantization"] is False
        assert converter.sha256(source / "model.bin") == digest
        try:
            converter.convert(source, output, hf)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing output was overwritten")
        try:
            converter.recover(np.array([[1., 1.125, 1.375, 2.]], np.float32))
        except ValueError:
            pass
        else:
            raise AssertionError("An incompatible grid was silently requantized")
        print("Affine WH converter: original BF16, metadata, unchanged records and rejection checks PASS")


if __name__ == "__main__":
    main()
