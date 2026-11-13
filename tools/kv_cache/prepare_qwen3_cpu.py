#!/usr/bin/env python3
"""Prepare the pinned official Qwen3-0.6B checkpoint for nntrainer CPU.

Only generated files under --output are written. Existing TEST_MODELS assets
are not used or modified. The numpy converter widens BF16 exactly to FP32 and
uses the current graph's positional up/gate FFN order.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO_ID = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
ASSETS = ("config.json", "generation_config.json", "tokenizer.json",
          "tokenizer_config.json", "merges.txt", "vocab.json", "LICENSE")


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build-kv-cpu/models"))
    parser.add_argument("--build", type=Path, default=Path("build-kv-cpu"))
    parser.add_argument("--download", action="store_true",
                        help="Download the pinned public checkpoint, without credentials")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--check-torch-converter", action="store_true",
                        help="Compare full output hashes with the torch converter")
    parser.add_argument("--torch-python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output = args.output.resolve()
    hf = args.output / "hf/qwen3-0.6b"
    fp32 = args.output / "qwen3-0.6b-fp32"
    quant = args.output / "qwen3-0.6b-q40"
    fp32.mkdir(parents=True, exist_ok=True)
    if args.download:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        from huggingface_hub import snapshot_download
        snapshot_download(REPO_ID, revision=REVISION, local_dir=hf, token=False,
                          allow_patterns=["*.json", "*.jinja", "*.safetensors",
                                          "merges.txt", "vocab.json", "README.md", "LICENSE"],
                          max_workers=3)
        (hf / "nntr_source_revision.json").write_text(json.dumps(
            {"repo_id": REPO_ID, "revision": REVISION}, indent=2) + "\n")
    if not (hf / "model.safetensors").is_file():
        parser.error("Checkpoint missing; first run with --download")
    source = json.loads((hf / "nntr_source_revision.json").read_text())
    if source != {"repo_id": REPO_ID, "revision": REVISION}:
        parser.error("Checkpoint revision metadata does not match the pinned source")
    for name in ASSETS:
        if (hf / name).exists():
            target = fp32 / name
            if target.is_symlink():
                target.unlink()
            shutil.copy2(hf / name, target)
    generation = json.loads((fp32 / "generation_config.json").read_text())
    generation["do_sample"] = False
    (fp32 / "generation_config.json").write_text(json.dumps(generation, indent=2) + "\n")
    cfg = dict(model_type="CausalLM", model_tensor_type="FP32-FP32",
               model_file_name="nntr_qwen3_0.6b_fp32.bin", fc_layer_dtype="FP32",
               embedding_dtype="FP32", lmhead_dtype="FP32", lora_rank=0,
               lora_alpha=0, lora_target=[], bad_word_ids=[], fsu=False,
               fsu_lookahead=2, num_to_generate=48, init_seq_len=128,
               max_seq_len=512, batch_size=1, tokenizer_file="tokenizer.json",
               sample_input="<|im_start|>user\nCompute 2 + 2.<|im_end|>\n"
                            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
               skip_prefill=False)
    (fp32 / "nntr_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    weight = fp32 / cfg["model_file_name"]
    converter = root / "Applications/CausalLM/res/qwen3/qwen3-0.6b"
    subprocess.run([sys.executable, str(converter / "weight_converter_layergraph.py"),
                    "--model_path", str(hf), "--output_name", str(weight),
                    "--data_type", "float32"], check=True)
    manifest = {"source": source, "source_weights_sha256": sha256(hf / "model.safetensors"),
                "converter_sha256": sha256(converter / "weight_converter_layergraph.py"),
                "fp32": {"path": str(weight), "bytes": weight.stat().st_size,
                         "sha256": sha256(weight)}}
    if args.check_torch_converter:
        check_weight = args.output / "torch-converter-parity.bin"
        subprocess.run([str(args.torch_python), str(converter / "weight_converter.py"),
                        "--model_path", str(hf), "--output_name", str(check_weight),
                        "--data_type", "float32"], check=True)
        check_hash = sha256(check_weight)
        manifest["torch_converter_sha256"] = check_hash
        if check_hash != manifest["fp32"]["sha256"]:
            raise RuntimeError("Torch and numpy converters produced different bytes")
        check_weight.unlink()
    if args.quantize:
        env = dict(os.environ, NNTR_ENGINE="cpu", NNTR_NUM_THREADS="8",
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="8")
        subprocess.run([str((args.build / "Applications/CausalLM/nntr_quantize").resolve()),
                        str(fp32), "--output", str(quant), "--fc_dtype", "Q4_0",
                        "--embd_dtype", "Q6_K", "--lmhead_dtype", "Q6_K",
                        "--isa", "X86", "--no_sidecar", "--output_bin",
                        "nntr_qwen3_0.6b_q40_embdq6k.bin"], check=True, env=env)
        weight = quant / "nntr_qwen3_0.6b_q40_embdq6k.bin"
        manifest["q40"] = {"path": str(weight), "bytes": weight.stat().st_size,
                           "sha256": sha256(weight), "isa": "X86",
                           "fc_dtype": "Q4_0", "embedding_dtype": "Q6_K",
                           "lmhead_dtype": "Q6_K", "activation_dtype": "FP32"}
    (args.output / "qwen3-cpu-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
