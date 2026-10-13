# SPDX-License-Identifier: Apache-2.0
"""Check the production HTP converter against the independent tiny serializer.

Requires torch 2.14.1, transformers 5.18.0, numpy and safetensors. Runs entirely
on CPU with a local random checkpoint; it never downloads model weights.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from transformers import Gemma4Config
import generate_gemma4_htp_moe_reference as reference


def check(output):
    output.mkdir(parents=True, exist_ok=True)
    model = reference.make_reference_model()
    expected = output / "independent.bin"
    reference.convert_weights(model, expected)

    checkpoint = output / "checkpoint"
    model.save_pretrained(checkpoint)
    Gemma4Config(text_config=model.config.to_dict()).save_pretrained(checkpoint)
    converter = (Path(__file__).resolve().parents[5] /
                 "Applications/CausalLM/res/gemma4/htp_weight_converter.py")
    actual = output / "converted.bin"
    subprocess.run([sys.executable, str(converter), "--model_path",
                    str(checkpoint), "--output_name", str(actual)], check=True)
    expected_bytes, actual_bytes = expected.read_bytes(), actual.read_bytes()
    assert actual_bytes == expected_bytes, "HTP converter changed tensor order or values"
    digest = hashlib.sha256(actual_bytes).hexdigest()
    manifest = json.loads(Path(str(actual) + ".manifest.json").read_text())
    assert manifest["sha256"] == digest
    assert manifest["bytes"] == len(actual_bytes)
    assert manifest["gemma4_execution_profile"] == "htp_fused"
    assert manifest["gemma4_weight_format"] == "htp_fused_v1"

    pinned = Path(__file__).resolve().parent.parent / "gemma4_htp_moe_tiny"
    pinned_digest = json.loads((pinned / "manifest.json").read_text())["files"][
        "nntr_gemma4_moe_tiny_fp32.bin"]["sha256"]
    assert digest == pinned_digest, "Regenerated weights differ from the pinned fixture"
    logits = reference.run_forward(model, reference.INPUT_IDS)
    expected_logits = json.loads((pinned / "reference_logits.json").read_text())
    max_error = max(abs(a - b) for a, b in zip(logits, expected_logits, strict=True))
    assert max_error < 1e-5, f"Regenerated logits differ: {max_error}"
    tokens = reference.run_greedy(model, reference.INPUT_IDS, reference.N_GEN)
    assert tokens == json.loads((pinned / "reference_tokens.json").read_text())
    result = {"bytes": len(actual_bytes), "sha256": digest,
              "pinned_fixture_match": True, "logits_max_abs_error": max_error,
              "greedy_tokens": tokens, "manifest_verified": True}
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out:
        check(args.out.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="nntr-gemma4-htp-converter-") as tmp:
            check(Path(tmp))
