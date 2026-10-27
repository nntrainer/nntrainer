#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
#
# One-shot reproduction and cross-verification of the CED port.
#
# Extracts a PyTorch reference from the HuggingFace checkpoint, converts the
# weights, and compares the nntrainer output against it in FP32 and after Q8_0
# (W8A8) and Q4_0 (W4A8) weight quantization. Exits non-zero if any check
# misses its budget, so it is usable as a gate.
#
# Usage:
#   verify_e2e.sh [--model-dir DIR] [--build-dir DIR] [--skip-reference]
#                 [--skip-quant]
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

MODEL_DIR="${HOME}/hdd/models/ced-tiny"
BUILD_DIR="${REPO_ROOT}/builddir"
SKIP_REFERENCE=0
SKIP_QUANT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --skip-reference) SKIP_REFERENCE=1; shift ;;
        --skip-quant)     SKIP_QUANT=1;     shift ;;
        -h|--help) sed -n '3,14p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

RUNNER="${BUILD_DIR}/Applications/CausalLM/nntr_causallm"
QUANTIZER="${BUILD_DIR}/Applications/CausalLM/nntr_quantize"
FAILURES=0

step() { echo; echo "=== $* ==="; }
fail() { echo "FAIL: $*"; FAILURES=$((FAILURES + 1)); }

# FP32 parity is float32 accumulation noise. The quantized budget is what an
# 8-bit weight costs on a sigmoid output, and is deliberately far below the
# 0.23 that 4-bit costs on the same model.
TOL_FP32=${TOL_FP32:-1e-4}
TOL_Q8=${TOL_Q8:-0.05}

step "provenance"
if [[ ! -f "${MODEL_DIR}/model.safetensors" ]]; then
    echo "checkpoint not found; fetch it first:" >&2
    echo "  mkdir -p ${MODEL_DIR} && cd ${MODEL_DIR}" >&2
    echo "  curl -fLO https://huggingface.co/mispeech/ced-tiny/resolve/main/model.safetensors" >&2
    echo "  for f in config.json configuration_ced.py modeling_ced.py feature_extraction_ced.py preprocessor_config.json; do" >&2
    echo "    curl -fLO https://huggingface.co/mispeech/ced-tiny/resolve/main/\$f; done" >&2
    exit 2
fi
# Recorded from the checkpoint this port was verified against.
EXPECT_ST=0e086f0cd62814c6def89001f3f25193f75955696f6975ef6800af31d00d6dd7
GOT_ST=$(sha256sum "${MODEL_DIR}/model.safetensors" | cut -d' ' -f1)
echo "model.safetensors ${GOT_ST}"
[[ "${GOT_ST}" == "${EXPECT_ST}" ]] && echo "                  matches the verified checkpoint" \
    || echo "                  DIFFERS from the verified checkpoint (${EXPECT_ST})"
python3 - <<'PY'
import importlib
for m in ("torch", "torchaudio", "transformers", "numpy"):
    try:
        print(f"{m:14s} {importlib.import_module(m).__version__}")
    except Exception as e:  # noqa: BLE001
        print(f"{m:14s} MISSING ({e})")
PY

step "build"
[[ -x "${RUNNER}" ]] || { echo "runner not found at ${RUNNER}" >&2; exit 2; }
ninja -C "${BUILD_DIR}" Applications/CausalLM/nntr_causallm \
    Applications/CausalLM/nntr_quantize >/dev/null \
    || { echo "build failed" >&2; exit 2; }
echo "runner and quantizer up to date"

REF_DIR="${MODEL_DIR}/nntr_ref"
if [[ "${SKIP_REFERENCE}" -eq 0 ]]; then
    step "PyTorch reference (deterministic noise input)"
    python3 "${SCRIPT_DIR}/extract_reference.py" --model-dir "${MODEL_DIR}" \
        --out-dir "${REF_DIR}" 2>&1 | grep -E '^\[(input|init_bn|patch|hidden|logits|top5)\]' \
        | sed 's/^/  /'
    step "PyTorch reference (tonal input)"
    python3 - "${MODEL_DIR}" <<'PY'
import math, os, sys
import numpy as np
try:
    import soundfile as sf
except ImportError:
    sys.exit("soundfile is required for the tonal fixture")
sr, n = 16000, (1012 - 1) * 160
t = np.arange(n) / sr
f0, f1 = 200.0, 3500.0
ph = 2 * math.pi * (f0 * t + (f1 - f0) / (2 * t[-1]) * t ** 2)
x = 0.5 * np.sin(ph) + 0.2 * np.sin(2 * ph) + 0.1 * np.sin(3 * ph)
for c in range(0, n, sr):
    seg = x[c:c + 80]
    seg += np.hanning(80)[:len(seg)] * 0.8
x = (x / np.abs(x).max() * 0.9).astype(np.float32)
sf.write(os.path.join(sys.argv[1], "chirp.wav"), x, sr)
print(f"  wrote {sys.argv[1]}/chirp.wav ({x.shape[0]} samples)")
PY
    python3 "${SCRIPT_DIR}/extract_reference.py" --model-dir "${MODEL_DIR}" \
        --wav "${MODEL_DIR}/chirp.wav" --out-dir "${MODEL_DIR}/nntr_ref_chirp" 2>&1 \
        | grep -E '^\[(input|logits|top5)\]' | sed 's/^/  /'
else
    step "PyTorch reference (skipped, reusing ${REF_DIR})"
fi
[[ -f "${REF_DIR}/ref_logits.bin" ]] || { echo "no reference logits" >&2; exit 2; }

step "weight conversion"
python3 "${SCRIPT_DIR}/weight_converter.py" --input "${MODEL_DIR}" \
    --output "${MODEL_DIR}/nntr_ced_tiny_fp32.bin" 2>&1 \
    | grep -vE '^\s+Layer ' | sed 's/^/  /'
cat > "${MODEL_DIR}/nntr_config.json" <<EOF
{
  "model_tensor_type": "FP32-FP32",
  "model_file_name": "nntr_ced_tiny_fp32.bin",
  "model_type": "Model",
  "embedding_dtype": "FP32",
  "fc_layer_dtype": "FP32",
  "batch_size": 1,
  "sample_input": "nntr_ref/input_values.bin",
  "init_seq_len": 252,
  "max_seq_len": 252,
  "num_to_generate": 0,
  "fsu": false,
  "skip_tokenizer": true
}
EOF

check_run() {  # label model_dir input ref tol
    local label="$1" dir="$2" input="$3" ref="$4" tol="$5"
    local log="${MODEL_DIR}/verify_${label//\//_}.log"
    if [[ -n "${input}" ]]; then
        CED_REF_TOL="${tol}" CED_REF_BIN="${ref}" "${RUNNER}" "${dir}" "${input}" \
            >"${log}" 2>&1
    else
        CED_REF_TOL="${tol}" CED_REF_BIN="${ref}" "${RUNNER}" "${dir}" >"${log}" 2>&1
    fi
    # `|` as the sed delimiter: labels contain a slash.
    grep -E '^\[(top1|CED_REF_BIN)\]' "${log}" | sed "s|^|  ${label}: |"
    grep -q 'CED_REF_BIN\] .*PASS' "${log}" || fail "${label}"
}

step "FP32 vs HuggingFace logits"
check_run "fp32/noise" "${MODEL_DIR}" "" "${REF_DIR}/ref_logits.bin" "${TOL_FP32}"
check_run "fp32/chirp" "${MODEL_DIR}" \
    "${MODEL_DIR}/nntr_ref_chirp/input_values.bin" \
    "${MODEL_DIR}/nntr_ref_chirp/ref_logits.bin" "${TOL_FP32}"

if [[ "${SKIP_QUANT}" -eq 1 ]]; then
    step "quantization (skipped)"
else
    for DT in Q8_0 Q4_0; do
        step "${DT} weight quantization"
        QDIR="${MODEL_DIR}-${DT}"
        rm -rf "${QDIR}"; mkdir -p "${QDIR}"
        cp "${MODEL_DIR}/config.json" "${QDIR}/"
        "${QUANTIZER}" "${MODEL_DIR}" --fc_dtype "${DT}" --isa X86 -o "${QDIR}" \
            --output_bin "nntr_ced_tiny_${DT}.bin" 2>&1 \
            | grep -E 'Source size|Output size|Compression' | sed 's/^/  /'
        # Q4_0 is expected to miss the 8-bit budget; report it rather than gate
        # on it, so the two sit side by side in one run.
        if [[ "${DT}" == "Q8_0" ]]; then
            check_run "${DT}/noise" "${QDIR}" \
                "${REF_DIR}/input_values.bin" "${REF_DIR}/ref_logits.bin" "${TOL_Q8}"
            check_run "${DT}/chirp" "${QDIR}" \
                "${MODEL_DIR}/nntr_ref_chirp/input_values.bin" \
                "${MODEL_DIR}/nntr_ref_chirp/ref_logits.bin" "${TOL_Q8}"
        else
            log="${MODEL_DIR}/verify_${DT}.log"
            CED_REF_TOL="${TOL_Q8}" CED_REF_BIN="${REF_DIR}/ref_logits.bin" \
                "${RUNNER}" "${QDIR}" "${REF_DIR}/input_values.bin" >"${log}" 2>&1
            grep -E '^\[(top1|CED_REF_BIN)\]' "${log}" \
              | sed "s|^|  ${DT}/noise (informational): |"
        fi
    done
fi

step "result"
if [[ "${FAILURES}" -eq 0 ]]; then
    echo "ALL GATED CHECKS PASSED"
    exit 0
fi
echo "${FAILURES} CHECK(S) FAILED"
exit 1
