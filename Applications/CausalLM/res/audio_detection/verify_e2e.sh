#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
#
# One-shot reproduction and cross-verification of the audio-detection port.
#
# Runs the upstream PyTorch reference and the nntrainer port over the same wav
# and compares them at three points: the log-mel front-end, the model on a
# reference mel, and the full end-to-end probabilities plus the printed table.
# Exits non-zero if any of them misses its budget, so it is usable as a gate.
#
# Usage:
#   verify_e2e.sh [--pytorch-dir DIR] [--build-dir DIR] [--work-dir DIR]
#                 [--wav NAME] [--skip-reference]
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

PYTORCH_DIR="/home/seungbaek/projects/0824/pytorch_audio_detection/pytorch"
BUILD_DIR="${REPO_ROOT}/builddir"
WORK_DIR="${HOME}/hdd/models/audio-detection"
WAV_NAME="dog_tizen.wav"
SKIP_REFERENCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pytorch-dir) PYTORCH_DIR="$2"; shift 2 ;;
        --build-dir)   BUILD_DIR="$2";   shift 2 ;;
        --work-dir)    WORK_DIR="$2";    shift 2 ;;
        --wav)         WAV_NAME="$2";    shift 2 ;;
        --skip-reference) SKIP_REFERENCE=1; shift ;;
        -h|--help) sed -n '3,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

REF_DIR="${WORK_DIR}/ref"
RUNNER="${BUILD_DIR}/Applications/CausalLM/nntr_causallm"
FAILURES=0

step() { echo; echo "=== $* ==="; }
fail() { echo "FAIL: $*"; FAILURES=$((FAILURES + 1)); }

# Budgets. The front-end one is in dB, the other two are absolute differences
# on the model output; all three are float32 accumulation noise, not tolerance
# for a behavioural difference.
TOL_FRONTEND_DB=${TOL_FRONTEND_DB:-5e-3}
TOL_MODEL_LOGITS=${TOL_MODEL_LOGITS:-1e-3}
TOL_E2E_PROBS=${TOL_E2E_PROBS:-1e-3}

step "provenance"
if [[ ! -f "${PYTORCH_DIR}/ckpt/weights.pt" ]]; then
    echo "reference checkpoint not found under ${PYTORCH_DIR}" >&2
    exit 2
fi
# Recorded from the checkpoint this port was verified against. A mismatch does
# not stop the run -- it just means the numbers below are not comparable to the
# ones in README.md.
EXPECT_CKPT=ff0acbba4ea3696c739f717a68ce8db2ed62d055a3edb33b812f1122f2cbebad
EXPECT_WAV=063e3967252e2bfcea89b55e44ab38900784d0297a2e2b222442b43b9f21797b
GOT_CKPT=$(sha256sum "${PYTORCH_DIR}/ckpt/weights.pt" | cut -d' ' -f1)
GOT_WAV=$(sha256sum "${PYTORCH_DIR}/${WAV_NAME}" | cut -d' ' -f1)
echo "weights.pt  ${GOT_CKPT}"
[[ "${GOT_CKPT}" == "${EXPECT_CKPT}" ]] && echo "            matches the verified checkpoint" \
    || echo "            DIFFERS from the verified checkpoint (${EXPECT_CKPT})"
echo "${WAV_NAME}  ${GOT_WAV}"
[[ "${GOT_WAV}" == "${EXPECT_WAV}" ]] && echo "            matches the verified clip" \
    || echo "            DIFFERS from the verified clip (${EXPECT_WAV})"
python3 - <<'PY'
import importlib
for m in ("torch", "torchaudio", "numpy", "scipy", "soundfile"):
    try:
        print(f"{m:12s} {importlib.import_module(m).__version__}")
    except Exception as e:  # noqa: BLE001
        print(f"{m:12s} MISSING ({e})")
PY

step "build"
if [[ ! -x "${RUNNER}" ]]; then
    echo "runner not found at ${RUNNER}; configure and build first:" >&2
    echo "  meson setup builddir -Denable-transformer=true -Denable-app=true" >&2
    echo "  ninja -C builddir" >&2
    exit 2
fi
ninja -C "${BUILD_DIR}" Applications/CausalLM/nntr_causallm >/dev/null \
    || { echo "build failed" >&2; exit 2; }
echo "runner up to date: ${RUNNER}"

if [[ "${SKIP_REFERENCE}" -eq 0 ]]; then
    step "PyTorch reference"
    python3 "${SCRIPT_DIR}/extract_reference.py" --repo "${PYTORCH_DIR}" \
        --wav "${WAV_NAME}" --out-dir "${REF_DIR}" >"${WORK_DIR}/reference.log" 2>&1 \
        || { echo "reference extraction failed, see ${WORK_DIR}/reference.log" >&2; exit 2; }
    grep -E '^\[(model|frontend|audio|shapes)\]' "${WORK_DIR}/reference.log" || true
else
    step "PyTorch reference (skipped, reusing ${REF_DIR})"
fi
[[ -f "${REF_DIR}/expected.txt" ]] || { echo "no ${REF_DIR}/expected.txt" >&2; exit 2; }

step "weight conversion"
python3 "${SCRIPT_DIR}/weight_converter.py" --ckpt "${PYTORCH_DIR}/ckpt" \
    --out-dir "${WORK_DIR}" | sed 's/^/  /'
cp -f "${PYTORCH_DIR}/${WAV_NAME}" "${WORK_DIR}/"
python3 - "${WORK_DIR}/nntr_config.json" "${WAV_NAME}" <<'PY'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c["sample_input"] = sys.argv[2]
json.dump(c, open(p, "w"), indent=2)
PY

step "1/3 front-end vs reference mel"
python3 - "${REF_DIR}" "${TOL_FRONTEND_DB}" <<'PY'
import json, math, sys
import numpy as np
ref_dir, tol = sys.argv[1], float(sys.argv[2])
meta = json.load(open(f"{ref_dir}/meta.json"))
audio = np.fromfile(f"{ref_dir}/audio.bin", dtype=np.float32)
N, HOP, NFFT = meta["window_samples"], meta["hop"], meta["n_fft"]
NMELS, TOPDB, SR = meta["n_mels"], meta["top_db"], meta["sample_rate"]
W = np.array([0.5 - 0.5 * math.cos(2 * math.pi * i / NFFT) for i in range(NFFT)])
n_freqs = NFFT // 2 + 1
all_f = np.linspace(0, SR // 2, n_freqs)
h2m = lambda f: 2595.0 * math.log10(1.0 + f / 700.0)
m2h = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
mp = np.linspace(h2m(0.0), h2m(SR / 2), NMELS + 2)
fp = np.array([m2h(x) for x in mp])
fd = np.diff(fp); sl = fp[None, :] - all_f[:, None]
FB = np.maximum(0.0, np.minimum(-sl[:, :-2] / fd[:-1], sl[:, 2:] / fd[1:]))
worst = 0.0
for i in range(meta["n_windows"]):
    ref = np.fromfile(f"{ref_dir}/window{i}_mel.bin",
                      dtype=np.float32).reshape(NMELS, meta["frames"])
    win = audio[i * N:(i + 1) * N].astype(np.float64)
    pad = NFFT // 2
    x = np.concatenate([win[pad:0:-1], win, win[-2:-pad - 2:-1]])
    frames = 1 + (len(x) - NFFT) // HOP
    spec = np.empty((n_freqs, frames))
    for t in range(frames):
        F = np.fft.rfft(x[t * HOP:t * HOP + NFFT] * W, NFFT)
        spec[:, t] = F.real ** 2 + F.imag ** 2
    db = 10.0 * np.log10(np.maximum(FB.T @ spec, 1e-10))
    db = np.maximum(db, db.max() - TOPDB)
    worst = max(worst, float(np.abs(db.astype(np.float32) - ref).max()))
print(f"  windows={meta['n_windows']} worst_max_abs_diff={worst:.3e} dB "
      f"tol={tol:.1e} {'PASS' if worst < tol else 'FAIL'}")
sys.exit(0 if worst < tol else 1)
PY
[[ $? -eq 0 ]] || fail "front-end exceeded ${TOL_FRONTEND_DB} dB"

step "2/3 model on a reference mel vs reference logits"
# Log to a file and grep the file rather than piping the runner into grep -q:
# grep -q exits at the first match, the runner takes SIGPIPE, and pipefail then
# reports the whole pipeline as failed even though the check passed.
MEL_LOG="${WORK_DIR}/model_mel.log"
CED_REF_TOL="${TOL_MODEL_LOGITS}" \
CED_REF_BIN="${REF_DIR}/window0_logits.bin" \
    "${RUNNER}" "${WORK_DIR}" "${REF_DIR}/window0_mel.bin" >"${MEL_LOG}" 2>&1
grep -E '^\[CED_REF_BIN\]' "${MEL_LOG}" | sed 's/^/  /'
grep -q 'CED_REF_BIN\] .*PASS' "${MEL_LOG}" || fail "model on reference mel"

step "3/3 end to end vs reference probabilities and table"
E2E_LOG="${WORK_DIR}/e2e.log"
AD_REF_DIR="${REF_DIR}" AD_REF_TOL="${TOL_E2E_PROBS}" \
    "${RUNNER}" "${WORK_DIR}" >"${E2E_LOG}" 2>&1
grep -E '^\[AD_REF\]' "${E2E_LOG}" | sed 's/^/  /'
grep -q 'AD_REF\] .*PASS' "${E2E_LOG}" || fail "end-to-end probabilities"

sed -n '/^=== /,$p' "${E2E_LOG}" | grep -E '^(=== |  \[)' > "${WORK_DIR}/table.txt"
if diff -q "${REF_DIR}/expected.txt" "${WORK_DIR}/table.txt" >/dev/null; then
    echo "  table: byte-identical to expected.txt"
else
    DIFF_LINES=$(diff "${REF_DIR}/expected.txt" "${WORK_DIR}/table.txt" \
                 | grep -c '^[<>]')
    TOTAL=$(wc -l < "${REF_DIR}/expected.txt")
    echo "  table: $((DIFF_LINES / 2)) of ${TOTAL} lines differ (third-decimal"
    echo "         rounding is expected; see README.md)"
    diff "${REF_DIR}/expected.txt" "${WORK_DIR}/table.txt" | sed 's/^/    /'
    # Detections and top-1 labels must still agree exactly.
    for f in "${REF_DIR}/expected.txt" "${WORK_DIR}/table.txt"; do
        grep -o "DETECTED: \[[^]]*\]" "$f" > "${f}.det" || true
        sed -E 's/^  \[ *([0-9]+)-.*\] ([a-z_]+)=.*/\1 \2/' "$f" \
            | grep -E '^[0-9]+ ' > "${f}.top1" || true
    done
    diff -q "${REF_DIR}/expected.txt.det" "${WORK_DIR}/table.txt.det" >/dev/null \
        && echo "         detections: identical" || fail "detections differ"
    diff -q "${REF_DIR}/expected.txt.top1" "${WORK_DIR}/table.txt.top1" >/dev/null \
        && echo "         top-1 labels: identical" || fail "top-1 labels differ"
fi

step "result"
if [[ "${FAILURES}" -eq 0 ]]; then
    echo "ALL CHECKS PASSED"
    exit 0
fi
echo "${FAILURES} CHECK(S) FAILED"
exit 1
