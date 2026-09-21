#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${IMAGE:-nntrainer-hexagon-dev:ubuntu24.04}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/Applications/CausalLM/res/qwen3/qwen3-0.6b}"
BUILD_DIR="${BUILD_DIR:-build-x86}"
THREADS="${THREADS:-4}"
PROMPT="Give me a short introduction to large language model."
DO_BUILD=1

usage() {
  cat <<'USAGE'
Run Qwen3-0.6B nntrainer inference inside an x86 Docker container.

Usage:
  run_qwen3_0_6b_x86_inference.sh [options] [prompt]

Options:
  --model-dir PATH   nntrainer model directory mounted as /model.
  --image NAME       Docker image to use. Default: nntrainer-hexagon-dev:ubuntu24.04 (tools/docker/Dockerfile)
  --build-dir PATH   Meson build directory. Default: build-x86
  --threads N        NNTR_NUM_THREADS value. Default: 4
  --build            Build Applications/CausalLM/nntr_causallm before running.
  -h, --help         Show this help.

Environment overrides:
  MODEL_DIR, IMAGE, BUILD_DIR, THREADS

Default prompt:
  Give me a short introduction to large language model.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
  --model-dir)
    MODEL_DIR="$2"
    shift 2
    ;;
  --image)
    IMAGE="$2"
    shift 2
    ;;
  --build-dir)
    BUILD_DIR="$2"
    shift 2
    ;;
  --threads)
    THREADS="$2"
    shift 2
    ;;
  --build)
    DO_BUILD=1
    shift
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  --)
    shift
    PROMPT="$*"
    break
    ;;
  -*)
    echo "Unknown option: $1" >&2
    usage >&2
    exit 2
    ;;
  *)
    PROMPT="$*"
    break
    ;;
  esac
done

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Model directory not found: ${MODEL_DIR}" >&2
  echo "Set MODEL_DIR or pass --model-dir to an nntrainer-converted Qwen3-0.6B directory." >&2
  exit 1
fi

for required in config.json generation_config.json nntr_config.json tokenizer.json nntr_qwen3_0.6b_fp32.bin; do
  if [[ ! -f "${MODEL_DIR}/${required}" ]]; then
    echo "Missing required model file: ${MODEL_DIR}/${required}" >&2
    exit 1
  fi
done

BINARY="/work/${BUILD_DIR}/Applications/CausalLM/nntr_causallm"
HOST_BINARY="${REPO_ROOT}/${BUILD_DIR}/Applications/CausalLM/nntr_causallm"
DOCKER_USER="$(id -u):$(id -g)"

if [[ "${DO_BUILD}" -eq 1 ]]; then
  if [[ ! -f "${REPO_ROOT}/${BUILD_DIR}/build.ninja" ]]; then
    docker run --rm --platform linux/amd64 \
      -v "${REPO_ROOT}:/work" \
      -w /work \
      --user "${DOCKER_USER}" \
      "${IMAGE}" \
      meson setup "${BUILD_DIR}" \
        -Denable-fp16=false \
        -Dthread-backend=omp \
        -Denable-transformer=true \
        -Denable-test=false \
        -Denable-tflite-backbone=false \
        -Denable-tflite-interpreter=false
  fi

  docker run --rm --platform linux/amd64 \
    -v "${REPO_ROOT}:/work" \
    -w /work \
    --user "${DOCKER_USER}" \
    "${IMAGE}" \
    ninja -C "${BUILD_DIR}" Applications/CausalLM/nntr_causallm
elif [[ ! -x "${HOST_BINARY}" ]]; then
  echo "Inference binary not found: ${HOST_BINARY}" >&2
  echo "Run this script with --build, or build Applications/CausalLM/nntr_causallm first." >&2
  exit 1
fi

echo "Docker image : ${IMAGE}"
echo "Model dir    : ${MODEL_DIR}"
echo "Build dir    : ${BUILD_DIR}"
echo "Threads      : ${THREADS}"
echo "Prompt       : ${PROMPT}"
echo

docker run --rm --platform linux/amd64 \
  -e NNTR_NUM_THREADS="${THREADS}" \
  -v "${REPO_ROOT}:/work:ro" \
  -v "${MODEL_DIR}:/model:ro" \
  "${IMAGE}" \
  "${BINARY}" /model "${PROMPT}"
