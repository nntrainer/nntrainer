# SPDX-License-Identifier: Apache-2.0
#
# nntrainer Hexagon development image (linux/amd64).
#
# One image for everything that does not need a phone:
#   * the regular x86 meson build (CausalLM app, unit tests)
#   * tools/hexagon/build_host_x86.sh   (lowering tests, nntr_hexpack, hexagon_ref_run)
#   * tools/hexagon/build_sim_test.sh / run_sim_test.sh   (needs the Hexagon SDK, mounted)
#   * tools/hexagon/build_skel.sh / build_host_test.sh    (needs SDK + Android NDK)
#
# The Hexagon SDK is NOT part of the image (Qualcomm licence). run.sh mounts
# it from the host at /opt/qcom/Hexagon_SDK/<version>. Android NDK r26d (the
# version the CI uses) is downloaded here because it is freely redistributable.
#
# Build:  docker build --platform linux/amd64 -t nntrainer-hexagon-dev:ubuntu24.04 tools/docker
# Use:    tools/docker/run.sh <command...>        (see run.sh and README.md)

FROM --platform=linux/amd64 ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG NDK_VERSION=r26d

# --- base toolchain + nntrainer build dependencies (mirrors .github/workflows/ubuntu_clean_meson_build.yml)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget unzip git gnupg software-properties-common \
        build-essential gcc g++ pkg-config \
        meson ninja-build cmake \
        libopenblas-dev libiniparser-dev libjsoncpp-dev libcurl4-openssl-dev \
        libglib2.0-dev libgstreamer1.0-dev libgtest-dev flatbuffers-compiler \
        libunwind-dev libssl-dev \
        python3 python3-dev python3-pip python3-numpy python3-matplotlib python-is-python3 \
        clang-format-14 \
        file less vim-tiny \
    && rm -rf /var/lib/apt/lists/*

# nnstreamer PPA: tensorflow2-lite-dev / ml-api packages used by the default meson options.
# If the PPA has no build for noble at the time you build, drop this block and pass
# -Denable-tflite-backbone=false -Denable-tflite-interpreter=false (run_qwen3_0_6b_x86_inference.sh already does).
RUN add-apt-repository -y ppa:nnstreamer/ppa && apt-get update && \
    apt-get install -y --no-install-recommends \
        tensorflow2-lite-dev nnstreamer-dev ml-api-common-dev ml-inference-api-dev \
    ; rm -rf /var/lib/apt/lists/*

# --- python: tokenizer for tools/hexagon/make_tokens.py (no torch needed)
RUN pip3 install --no-cache-dir --break-system-packages \
        "transformers>=4.40" sentencepiece tokenizers safetensors

# --- Android NDK (host harness cross-build, tools/hexagon/build_host_test.sh)
RUN cd /opt && \
    wget -q "https://dl.google.com/android/repository/android-ndk-${NDK_VERSION}-linux.zip" && \
    unzip -q "android-ndk-${NDK_VERSION}-linux.zip" && \
    rm "android-ndk-${NDK_VERSION}-linux.zip"
ENV ANDROID_NDK=/opt/android-ndk-${NDK_VERSION}
ENV ANDROID_API=31

# --- ncurses 5 for hexagon-sim (SDK 6.4 tools link libncurses.so.5; Ubuntu 24.04 ships only 6)
ARG NCURSES5_VER=6.3-2ubuntu0.3
RUN cd /tmp && \
    for p in libtinfo5 libncurses5; do \
      wget -q "http://archive.ubuntu.com/ubuntu/pool/universe/n/ncurses/${p}_${NCURSES5_VER}_amd64.deb"; \
    done && \
    dpkg -i libtinfo5_${NCURSES5_VER}_amd64.deb libncurses5_${NCURSES5_VER}_amd64.deb && \
    rm -f /tmp/*.deb

# --- Hexagon SDK mount point (populated by run.sh -v, or by setup_wizard.sh with qpm-cli)
RUN mkdir -p /opt/qcom/Hexagon_SDK /opt/qcom/hexkl_addon
ENV HEXAGON_SDK_BASE=/opt/qcom/Hexagon_SDK

# The simulator writes osam.cfg etc. into the cwd; the repo is mounted at /work.
WORKDIR /work

COPY entrypoint.sh /usr/local/bin/nntr-entrypoint
RUN chmod +x /usr/local/bin/nntr-entrypoint
ENTRYPOINT ["/usr/local/bin/nntr-entrypoint"]
CMD ["bash"]
