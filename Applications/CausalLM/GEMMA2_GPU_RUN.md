# Gemma2-2B GPU (layer-graph) — canonical build + run config

The configuration that reaches **gn-parity prefill (~894 TPS, best 901, M=1024)** on
Adreno 840 (SD8-Elite, S26 Ultra) with the 2026-06-15 commits
(`1d6dc648` decode active-row, `a0cb675d` addition lws, `2e64ef5d` FC_QUANT_DIRECT).

## 1. Build (host, Android NDK)

```sh
# NDK: 27.0.12077973 works (the original 27.2.12479018 was lost in PC migration).
export ANDROID_NDK=$HOME/Android/Sdk/ndk/27.0.12077973
ROOT=<repo>/nntrainer

# (only if a .cl kernel was edited) regenerate the embedded kernel strings:
bash tools/regen_cl_kernels.sh builddir

# libnntrainer.so
$ANDROID_NDK/ndk-build NDK_PROJECT_PATH=$ROOT/builddir \
  APP_BUILD_SCRIPT=$ROOT/builddir/jni/Android.mk \
  NDK_APPLICATION_MK=$ROOT/builddir/jni/Application.mk -j$(nproc) nntrainer
# ndk-build does not strip+install the named target -> do it manually:
$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip --strip-unneeded \
  $ROOT/builddir/obj/local/arm64-v8a/libnntrainer.so \
  -o $ROOT/builddir/libs/arm64-v8a/libnntrainer.so
cp $ROOT/builddir/libs/arm64-v8a/libnntrainer.so \
  $ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so

# nntrainer_causallm (the app binary; links the prebuilt libnntrainer.so above).
# NOTE: layer/model edits (rms_norm_gpu.cpp, gemma2_causallm.cpp) live HERE, not in
# the lib. If you add a public symbol to a lib header, also sync the INSTALLED copy:
#   cp nntrainer/tensor/cl_operations/<hdr>.h builddir/android_build_result/include/nntrainer/
$ANDROID_NDK/ndk-build NDK_PROJECT_PATH=$ROOT/Applications/CausalLM \
  APP_BUILD_SCRIPT=$ROOT/Applications/CausalLM/jni/Android.mk \
  NDK_APPLICATION_MK=$ROOT/Applications/CausalLM/jni/Application.mk \
  NNTRAINER_ROOT=$ROOT -j$(nproc) nntrainer_causallm
```

## 2. Push to device

```sh
D=/data/local/tmp/nntrainer/causallm
adb push builddir/libs/arm64-v8a/libnntrainer.so $D/libnntrainer.so
# ⚠️ The app binaries from `ndk-build nntrainer_causallm` land in
#    Applications/CausalLM/obj/local/arm64-v8a/ (the libs/ copies can be STALE
#    and are NOT auto-refreshed). PUSH FROM obj/. And the app links the
#    per-app plugin libcausallm_core.so (contains mha_core.cpp,
#    tie_word_embedding.cpp, the model layers) — it MUST be pushed too, or
#    layer/model edits silently do not take effect on device.
O=Applications/CausalLM/obj/local/arm64-v8a
adb push $O/libcausallm_core.so $D/libcausallm_core.so
adb push $O/nntrainer_causallm $D/nntrainer_causallm
adb shell chmod +x $D/nntrainer_causallm $D/libcausallm_core.so
```

## 3. Models on device (`/data/local/tmp/nntrainer/causallm/models/`)

| dir | weights | lm_head | use |
|---|---|---|---|
| `gemma2_lg` | `nntr_gemma2_2b_qint4_fp16.bin` (1.34 GB, QINT4 FC) | Q4_0 **host** | **prefill perf (the ~894/901 config)** |
| `gemma2_lg_q6k` | `m_qint4_embdq6k.bin` (QINT4 FC + Q6_K embd) | Q6_K **GPU** | decode (faster: 13.5 vs 7 TPS) |

Prompts: `prompt_1p2k.txt` (M=1024), `prompt_1k.txt` (~843 Gemma2 tokens).

## 4. Canonical run command (the gn-parity config)

```sh
D=/data/local/tmp/nntrainer/causallm; cd $D
export LD_LIBRARY_PATH=$D
export NNTR_NUM_THREADS=4
export NNTR_FC_INT8_GPU=1      # v8c int8xint4 FC GEMM on GPU
export NNTR_MHA_GPU=1          # attention on GPU
export NNTR_GPU_SVM_POOL=1     # SVM activation pool
export NNTR_KV_IMG_ATTN=1      # image2d KV attention (d=256)
export NNTR_GPU_CLMEM_POOL=1   # cl_mem residency  <-- REQUIRED for coherent + fast
export NNTR_MHA_GPU_DECODE=1   # GPU attention at DECODE too (not just prefill):
                              # routes step==1 through the OHWI image-attention
                              # path. Decode +19% at 1024 ctx (10.7->12.8 TPS,
                              # token-identical md5 0b2170b7), since the CPU NEON
                              # decode attention (mha_core compute_kcaches) over
                              # the growing KV cache is the long-context decode
                              # bottleneck. FP16 KV only (default); falls back to
                              # CPU on not-ok. Use gemma2_lg_q6k (GPU lm_head).
# NNTR_FC_QUANT_DIRECT is default ON now (commit 2e64ef5d); =0 restores the staging copy.
./nntrainer_causallm models/gemma2_lg "$(cat prompt_1p2k.txt)"
```

Measured (Adreno 840, M=1024, cooled best-of-3): **prefill 888 / 896 / 901 TPS**
(token-identical md5 `a6710b4d`). gpu_native chain reference ~988.

> ⚠️ Without `NNTR_GPU_CLMEM_POOL=1` the run is pure-SVM: prefill ~764 and the
> generation degenerates (greedy-collapse). cl_mem residency is mandatory.
> ⚠️ Single-device runs are thermal-noisy — cool the device (the TPS rises as it
> cools); best-of-3 after a long cooldown.

## 5. x86 Intel Arc (OpenCL) — laptop / desktop

Verified COHERENT on **Intel Arc Graphics [0x7d55]** (Meteor Lake-P, Core Ultra 9
185H) with the 2026-06-16 HEAD. The OpenCL context manager filters for Intel vendor
`0x8086`, so a co-installed NVIDIA / pocl-CPU platform is auto-avoided — no manual
device selection needed.

### Build (native meson — NOT the NDK flow above)

```sh
ROOT=<repo>/nntrainer; cd $ROOT
# build_cl is the OpenCL+FP16 x86 build dir (enable-opencl=true, enable-fp16=true).
# Incremental rebuild of the runner + its lib deps (~80 s):
ninja -C build_cl Applications/CausalLM/nntr_causallm
```

> ⚠️ **The build MUST be at HEAD.** A stale `build_cl` (e.g. predating
> `99b8c7b5` "tokenize add_special_tokens=true: leading BOS") makes Gemma2 emit
> `<pad><pad>...` — Gemma2 degenerates without the leading BOS. Other required
> commits: `83ecdaca` (Q6_K lm_head GEMV in the layer graph), `db87c78e`/`a0cb675d`.
> If you pulled the model+runtime from a device built at HEAD, rebuild `build_cl`.

### Run env (differs from Adreno — Intel needs the BUFFER paths)

```sh
cd <repo>/nntrainer
NNTR_GPU_SVM_POOL=1 \
NNTR_V8C_BUF=1 \          # Intel NEO cannot do the image-read v8c FC -> BUFFER FC (required)
NNTR_MHA_GPU=1 \
NNTR_FC_GPU=1 \
NNTR_FC_INT8_GPU=1 \
NNTR_GPU_CLMEM_POOL=1 \   # cl_mem residency -- REQUIRED for coherence (else greedy-collapse)
  ./build_cl/Applications/CausalLM/nntr_causallm <model_dir> "<prompt>"
```

- Intel does **not** use `NNTR_KV_IMG_ATTN` (image2d attention is the Adreno path);
  the buffer attention path is used via `NNTR_MHA_GPU` + `NNTR_V8C_BUF`.
- Both `NNTR_V8C_BUF=1` and `NNTR_GPU_CLMEM_POOL=1` are mandatory on Intel; dropping
  either gives garbage (`<pad>`) or greedy-collapse respectively.
- Local model dir: a device model pulled to host. Fix `nntr_config.json`'s
  `tokenizer_file` to the **local** absolute path (device pull leaves a `/data/...` path).
- Convenience wrapper: `.claude/scripts/run_gemma2_x86.sh "<prompt>"`.

### Measured (Intel Arc, `gemma2_lg_q6k`)

| metric | value |
|---|---|
| prefill, M=1024 (`prompt_1p2k.txt`) | ~693 TPS |
| prefill, M=842 (`prompt_1k.txt`) | ~674 TPS *(after the M_pad fix below; was ~175)* |
| decode (short ctx, 32 tok) | ~9.5 TPS |
| decode (Q6_K lm_head GEMV) | ~37 ms/call — dominates decode on Intel |
| sample | continues the prompt coherently ("...stored program architectures...") |

Intel M=1024 prefill (~693) is ~80% of Adreno (~810–890 @ M=1024); the remaining gap
is the Intel-no-XMX FC ceiling + buffer (vs Adreno image2d/QCOM-residency) attention,
not a correctness issue. (For ~875 on Intel, the separate `gpu_native` binary path is
needed — not built in `build_cl` by default.)

## 6. Intel vs Adreno — the device differences (read this to avoid confusion)

The layer-graph runs on BOTH the Adreno (Android/ARM, image path) and Intel Arc
(x86, buffer path). They share the kernels but take **different code paths**, gated
by env. Keep this table straight when reading perf numbers across sessions:

| aspect | Adreno 840 (Android) | Intel Arc (x86) |
|---|---|---|
| FC GEMM (v8c int8×int4) | **image2d** (`read_imageui` texture) | **buffer** (`NNTR_V8C_BUF=1`; NEO can't `read_imageui`) |
| attention | **image** 3-kernel (`NNTR_KV_IMG_ATTN=1`) | **flash** Block-Q + subgroup-reduce (`flash_attention_prefill_f16_blockq`, FBQ_SG) |
| attention env | `NNTR_KV_IMG_ATTN=1` | `NNTR_V8C_BUF=1` (NO `KV_IMG_ATTN`) |
| coherence req | `NNTR_GPU_CLMEM_POOL=1` | `NNTR_GPU_CLMEM_POOL=1` + `NNTR_V8C_BUF=1` |
| M=1024 prefill | ~810–890 TPS | ~693 TPS |
| flash kernel | fallback only (image wins 3× here) | the primary path |

### ⚠️ The non-power-of-2 M prefill cliff = **Intel-only**, and it's now fixed

A prompt whose token count is not a "nice" size (e.g. 842, 668) used to make Intel
prefill **~4× slower** (842: 175 TPS vs 1024: 693 TPS). Root cause (profiled, not
the flash kernel — that scales fine):

- The **buffer-path v8c FC GEMM** dispatches `gws` M-axis = `M_pad / V8C_TM` (TM=4).
  For the large-N FFN GEMM (N=9216, gate/up_proj) a **non-power-of-2 M-workgroup
  count** (842 → 211 groups, prime) maps poorly to the Intel EU array and runs
  **~4.7× slower** than 1024 → 256 groups. Other shapes (N≤2304) are unaffected.
- **Adreno does NOT have this cliff** — its image/texture-cache GEMM is insensitive
  to the M-workgroup count (measured: 668→714, 842→798, 1024→811 TPS, all healthy).

**Fix (committed):** `dotCl_v8c` rounds `M_pad` up to a coarser granularity (default
**64** on the `NNTR_V8C_BUF` path; `NNTR_FC_MPAD_ALIGN` overrides). The padded rows
are computed but never read back (M-valid store guard), so output is **token-identical**.
842 prefill 175 → ~674 TPS. Gated to the buffer path, so **Adreno is byte-identical**
(keeps `V8C_TM`=4). The coarse align only applies to prefill (`M ≥ align`); decode
(M=1) keeps `V8C_TM` so it never blows up to a 64-row FC.
