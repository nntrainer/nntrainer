# Android build and regression check for the ScreenAI / YOLO / W8A8 branch

This branch rebases nntrainer/nntrainer#4287 (SigLIP2 encoder, BERT decoder,
ScreenAI caption, YOLOv11) and nntrainer/nntrainer#4292 (W8A8 int8 conv
engine, YOLOv7 / YOLOv7-Pose, FastViT keyword, CED audio tagging) onto main
as four commits. This note is the recipe for building both `main` and this
branch for a handset and comparing memory and speed on the existing LLM
path, plus how to run the new models. x86 unit tests already pass on both
(main 40/40, branch 41/41); the Android fp16 build is what still needs a
device.

## 1. Build both trees with the same options

```bash
# one worktree per tree, same NDK, same options
git worktree add ../nntr-main  upstream/main
git worktree add ../nntr-pr    claude/inspiring-ritchie-m2qh6b

export ANDROID_NDK=/path/to/ndk            # r26 or later
for T in ../nntr-main ../nntr-pr; do
  ( cd $T/Applications/CausalLM && \
    ./build_android.sh --clean --nntr-threads=6 -Denable-fp16=true )
done
```

- `build_android.sh` forwards every `-D...` option to the engine build; keep
  the list identical for both trees (`-Denable-fp16`, `-Dmmap-read`,
  `-Denable-opencl`, `--arm-arch`). Check with
  `meson introspect <builddir> --buildoptions` if in doubt.
- `Applications/CausalLM/json.hpp` (nlohmann 3.11.x) and the `xgrammar`
  submodule (`git submodule update --init --recursive Applications/CausalLM/xgrammar`)
  are needed by both trees, exactly as on main.
- Push each tree to its own directory so the two never share a `.so`:

```bash
adb push <main-out>/ /data/local/tmp/nntr_main/
adb push <pr-out>/   /data/local/tmp/nntr_pr/
```

The branch adds these executables next to `nntr_causallm` / `nntr_quantize`:
`yolov11_infer`, `yolov7_infer`, `yolov7_tiny_infer`, `yolov7_pose_infer`,
`fastvit_keyword_infer`. ScreenAI caption and CED run through `nntr_causallm`.

## 2. Model packs

- QS4CX / Q4_0 / Q6_K / FP16 packs made on main load unchanged on the branch.
- **Q8_0 packs made on the original #4287 / #4292 branches must be
  re-quantized.** The on-disk `QScheme` id of Q8_0 moved from `0x6` to `0x7`
  so that main's `QS4CX = 0x6` files keep loading; a `.bin` written by the
  old branches reads back as the wrong scheme. safetensors packs carry the
  dtype by name and are unaffected.
- Q8_0 FC weights are written per-block in `q8_0x4` on every ISA, so a pack
  quantized on x86 runs on ARM and the other way round. x86 Q8_0 packs made
  before this change hold plain rows and must be re-quantized. Setting
  `NNTR_W8A8_PERCH=1` at `nntr_quantize` time selects the #4292 per-channel
  encoding (plain rows); whatever was set at quantize time must also be set
  at run time.
- `nntr_quantize --conv_dtype Q8_0 ...` quantizes conv filters (YOLO,
  FastViT); `--act_dtype` left empty follows the source model's activation
  type.

## 3. LLM regression run (same pack, both trees)

Pin the loader and thread knobs so both trees run the same way, then
alternate main / branch runs (page-cache state matters), three runs each,
compare medians.

```sh
#!/system/bin/sh
# run.sh <bin_dir> <model_dir> <tag>
BIN=$1; MODEL=$2; TAG=$3
export LD_LIBRARY_PATH=$BIN:$BIN/layers
export NNTR_LOAD_PREFETCH=1 NNTR_MALLOC_TRIM=1 NNTR_DETERMINISTIC=1
$BIN/nntr_causallm $MODEL "$(cat /data/local/tmp/prompt_1k.txt)" > out_$TAG.log 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do
  awk '/VmHWM|VmRSS|RssAnon|RssFile/{printf "%s %s ", $1, $2} END{print ""}' \
      /proc/$PID/status >> mem_$TAG.log
  sleep 0.2
done
grep -E "prefill:|generation:|peak memory|honest" out_$TAG.log
```

What to compare:

| metric | where | note |
|---|---|---|
| prefill / generation TPS | `nntr_causallm` summary block | median of 3, alternate trees |
| init time | `init` line of the summary | cold (`echo 3 > /proc/sys/vm/drop_caches`, root) and warm |
| `RssAnon` right after load, before the first token | `mem_<tag>.log` | real heap; the KV cache lands here |
| `RssAnon` peak during decode | `mem_<tag>.log` | |
| `honest RssAnon+accel` | summary block | the footprint sampler's per-run peak |
| generated text | `diff out_main.log out_pr.log` | must be byte-identical with greedy sampling |

Do not read `ru_maxrss` / "Max Resident Set Size" as the steady-state
footprint: every load worker maps the whole weight file, so that number
keeps the load-time `RssFile` peak (GBs for a 3 GB pack) for the life of
the process. `NNTR_LOAD_PREFETCH` only fills the page cache and is not in
RSS.

Known main-side item while reading `RssAnon`: `KVCacheManager::allocate()`
on main zero-fills every layer's K/V slab (`max_seq_len` rows) at load, so
`RssAnon` after load already contains the whole KV cache
(`layers x 2 x max_seq_len x kv_width x 2 B`). The 2026-06 fix that removed
that memset (`3ded55ffe`) was undone by `d1ff9ee25`; it is the same on this
branch, so it does not bias the A/B, but it is why a large `max_seq_len`
shows GBs before any token.

## 4. New models (branch only, record absolute numbers)

```sh
export LD_LIBRARY_PATH=/data/local/tmp/nntr_pr:/data/local/tmp/nntr_pr/layers
B=/data/local/tmp/nntr_pr

# ScreenAI caption: SigLIP2 encoder + BERT decoder over one image
$B/nntr_causallm --caption-run <screenai_model_dir> <image.jpg>

# YOLOv11 / YOLOv7 / YOLOv7-tiny / YOLOv7-Pose: prints "Inference time" and "Peak RSS"
$B/yolov11_infer     <res_dir> <image.jpg | input.bin>
$B/yolov7_infer      <res_dir> <image.jpg | input.bin>
$B/yolov7_pose_infer <res_dir> <image.jpg | input.bin>

# FastViT keyword: prints backbone / head / total inference time
$B/fastvit_keyword_infer <res_dir> <input.bin>

# CED audio tagging (see res/ced/README.md for the reference check)
CED_REF_BIN=<ref_logits.bin> $B/nntr_causallm <ced_model_dir> <input_values.bin>
```

The YOLO runners take `YOLO_APPROX_SILU=1` to use the approximate SiLU; the
default is the exact one. The W8A8 int8-resident conv path is enabled with
`NNTR_W8A8=1` (and `NNTR_W8A8_PERCH=1` for per-channel packs).

## 5. What to look at if something moved

- Decode TPS down on the branch: the W8A8 commit adds `std::getenv` calls on
  the conv / activation / concat / addition forward paths. They are cheap
  but per-layer per-step; caching them is the first thing to try.
- Prefill TPS: `ActiFunc::gelu` now splits tensors of 32K+ floats across the
  thread pool and the FP32 width-axis LayerNorm is a fused two-pass kernel.
  Both should be neutral or faster; a slowdown points at a thread-count
  difference between the two builds.
- Attention: `cross_attention` and `attn_fp32_accum` are opt-in mha_core
  properties set only by the SigLIP2 / BERT decoder models. The LLM path is
  main's code; a difference there is a merge problem in `mha_core.cpp`.
- Memory on CNN models: the conv im2col scratch is allocated once in
  `finalize()` for the whole batch instead of per forward.

## 6. Suggested pass criteria

| metric | tolerance vs main |
|---|---|
| prefill / decode TPS | within -3 % |
| init time | within 10 % |
| `RssAnon` after load | not above main |
| honest footprint peak | within +2 % |
| generated tokens | identical |
