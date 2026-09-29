# 60 — CPU baseline: Qwen3-0.6B on the S25 Ultra CPU from `main`

Issue: #60 (`prio:p1`). Contract: `0000-agent-system-and-env.md`. Branch
`hvx/60-cpu-baseline`. The deliverable is a *measurement handoff*, not a
kernel: nothing under `nntrainer/tensor/hexagon/` changes, no ABI bump,
no simulator run ("no DSP bytes changed").

## 1. Goal and gate

Issue text: "a device measurement handoff (`docs/measurements/60-cpu-baseline.md`)
that the user runs once, with the exact `main` commit, the Android CPU build
recipe, the binary, the model/quantisation (the fair comparison to the HVX
W8A8 rows, both if two are reasonable), rows at 512 / 1024 / 4096 with
prefill and decode tok/s by the 64-token method, unit serial, SoC clock
state, battery temperature before/after, and an accuracy column."

Made measurable — the handoff is *filled* when every cell below has a
number and the supervisor can replace the `CPU | pending` row of
`HEXAGON_BENCHMARK.md` and rewrite the stage-1 goal cells as multiples:

| variant | threads | ctx | prefill tok/s | decode tok/s (64 tok) | init_len | gen_cnt | batt °C before/after | cpu7 MHz median | first-token match | common-prefix tokens vs x86 ref / vs DSP #25 B |
|---|---|---|---|---|---|---|---|---|---|---|
| A `QS4CX-FP32` (primary) | 8 | 512 / 1024 / 4096 | median of 3 runs | median of 3 runs | must equal ctx | must be 64 | | | | |
| A4 same, thread sweep | 4 | 512 | 1 run | 1 run | | | | | | |
| B `Q4_0-FP16` (secondary, optional) | 8 | 512 / 1024 / 4096 | 1 run | 1 run | | | | | | |

Bounds that make a row valid: `init_len == ctx` (the prompt was not
shortened by the tokenizer round trip; if it differs, the row is kept and
the delta written next to it), `gen_cnt == 64` (EOS banned, see §3),
spread of the three A run-means ≤ 5 % (else one cooled re-run), battery
temperature before each cell ≤ 32 °C. There is no pass/fail on the tok/s
value itself: the number *is* the baseline. Device time ≤ 30 min (§4,
step 6 estimates 20).

## 2. Where it lives

Everything is read from `origin/main` @ **`2d1e4974`** ("[Tensor] Include
<cstdint> where QScheme is declared", 2026-09-18, fetched 2026-09-18). No
file on `main` is edited; the implementer checks it out as a second
worktree and builds it as is. Verified references (`path:line` on `main`):

* `Applications/CausalLM/main.cpp:401` — `argv[2]` is used raw (no chat
  template; the README's "auto-wrapped" claim is stale); `:434`
  `do_sample = generation_cfg.value("do_sample", false)` (Qwen3's HF
  `generation_config.json` says `true` → must be overridden); `:430-432`
  `initialize()/load_weight()/repack_weight()`.
* `Applications/CausalLM/models/causal_lm.cpp` — `:86` `bad_word_ids`,
  `:89-91` `lmhead_dtype` defaults to `embedding_dtype`, `:141` KV cache is
  **FP16** under `ENABLE_FP16` (same as the DSP), `:225-245` tied lm_head =
  `tie_word_embeddings` layer with `shared_from embedding0`, `:301-345`
  `generate()`: bad words → `-INFINITY` (`llm_util.cpp:68-73`), argmax when
  `!do_sample`; `run()` `:446-460` truncates the prompt to
  `max_seq_len - num_to_generate` (the lever that pins `init_len`),
  `:520-605` prefill timer (one `incremental_inference` over the whole
  prompt + the first argmax), `:618-679` the 64-step generation loop,
  `:691-703` prints `prefill: N tokens, M ms, X TPS` and
  `generation: G tokens, M ms, Y TPS`.
* `Applications/CausalLM/models/transformer.cpp:130-146` — reads
  `model_tensor_type`, `init_seq_len`, `max_seq_len`, `num_to_generate`,
  `embedding_dtype`, `fc_layer_dtype`; `:356-365` `repack_weight()` calls
  every layer's `pack()`.
* Precision paths: `nntrainer/layers/fc_layer.cpp:183-192` (`QINT8` =
  per-tensor affine quantizer, `:227-231` dequantises to float before the
  dot — storage only, no int8 kernel), `:218-222` packs `QS4CX`;
  `nntrainer/tensor/float_tensor.cpp:743-768` FP32-activation dot
  dispatch, `:1054-1097` `dotQs4cx` → KleidiAI `qai8dxp` × `qsi4cxp`
  (kernel index 2 = `1x8_qsi4cxp8x8_1x8x32_neon_dotprod` GEMV at M=1,
  8 = `4x8_qsi4cxp8x8_8x8x32_neon_i8mm` GEMM);
  `nntrainer/tensor/half_tensor.cpp:718-745` FP16-activation dot supports
  only `Q4_0` / `Q6_K` (so `QS4CX` needs FP32 activations);
  `nntrainer/models/model_common_properties.h:205-223` the
  `model_tensor_type` strings (`QS4CX-FP32`, `Q4_0-FP16`, …);
  `nntrainer/layers/layer_devel.h:423-445` the FP32 → QS4CX producer used
  by `save_weight` (per-channel symmetric scales, `quant_qs4cx_f32`);
  `nntrainer/tensor/cpu_backend/arm/kai_interface/kleidiai_interface_qai8dxp_qsi8cxp.cpp`
  a true **W8A8** KleidiAI kernel (`qsi8cx` weights × `qai8dx`
  activations) that no `DataType`, tensor class or loader reaches.
* Quantiser: `Applications/CausalLM/quantize.cpp:104-106` dtype map
  (`FP32 FP16 Q4_0 Q6_K Q4_K QS4CX`), `:175` writes
  `model_tensor_type = <fc>-FP32`, `:476` / `:544` map `embedding0` /
  `output_of_causallm`, `:814` `save_weight(..., layer_dtype_map, isa)`.
  `Applications/CausalLM/layers/embedding_layer.cpp:530,687-731` embedding
  rows for `Q6_K` / `Q4_0` / `QS4CX`; `layers/lm_head.cpp:196-209` the tied
  lm_head is `LmHeadLayer`, which has **no `pack()`**, so a `QS4CX` tied
  lm_head would hit `QS4CX_Tensor::getPackedData()`'s "pack before run
  model" throw (`qs4cx_tensor.cpp:95-100`) — lm_head/embedding stay `Q6_K`.
* Build recipe: `.github/workflows/causallm_android.yml` (NDK r26d, Rust
  stable + `aarch64-linux-android`, `Applications/CausalLM/build_android.sh`);
  `build_android.sh:117-119` wipes `builddir` and runs
  `tools/package_android.sh` (defaults `-Dplatform=android -Denable-fp16=true
  -Dnntr-num-threads=4`, `--arm-arch` default `armv8.2-a` →
  `-march=armv8.2-a+fp16+dotprod+i8mm`, `tools/cross/android_armv8-2-a.json`),
  `:136-137` builds `lib/libtokenizers_android_c.a` with cargo if missing
  (only the x86 `libtokenizers_c.a` is tracked), `:177` `ndk-build`
  of `causallm_core nntrainer_causallm nntr_quantize nntr_safetensors_info`.
  `install_android.sh:204-212` writes `run_causallm.sh` with
  `NNTR_NUM_THREADS=4` hard-coded; `nntrainer/utils/thread_manager.h:94-105`
  reads the env var first, then the compile flag.
* Model files: `Applications/CausalLM/res/qwen3/qwen3-0.6b/weight_converter.py`
  (torch + transformers; HF safetensors → fp32 `.bin`, 2,384,199,680 B per
  HEXAGON.md §5.4's fallback log), `gguf_to_nntrainer.py` (numpy only; GGUF
  → `Q4_0` + `Q6_K`, an alternative producer for B). The HF snapshot is
  already on the Mac at `Applications/CausalLM/res/qwen3/qwen3-0.6b/hf/`
  (`model.safetensors` 1,503,300,328 B) and on the workstation at
  `models/qwen3-0.6b-hf/` (#23).
* Existing measurement tooling on `main` (reused as reference, not run by
  agents): `Applications/CausalLM/benchmarks/benchmark_android.py`
  (injects `sample_input` into `nntr_config.json`), `device_utils.py`
  (`/sys/class/thermal/thermal_zone0/temp`, cool-down wait),
  `repeat_perf_test.py` (governor write needs root — not on this unit).
* On `hvx_impl` (this branch): `tools/hexagon/make_tokens.py` (`--limit`
  also writes the detokenised `.txt`), `test/hexagon/hexagon_e2e_test.cpp:261-333`
  (`E2E step … top1=`, `E2E gen <ids>`, `E2E ppl`), the #25 B logs under
  `logs/hexagon/` (the DSP `E2E gen` ids for 512 / 1024 / 4096),
  `docs/measurements/23-sdk64-baseline.md` (token files: `t512_23.i32` md5
  `10bb428f…` from `models/eval_23.txt` md5 `a838b25d…`, whose text is
  reproduced there; `t1024.i32` / `t4096.i32` are the workstation's P4
  files with no source text), `tools/docker/Dockerfile:31-53` (no Rust, no
  torch, NDK r26d baked in), `tools/docker/run.sh:28` (`REPO_ROOT` is
  the script's own tree → needs a knob to mount the `main` worktree).

New files, all on `hvx_impl`: `tools/hexagon/cpu_baseline.py`,
`docs/measurements/60-cpu-baseline.md`, a two-line knob in
`tools/docker/run.sh` (`NNTR_WORK_DIR`) and the Dockerfile additions of
step 1. No consumer of the wire format moves.

## 3. Design

**Precision (user constraint 2026-09-18: closest to W8A8, never fp16/fp32 as
the primary row).** The HVX row is int8 per-channel symmetric weights for
every projection and the tied lm_head, per-token int8 symmetric activations
(int16 for `down_proj`), int32 accumulate, fp16 elsewhere, fp16 KV. What
`main` can run on the S25 Ultra CPU, settled from the code above:

| candidate | weights | activations | reachable from the app? |
|---|---|---|---|
| **`QS4CX-FP32`** | int4 **per-channel symmetric** (KleidiAI `qsi4cx`) | **per-token** (per-row) int8 asymmetric, dynamic (`qai8dx`), int32 acc, fp32 out | yes: `nntr_quantize --fc_dtype QS4CX`, `dotQs4cx`, packed at load |
| `Q4_0-FP16` | int4 block-32 symmetric (ggml) | int8 block-32 (`q8_0`), fp16 elsewhere | yes: the README's recommended device config |
| `QINT8` | int8 per-tensor affine | fp32 (weight dequantised before the dot) | not in `nntr_quantize`'s map; no int8 kernel — storage only |
| `qsi8cx` × `qai8dx` (W8A8) | int8 per-channel symmetric | per-token int8 asymmetric | **kernel exists, nothing reaches it** (no `DataType`, tensor, loader) |
| `FP16-FP16` | 16-bit | 16-bit | yes, but 2× the HVX byte stream: a different question |

**Chosen: A = `QS4CX-FP32` for all FC layers, `Q6_K` embedding + tied
lm_head, 8 threads**, because it is the only path on `main` with the same
quantisation *scheme* as the HVX kernels (per-channel weight scale,
per-token activation scale, integer dot with int32 accumulate) — the axis
the user named. Stated gaps, to be printed in the handoff next to the row:
(1) weights are 4-bit, so the CPU streams ≈ 0.22 GB of FC weights per step
against the DSP's 0.44 GB — the CPU row is *favoured* on bandwidth;
(2) activations are asymmetric with a zero point, the DSP's are symmetric;
(3) `down_proj` activations are int8, the DSP's int16; (4) lm_head is
6-bit `Q6_K` (the tied `LmHeadLayer` cannot pack `QS4CX`), the DSP's int8;
(5) non-matmul math is fp32 here, fp16/qf on the DSP; KV is fp16 on both.
**B = `Q4_0-FP16`, `Q6_K` embedding** is the optional secondary row: it is
`main`'s fastest CPU configuration (README table, S26 Ultra: 755 / 80 tok/s
at 437 tokens, 8 threads) and the number a reader would compare with the
GENIEX q4_0 CPU rows; it is measured once per context after A, and only
after A is complete. An fp16-weight row is not in this handoff (a 2×-bytes
comparison answers nothing about the W8A8 goal); if the user wants it
later it is one more `nntr_quantize --fc_dtype FP16` directory and three
runs.

**Rejected alternative: wire the existing W8A8 KleidiAI kernel** (`qsi8cx`
weights, `qai8dx` activations) so the CPU row is a true W8A8 — a new
`DataType::QS8CX`, tensor class, `fc_layer` pack, `save_weight` producer
and `dotQs8cx`, i.e. nntrainer core on `main`. Out of the Hexagon
subtree by the contract (§4: "anything in nntrainer core is filed as an
issue"), and it would make the baseline a number *we* produced rather than
`main`'s. Filed as a follow-up candidate in §6, not done here.

**Speed method.** The app has no per-step timer, so the row reads the two
summary lines of `run()`: prefill tok/s = `N / prefill ms` (the same
definition as the HVX rows: prompt tokens over the summed chunk time; the
CPU prefill is one call, no chunking), decode tok/s = the `generation:`
TPS, i.e. the **mean over the 64 generated tokens** (each step =
`incremental_inference` + argmax over 151,936 logits + detokenise/print),
where the HVX rows use the median over steps 2–64 of the RPC wall. The
estimator difference is recorded in the handoff and the benchmark note;
it is bounded by the run-to-run spread, which is why A takes three runs
per cell and reports the median of the run means with min/max. No source
change on `main` for a per-token log: the contract forbids it and it
would no longer be "`main`".

**Prompt equality with the HVX rows.** `nntrainer_causallm` takes text,
the harness takes ids. For 512 the source text exists (`models/eval_23.txt`,
tokenises to > 512 ids with the same `tokenizer.json`; `make_tokens.py`
uses `add_special_tokens=False`, the app inserts no BOS —
`causal_lm.cpp:433` is commented out): set `max_seq_len = 512 + 64`,
`num_to_generate = 64`, feed the whole text and let `run()` truncate to
exactly 512 (`:446-460`, it warns and keeps the head), so `init_len ==
512` and the ids equal `t512_23.i32`. For 1024 / 4096 only the P4 id
files exist: the script detokenises them, appends a filler paragraph so
truncation lands on the ctx, and **verifies** `encode(text)[:ctx] == ids`
in the container (`PROMPT_MATCH n/ctx`); a mismatch is reported, the
speed row stays valid, the accuracy column then says "prompt differs at
k". The prompt goes into `sample_input` of a per-context
`nntr_config.json` (as `benchmark_android.py` does), never onto the adb
command line (a 4096-token text as an argv is a quoting hazard).

**Forcing 64 greedy tokens.** `generation_config.json` gets
`do_sample: false` (argmax, like `hexagon_e2e_test`); `bad_word_ids:
[151643, 151645]` (`<|endoftext|>`, `<|im_end|>`) so EOS cannot cut the
window (`generate()` sets them to `-INFINITY`, then argmax). `gen_cnt` is
recorded and must read 64.

**Accuracy column (what is compared, since `--eval` PPL is not consumable
by `main`'s CLI).** Three greedy 64-token continuations of the same prompt:
the x86 reference `hexagon_ref_run … --chunk 128 --steps 64` (W8A8 oracle),
the DSP #25 B `E2E gen` line (already on disk), and the CPU output text.
The report step re-tokenises the CPU text with the HF tokenizer and
prints (a) first-token top-1 agreement (the CPU's first generated token vs
the `top1=` of the last prefill chunk in the two id lists) and (b) the
longest common id prefix vs each reference. Two correct but differently
quantised implementations diverge after a handful of tokens (HEXAGON.md
§8.1), so this is a *sanity* column with no pass bar; the x86 CPU run of
step 2 (same QS4CX model on the x86 fallback kernel) adds a fourth text
so a device-only anomaly (e.g. the i8mm kernel) would show as CPU-x86 ≠
CPU-device. A teacher-forced PPL on the CPU would need a small host
program around `CausalLM::setLogitsProcessor` (`transformer.h:79-113`:
record `log_softmax[ref]` in `process()`, force the reference id) — ~80
lines against `libcausallm_core.so`, a separate issue if the supervisor
wants the PPL column filled (§6).

**Thermal / clock protocol** (recorded, not controlled — no root): screen
off, no foreground apps, phone on the same USB as every HVX row; before
each cell wait until `dumpsys battery | grep temperature` ≤ 320 (32.0 °C)
and `thermal_zone0` ≤ 40 °C (`device_utils.wait_for_cooling` thresholds),
record both before and after; `getprop ro.serialno`, `ro.hardware.chipname`,
`cpu*/cpufreq/{cpuinfo_max_freq,scaling_governor}` once per session;
during each run a background `adb shell` samples `cpu0` and `cpu7`
`scaling_cur_freq` every 0.5 s and the report prints min / median / max
(the CPU analogue of `pcycles_per_us`, HEXAGON.md §7 rule 9). Thread
count: `NNTR_NUM_THREADS` and `OMP_NUM_THREADS` exported explicitly (the
device `run_causallm.sh` hard-codes 4, so the script runs the binary
directly with `LD_LIBRARY_PATH` set) and echoed into the log; no
`taskset` (all 8 cores, the README's method), cores identified by the
recorded max frequencies.

## 4. Steps

Rung numbers refer to `.claude/skills/hexagon-gates`. Because no file under
`nntrainer/tensor/hexagon/htp/`, `test/hexagon/sim_*`, the packer or the
lowering changes, rungs 2–3 are skipped; the PR states "no DSP bytes
changed" with `git diff hvx_impl --stat -- nntrainer/tensor/hexagon/htp`
empty.

1. **Container prerequisites** (`tools/docker/Dockerfile`, `tools/docker/run.sh`).
   Add `cmake`, rustup (stable, `rustup target add aarch64-linux-android`)
   and the CPU torch wheel (`pip3 install torch --index-url
   https://download.pytorch.org/whl/cpu`) — the first two for
   `build_tokenizer_android.sh`, the last for `weight_converter.py`. Add
   `NNTR_WORK_DIR` to `run.sh` so the `main` worktree can be mounted at
   `/work` (`REPO_ROOT="${NNTR_WORK_DIR:-$REPO_ROOT}"`), keeping `/model`
   at the HF-holding directory. Create the worktree:
   `git worktree add ../nntrainer-main 2d1e4974` (read-only; `git status`
   clean is part of every later gate).
   Gate: `tools/docker/run.sh --build-image`; inside,
   `rustc --version && rustup target list --installed | grep aarch64-linux-android && python3 -c 'import torch'`.
   Fallback if the user prefers not to grow the image: the CI recipe on the
   workstation (NDK r26d + Rust), recorded as such in the handoff.
2. **Model files** (container, x86, `NNTR_WORK_DIR=../nntrainer-main`).
   `python3 Applications/CausalLM/res/qwen3/qwen3-0.6b/weight_converter.py`
   from `/model/hf` → `nntr_qwen3_0.6b_fp32.bin` + the fp32
   `nntr_config.json` (2,384,199,680 B expected). Build `main` for x86
   (`meson setup build-x86-main -Denable-transformer=true -Denable-fp16=true
   -Dthread-backend=omp && ninja -C build-x86-main`, the README §2 recipe),
   then
   `nntr_quantize <fp32 dir> -o <A dir> --fc_dtype QS4CX --embd_dtype Q6_K`
   and `nntr_quantize <fp32 dir> -o <B dir> --fc_dtype Q4_0 --embd_dtype Q6_K --isa ARM`
   (Q4_0 is ISA-specific, README "Q4_0 platform dependency"; QS4CX packs at
   load, so one file serves x86 and the device). Copy `tokenizer.json`,
   `config.json`, `generation_config.json` (with `do_sample: false`).
   Gate (rung 1, x86): both directories load and generate 64 fluent tokens
   on x86 with `build-x86-main/Applications/CausalLM/nntr_causallm <A dir>`
   on the 512 prompt (the QS4CX x86 fallback kernel
   `gemm_qai8dxp_qsi4cxp_rhs_unpacked` exists); record `md5sum` and byte
   size of every `.bin` and the generated `nntr_config.json` dtype lines
   (`"model_tensor_type": "QS4CX-FP32"`, `"fc_layer_dtype": "QS4CX"`,
   `"embedding_dtype": "Q6_K"`). Also on x86, the accuracy reference:
   `build_x86_hexagon/hexagon_ref_run /work/build_x86_hexagon/qwen3_full --tokens t512_23.i32 --chunk 128 --steps 64`
   (hvx_impl tree, HEXAGON.md §5.1) → the 64 reference ids for 512; for
   1024 / 4096 the same on the workstation's token files if the
   implementer has them, else the DSP #25 B `E2E gen` lines alone.
3. **Android build of `main`** (container, `NNTR_WORK_DIR=../nntrainer-main`,
   `ANDROID_NDK` from the image): `cd Applications/CausalLM && ./build_android.sh`
   (no `--cache`, no `-Denable-hexagon`; defaults = the CI job).
   Gate (rung 4 analogue): `jni/libs/arm64-v8a/{nntrainer_causallm,libcausallm_core.so,libnntrainer.so,libccapi-nntrainer.so,libc++_shared.so}`
   exist, `md5sum` recorded;
   `strings libnntrainer.so | grep -c qsi4cxp8x8_8x8x32_neon_i8mm` ≥ 1
   (the i8mm kernel is compiled in, i.e. `-march` carried `+i8mm`);
   `git -C ../nntrainer-main status --porcelain` empty except the untracked
   build outputs.
4. **`tools/hexagon/cpu_baseline.py`** (hvx_impl, ≤ ~250 lines, stdlib +
   numpy + transformers, run by the user on the workstation; agents never
   call adb). Sub-commands: `prepare` (from the HF dir, the A/B model dirs
   and the three token files, write `qwen3-0.6b-{A,B}-{512,1024,4096}/`
   with `nntr_config.json` (`sample_input`, `init_seq_len = ctx`,
   `max_seq_len = ctx + 64`, `num_to_generate = 64`, `batch_size = 1`,
   `fsu = false`, `bad_word_ids = [151643, 151645]`, dtype fields from the
   quantiser's config, `model_file_name` pointing at the shared `.bin`,
   `tokenizer_file` relative) and `generation_config.json` with
   `do_sample: false`; print `PROMPT_MATCH n/ctx` per context), `run`
   (push once, then per cell: cool-down wait, before-temps, background
   `scaling_cur_freq` sampler, `adb shell "cd <dir> && export
   LD_LIBRARY_PATH=. NNTR_NUM_THREADS=<t> OMP_NUM_THREADS=<t> &&
   ./nntrainer_causallm <model dir>"`, after-temps; log to
   `logs/hexagon/cpu_<stamp>.log`), `report` (parse `prefill:` /
   `generation:` / `Peak memory` / the `[CausalLM] WARNING … truncating`
   line, compute the tables of §1 including the accuracy prefixes from
   the reference id files). Gate: `python3 -m py_compile`; `prepare`
   inside the container prints `PROMPT_MATCH 512/512` for `t512_23.i32`
   (and `n/1024`, `n/4096` if the P4 files are available on the Mac —
   they are workstation files, so this half of the gate may only run at
   handoff time and the script must tolerate it); `report --selftest` on
   a canned log reproduces a known table.
5. **Handoff `docs/measurements/60-cpu-baseline.md`** (`hexagon-handoff`
   skill): the `main` sha, every artifact with md5 (binaries from step 3,
   `.bin`s from step 2, token files, the three config directories, the
   reference id files), the three commands (`prepare`, `run --variant A
   --threads 8 --ctx 512,1024,4096 --repeat 3`, `run --variant A --threads
   4 --ctx 512`, `run --variant B --threads 8 --ctx 512,1024,4096`,
   `report`), expected log lines (`prefill: 512 tokens, … TPS`,
   `generation: 64 tokens, … TPS`, `Peak memory usage`, the truncation
   warning), the empty tables of §1 with the estimator note, and the
   protocol of §3. **This is the step where the device is unavoidable**;
   variants = A (3 ctx × 3 runs), A4 (1 run), B (3 runs) — 13–16 runs,
   ≈ 4 s load + 1 / 3 / 20 s prefill + ≈ 1.5 s decode each, two 0.35 GB
   pushes ≈ 1 min, cooling waits ≤ 10 min: **≈ 20 min**, under the
   30-min cap. Label `state:needs-measurement`.
6. **Read-back** (supervisor, after `state:measured`): md5 check, fill the
   benchmark CPU row from A (8 threads), note B and A4 in the same cell's
   note, rewrite the stage-1 goal cells as multiples of A's decode at 512 /
   4096 and prefill at 512, log the decision (§6).

Order of the user's session is fixed as A → A4 → B so the primary row is
never lost to a late crash; if a B or 4096 cell fails (OOM, throw), the
handoff says to note the last log line and continue.

## 5. Risks

* **Untested path on this device.** `QS4CX` on Qwen3 at 4096 tokens has
  no record on `main`'s CI (which only builds). A crash or a garbage
  continuation shows up in the accuracy column and `gen_cnt`; the fallback
  is to read B for that cell and file the QS4CX failure upstream. The x86
  run of step 2 catches config-level problems before the phone is used.
* **Estimator gap (mean vs median).** Visible through the three-run
  spread column; if it exceeds 5 % the cooled re-run rule applies and the
  benchmark note carries the spread. The DSP rows' own 512 drift is ±5 %
  by construction (HEXAGON.md §8.2), so the comparison is not made
  finer than that.
* **Thermal and clock.** 8 threads for 20 s of 4096 prefill is more
  sustained CPU load than any HVX row put on the host; the battery
  temperature before/after and the `cpu7` frequency samples make a
  throttled run visible, and the cool-down wait before each cell keeps
  the cells comparable. No governor control (unrooted), same as the DSP
  rows (§7 rule 9: record, compare within a session).
* **Prompt round trip** for 1024 / 4096 (ids → text → ids): reported by
  `PROMPT_MATCH`; the speed row is unaffected by a few differing ids, the
  accuracy column is annotated.
* **Bandwidth asymmetry of the chosen precision.** A streams half the FC
  bytes of the HVX row (4-bit); the handoff prints the gap list of §3
  above the table so the multiple is read with it. This is the reason the
  W8A8 KleidiAI wiring is filed as a follow-up rather than silently
  accepted.
* **Stale artifacts.** The device run records `md5sum` of the pushed
  `nntrainer_causallm`, `libnntrainer.so` and the two `.bin`s from
  `adb shell md5sum`, next to the container md5s; a mismatch voids the
  row (environment-gap rule).
* **Thread env.** `run_causallm.sh` on the device hard-codes 4 threads;
  the script bypasses it and echoes the exported values into the log so
  a 4-thread "8-thread" row cannot go unnoticed.

## 6. Docs to update

* `docs/backend_guide/HEXAGON_BENCHMARK.md`: replace the
  `nntrainer | (issue #60) | … | CPU, main | pending` row with three rows
  (A at 512 / 1024 / 4096, precision `QS4CX-FP32 + Q6_K head`, 8 threads,
  unit serial, battery °C, the estimator note, B and A4 in the note);
  the stage-1 W8A8 `decode @512`, `decode @4096` and `prefill @512` goal
  cells become multiples of A; a Log line dated with the fill; the
  Method paragraph gains one sentence on the CPU estimator (mean over 64)
  and the precision gap list.
* `docs/backend_guide/HEXAGON.md`: §5.4's "CPU, fp32 (fallback) 52.0 /
  18.7 tok/s" table gets a pointer to the #60 rows (that figure is the
  M5-era fp32 app run on a different prompt); a short "CPU baseline
  (#60)" block at the end of §8.2 with the A / A4 / B numbers and the
  accuracy prefixes; §9 gains the follow-up candidate "wire the
  `qsi8cx` × `qai8dx` KleidiAI kernel as `DataType::QS8CX` so the CPU
  baseline can be a true W8A8 — nntrainer core, filed upstream" and,
  optionally, "teacher-forced CPU PPL via `LogitsProcessor`".
* `docs/plans/0000-agent-system-and-env.md` §1: the sentence "The
  `nntrainer` CPU rows are skipped for now" is superseded (supervisor).
* `docs/backend_guide/hexagon-guide/06-performance.html`: the guide
  writer adds the CPU baseline to the history table and the chart after
  the fill.
