# Prompt kit — what to actually paste into the implementing agent

Companion to `MHA_HTP_U8_TASKS.md`. That file holds the *task bodies*; this one
holds the boilerplate that wraps every one of them, the fully assembled first
message, and the replies to send when the agent comes back with something.

Rule of thumb: **one task = one fresh session.** Not because context runs out, but
because a session that has already declared Task N done will defend that claim
while doing Task N+1.

---

## 1. PREAMBLE — paste this at the top of every task, unchanged

```
You are implementing one task in a multi-session plan for nntrainer's Hexagon HTP
backend. The repo is at /home/leeseunghui/workspace/nntrainer.

BEFORE DOING ANYTHING:
1. Read MHA_HTP_PLAN.md in full.
2. Read MHA_HTP_U8_TASKS.md in full.
3. Then, in your first reply and before writing any code, do three things:
   a) restate MHA_HTP_U8_TASKS.md §1.8's constraint list in your own words;
   b) state which of §1.5's four scale axes applies to each of Q, Kt, P and V;
   c) list the files you intend to create or modify, and stop for one round if
      that list includes anything the task told you not to touch.

RULE ZERO -- these are not negotiable, and violating one means the task failed
even if the tests are green:

1. Do not modify any reference implementation:
   - compute_kcaches_fp32_reference (Applications/CausalLM/layers/mha_core.cpp:42)
   - compute_vcache_fp32_transposed_reference (same file, :73)
   - nntrainer::softmax_row / nntrainer::softmax_row_inplace
   - the scalar reference in test/unittest/unittest_hvx_mm_u8i4.cpp
   If a test fails, the new code is wrong, not the reference.

2. Do not change a tolerance, a shape, or a test matrix from the ones written in
   the task. They were fixed before any code existed, specifically so they cannot
   be adjusted to fit a result.

3. Do not invent a HexKL, HVX or QuRT API. Every call must already exist in
   ~/workspace/hxkl-beta2/hexkl_addon/include/hexkl_micro.h or in a header this
   tree already includes. If you need something that does not exist, stop and
   report it instead of writing a plausible-looking call.

4. Do not modify any of these -- all are device-verified and no task in this plan
   requires changing them:
     nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_dma.{c,h}
     nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i8_dma.{c,h}
     nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4.{c,h}
     nntrainer/tensor/htp_backend/hmx/hexkl_dma_ring.{c,h}
     nntrainer/tensor/htp_backend/hvx/hvx_quant_u8.c
     nntrainer/tensor/htp_backend/hvx/hvx_dequant_i32.c
   If you believe one must change, stop and report which line and why.

5. If a test fails, do not "fix" the test, loosen a bound, skip a case, or narrow
   the matrix. Stop and report the failure with its complete output.

6. Do exactly this task. Do not start the next one, do not refactor code you
   happened to read, do not "improve" anything outside the stated scope. If you
   notice a real problem outside scope, write it in your final report instead of
   fixing it.

HOW TO REPORT
- Paste the COMPLETE output of every command you run. Not a summary, not "all
  tests passed", not an excerpt. If the output is long, paste it anyway.
- If you ran a command and it failed, paste that too. A clean-looking report with
  a hidden failed run is the single worst outcome of this task.
- State the line count of your diff (`git diff --stat`).

NUMERICAL FAILURE SIGNATURES -- use these to diagnose, not to adjust bounds:
- At i8, max relative error around 1e-3 or below is healthy. At i4, around 1e-2
  is healthy.
- An error roughly 10x larger at i4 than at i8 on the same shape means the width
  is doing what it should. That is not a bug.
- An error that is THE SAME at both widths is not a quantization problem. Look at
  layout, masking, or an index calculation.
- >= 1e-1, or an error pattern that is blocky/structured rather than scattered,
  is a layout bug.
Report which of these cases you are looking at. Do not change the tolerance.
```

---

## 2. The first message — Task 1, fully assembled, copy-paste ready

Paste §1's PREAMBLE, then this:

```
=== TASK 1: KV block quantizer, parametric over i4 and i8 (HOST ONLY) ===

GOAL
New files nntrainer/tensor/htp_backend/hmx/hexkl_kv_quant.{c,h}: turn appended
fp16 K and V rows into exactly the inputs hexkl_weight_u8i4_register and
hexkl_weight_u8i8_register want, one KV block at a time.

Pure arithmetic: no HVX intrinsics, no HexKL calls, no DSP-only headers. This file
must compile and run on the HOST. There is no device work in this task.

  typedef enum { HEXKL_W_I4 = 4, HEXKL_W_I8 = 8 } hexkl_w_width;

  /* Kt block: logical [head_dim][T], N = kv position within the block.
     Symmetric per-N (per kv position) quantization. */
  void hexkl_kvq_pack_kt_block(const uint16_t *k_rows_f16, uint32_t n_rows_valid,
                               uint32_t T, uint32_t head_dim, uint32_t nch,
                               uint32_t head, hexkl_w_width w,
                               int8_t *out_rm, float *out_scale,
                               int32_t *out_colsum);

  /* V block: logical [T][head_dim], N = head_dim column.
     Symmetric per-N (per column) quantization over THIS BLOCK's rows only. */
  void hexkl_kvq_pack_v_block(const uint16_t *v_rows_f16, uint32_t n_rows_valid,
                              uint32_t T, uint32_t head_dim, uint32_t nch,
                              uint32_t head, hexkl_w_width w,
                              int8_t *out_rm, float *out_scale,
                              int32_t *out_colsum);

Both read the CPU cache layout [kv_position][nch][head_dim] in fp16: row r of
head h starts at (r*nch + h)*head_dim. See mha_core.cpp:61 and :95.

The width is a RUNTIME parameter and there is ONE code path -- not two functions,
not a macro expansion, not a template. The only differences are the clamp range
and the scale denominator:
    i4 -> [-8, 7]        i8 -> [-127, 127]
i8 deliberately excludes -128 so the symmetric scale stays exact. Both widths
write out_rm as one int8 container per value: that is what rm_to_wh_i4 and
rm_to_wh_i8 both take, and the packing down to 4 bits happens inside the WH bake,
not here. Round to nearest, ties away from zero.

n_rows_valid < T means a partial tail block. Zero-fill every unused element of
out_rm, and set the corresponding out_scale entry to 1.0f and out_colsum to 0.
A garbage tail is a silent wrong answer later; the mask alone will not save it.

MUST NOT TOUCH any existing file. This task adds two source files and one test
file, plus one line in test/unittest/meson.build.

BUDGET: about 250 lines across the two new files. If you exceed it, stop and
report why rather than continuing.

ACCEPTANCE
New host gtest test/unittest/unittest_hexkl_kv_quant.cpp covering the cross
product of:
    T             = 32, 64, 256
    head_dim      = 64, 128
    nch           = 1, 8
    head          = 0 and nch-1
    n_rows_valid  = 1, T-1, T
    width         = HEXKL_W_I4, HEXKL_W_I8
and proving all of:
  a) ROUND TRIP: dequantizing out_rm with out_scale reproduces the input within
     the theoretical quantization bound for that width. COMPUTE that bound from
     the width and the row's dynamic range in the test. Do not hardcode a number.
  b) COLSUM: out_colsum[n] equals a plain independent sum of column n of out_rm,
     computed by a separate loop written in the test.
  c) TAIL: every element past n_rows_valid in a partial block is exactly 0, and
     its scale is exactly 1.0f.
  d) RANGE: no value falls outside the width's range.
  e) CONSISTENCY: i4 and i8 agree in sign and in relative order on every element
     (i8 is a refinement of i4, not a different mapping). If this fails, one of
     the two ranges or one of the two scale denominators is wrong.

Register the test in test/unittest/meson.build the way the neighbouring tests
are registered. Do not restructure that file.

RUN AND REPORT
  cd builddir && meson test unittest_hexkl_kv_quant --print-errorlogs
Paste the complete output. Then state how many parameter combinations actually
ran. That count must equal the cross product above; if it does not, say so
explicitly and explain which axis collapsed.
```

---

## 3. Tasks 2–11: the assembly rule

```
[§1 PREAMBLE]  +  [the task block from MHA_HTP_U8_TASKS.md §3, with its
                   "[paste Rule zero]" and "[paste the failure-signature table]"
                   lines DELETED -- the preamble already contains both]
```

Two adjustments by task:

- **Tasks 2 and 3** are test-side only. Add one line: *"There is no device work in
  this task and nothing under nntrainer/ may change."*
- **Tasks 4–11** touch the device. Append the device block below, because the
  agent will otherwise waste a session on the environment:

```
DEVICE ENVIRONMENT -- use exactly this, do not improvise:

  export HEXAGON_SDK_ROOT=~/workspace/Hexagon_SDK/6.4.0.2
  export DEFAULT_HEXAGON_TOOLS_ROOT=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.04
  export HEXKL_ROOT=~/workspace/hxkl-beta2/hexkl_addon
  export HEXKL_SDK_VER=6.4.0.2
  cd test/htp && bash build.sh

- Use the BETA2 addon path above. The addon under
  Hexagon_SDK/6.4.0.2/addons/hexkl_addon is beta1, whose hexkl_micro_hw_init
  takes 2 arguments instead of 3. Mixing them fails at the DSP compile with
  "too many arguments to function call". That is a library version mismatch, not
  a code bug -- do not try to fix the code.
- Do NOT try to fix test/htp/build.sh's setup_sdk_env.source. It fails in this
  environment ("missed components"); the two exports above are exactly what it
  would have set.
- Run through test/htp/run_u8i4_layer_on_device.sh. EXTEND that script; do not
  write a second runner. It already handles the googletest submodule, the
  test/jni/googletest symlink, and the iniparser wrap-git placeholder.
- Pass NNTRAINER_ROOT=/home/leeseunghui/workspace/nntrainer explicitly on every
  ndk-build invocation. This shell's profile exports a different NNTRAINER_ROOT
  which silently wins over Android.mk's default. The same profile pins
  LD_LIBRARY_PATH to an unrelated build's libnntrainer.so, which makes a fresh
  binary segfault in a way that looks like heap corruption but is an ABI
  mismatch -- override it with the builddir's own nntrainer:api/ccapi:api/capi.
- Device is R3CY10WM83Y.
- If a DSP call hangs, or fails with no application-level detail, run
      adb logcat -d | grep adsprpc
  FIRST and paste the result. The kernel driver reports PD crashes with the
  crashed function name and a call trace, independently of FARF (which needs a
  .farf file next to the binary to reach logcat at all). Do not start guessing
  before you have looked at that.
- Anything you malloc and then index as an HVX vector type must be HVX_UVector*,
  never HVX_Vector*. malloc carries no 128-byte alignment guarantee and an
  aligned vector store on an unaligned base crashes the entire unsigned-PD DSP
  process. This applies to DSP-heap malloc, not only to FastRPC parameters.
- Run the device test THREE consecutive times and paste all three outputs.
  Thermal state moves the numbers and a single run is not evidence.
```

---

## 4. Replies to send when the agent comes back

The seven things it will say, and what to send. Keep these short — a long reply
invites negotiation.

**"All tests pass."** (no output pasted)
> Paste the complete, unedited output of the command, including the command line
> itself. I am not able to accept the result without it.

**"The tolerance was slightly exceeded so I relaxed it to X."**
> Revert the tolerance to the value in the task. Rule zero item 2. Then report
> the failing configuration, the observed value, and which of the numerical
> failure signatures it matches. Do not change the code yet.

**"I need to modify `hexkl_mm_u8i4_dma.c` to make this work."**
> Rule zero item 4. Tell me the exact line and what you need from it, and stop.
> Do not modify it. In this plan every task has been checked against those files
> and none of them needs a change, so the likely answer is that the call is being
> used with the wrong arguments rather than that it is missing a feature.

**"I could not run on device, so I verified by inspection."**
> Verification by inspection is not verification. If the device or the build is
> genuinely unavailable, say what failed and paste the error; that is a legitimate
> blocked state and I will unblock it. Do not report the task as complete.

**"While I was there I also refactored / renamed / cleaned up ..."**
> Revert everything outside this task's stated scope, then re-run the acceptance
> command and paste the output. Rule zero item 6. If the cleanup is worth doing,
> put it in your final report as a note and I will schedule it.

**"Should I continue with the next task?"**
> No. Stop here.

**"The slowdown is probably caused by X."** (no measurement)
> Do not act on that. Add the per-stage breakdown first and show me the numbers.
> The last time this project acted on a performance hypothesis before measuring
> the breakdown, the hypothesis was wrong: FastRPC marshalling was guessed to be
> dominant and measured at 16-32%, while the real cost sat elsewhere at 92%.

---

## 5. Accept checklist — all of these before you say "next task"

1. The three opener items (a/b/c) were actually answered, and the restatement of
   §1.8 is correct rather than paraphrase-shaped.
2. Complete command output pasted, with the command line visible.
3. The number of parameter combinations that ran equals the cross product in the
   task. Check the arithmetic yourself; this is the check most likely to fail
   quietly.
4. `git diff --stat` is within the task's budget, and every touched file is on
   the task's allowed list.
5. For device tasks: three runs, all pasted.
6. For tasks with a bit-exactness gate (4, 8, 10): the comparison is bitwise, not
   "within tolerance". If the report says "matches within 1e-6" where the task
   said bitwise, the gate was not run.
7. Nothing on Rule zero item 1's or item 4's list appears in the diff. Grep, do
   not trust the summary.

---

## 6. If a session dies mid-task

Start a fresh session with §1's PREAMBLE, the same task block, and this appended:

```
A previous session started this task and did not finish. Do not assume its work
is correct or complete.

First: run `git status` and `git diff`, and report what already exists. Then state
which parts of this task's ACCEPTANCE list are already satisfied and which are
not, with evidence for each claim -- an existing file is not evidence that it is
correct. Only after that, continue.

If what exists contradicts the task as written, the task wins. Delete the
contradicting work rather than adapting the task to it.
```
