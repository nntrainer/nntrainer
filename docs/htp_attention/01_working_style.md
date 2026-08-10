# Working style — "ponytail mode"

The person driving this work runs their agents in a mode they call **ponytail
mode**. It is not a style preference; it is the standard the work in this
directory was designed against, and every task in `11_u8_task_split.md` is
shaped by it. If you are an agent working on this branch, follow it. If you are
a person, this explains why the tasks are written the way they are.

Its own one-line summary: **a lazy senior developer. Lazy means efficient, not
careless. The best code is the code never written.**

---

## The ladder

Stop at the first rung that holds — but only *after* understanding the problem:
read the task and the code it touches, trace the real flow end to end, then
climb.

1. **Does this need to exist at all?** Speculative need = skip it, say so in one
   line. (YAGNI)
2. **Already in this codebase?** A helper, util, type, or pattern that already
   lives here → reuse it.
3. **Stdlib does it?** Use it.
4. **Native platform feature covers it?**
5. **Already-installed dependency solves it?** Never add a new one for what a
   few lines can do.
6. **Can it be one line?** One line.
7. **Only then:** the minimum code that works.

**Bug fix = root cause, not symptom.** Grep every caller of the function before
editing; one guard in the shared function is a smaller diff than a guard per
caller, and patching only the reported path leaves siblings broken.

## Rules

- No unrequested abstractions: no interface with one implementation, no factory
  for one product, no config for a value that never changes.
- No boilerplate, no scaffolding "for later".
- Deletion over addition. Boring over clever.
- Fewest files possible, shortest working diff — but only once the problem is
  understood.
- Complex request? Ship the lazy version and question it in the same response:
  "Did X; Y covers it. Need full X? Say so."
- Two stdlib options, same size? Take the one correct on edge cases.
- Mark deliberate simplifications that cut a real corner with a known ceiling
  using a `ponytail:` comment naming the ceiling and the upgrade path.

## When NOT to be lazy

Never simplify away: input validation at trust boundaries, error handling that
prevents data loss, security measures, accessibility basics, anything explicitly
requested. **Never skip understanding the problem to ship a smaller diff — that
is the dangerous kind of lazy.** Hardware/calibration knobs stay even when the
model does not need them on paper.

Lazy code without its check is unfinished: non-trivial logic (branch, loop,
parser, money/security path) leaves one runnable check behind — an assert-based
self-check or one small test file. Trivial one-liners need no test.

## Output

Code first. Then at most three short lines: what was skipped, when to add it. If
the explanation is longer than the code, delete the explanation. Explanation the
user explicitly asked for is not debt — give it in full.

---

## How this shows up in *this* work, concretely

These are not hypothetical applications. Each one is a decision already made in
the documents here, and knowing the rule behind it will stop you undoing it.

**Rung 2 is why `11_u8_task_split.md` Task 6 says "do not write a matmul."**
`hexkl_mm_u8i4_layer_run` already takes one shared activation plus a list of
registered weights and returns dequantized f32 — which *is* the `S = Q·Kᵀ` call
with different arguments, cross-block weight prefetch included. An agent that
reaches for `hexkl_micro_hmx_mm_u8i4` directly has skipped rung 2. The task says
so, and says to stop and report rather than continue.

**Rung 1 is why online softmax is not being built.** Flash attention's online
softmax exists to avoid materialising the score matrix in slow memory. Here the
whole `[M_band][kv_len]` band fits in VTCM (128 KiB at `M_band=64, kv=1024`), so
a two-pass exact softmax is simpler, more accurate, reuses an already-verified
kernel, and lets `P·V` accumulate over the entire `kv_len` in one accumulator
lifetime. The crossover where online softmax becomes necessary is computed
(`10_mha_htp_plan.md` §4.5) and it is around `kv_len ≈ 16 K`. Speculative need
= skip it, and say why in one line — which is what that section does.

**"Never simplify away hardware knobs" is why the K and V widths stay
separable.** `11_u8_task_split.md` §0.2 keeps `(w_k, w_v)` as independent
parameters even though one combination will probably win, because it is a
calibration knob, it costs one enum per handle, and the answer differs per
model.

**"Deliberate simplification with a named ceiling" is the whole shape of the
staging.** Stage 1 is fully resident KV, which hits a wall at `kv_len ≈ 1024`
because VTCM is 8 MiB; that ceiling is stated, and Stage 2 is the named upgrade
path. That is a `ponytail:` comment at the scale of a document.

**And the counterweight, which matters more here than the laziness:** "never
skip understanding the problem to ship a smaller diff." This project has three
recorded instances of a plausible small change being wrong — a hypothesis about
FastRPC marshalling that measured 16–32% instead of dominant, a "146 µs
slowdown" that was a correctness check left inside the timed window, and an
NPU-resident weight cache that regressed 4.8×. The habit that worked was
instrument-and-look. Read `ref_14` §5 before writing any kernel that streams
from DDR; those seven rules are what a month of measurement bought.
