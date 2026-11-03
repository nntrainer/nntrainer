# 61. HMX FC 경로의 a16w4 (계획서, 2026-10-18)

**상태 (2026-10-26): §3의 1–3단계와 5단계의 host 검사가 `defa7002c`에 들어갔다.** 4단계(skel 진입점, host
o-proj 배선)는 아직이다. Gate 0은 건너뛰었다. 대신 단말(v79) 벤치에서 a16w4 출력이 double 기준값과
f32 반올림 수준으로 일치하고, 같은 입력의 a8w4보다 오차가 몇 자릿수 작은 것을 확인했다.
- 커널: `hexkl_mm_opts.act16` → `hexkl_mm_u8i4_layer_run`이 두 번 sweep한다. 양자화:
  `hvx_quant_params_pack_u16_ah`. 합치기: `hvx_dq_tiles_job.lo_off`. 고정 encoding을 쓰는 producer를
  위해 `act_ah_prepacked`로 두 plane을 미리 받는 경로도 있다.
- 검사: `fc_layer_host_check.c`의 a16 케이스, `quant_u16_host_check.c`(실제 HVX 코드를 hvx_emu에서 실행,
  비트 단위 비교, 변형 검사 포함).
- 속도 작업 (`9a7a9bc74`):
  - accumulator를 라이브러리 int32 읽기 대신 convert 3회로 int24 byte plane에 쓴다
    (`hvx_dq_tiles_job.planes24`, K ≤ 4111). a8 경로도 같이 쓴다.
  - 패킷 루프의 정렬 검사는 호출당 한 번으로 줄였다.
  - dequant는 정렬 store를 쓴다.
  - 기각한 것: `weight.n:2x`는 처리량 모드가 아니라 int8 weight 읽기였다.
- 4단계 연결 시 지킬 것: 호출당 행 수는 staging buffer마다 결과 tile 32개가 남게 잡는다
  (K=2816 a16에서 768행). 남는 tile이 8개면 dequant worker를 기다리는 시간이 크게 늘어난다. a8의
  `fcMaxRows`에도 같은 효과가 있었다.
- 남은 격차: HMX가 노는 두 구간, 즉 시작 전 양자화와 f32 출력을 쓰는 worker 대기.
- HMX 탐침 결과 (v79, 2026-10-22, `run_main_on_hexagon`):
  - HMX 클럭 투표(`HAP_power_set_HMX_v2`)가 없으면 패킷이 몇 배 느리다. 투표를 더 세게 해도 그대로이므로
    투표한 상태의 패킷 시간이 하한이다. operand를 재사용해도, `:deep`을 써도 같다. HMX 시간을 재는
    탐침은 반드시 `nntr_hvx_open`처럼 투표해야 한다.
  - convert(`mxmem(..):after:retain:cm.ub = acc`) 다음의 clear에는 fence가 필요 없다. 64개 tile이
    int32 읽기와 비트 단위로 같았다.
  - u16 출력은 두 단계 형식으로만 가능하다: `bias = mxmem2` 256 B 블록(word c: fp16 scale,
    word c+32: bias), `cvt.uh = acc(Rt):2x1|2x2`, `mxmem(Rd, Rs) = cvt`.
  - `:2x1`은 2 KiB, 즉 u16 1024개를 쓴다. 4행 묶음마다 두 행 분량을 열 순서로 번갈아 쓰는 배치다.
    값은 대략 `acc · s · 257/512`이지만, scale 4에서는 "행 하나를 그대로 쓴다"는 해석과 "인접한
    두 행을 평균한다"는 해석에 원소가 반씩 맞았다. 행 두 개를 하나로 줄이는(2:1) 모드로 보인다.
  - `Rt`는 bit 12, 13, 16, 17, 18이 결과를 바꾼다. 의미는 미해독이다.
  - convert를 두 번 한 뒤 한 번 저장하면 0이 나온다.
  - 따라서 FC의 64×32 출력 tile을 u16으로 바로 꺼내는 방법은 아직 없다.

### Integer FC extension (2026-11-02)

`hexkl_mm_u16i4_layer_run_i32` adds a separate integer-only entry alongside
`hexkl_mm_u8i4_layer_run`. It reuses the two HMX sweeps, weight DMA and double
buffering, but bypasses dynamic quantization and the floating-point epilogue.
The scope is the FC kernel and its requantization helper; FastRPC/model wiring,
normalization, nonlinear activations and attention are not converted here.

- Input: row-major UINT16 codes with one fixed asymmetric encoding for the
  tensor, supplied as integer `zero_point`; K must be a multiple of 64.
  `hvx_pack_u16_ah_group` splits bytes with HVX without consulting float
  encodings; ready row bands can be consumed while later bands are packed.
- Weight: offline WH-tiled signed INT4 with weight zero-point zero and
  precomputed INT32 column sums. No weight packing occurs in the timed FC;
  DDR-to-VTCM transfers are included. The registry's float scales/bias are ignored.
- Output: concatenated M × N_i INT32 matrices, exactly
  `sum_k((a_q - z_a) * w_q) + bias_i32`. Bias is prepared offline in units
  `s_a * s_w[c]`; NULL means zero. Each sweep must fit INT32. The epilogue
  uses exact widened intermediates where needed; proven bounded cases use
  INT32 with explicit overflow checks. On
  overflow the call returns `AEE_EFAILED`; discard the entire output.
  It never wraps or silently saturates accumulators.
- Readout uses three byte planes for K <= 4111 (4096 with the K alignment)
  and four byte planes for larger K. Both are assembled in the HVX worker
  while the caller issues subsequent HMX operations. The constants are supplied
  as binary16 bit patterns, not computed from FP32. On Hexagon the new
  epilogue unpacks four rows together and uses HVX low/high word pairs for
  exact widened integer addition, zero-point correction and overflow checks.
  Column-sum/bias correction is shared across a worker's rows. Host builds,
  or `-DHEXKL_INTEGER_SCALAR`, retain the scalar implementation.

`hexkl_requantize_i32_u16` in `hmx/hexkl_integer.h` maps those accumulators to
another asymmetric UINT16 tensor using only integer arithmetic:

```
a_next = clamp(round_ties_away(acc * multiplier / 2^shift) + z_next, 0, 65535)
multiplier / 2^shift approximates s_a * s_w[c] / s_next
```

The caller supplies precomputed parameters, one set for the tensor or one per
output channel. Multiplier is 0..INT32_MAX; shift is 0..62. For chaining into
the FC entry, every channel must represent the same `s_next` and `z_next`;
per-channel multipliers account for the previous FC's weight scales, not
independent next-layer encodings. Source and destination buffers must not
overlap. Rounding and saturation are explicit and are not a claim of
bit-exact equivalence to an undocumented QNN requantization implementation.

Example (scale ratio 3/256 and next zero-point 30000, already calibrated):

```c
const hexkl_requant_u16 next = {3, 8, 30000};
const hexkl_requant_u16 *params[] = {&next};
const uint32_t counts[] = {1};
int rc = hexkl_mm_u16i4_layer_run_i32_requant(
  table, vtcm, vtcm_size, config_off, M, K, &handle, 1,
  input_u16, input_zero_point, NULL, accumulator_i32, pool,
  params, counts, next_input_u16);
if (rc != AEE_SUCCESS)
  return rc; // discard both outputs on accumulator overflow
```

This fused entry preserves the requested INT32 output and also emits UINT16
from the same registers, avoiding a second read of the INT32 matrix. Each
handle supplies either one requantization parameter set or N_i sets. The
separate `hexkl_mm_u16i4_layer_run_i32` and
`hexkl_requantize_i32_u16_pooled` entries remain available.

`fc_integer_host_check.c` checks direct INT64 dots (not a float reference),
nonzero column sums, both readouts, multiple row/tile/handle batches,
zero-points 0/32768/65535, bias, INT32 endpoints, overflow/cancellation, and
an FC → requantization → FC chain. `quant_u16_host_check.c` also executes the
real byte-only HVX pack on emulation, including padding/tails and pool lanes.
The HVX requantization body processes 64 outputs per iteration, using exact
64-bit products represented by two word vectors. It keeps rounding and
saturation identical to the scalar specification, including INT32_MIN and
all shifts 0..62. Bounded L2 prefetch avoids demand-load stalls. The pooled
entry divides independent rows over the existing worker pool. Small tails
and per-channel parameter gathers remain scalar; per-tensor parameters are
broadcast once per row.

`integer_fc_device_bench.c` runs actual DSP arithmetic checks: 49,664
requantized values against an independent quotient/remainder specification,
both accumulator layouts, widened correction/overflow, partial row groups,
and pooled per-channel requantization with uneven rows and invalid inputs.
The FC tests cover positive and negative four-plane readout beyond int24,
K=4096/4160/8192/32768, cancellation and true INT32 overflow. Guarded outputs
exercise paired stores, odd tile offsets, unaligned pointers and tails. The benchmark
compares all output hashes between scalar and HVX binaries, checks direct
INT64 dot-product samples, and checks every requantized output. These checks
establish kernel behavior, not whole-model accuracy.

Reproduce with the SDK/HexKL environment used by `test/htp/build.sh`:

```sh
HEX_ARCH=v81 bash test/htp/run_integer_fc_device_bench.sh DEVICE_SERIAL /tmp/integer-fc-bench
```

The script builds both variants, alternates their order over three processes per shape
and variant, and saves raw logs, binary hashes, individual samples and medians.
Each process records six samples per shape after two warmup rounds; each
sample averages three calls after one additional warmup of that path.
FC timings include UINT16 packing, weight DMA, two HMX sweeps and output
processing; registration and host/FastRPC transfer are excluded. The current
chain metric calls the fused FC + requantization entry. The initial results
below predate fusion and measured FC followed by pooled requantization.
Requantization timings use one per-tensor parameter set `{3, 8, 30000}`.
Both builds use the same worker pool, input codes, packed weights and zero
point. The FP32 baseline is the retained original entry in the same binary,
not a separately built, unmodified PR binary. It ends at FP32 output and
therefore does less work than the FC + UINT16 requantization chain.

**Initial scalar-to-HVX measurements, before the overlap changes below.**

Measured on 2026-11-02: S26 Ultra SM-S948N / SM8850, v81, 8 MiB VTCM,
7 pool workers (the pooled requantization caller is an eighth lane), SDK
6.4.0.1, Hexagon Clang 19.0.04, HexKL 6.4.0.2, `-O3`. The session uses its
existing DSP/HMX clock votes; actual sustained clocks were not sampled.
Battery was 100%, USB powered, 24.9 °C at the start. All times below are
microseconds, medians of 18 samples (three calls per sample). Each shape
runs in its own process to stay within the SDK runner’s 10-second timeout.

| M × K × N | Existing FP32 FC | Scalar INT32 FC | HVX INT32 FC | Scalar/HVX speedup | HVX FC + requant |
|---|---:|---:|---:|---:|---:|
| 128 × 2816 × 4096 | 232.0 | 18450.6 | 194.2 | 95.0× | 295.7 |
| 512 × 2816 × 4096 | 868.4 | 73809.3 | 818.5 | 90.2× | 1155.4 |
| 768 × 2816 × 4096 | 1227.5 | 110665.4 | 1174.9 | 94.2× | 1683.6 |
| 128 × 4096 × 4096 | 285.7 | 18464.8 | 269.5 | 68.5× | 366.5 |
| 512 × 4096 × 4096 | 1078.5 | 73723.7 | 1047.5 | 70.4× | 1378.4 |
| 67 × 4160 × 256 | 342.8 | 966.9 | 345.8 | 2.8× | 348.1 |

For the five N=4096 prefill shapes, the optimized INT32 FC takes 2.9–16.3%
less time than the retained FP32 output path. Including the separate
pooled requantization takes 27.4–37.2% more time than stopping at FP32.
The K=4160 INT32-readout case is about 0.9% slower for FC alone. This test
does not separately time the HMX readout. No whole-model speedup is claimed.

| Output shape (K=2816) | Scalar requant, 1 lane | HVX requant, 1 lane | Scalar requant, pool | HVX requant, pool |
|---|---:|---:|---:|---:|
| 128 × 4096 | 2906.9 | 324.9 | 486.5 | 62.2 |
| 512 × 4096 | 12970.4 | 1348.7 | 2051.4 | 353.9 |
| 768 × 4096 | 19158.8 | 2024.1 | 3091.2 | 524.5 |

The standalone requantization and chain numbers have different cache states;
do not substitute their sum for the measured chain time. Per-channel
requantization is correctness-tested here; the timing tables use per-tensor
parameters. The kernel has no scalar work proportional to every FC output
on the Hexagon path, but parameter gathers and short requantization tails
still use scalar code.

All 36 completed device runs passed, with identical INT32 hashes between
scalar and HVX variants for every shape; every requantized value matched the
independent integer specification. The maximum difference from the legacy
FP32 result after scaling was 3.052e-5. Host checks, including the old FP32 FC
path and byte-plane mutation detection, and ASan/UBSan also passed. Both
v79 and v81 DSP builds passed; device timings here are exclusively v81.

Artifacts: [individual samples](benchmarks/integer_fc_20261102/samples.csv),
[medians](benchmarks/integer_fc_20261102/medians.csv),
[raw device logs](benchmarks/integer_fc_20261102/runs.txt),
[environment](benchmarks/integer_fc_20261102/environment.txt), and
[binary hashes](benchmarks/integer_fc_20261102/binaries.sha256).

### Overlap and fused integer requantization (2026-11-02)

The runtime assumes WH-tiled weights and column sums are prepared offline.
The test fixture creates that image before timing; the FC never packs weights.
It still includes each handle's DDR-to-VTCM transfer, once per call.

- Activation packing runs as four-row background jobs. HMX waits only for
  its next 64-row band. Available VTCM selects up to eight handles per weight
  panel, reusing each ready activation band before advancing. Calls with
  at most two row bands retain one-handle panels to reduce initial DMA latency.
- Two accumulator banks collect up to 32 tiles across row/handle boundaries.
  HMX fills one while foreground HVX workers assemble/correct the other.
  Four-row groups across the entire bank are balanced over the workers;
  foreground output work takes priority over background activation packing.
- The fused epilogue writes both INT32 and requantized UINT16. It shares
  the corrected accumulator in registers, uses paired 128-byte UINT16 stores
  and avoids rereading INT32 from DDR. Larger-K readout also emits byte
  planes directly, moving its previously synchronous zip to those workers.
- Requantization remains exact. Uniform shifts <=16 have an integer
  saturation shortcut; other shifts and per-channel parameters use widened
  products. The timing tables use uniform `{3, 8, 30000}` parameters, so
  they do not establish performance for arbitrary per-channel parameters.

Comparison against the saved **previous HVX integer implementation**,
using the same device/configuration as above (microseconds, 18 samples):

| M × K × N | Previous INT32 | Current INT32 | Previous FC + requant | Fused FC + requant | Chain latency reduction |
|---|---:|---:|---:|---:|---:|
| 128 × 2816 × 4096 | 193.8 | 190.6 | 295.3 | 225.7 | 23.6% |
| 512 × 2816 × 4096 | 814.6 | 760.7 | 1152.7 | 831.0 | 27.9% |
| 768 × 2816 × 4096 | 1172.2 | 1061.1 | 1683.9 | 1159.1 | 31.2% |
| 128 × 4096 × 4096 | 268.5 | 280.4 | 368.6 | 300.0 | 18.6% |
| 512 × 4096 × 4096 | 1050.0 | 996.5 | 1382.7 | 1047.4 | 24.3% |
| 67 × 4160 × 256 | 345.9 | 132.0 | 348.2 | 133.7 | 61.6% |

The short M=128, K=4096 INT32-only case regresses 4.4% against the preceding
HVX implementation; its fused chain improves 18.6%. Larger prefill chains
improve 24.3–31.2%. These are improvements over an already vectorized integer
kernel, not the scalar implementation and not the original PR.

Full hiding is not achieved. The first input/weight preparation, final
output drain, and some intermediate waits remain exposed. The separate
instrumented profiles record nonzero packing, epilogue and DMA waits.
`hmx_readout_us` includes both sweeps and accumulator readout; it is not
pure matrix instruction time. Instrumentation and its different cache state
change total latency, so those profiles are diagnostics, not additive parts
of the uninstrumented medians or proof that the HMX lower bound was reached.

Artifacts: [samples](benchmarks/integer_fc_overlap_20261102/samples.csv),
[medians](benchmarks/integer_fc_overlap_20261102/medians.csv),
[profiles](benchmarks/integer_fc_overlap_20261102/profiles.csv),
[logs](benchmarks/integer_fc_overlap_20261102/runs.txt), and
[validation](benchmarks/integer_fc_overlap_20261102/validation.txt).

### Direct comparison with the unchanged #4471 PR

The baseline below is a separate, clean build of PR commit
`3881042471377ec3c083afd06e55a04ab6934cbe`. It is not the retained FP32 entry
in the modified binary. `pr4471_fc_device_bench.c` links only the original
FP32 API, using the same input generator, fixed UINT16 encoding, WH weight
setup, clock votes, warmup and three-call samples as the integer harness.
Baseline and candidate processes alternate for each shape, reversing order
in round two. All 36 processes passed. Each shape has 18 samples per variant.

The original PR outputs FP32. The current FC outputs exact INT32; the fused
entry additionally outputs UINT16. These are kernel latency comparisons
with those explicitly different output contracts, not whole-model timings.
All times are microseconds; negative reductions mean a regression.

| M × K × N | Original PR FP32 | Current INT32 | INT32 latency reduction | Current INT32 + requant | Chain latency reduction vs PR |
|---|---:|---:|---:|---:|---:|
| 128 × 2816 × 4096 | 227.3 | 189.6 | 16.6% | 226.4 | 0.4% |
| 512 × 2816 × 4096 | 872.6 | 761.1 | 12.8% | 832.3 | 4.6% |
| 768 × 2816 × 4096 | 1227.2 | 1064.5 | 13.3% | 1158.9 | 5.6% |
| 128 × 4096 × 4096 | 277.3 | 280.6 | -1.2% | 302.8 | -9.2% |
| 512 × 4096 × 4096 | 1072.8 | 996.0 | 7.2% | 1044.5 | 2.6% |
| 67 × 4160 × 256 | 341.7 | 132.2 | 61.3% | 134.0 | 60.8% |

The representative 512 × 2816 × 4096 FC is 1.15× as fast (12.8% less time)
than the original PR; including requantization saves 4.6%. For M=128,
K=4096, INT32 alone takes 1.2% more time and the fused chain takes 9.2% more.
The K=4160 case benefits especially from moving the four-byte readout zip
off the HMX issuing thread. No blanket speedup across all shapes is claimed.

Reproduce the unchanged baseline with an isolated checkout and the SDK
variables used by `build.sh`:

```sh
git worktree add --detach /tmp/pr4471-original 3881042471377ec3c083afd06e55a04ab6934cbe
bash test/htp/build_pr4471_fc_device_bench.sh /tmp/pr4471-original /tmp/pr4471-bin
# Prepare /tmp/current-bench/hvx with run_integer_fc_device_bench.sh above.
NNTR_BENCH_BASELINE_KIND=fp32 bash test/htp/compare_integer_fc_device_bench.sh \
  DEVICE_SERIAL /tmp/pr4471-bin /tmp/current-bench/hvx /tmp/pr4471-comparison
```

The comparison script also accepts two saved integer benchmark directories
without `NNTR_BENCH_BASELINE_KIND`; that mode checks identical INT32 hashes
across variants. All comparison scripts preserve raw logs and binary hashes.

Artifacts: [samples](benchmarks/integer_fc_vs_pr4471_20261102/samples.csv),
[medians](benchmarks/integer_fc_vs_pr4471_20261102/medians.csv),
[raw logs](benchmarks/integer_fc_vs_pr4471_20261102/runs.txt),
[environment](benchmarks/integer_fc_vs_pr4471_20261102/environment.txt),
[binary hashes](benchmarks/integer_fc_vs_pr4471_20261102/binaries.sha256), and
[source hashes](benchmarks/integer_fc_vs_pr4471_20261102/source.sha256).

### Final W4A16 optimization pass (2026-11-02)

This pass keeps offline WH weights, two u8×i4 HMX sweeps, asymmetric UINT16
input, exact INT32 output, and optional fused integer UINT16 requantization.
It changes four parts of the existing integer path:

1. Activation groups read 64 codes from each row once and shuffle them into
   two K tiles. Full aligned vector stores also cover partial/padded rows.
   Disassembly showed the preceding tail's 32-byte `memcpy` becoming many
   scalar byte stores; the new implementation has no such tail copies.
2. A small HMX sweep function keeps the K-loop stride and descriptors in
   registers, eliminating spills from the large shared layer dispatcher.
3. For K <=4096, `65535 * 8 * K <= INT32_MAX` proves the uncorrected UINT16
   dot fits INT32. A correction fitting INT32 uses ordinary/saturating adds
   to detect overflow; other corrections and larger K keep widened exact
   arithmetic. Bias/zero-point overflow is still reported, never accepted
   as a saturated FC output. No assumption about small activation values
   or correct cancellation is used to skip overflow checks.
4. For that bounded dot and uniform requant shift <=16, adjacent output
   tiles are processed together. Their UINT16 vectors are combined before
   moving to the next row group, avoiding the full-band temporary. Odd
   tiles, other shifts, per-channel parameters and larger K retain their
   exact fallback. Both INT32 and UINT16 outputs are still written.

A separate experiment packed address updates into the HMX issue packet;
measured gains were negligible, so it was discarded. This final pass ends
with the paired output processing; no model wiring or additional parameter-
specific optimization is included.

**Unchanged #4471 PR versus final kernel.** Same clean baseline commit
`3881042471377ec3c083afd06e55a04ab6934cbe`, device and method described above:
S26 Ultra v81, 18 samples per shape, alternating processes, offline weight
packing excluded and weight DMA included. Times are microseconds. Baseline
outputs FP32; final columns output INT32 and INT32+UINT16 respectively.
The chain uses the uniform `{3, 8, 30000}` requantization parameters.

| M × K × N | Original PR FP32 | Final INT32 | INT32 latency reduction | Final FC + requant | Chain latency reduction |
|---|---:|---:|---:|---:|---:|
| 128 × 2816 × 4096 | 225.5 | 185.0 | 18.0% | 188.1 | 16.6% |
| 512 × 2816 × 4096 | 871.8 | 710.0 | 18.6% | 704.9 | 19.1% |
| 768 × 2816 × 4096 | 1227.4 | 989.5 | 19.4% | 994.5 | 19.0% |
| 128 × 4096 × 4096 | 279.9 | 267.4 | 4.5% | 274.5 | 1.9% |
| 512 × 4096 × 4096 | 1068.9 | 930.8 | 12.9% | 939.2 | 12.1% |
| 67 × 4160 × 256 | 341.6 | 28.9 | 91.5% | 30.4 | 91.1% |

**Incremental gain over the preceding integer kernel**, measured separately
with alternating before/after processes. The unchanged INT32 hashes match
for every shape. Times are microseconds.

| M × K × N | Previous INT32 | Final INT32 | Previous FC + requant | Final FC + requant | Chain latency reduction |
|---|---:|---:|---:|---:|---:|
| 128 × 2816 × 4096 | 190.3 | 185.1 | 222.4 | 186.8 | 16.0% |
| 512 × 2816 × 4096 | 762.5 | 710.1 | 833.4 | 704.3 | 15.5% |
| 768 × 2816 × 4096 | 1065.6 | 990.1 | 1157.6 | 993.7 | 14.2% |
| 128 × 4096 × 4096 | 282.3 | 270.4 | 303.6 | 275.9 | 9.1% |
| 512 × 4096 × 4096 | 994.1 | 928.4 | 1043.4 | 938.8 | 10.0% |
| 67 × 4160 × 256 | 132.0 | 29.0 | 133.9 | 30.5 | 77.2% |

The 512 × 2816 × 4096 fused chain falls from 833.4 to 704.3 us (15.5%).
INT32-only and fused calls use different epilogue schedules; their measured
difference is not a standalone requantization cost. The slight fused win
at that shape also includes normal measurement variation. Per-channel and
large-shift parameters are correctness-tested but are not timed by these
tables; the specialized paired path does not apply to them.

Hiding remains incomplete: first weight/input preparation, final output
completion and some intermediate waits are still exposed. Profiles are
separate instrumented calls, and HMX time includes readout. For example,
the representative large shape still has nonzero epilogue/DMA waits. The
large K=4160 improvement is principally removal of the activation tail's
scalar stores, not elimination of either HMX sweep.

All 72 final device runs, host checks, byte-plane mutation detection,
ASan/UBSan, and v79/v81 DSP builds passed. The DSP build artifact is restored
to the final v81 binary after the v79 compile check. No whole-model accuracy
or end-to-end inference performance is inferred from these FC measurements.

Artifacts: [PR comparison medians](benchmarks/integer_fc_max_20261102/vs_pr4471/medians.csv),
[PR raw logs](benchmarks/integer_fc_max_20261102/vs_pr4471/runs.txt),
[previous-version medians](benchmarks/integer_fc_max_20261102/vs_previous/medians.csv),
[previous-version raw logs](benchmarks/integer_fc_max_20261102/vs_previous/runs.txt),
[profiles](benchmarks/integer_fc_max_20261102/vs_pr4471/profiles.csv),
[validation](benchmarks/integer_fc_max_20261102/validation.txt), and
[source hashes](benchmarks/integer_fc_max_20261102/source.sha256).

### Integer FC follow-up: waits and direct AH output (2026-11-03)

Implemented in the same worktree, relative to the final integer kernel from
2026-11-02, not the original PR's FP32-output API:

- Each worker publishes one overflow summary; the caller retires each completed
  batch once. Removed whole-job-array initialization and repeated scans of both
  staging banks. Overflow remains an error, including the last partial batch.
- Weight handles have descriptor-level readiness. HMX can start on the first
  ready handle without waiting for the whole panel; later panels stay queued.
  A partial-batch readout fence protects weight-buffer reuse.
- `hexkl_mm_u16i4_layer_run_i32_ah` accepts packed input and/or emits directly
  packed requantized output. INT32 output is retained. AH output concatenates
  handle channels into a single logical tensor, with padded rows initialized.
  The consumer DMA-loads the two planes by row band and does no HVX packing.
- Packed epilogues keep four rows' intermediate vectors in registers. Explicit
  loop unrolling removed the array spills found in the first device build.

The packed buffer is external DDR, 128-byte aligned, with checked capacity and
non-overlap requirements. The image is low plane followed by high plane, each
`ceil(M/64)*64*width` bytes. Its per-plane offset for logical `(r,c)` is
`(r/64)*64*width + (c/32)*2048 + (r%64)*32 + c%32`. Input width is K; output width
is the sum of handle widths. The next FC needs one common output encoding and
its matching zero-point. Per-channel requant multipliers remain supported.
This is an explicit opt-in API, not VTCM-resident handoff or model/FastRPC wiring.

**Matched single-FC comparison.** S26 Ultra v81, three alternating processes
per build and shape, six samples per process, three timed calls per sample.
Two warmup rounds; same offline WH weights and input codes; both INT32 and
UINT16 outputs retained. Offline weight packing/registration and FastRPC are
excluded; DMA, activation preparation and final completion are included.
These are medians with profiling disabled; the baseline binary is the saved
2026-11-02 final integer kernel.

| M x K x N | Before FC+requant (us) | After FC+requant (us) | Reduction |
|---|---:|---:|---:|
| 128 x 2816 x 4096 | 181.293 | 172.231 | 5.0% |
| 512 x 2816 x 4096 | 709.540 | 677.239 | 4.6% |
| 768 x 2816 x 4096 | 997.092 | 955.096 | 4.2% |
| 128 x 4096 x 4096 | 269.453 | 263.038 | 2.4% |
| 512 x 4096 x 4096 | 942.265 | 903.507 | 4.1% |
| 67 x 4160 x 256 | 30.425 | 29.835 | 1.9% |

**Two-FC chain.** The producer has the shape shown below; the consumer is
`M x N x 256`. The row baseline uses the original concatenated handle outputs,
a parallel HVX join into an M x N tensor, and the consumer's normal packing.
The packed path writes global-channel AH tiles directly and DMA-loads them in
the consumer. Timings include both FCs and intermediate layout work. Therefore
the gain includes eliminating the handle join as well as the consumer packing;
it is not an isolated packing speedup. The after-row column separates the
single-FC improvements from the layout change. Common requant `{3,8,30000}`
is timed; per-channel parameters and shifts 0..62 are correctness-tested.

| Producer M x K x N | Before row chain (us) | After row chain (us) | After packed chain (us) | Reduction vs before |
|---|---:|---:|---:|---:|
| 128 x 2816 x 4096 | 319.800 | 318.741 | 221.423 | 30.8% |
| 512 x 2816 x 4096 | 1166.276 | 1127.657 | 833.064 | 28.6% |
| 768 x 2816 x 4096 | 1668.559 | 1615.070 | 1168.976 | 29.9% |
| 128 x 4096 x 4096 | 409.809 | 407.083 | 320.486 | 21.8% |
| 512 x 4096 x 4096 | 1395.860 | 1362.370 | 1049.939 | 24.8% |
| 67 x 4160 x 256 | 43.212 | 42.873 | 42.570 | 1.5% |

At 512 x 2816 x 4096, the new row chain is 1127.657 us and the packed chain
833.064 us: the layout change alone cuts 26.1%, while the full change versus
yesterday's 1166.276 us row chain cuts 28.6%. Small single-handle shapes have
much less to save. Direct AH output itself is not free; compare the whole chain.

All 36 single-FC and 36 chain comparison processes passed. Complete consumer
INT32 hashes match before/after in all six shapes. Candidate runs additionally
compare every producer and consumer INT32 result, all requantized UINT16 codes,
padding and common/per-channel cases. Guarded single-tile output, M tails,
K=4096/4160/8192/32768, cancellation, genuine overflow and last-row/last-column
failure in a final partial batch passed on device. Integer host checks including
multi-handle packed input/output and rejection-without-write passed under
ASan/UBSan; legacy FC host checks passed. DSP v79/v81 builds and clang-format 14
passed; the build output was restored to the measured v81 skel after v79 validation.

Remaining costs include HMX readout, first input/weight transfers, worker work
and final completion. DMA wait reduction can expose packing waits elsewhere:
the profile counters must not be interpreted as independent savings. No claim
of fully hidden non-HMX work or whole-model end-to-end speedup is made.

Artifacts: [single-FC medians](benchmarks/integer_fc_followup_20261103/fc-comparison/medians.csv),
[two-FC medians](benchmarks/integer_fc_followup_20261103/chain-comparison/medians.csv),
[chain components](benchmarks/integer_fc_followup_20261103/chain-comparison/components.csv),
[source hashes](benchmarks/integer_fc_followup_20261103/source.sha256),
and the raw comparison/build/validation logs alongside them.

**Direct comparison with original PR #4471.** Re-ran the saved, unmodified
`3881042471377ec3c083afd06e55a04ab6934cbe` binary against the final candidate on
the same S26 Ultra, with the same alternating-process protocol (18 samples per
build and shape). The original returns scaled FP32; the current API returns
corrected INT32 and optionally requantized UINT16. These output contracts differ.
The current columns below use row-major output; the direct AH two-FC chain is
measured separately above. All times are microseconds; reduction is relative to
the original FP32 time and includes current UINT16 requantization.

| M x K x N | PR #4471 FP32 | Current INT32 | Current INT32 + requant | Reduction |
|---|---:|---:|---:|---:|
| 128 x 2816 x 4096 | 224.262 | 172.622 | 169.505 | 24.4% |
| 512 x 2816 x 4096 | 872.023 | 677.673 | 676.562 | 22.4% |
| 768 x 2816 x 4096 | 1235.668 | 940.938 | 953.368 | 22.8% |
| 128 x 4096 x 4096 | 274.974 | 254.288 | 255.520 | 7.1% |
| 512 x 4096 x 4096 | 1071.745 | 896.892 | 904.263 | 15.6% |
| 67 x 4160 x 256 | 341.649 | 28.637 | 30.148 | 91.2% |

At 512 x 2816 x 4096, the reduction is 22.4% (1.289x speedup). INT32-only is
22.3% lower than the original. The K=4160 tail case principally benefits from
removing scalar activation-tail stores; its gain is not representative of the
larger shapes. INT32-only and fused requant timings have different schedules
and normal measurement variation; their difference is not isolated requant cost.

All 36 comparison processes passed. Output hashes are stable within each API;
the FP32 and integer hashes are not compared to each other. The earlier 28.6%
two-FC gain uses yesterday's integer kernel as its baseline. A two-FC chain
comparison against the original FP32 PR has not been measured.

Artifacts: [PR comparison medians](benchmarks/integer_fc_followup_20261103/vs-pr4471/medians.csv),
[derived reductions](benchmarks/integer_fc_followup_20261103/vs-pr4471/comparison.csv),
[raw runs](benchmarks/integer_fc_followup_20261103/vs-pr4471/runs.txt), and
[protocol and provenance](benchmarks/integer_fc_followup_20261103/vs-pr4471/report.txt).

규칙은 `01_working_style.md`, 검증 순서는 `hexagon-gates` skill.

이 문서 시점 브랜치 `htp/gemma-e2b`, HEAD `f696c9347`. 줄 번호는 그 시점 기준이다.

참고 커밋 (plan 24 문서 `24_attention_a16_plan.md`는 이 트리에 없다):
- `48f04ddb4` a16 / kv8 attention 커널 — u8×i8 array 위의 16-bit Q·P, 이 계획의 원본
- `7d9490881` FC 호출이 u16 activation을 quant pass 안에서 dequant — §2에서 고칠 자리
- `f5a70c4d5` attention의 u16 context를 o-proj로 넘기는 배선
- `0547df38f` readout 합치기를 convert 한 tile 뒤로 미룬 최적화 — §3-2 파이프라인에 참고

---

## 0. 왜, 그리고 정말 필요한가

HTP context의 FC는 a8w8(`hexkl_mm_u8i8_dma.c`)과 a8w4(`hexkl_mm_u8i4_dma.c`)뿐이다. 예외가 attention core의
a16 / kv8 커널(`hexkl_attn_q2.c`, plan 24)이다. 이 문서는 그 a16 방식을 u8×i4 FC에 옮기는 계획이다.

지금 o-proj는 attention의 u16 context를 받아, DSP의 quant pass 안에서 f32로 dequant한 뒤 **u8로 다시
양자화**한다(`htp_compute_ops.cpp:1797-1824`의 `u16_in` 분기 → `nntr_hvx_mm_u8i4_layer_res_add_u16`,
`hexkl_mm_opts.h:70-79`의 `act_u16`). a16w4가 얻는 것은 이 u8 재양자화 손실을 없애는 것뿐이다.
속도는 얻지 못하고 잃는다(§5).

**Gate 0 (host, DSP 코드 작성 전):** 레이어별로 a8w4 / a16w4 / f32 출력 오차를 CPU에서 비교하고,
Gemma-4 E2B PPL 차이를 본다. 의미 있는 이득이 없으면 여기서 멈추고 이 문서에 결과만 적는다.

---

## 1. 수학 — attention의 a16 트릭 그대로

`hexkl_attn_q2.c:43-48`:

```
u   = 256·u_hi + u_lo                       (u16 activation을 두 u8 plane으로)
A·W = 256·(A_hi·W) + (A_lo·W)               (u8×i4 sweep 두 번, 각자 누산)
y   = s_a·s_w·(256·acc_hi + acc_lo − zp·colsum_w) + bias
```

attention과 다른 점:

- **readout이 더 단순하다.** attention은 convert unit plane(QK int16, PV int24 + 2^s shift)으로 읽는다.
  FC 경로는 이미 `hexkl_micro_hmx_acc_read_int32`로 int32를 직접 읽으므로(`hexkl_mm_u8i4_dma.c:783`)
  shift 계획이 필요 없다.
- **합치기는 f32에서.** sweep 하나의 |acc| ≤ 255·8·K는 K=8192에서도 int32 안이다. 하지만 256배하면
  K≳1024부터 int32를 넘는다. 기존 dequant가 colsum을 이미 f32로 다루므로(`hvx_dequant_i32.c:118`)
  같은 자리에서 `256·f(hi) + f(lo)`로 합친다. f32 반올림(상대 2^-24)은 16-bit 양자화 step보다 훨씬 작다.
- **zero point는 쪼개지 않는다.** zp·colsum 항은 dequant에서 한 번에 뺀다. 16-bit zp·colsum도 f32라
  넘치지 않는다.

---

## 2. 함정 — o-proj 입력은 head마다 encoding이 다르다

attention 출력은 head마다 (scale, zp)를 갖는다(`act16_enc`, `act16_hd`). K 방향으로 scale이 바뀌면
하나의 누산에서 scale을 밖으로 뺄 수 없다. head마다 따로 누산·readout하는 것은 head 수만큼 acc read가
늘어 비싸다.

**해결:** 행마다 u16 encoding 하나로 재양자화한다. quant pass가 이미 u16 → f32를 `pre_scratch`에 만들고
있으므로, 출력만 u8 대신 **per-row u16**으로 바꾸면 된다. f32 입력(qkv, FFN)도 같은 경로를 탄다.
16-bit라 head를 합쳐도 여유가 충분하다. 같은 재양자화를 attention decode의 GQA fold가 이미 한다
(`hexkl_attn_q2.c:263-290`, `q_dma_push`).

---

## 3. 구현 단계 (가장 짧은 diff)

1. **HVX quant pack** — `hvx_quant_u8.c`
   - `hvx_quant_params_pack_u8_ah`(`hvx_quant_u8.h:83`)에 16-bit 모드: per-row scale/zp를 [0, 65535]로,
     AH tile을 lo plane / hi plane 두 장으로 쓴다.
   - 바이트 분리는 `stage_qprep`의 `vdeal`(`hexkl_attn_q2.c:325-337`)을 재사용한다.
2. **커널** — `hexkl_mm_u8i4_dma.c` layer 경로
   - `hexkl_mm_opts`에 `act16` 플래그.
   - 출력 tile 루프(`:770-789`)를 hi sweep → acc read → clear → lo sweep → acc read로. `mm_u8i4_tile`은 그대로.
   - VTCM: activation 영역 ×2(`act_bytes`), staging ×2 → `acc_tiles` 절반(`:582`).
3. **Dequant** — `hvx_dequant_i32.c`
   - `DQ_TILE_ROW`가 lo tile도 읽어 `256·f(hi) + f(lo)`. `hvx_dq_tiles_job`에 lo tile offset 추가.
   - `!acc_layout->usable` fallback(`acc_scratch`)은 a16에서 `AEE_EUNSUPPORTED`로 거절한다(`out_off`와
     같은 방식). `ponytail:` 주석으로 천장과 업그레이드 경로를 남긴다.
4. **Skel / host 배선**
   - 새 entry 대신 `nntr_hvx_mm_u8i4_layer_res_add_u16`에 a16 모드 인자 하나(`test/htp/nntr_hvx.idl`).
   - host는 `gemm_qs4cx_res_add_fp32`의 `u16_in` 분기에서 켠다. 1차 대상은 o-proj만.
5. **검증**
   - `test/htp/host/fc_layer_host_check.c`에 a16w4 케이스: hi/lo 두 sweep 합이 u16×i4 int64 직접 dot와
     정수 단계에서 bit-exact, 최종 f32가 기준과 일치.
   - 그 뒤 `hexagon-gates` 순서. 기기에서 o-proj 시간과 PPL. 측정 instrumentation은 커밋하지 않는다.

---

## 4. 범위 밖

- **Decode (M=1):** HVX GEMV 경로라 이 HMX 커널을 타지 않는다. 그 경로가 이미 f32 activation을 받는지는
  확인하지 않았다 — 손대기 전에 확인할 것.
- **MoE 커널** `hexkl_mm_u8i4_moe.c`: Gate 0 결과를 보고 결정.
- **a16w8 FC** `hexkl_mm_u8i8_dma.c`: 같은 패턴이라 이 작업 뒤에 거의 그대로 옮길 수 있다.

---

## 5. 예상 비용과 미확인 사항

- HMX packet과 acc read가 2배, activation DMA와 VTCM도 2배. prefill o-proj의 HMX 구간은 약 2배로
  예상한다 — **가정이다.** 착수 시 a8w4 o-proj의 단계별 시간(`HEXKL_PROBE_*`)을 먼저 재고 판단한다.
- **v81 HMX에 네이티브 16-bit 정수 activation은 없다 (2026-10-18 확인).** 그래서 §3의 두 번 sweep이
  유일한 경로다.
  - SDK 6.4.0.2 어셈블러(`-mv81 -mhmx`)가 받는 activation 타입은 `.ub`와 `.hf`뿐이다. `.uh`/`.h`/`.b`는
    "invalid operand"로 거절된다.
  - QNN 2.47 `libQnnHtpV81Skel.so`의 "16-bit activation" 커널(`hmx_v73_convhnh*`, `hbh*`)도 SDK objdump로
    보면 `activation.ub` × `weight.n`/`.b`다. 16-bit는 출력 readout(`cvt.uh = acc:2x2`)에만 있다. skel
    전체의 패킷 조합은 `ub×b`, `ub×n`, `ub×c`, `hf×hf` 네 가지뿐이다. `native_hmx_a16w4`는 런타임 옵션
    문자열이고 커널이 아니다.
  - 어셈블러가 받는 `activation.hf × weight.n`을 S26 Ultra(v81) cDSP에서 실행해 봤다(HexKL micro +
    `run_main_on_hexagon`). int4로 디코딩되지 않는다. WH i4 tile의 바이트를 **fp16 비트 그대로** 읽는다:
    k=0의 nibble q는 q·2^-24(fp16 subnormal), k=1의 nibble은 halfword의 bit 8–11, k=5의 nibble은 bit
    12–15(지수부)에 들어간다. 대조군 hf×hf는 fp16 정밀도로 기준값과 일치했다. 따라서 fp16×int4
    네이티브 모드로는 쓸 수 없다.
  - 같은 실험에서 패킷당 시간은 hf×hf, hf×n, ub×n이 모두 같았다. 같은 tile을 반복한 측정이다.
    hf tile은 32행, ub tile은 64행이므로 행당 처리량은 ub가 2배다. 따라서 u8 두 번 sweep의 HMX 비용은
    hf 한 번 sweep과 같은 수준이다.
  - 실험 소스는 커밋하지 않았다. 다시 하려면 HexKL `examples/hexkl_micro_hmx_mm_f16`을 바탕으로,
    빌드 플래그에 `-mhmx`를 추가하면 된다.
