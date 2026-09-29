# 65 — W4 on HTP: port PR nntrainer/nntrainer#4327's HMX u8×i4 path into the graph executor, Qwen3-0.6B e2e against GENIEX_QAIRT

Issue: dlwlzzero/nntrainer#65 (`prio:p0`, user decision 2026-09-18: the new top goal; the A8W8 HVX-only
line ends with #26 #41 #57 #58 #59 #60). Contract: `docs/plans/0000-agent-system-and-env.md`.
Branch: `hvx/65-w4-htp-port` from `hvx_impl @ 2c880173`; the upstream PR is read as `pr-4327`
(`git fetch https://github.com/nntrainer/nntrainer.git pull/4327/head:pr-4327`, tip `5ae9830e`,
merge-base with `hvx_impl` = `f97c2e26`, 141 files / +25,439). Nothing from it is merged; files are
ported one by one into `nntrainer/tensor/hexagon/htp/hmx/` with their SPDX / copyright headers kept
(Apache-2.0, authors SeungHui Lee and dlwlzzero) and a `@see` line naming the PR. Line numbers below
are `hvx_impl @ 2c880173` (before #58 merges; `htp_graph.c` shifts a few lines after it).

Planner probe (2026-09-18, the one exploratory build): all 14 `nntrainer/tensor/htp_backend/{hmx,hvx}/*.c`
of the PR compile with the container's SDK 6.4.0.2 / hexagon-clang 19.0.04 for v79 against the
**installed HexKL addon** (`~/Qualcomm/hexkl_addon`, mounted at `/opt/qcom/hexkl_addon` by
`tools/docker/run.sh:33,60`; `lib/hexagon_toolv19_v79/libhexkl_micro.a`, 122 KB, **1.0 beta1**:
`hexkl_micro_hw_init(uint8_t **, uint32_t *)` is the 2-arg form, the PR's `test/htp/hvx_add_f32.c`
uses beta2's 3-arg form) and link into a `.so` with only libc / QuRT / `HAP_perf` symbols left
unresolved. The `hmx/` and `hvx/` sources themselves never call `hw_init` / `hmx_lock`, so beta1 is
sufficient for this plan. HexKL's own `examples/hexkl_micro_hmx_mm_u8i4_i32` is documented as a
**Hexagon-simulator** harness (`run_simulator.sh`), so an HMX simulator gate exists.

## 1. Goal and gate

The issue's three points, made measurable. Every device number is a DSP Mcyc or a tok/s on a named
unit from pass 2 of a reversed double pass (HEXAGON.md §7 rule 9); prefill = 512 ÷ Σ`us` of the four
`n=128` calls, decode = median of the 63 `n=1` steps (`hexagon_e2e_test --chunk 128 --steps 64`).

| # | Acceptance (issue) | Pass means |
|---|---|---|
| G1 | the W4 (int4 weight × u8 activation, HMX) path runs Qwen3-0.6B e2e through our executor | `hexagon_e2e_test` on a `w4cx_down8` image (§3.2) with the `MATMUL_W4A8` ops on HMX: `E2E init ok`, 4 + 63 steps, `E2E gen` a coherent continuation; `--eval` on `t512_23.i32` within **±2 %** of the x86 W4 reference PPL and the x86 W4 reference itself ≤ **1.10 ×** the x86 W8 reference on the same prompt (41.5466 / 33.0195 on `t512_23` / the P4 prompt; stop line, §4 S1) |
| G1' | the kernel, not the row: prefill projections leave the HVX `vrmpy` regime | handoff H1 (§4 S2), FARF field `mm4` (new, the W4A8 kind) at m=128: `mm4(B) ≤ 0.25 × mm8(A)` summed over the four prefill calls — the HVX tiled kernel is ≈ 15 % of prefill512 in the P3 simulator split and the whole prefill chunk is 478 ms/128 tokens on device (HEXAGON.md §8.2/§8.3); HMX's measured 3.5 TMAC/s issue rate + one `acc_read` per (64-row, 32-col) tile puts the six projections of 512 tokens at ≈ 0.1–0.16 s DSP (PR doc 34 §1/§3: 619 µs micro-mm + 380 µs acc_read for a 1024×1024×2048 call; 0.019 µs per micro-mm, 0.37 µs per acc_read) |
| G2 | prefill / decode tok/s at 512 / 1024 / 4096 recorded in HEXAGON_BENCHMARK.md | three new rows `nntrainer / w4a8 / NPU, HVX + HMX` filled from H1 (B) and H2 (B), each with PPL / top-1 and the FARF split; read against the **GENIEX_QAIRT** cells 8,207 / 121, 7,503 / 112, 3,546 / 57.6 and against the #60 CPU floor (the multiple is written once #60 is filled). Prefill @512 **≥ 1,000** (contract §1 stage 2 interim) is the milestone-2 cell (after #26), not H1's gate: with the six projections on HMX, prefill is bounded by ATTN (27.5 % of prefill512 pre-P4 and growing with context) and `down` (HVX int16), so H1 is expected at ≈ 350–500 tok/s and is read as `mm4`, not tok/s |
| G3 | decode not regressed by moving the six projections to HMX at m=1 | H1: `Mcyc(B) ≤ Mcyc(A)` at 512 and `mm4(B) − mm4(C) ≤ 2.0 Mcyc` (C = DMA-only floor): HMX at m=1 pads 1 row to 64 (0.019 µs × 344 k micro-mms + 0.37 µs × 10.8 k acc_reads ≈ 10.5 ms for 176 MB of i4 tiles ≈ 17 GB/s-equivalent, against 4.8 ms for the same bytes on the 37 GB/s DMA ring). The gap decides §4 S3's m=1 dispatch (HVX 4-bit stream = #51 folded in) |
| G4 | environment | A (W8 control, `hvx_impl` with `-DHTP_PROF_FARF`) at 512 pass 2 within ±5 % of **54.6 Mcyc** (#25 B on `R3CY10WM83Y`); outside → the session is void |
| G5 | H2 (full 4-bit image, §4 S3) | decode stream phase (`mm4`+`mm16`+`lg`) ≤ 0.55 × A's `mm8`+`mm16`+`lg` at 512 (#51's gate, ≈ 307/596 with margin); decode @512 / @4096 tok/s recorded against 121 / 57.6 and #60; @4096 read inside one order with #58 merged (rule 9(e)) |

Not a gate, recorded: the 8,207 / 121 cells are a w4a16 QNN graph on the same silicon; 121 tok/s = 8.3 ms
per step = ≈ 300 MB at 36 GB/s, i.e. the whole step is the weight stream at the DMA ceiling with
attention ≈ 0. Our reachable decode with the HVX 4-bit stream (#51 folded in) + #58 attention is
≈ 15–16 ms ≈ 60–66 tok/s at 512; with HMX at m=1 ≈ 40–46. Prefill 8,207 needs u8 op boundaries and
HMX/HVX overlap (PR doc 35: 1.59–1.85× on top of the kernel) plus attention on HMX — §4 S4, later issues.

## 2. Where it lives

### 2.1 Inventory of PR #4327 (what exists, what was measured, what is WIP)

| PR component | What it is | Measured (PR docs; S25 Ultra `R3CY10WM83Y`, v79, SDK 6.4.0.2, HexKL beta2) | Disposition here |
|---|---|---|---|
| `hmx/hexkl_mm_u8i4_dma.{c,h}` (367 l) | weight registry (WH-baked bytes in DSP heap, ≤ 512 handles) + `layer_run`: one u8 activation, N handles, whole-weight VTCM double buffer, cross-handle DMA prefetch, in-place tile dequant, `accumulate` | doc 34: q_proj 1024×2048, M=64: DSP 156 µs (quant 22 / drain 35 / mm 40 / acc_read 26 / dequant 33), wall 516; M=1024: 2,229 µs (quant+dequant 53 %); i4 x6-grouped 1,662 µs → **≈ 1.3 TMAC/s with f32 boundaries, 3.5 TMAC/s micro-mm alone**; M=1 x6 73 µs DSP | **adapt** (§3): the tile loop (`acc_clear` → `mm_u8i4` × k-tiles → `acc_read_int32` → in-place dequant) and its VTCM arithmetic become `ops/hmx-matmul.c`; the registry, heap residency, malloc-per-call and f32 in/out are replaced by our image + ACT slots |
| `hmx/hexkl_mm_u8i8_dma.{c,h}` | u8×i8 mirror (1024 B tiles, `rm_to_wh_i8`) | doc 34: parity with i4 at M=64 apart from the weight DMA | **keep as the compile-time width knob** of the same kernel (per-tensor i8 fallback if S1's PPL band fails for a tensor) |
| `hmx/hexkl_acc_tile.{c,h}` | ramp-probes the undocumented 64×32 int32 accumulator layout so dequant reads the VTCM tile in place (deleted the 52.8 µs/tile `copy_32b_to_submatrix`, 75–96 % of attention time) | doc 31 §3, doc 34 §4 A | **reuse as-is** |
| `hmx/hexkl_dma_ring.{c,h}` | one global static dmlink ring (256 descriptors), 2-D pushes, `MAX_DMA_ROW_BYTES` 256 KB | doc 34 §4 C: prefetch hides the weight DMA (75 → 55 µs/mm) | **replace** by our per-worker graph-lifetime `dma/dma-queue.h` (#25; same ggml lineage, already carries VTCM-bypass and the cross-op prefetch record) |
| `hvx/hvx_quant_u8.{c,h}`, `hvx/hvx_dequant_i32.{c,h}` | per-row asymmetric u8 params + AH-tile packer (f32 → u8, 2048 B tiles); i32 → f32 dequant `(acc − zp·colsum)·sx·sw + bias`, in-place tile variant | doc 34: 22 µs / 33 µs at 64 rows | **adapt**: fp16 input, our rule-6 int8 quantiser + 128 offset (zp fixed, `hvx-quant.h:106`), fp16 output via `Vhf_equals_Wqf32`, arithmetic re-expressed with qf helpers (§7 rule 1 review list); the AH tile layout and the in-place tile walk are kept |
| `hvx/hvx_worker_pool.{c,h}` | fork-join QuRT pool, unit 0 on the caller | doc 35 §4(c): needs `submit/wait` for overlap | **drop**: our `worker_pool.{h,c}` (rule 7: every worker holds an HVX unit; the caller has none) |
| `hmx/hexkl_attn_u8.{c,h}`, `hexkl_kv_quant`, `hexkl_attn_dtype`, `hvx_softmax*`, `hvx_exp_f32.h` | fused u8 SDPA: KV registered as i4/i8 blocks (T=256 positions), Q·Kᵀ and P·V through `layer_run`, blocked (not online) softmax, s_band/o_band in DDR | doc 35 §1: SDPA 4.07 / 6.09 ms per layer at kv 512 / 1024 prefill vs QNN's whole block 1.45 / 3.21; doc 32 §5: s_band in VTCM rejected (+30 % softmax under 6-thread bank contention) | **defer** to a follow-up issue after H1 shows the prefill ATTN share; our fp16 KV + #58 kernel stays. The KV quantiser's two placement defects (doc 00 §5) are already fixed in the PR (`.c`, self-contained fp16 decode) |
| `hmx/hexkl_probe.{c,h}`, `hexkl_mm_opts.h` | five per-stage µs counters; opts struct | the instrument behind every PR number | **reuse the idea**: the counters become fields of our `HTP_PROF_FARF` line (`mm4` split) behind `-DHTP_HMX_PROBE` |
| `test/htp/*` (IDL `nntr_hvx`, session with `hw_init` + `hmx_lock` + `HAP_power` vote), `test/unittest/unittest_hvx_*.cpp`, `tools/htp_{fc,attn}_report.py` | per-op FastRPC harness and gtests | doc 34 §2: transport 326 µs per call; `HAP_power_set` vote = the DSP half of a 90 → 3,900 µs transport spread | **not ported** (our graph executor is one RPC per chunk / token; the `HAP_power` vote is #41's lever — noted there) |
| `htp_backend.{h,cpp}`, `htp_compute_ops.cpp`, `htp_q4_0_convert.{h,cpp}`, `nntrainer/htp_context.cpp`, CausalLM LFM2-MoE (`Applications/CausalLM/models/lfm2_moe/*`, `tie_word_embedding.cpp`, `main.cpp`, `quantize.cpp`), `build_android.sh`, `install_android.sh`, `jni/*` | per-op `ComputeOps` seam (one FastRPC per `dot()`), Q4_0x4 → qs4cx requantiser (SNR 22.8 dB), LFM2-8B-A1B MoE model | doc 41 §0/§2/§3: **the model path has never run** (four dispatch blockers B1–B4, `M > 1` gate, workspace tensors without `ct_data`, x4/x8 repack ambiguity); Tier 1 = 176 FastRPC calls per token ≈ 57 ms transport — "a regression before compute" | **not cherry-picked**: it targets LFM2 MoE through per-op dispatch, which our graph-granularity executor (`engine="htp"` in `Applications/CausalLM/hexagon/hexagon_backend.cpp`, HEXAGON.md §5.4) supersedes; the requantiser's per-channel scheme is reproduced host-side in `make_w8cx_bin.py` from the fp checkpoint (no Q4_0 detour) |
| `docs/htp_attention/*` (28 files) | design ledger | doc 13: PR ①/② device gates cleared (u8i4 layer x4 3.31× over the harness); doc 35/36/37: u8 op boundaries then HMX/HVX overlap (1.59–1.85×, unbuilt); doc 40/41: MoE Stages 1–3 done, E0–E4 / P0–P3 unbuilt | numbers cited above; the docs are not copied |

### 2.2 Files and functions that change (ours)

| File:line | Today | Change |
|---|---|---|
| `nntrainer/tensor/hexagon/htp/nntr_htp_common.h:18-19` | `NNTR_HTP_ABI_VERSION 4u` | **v5**: `NNTR_HTP_OP_MATMUL_W4A8 = 9` (`:44-55`, `KIND_COUNT 10`), layout ids `NNTR_HTP_WEIGHT_LAYOUT_W4CX_DOWN8 = 2` (S2) and `W4CX = 3` (S3) next to `TILED32` (`:23`); `nntr_htp_w4_tile_off()` beside `nntr_htp_tile_off`; extent table entries for the new kind (`nntr_htp_op_extent`, #36) including the fp32 scale + int32 colsum refs (`in1` = tiles, `in2` = scale, `param0` = colsum offset) |
| `nntrainer/tensor/hexagon/htp/htp_graph.c:29` | `HTP_GRAPH_VTCM_BYTES` 4 MB | request 8 MB when `HTP_HMX` (v79 has 8 MB; `HAP_compute_res_query_VTCM` first, 4 MB fallback); the arena is split at init into the HMX region (act tiles for `max_chunk` rows × `k_max`/32 × 2048 B ≈ 384 KB, two weight strips ≤ 512 KB each, two 8 KB result tiles, `hexkl_micro_hmx_config_size()` at the top, 256-aligned) and the per-worker half-slabs the HVX tiled kernels keep |
| `htp_graph.c:155-168` | `HAP_compute_res_attr_set_vtcm_param` + acquire | add `HAP_compute_res_attr_set_hmx_param(&rattr, 1)`; log `hexkl_micro_get_version` once; `htp_graph_destroy:315-316` releases as today |
| `htp_graph.c` (new, after the buffers are mapped, before `:174`) | — | **in-place WH bake**: for every `MATMUL_W4A8` op, per 512 B tile: unpack nibbles → 1 KB int8 staging in VTCM → `hexkl_micro_hmx_rm_to_wh_i4(vtcm, off, staging, 0, 0, 32)` → write the 512 B WH tile back over its source; runs on worker 0 under `wp_run`; sets `g->baked`; ≈ 176 MB / (12.2 ms per 2 MB, PR doc 34 §2 — that figure includes malloc + memcpy) ≈ 1.1 s; a second `init()` on the same fd is rejected (`AEE_EALREADY`-style rc) — the host re-packs per session, as the app already does |
| `htp_graph.c:130-137` (`next_mm[]`), `:122` (`stream_bytes`) | tiled W8A8 / LOGITS only | the new kind joins both (its strips prefetch across ops exactly like W8A8's chunk 0; bytes = tiles + scale + colsum) |
| `htp_graph.c:217-240` (`HTP_PROF_FARF`) | `mm8 mm16 lg attn rest` | `mm4` field; `-DHTP_HMX_PROBE` adds `q4 hmx acc dq dr` (quant / micro-mm / acc_read / dequant / drain µs) after it; `tools/hexagon/summ_farf_prof.py:26` `FIELDS` gains them |
| `nntrainer/tensor/hexagon/htp/worker_pool.c:40-44` (`worker_main`) | `qurt_hvx_lock` at thread entry | worker 0 additionally takes `HAP_compute_res_hmx_lock(ctx_id)` on its **first** HMX op (lazily: the compute-res id exists only after `:155`) and releases it in a `wp_run` job from `htp_graph_destroy`; HMX instructions are issued by worker 0 only (§3, §5 risk 3) |
| `nntrainer/tensor/hexagon/htp/ops/htp_ops.h` (`struct htp_exec_ctx`, `htp_op_table`) | 9 kinds | `hmx` sub-struct (arena offsets, `acc_layout`, act-tile cache key `{in0.offset, n_tokens}` so q/k/v and gate/up quantise `t` once — the PR's x3/x6 grouping without a lowering change), `hvx_op_matmul_w4a8` |
| `nntrainer/tensor/hexagon/htp/ops/hmx-matmul.c` (**new**, ported from `hexkl_mm_u8i4_dma.c` `layer_run` + `hvx_quant_u8.c` + `hvx_dequant_i32.c` + `hexkl_acc_tile.c`) | — | `quant_worker`-style u8 AH packing on all workers (rule 6 quantiser + 128, per-token `sx`), n-strip streaming through worker 0's `dma-queue` (`hvx-matmul.c:138 mm_dma_push`, `:189 mm_pf_kick` reused; strip = c n-tiles × K/32 tiles × 512 B, rows ≤ 256 KB per 2-D push — the PR's DMA row-size limit), HMX tile loop on worker 0, in-place tile dequant to fp16 ACT rows (qf32 epilogue in the rule-5 order `(((float)(acc − 128·colsum[n])) · sw[n]) · sx[t]`), `-DHTP_HMX_STREAM_ONLY` (DMA only), `-DHTP_HMX_W8` (u8i8 tiles, 1024 B) |
| `nntrainer/tensor/hexagon/htp/hmx/` (**new dir**) | — | `hexkl_acc_tile.{c,h}` verbatim; `hexkl_micro.h` is *not* copied (SDK-style `-I /opt/qcom/hexkl_addon/include`) |
| `nntrainer/tensor/hexagon/htp/ops/hvx-matmul.c:494` (`hvx_op_matmul_w8a8`), `:326 mm_worker`, `:224 mm_worker_vtcm` | HVX tiled kernel | unchanged in S2 (W8 images keep working: the two layouts coexist behind `weight_layout`); S3 adds the m=1 W4 tiled kernel (`vrmpyacc_VwVubVb` on unsigned nibbles + `8·Σx` correction — #51 PR 2's design, but per-channel, no group scales) and `hvx_op_matmul_w4a8` dispatches `m == 1 → HVX, else → HMX` under `-DHTP_W4_DECODE_HVX` |
| `nntrainer/tensor/hexagon/htp/hvx/hvx-quant.h:106` | `htp_quant_row_fp16` int8 | unchanged; the u8 tile packer adds 128 after it (bit-exact reuse of rule 6) |
| `tools/hexagon/build_skel.sh:43-46, 48-62`, `build_sim_test.sh:31-48` | glob `ops/*.c hvx/*.c dma/*.c` | add `hmx/*.c`, `-I "$HTP_DIR/hmx" -I "$HEXKL_ADDON_ROOT/include"`, link `$HEXKL_ADDON_ROOT/lib/hexagon_${DEFAULT_TOOLS_VARIANT}_${HEX_ARCH}/libhexkl_micro.a`, `HEXKL_ADDON_ROOT` default `/opt/qcom/hexkl_addon` (`tools/docker/run.sh:33,60` already mounts it); `-DHTP_HMX=1` when the lib is present, else the W4A8 kind is rejected at `init()` with rc 4 (a skel without HexKL never runs a W4 image silently) |
| `Applications/CausalLM/hexagon/qwen3_lowering.cpp:74-88` (per-layer offsets), `:152,164,176,234,267,279` (six `MATMUL_W8A8` emits), `:303` (`W8A16`), `:338` (`LOGITS`) | one layout | `HexModelConfig` gains `weight_layout`; the six emits become `MATMUL_W4A8` with `wq_cs` etc. offsets when `w4cx*`; `down` / `embed` / `LOGITS` unchanged in S2, 4-bit in S3 |
| `nntrainer/tensor/hexagon/host/graph_lowering.{h:72-86,cpp:57-87}` | `write_tiled` → `nntr_htp_repack_tiled32` | `write_w4cx` (nibble tiles 32 n × 32 k, n-tile outer, k-tile inner, host-computed `colsum[N]` int32 and `scale[N]` fp32); `HexLayerWeights` gains the W4 pointers |
| `Applications/CausalLM/hexagon/hex_image.cpp:28,63-68`, `qwen3_w8cx_bin.{h,cpp}`, `hexagon_backend.cpp` | `weight_layout=tiled32` | `w4cx_down8` / `w4cx`; the `.bin` reader recognises the W4 record (`bits` field in a small header, or a sibling `.w4cx.bin` — S1 decides); the app packs W4 from it |
| `tools/hexagon/make_w8cx_bin.py:73 quant_w8cx, :104-108` | int8 per-channel | `make_w4cx_bin.py` on the `hvx_w4cx` branch (symmetric RTN, `scale = amax/7`, values in [−7, 7] inside HexKL's [−8, 7]), `--i8-tensors down,embed` (the `w4cx_down8` split), colsum written per tensor; the W8 output stays byte-identical |
| `test/hexagon/sim/ref_ops.{h:28-43,c}` | `ref_matmul_w8a8` | `ref_matmul_w4a8` (int8 quant + 128 → exact int32 with the colsum term, fp16 epilogue); `ref_graph_forward` dispatches the new kind; `hexagon_ref_run --eval` reads the pre-bake image |
| `test/hexagon/sim/test_hmx.c` (**new**), `test_matmul.c`, `test_graph.c`, `test_profile.c`, `sim_model.c` | 13 tests | `hmx`: acc-tile probe `usable=1`, one 64×32×32 mm vs scalar, WH relocate round trip (bake in VTCM → DDR → DMA back → identical result — the PR verified this on device, doc 13 §3b; the spec's H1 spike question), **WH ramp probe printing the permutation** (S3's input); `matmul_w4a8_m1/m8/m128` STAT `0/0` on the int32 accumulator, ≤ 2e-3/1e-3 on fp16; `graph_w4_*`, `profile acc_w4`; negatives: `k % 32`, `n % 32`, a W4A8 op on a `tiled32` header → rc 5 |
| `tools/hexagon/find_divergence.py:44-63` | kind names from `--list-ops` | knows `MATMUL_W4A8` (fp16 output) |
| `docs/backend_guide/HEXAGON.md`, `HEXAGON_BENCHMARK.md`, `NOTICE` | — | §6 |

ABI: **v4 → v5** in S1 (one bump for the whole issue: new kind, two layout ids, colsum/scale refs).
Consumers that move with it, in the same PR: `lower_qwen3`, `pack_weights`, `.hexcfg`
(`read_hexcfg` rejects v4 images that claim a W4 layout), `ref_graph_forward`, `sim_model`, the sim
`graph` test, `find_divergence.py`, `nntr_hexpack`, `hexagon_backend.cpp`, HEXAGON.md §1.4 / §2.1.
The KV layout, ATTN scratch, IDL and `forward()` signature do not change.

## 3. Design

### 3.1 Chosen: HMX for the six per-layer projections inside the existing graph executor; everything else stays

* **One RPC per chunk / token stays.** The PR's whole cost structure (326 µs transport per call, x3/x6
  grouping, `HAP_power` votes, weight registries) exists because it dispatches per op over FastRPC.
  Our executor already amortises all of that (0.26 ms per call, HEXAGON.md §1), so what is ported is the
  **tile loop and its VTCM arithmetic**, not the session, registry or ring.
* **Weights live in the mapped WEIGHTS image, WH-baked in place at `init()`.** WH tiles are position
  independent (PR: baked in VTCM, memcpy to heap, DMA back, device-verified — the spec's H1 gate is
  answered), a 32×32 int4 tile is 512 B before and after the bake, and `rm_to_wh_i4` accepts a 32-column
  source, so the bake is byte-neutral and needs no DSP heap (the PR's `-v2` heap cache crashed above
  256 MiB resident, doc 13 §3 T2). Memory: 176 MB of i4 projection tiles + 2.8 MB scale/colsum +
  88 MB int8 `down` + 156 MB int8 `embed`/lm_head ≈ **423 MB** for `w4cx_down8` (598 today), ≈ 307 MB
  for `w4cx` (S3). The image's tile order (n-tile outer, k-tile inner) makes an n-strip one contiguous
  2-D DMA, like tiled32 today.
* **Activation: our rule-6 int8 quantiser + 128.** `x_u8 = x_i8 + 128` keeps the bit-exact per-token
  quantiser (`htp_quant_row_fp16`) and the reference (`ref_quant_row`) unchanged; the correction is
  `128·colsum[n]` (spec 03's formula), exact in int32, so the integer path stays bit-exact against
  `ref_ops.c`. The PR's asymmetric u8 (min/max, zp) buys < 1 bit on symmetric activations and would
  need a new tie-rule quantiser and reference; rejected for S2 (revisit only if S1's PPL band fails on a
  projection). `down` stays int16 activation on HVX (HEXAGON.md §2.3: int8 there costs 6 % PPL; HMX has
  no 16-bit activation path), `embed` / `LOGITS` stay HVX (last-token prefill gains nothing from HMX;
  their 4-bit tiles are S3, a #51 fold-in).
* **Threads.** HMX instructions are issued by **worker 0 only** (the HMX unit is one per core and the
  lock is per thread; the PR locks on the FastRPC thread and issues from later FastRPC calls, doc 35
  §4(a) — we do not rely on that). The other five workers pack the u8 AH tiles (S2) and, in S4, dequant
  tile i while worker 0 issues tile i+1 (the PR's doc 35 §4/§5 two-stage pipeline with two result
  tiles, on our barrier pool: worker 0 finishes a strip, the pool dequants it while worker 0 starts the
  next — one `wp_run` per op, no new thread). Rule 7 holds: nothing vectorised runs on the RPC thread.
* **Weight streaming through our DMA queue, strip-granular.** The PR double-buffers *whole* weights
  (`wb_max × 2`; `AEE_ENOMEMORY` at M ≥ 1024 for its shapes, doc 41 §3.4). We stream n-strips into two
  VTCM slots through worker 0's graph-lifetime queue and kick the next op's strip 0 across ops exactly
  as `mm_pf_kick` does for W8A8, so the #25 cross-op prefetch and `HTP_MM_STREAM_ONLY`-style floor
  variants carry over unchanged in spirit.
* **Format `w4cx` (4-bit, per-channel scale, "cx" as in W8_CX)** rather than #51's `w4g128`: HexKL's
  u8i4 accumulates the whole K in one int32 per (row, column), so a per-group scale would need an
  `acc_read` every 4 k-tiles — 8× the acc_reads, ≈ 3 ms against 0.62 ms of micro-mm on the doc-34
  shape, i.e. HMX prefill 5× slower. Per-channel int4 is what the PR, QNN's u8i4 FC and HexKL all
  assume; its accuracy on Qwen3-0.6B is **unknown** and is the first gate (S1, x86 only). If a tensor
  fails the band it stays i8 (`-DHTP_HMX_W8` tiles for that op; the layout id records the split).

### 3.2 Qwen3-0.6B specifics vs the PR's LFM2-MoE target

| Qwen3-0.6B | consequence |
|---|---|
| dense, 28 layers, 7 projections/layer (q 1024→2048, k/v 1024→1024, o 2048→1024, gate/up 1024→3072, down 3072→1024), all K, N multiples of 32 and 128 | no expert routing, no per-expert M; the six HMX ops per layer are static in the op-list; `max_chunk` 128 = 2 row blocks, decode = 1 row block with 63 padding rows |
| GQA 16/8, head 128, tied lm_head (vocab 151,936) | attention untouched (fp16 KV, #58); the tied table stays one tensor (`embed` gathers rows, `LOGITS` streams it) — a WH-baked table cannot be gathered, so it stays tiled32 int8 (S2) and becomes 4-bit *tiled32-nibble* for the HVX kernels (S3), never WH |
| fp checkpoint → our own producer | no Q4_0x4 → qs4cx requantiser (PR `htp_q4_0_convert`); `make_w4cx_bin.py` quantises from bf16 |
| 4 MB VTCM today, 8 MB on v79 | HMX act tiles for 128 tokens × K 3072 = 384 KB; strips ≤ 512 KB × 2 — fits either budget |

### 3.3 Rejected alternative

**Cherry-pick the PR's per-op path** (`HtpComputeOps::gemm_q4_0_accel_fp32` + its skel + `engine="htp"`
per layer) and drive Qwen3 through `FloatTensor::dot`. Rejected: 7 FastRPC calls per layer per step =
196 per token ≈ 64 ms of transport at the PR's measured 326 µs (doc 41 §3) before any compute — 4× our
whole decode step; the PR's own doc 41 calls Tier 1 "a regression before you run it"; and it would leave
two executors, two images and two accuracy references in the tree. A second rejected variant, HMX
*everywhere* including m=1 as the shipping decode (`spec hexagon-hmx/00-overview`): G3's arithmetic
puts HMX at m=1 at ≈ 17 GB/s-equivalent against the 37 GB/s ring, so m=1 goes to the HVX 4-bit stream
(S3) once H1 confirms the gap.

## 4. Steps

Each step is one PR-sized change with its gate (`.claude/skills/hexagon-gates`); rungs 2–3 once per PR,
right before opening it. **The smallest thing that produces a device number is S2's H1.**

| # | Step | Gate |
|---|---|---|
| S0 | **Build + resources** (no ABI, no image): `build_skel.sh` / `build_sim_test.sh` link `libhexkl_micro.a` (`HEXKL_ADDON_ROOT`, `-DHTP_HMX=1`); `htp_graph_init` acquires VTCM + HMX (`set_hmx_param`), logs the HexKL version, splits the arena; worker 0's lazy `HAP_compute_res_hmx_lock`; `hmx/hexkl_acc_tile.{c,h}` ported; `test_hmx.c` (acc-tile probe, one micro-mm vs scalar, WH relocate round trip, WH ramp permutation print) | rung 1 unchanged; `HEX_ARCH=v79 run_sim_test.sh hmx` → `SIM_TEST hmx PASS` — **if hexagon-sim does not execute HMX**, run HexKL's `examples/hexkl_micro_hmx_mm_u8i4_i32/run_simulator.sh --hex-arch v79` to confirm, then `hmx` degrades to "links and skips" and every HMX numeric gate below becomes device-only (a new §7 rule); `profile acc` STAT bit-identical (the arena split must not move the HVX slabs' results); skel + harness build (rung 4), skel md5 recorded |
| S1 | **W4 producer, image, reference** (no DSP bytes): `make_w4cx_bin.py --i8-tensors down,embed`, `.bin` record, `Qwen3W8cxBin` W4 read, `w4cx` packer + `nntr_htp_w4_tile_off`, ABI v5 (kind 9, layout ids, extent table), `lower_qwen3` emits `MATMUL_W4A8`, `.hexcfg`, `ref_matmul_w4a8`, `ref_graph_forward`, `find_divergence.py`, `test_lowering` | `LOWER_TEST PASS`, `W8CX_BIN_TEST PASS`, the **W8 image byte-identical** (md5) and its `--eval` unchanged (33.0195 / 184); x86 `--eval` of the `w4cx_down8` image on the P4 prompt and `t512_23`: **≤ 1.10 × W8** → S2 proceeds with all six on i4; 1.10–1.25 × → the per-tensor sweep (x86, one tensor at a time back to i8) picks the `i8-tensors` set that gets under 1.10 and the layout id records it; > 1.25 × even with q/k/v/o i8 → `needs-user` (per-channel int4 is not viable for this model on HMX; the options are w4g128 on HVX only = #51 as planned, or HMX u8i8 W8A8 prefill = the original stage 2). Record the PPL of every tried set in the plan |
| S2 | **HMX kernel + H1**: `ops/hmx-matmul.c` (u8 AH packing on the pool, strip DMA via worker 0's queue with cross-op kick, HMX tile loop, in-place dequant to fp16, the act-tile cache across q/k/v and gate/up, `HTP_HMX_STREAM_ONLY`, `HTP_HMX_PROBE`, `HTP_HMX_W8`), in-place bake at init, `mm4` FARF field, `sim_model` op sequence, `matmul_w4a8_*` / `graph_w4_*` / `profile acc_w4` cases; HMX serves m=1 too in this step | `run_sim_test.sh matmul` per step; before the PR: `profile acc` **and** `profile acc_w4`, 13 + `hmx` = 14 tests, int32 STATs `0/0` at m=1 / 8 / 128, fp16 within 2e-3/1e-3, `graph_w4_prefill/decode` against `ref_graph_forward`; rung 4 → **handoff H1** (`docs/measurements/65-w4-hmx-h1.md`, ≤ 4 skels): A = `hvx_impl` W8 control (`-DHTP_PROF_FARF`), B = W4 HMX, C = B + `HTP_HMX_STREAM_ONLY`, D = B + `HTP_HMX_PROBE`; 512 sweep A/B/C/D, B at 512 / 1024 / 4096, `--eval` A/B (+ D digit-identical to B); gates G1, G1', G3, G4 |
| S3 | **Full 4-bit image + decode path + H2** (two PRs): (a) `w4cx` for `embed` / `LOGITS` (tiled32-nibble for the HVX gather / stream kernels) and `down` (row-major nibbles, int16 activation — #51 PR 3's kernel, per-channel scale); (b) the m=1 HVX W4A8 kernel reading **WH tiles** through the permutation `test_hmx` printed (a fixed intra-512 B shuffle: `vdelta` / `vrdelta` or a `vlut` per tile, cheap on a DMA-bound path) and the `m == 1 → HVX` dispatch — built only if H1's `mm4(B) − mm4(C) > 2 Mcyc` (G3); if the WH permutation is not expressible in ≤ 4 vector ops per tile, decode stays on HMX and the two-copy alternative goes to `needs-user` | sim: `logits` / `embed` / `matmul` W4 cases `0/0` on the integer paths, `matmul_dma` W4 strips at 4 MB / 256 KB / 64 KB with DDR = DMA `memcmp`; x86 `--eval` of `w4cx` within the S1 band; **handoff H2**: A = H1's B, B = full `w4cx` (+ HVX m=1 if built), C = B stream-only; gates G5 (stream phase ≤ 0.55 × W8 control, B − C ≤ 1 Mcyc on `mm4`), decode / prefill rows at 512 / 1024 / 4096, `--eval` per variant; benchmark rows filled |
| S4 | **Prefill toward 8,207** (new issues filed from H1's D split, not this plan's gate): (i) dequant / HMX overlap (two result tiles, pool dequants strip i while worker 0 issues i+1; PR doc 35 ceiling 1.3–1.6× on the HMX phase); (ii) u8 op boundaries between fused ops (skip fp16 round trips where the consumer is another W4A8 op); (iii) `down` on HMX with u8 activation *if* an x86 sweep shows the int16 requirement is per-token-outlier and a per-token u8 + int16 fallback split holds the band; (iv) attention on HMX (PR `hexkl_attn_u8` + KV quant) after #26 measures the prefill ATTN share | each its own plan; device handoffs at 512 only |

**S0 outcome (#68, 2026-09-18, branch `hvx/65-s0-hmx-build`).** hexagon-sim **executes HMX**: the
plain `-mv79` core is `v79na_1` (HMX v3 coprocessor), HexKL's own `examples/hexkl_micro_hmx_mm_u8i4_i32`
passes bit-exact on it, and `run_sim_test.sh hmx` → `SIM_TEST hmx PASS` with `lock rc=0 acc_layout
usable=1 base=0 row_stride=32`, `hmx_mm_copy` / `hmx_mm_inplace` / `hmx_mm_reloc` STAT `max_abs=0`,
`wh_reloc bytes_same=1`; so the "links and skips" degrade and the device-only §7 rule are **not** needed
(the acc-layout line is still re-read on the device in H1, §5 risk 1). The simulator grants 8 MB VTCM
(4 MB HVX + 4 MB arena; `cfg_off` 4177920, result tiles at 4161536 / 4169728, 4161536 B free for S2).
HexKL is **1.0.0-beta1 hexagon v79**. The WH bake is a pure bit permutation of the tile index: element
`(k, n)` of a 32×32 int4 tile lands at byte `128·(k>>3) + 4·n + (k&3)`, nibble `(k>>2)&1` (`n → dst bits
3..7`, `k → bits 1, 2, 0, 8, 9`) — S3 (b)'s reader is a fixed intra-128 B shuffle, expressible in
≤ 4 vector ops per tile. `profile acc` STAT and every unit STAT bit-identical to the #25 `hvx_impl`
record; the `-DHTP_HMX=0` build is object-identical to `hvx_impl` (12/12 DSP objects, `dsp_obj_md5.sh`).

**S1 outcome (#69, 2026-09-21, branch `hvx/69-w4cx-producer-abi5`).** Everything in the S1 row is
built and its mechanical gates pass: `make_w4cx_bin.py --i8-tensors down,embed` (64-byte
`W4CX` header, int4 codes one per byte, the W8 output byte-identical — md5 `7562313b…`), `Qwen3W8cxBin`
reads both and `apply_layout()` picks `tiled32` / `w4cx_down8`, ABI **v5** (kind 9 `MATMUL_W4A8`,
layout ids 2 / 3, `nntr_htp_w4_tile_off` / `nntr_htp_repack_w4cx` / `nntr_htp_w4_get`, extent row,
validator rules incl. the `param0` colsum bound; `W4CX` = 3 stays rejected until S3), `lower_qwen3`
emits `MATMUL_W4A8` per int4 class with `param0 = colsum` (a mixed i8 set is per op, recorded as
`i8_tensors=` in the `.hexcfg`, so the sweep needs no new layout id), `ref_matmul_w4a8` (u8 × i4 +
`128·colsum`, bit-exact against numpy from the `.bin` codes on the 1-layer image), `LOWER_TEST PASS`,
`W8CX_BIN_TEST PASS` (+ `W4CX_BIN_TEST PASS` on the W4CX file), the oplist header check, the
W8 image byte-identical (`5abf61be…`, 598,623,744 B), the `w4cx_down8` image 423,724,544 B. DSP
bytes: only `executor.o` (version string) and `htp_graph.o` (NULL table slot + init reject) differ
from `hvx_impl`, 9/11 objects identical (`dsp_obj_md5.sh`); `run_sim_test.sh graph` on v79 passes.

**The accuracy gate fails at the plan's stop line.** The P4 / `t512_23` prompt files were not on the
build machine, so the band is read on a fresh 512-token prompt (*Pride and Prejudice* ch. 1, token
md5 `8186fa8b…`; the ratio is what the gate is about — the user can re-run both commands on the
original files, HEXAGON.md §8.1). x86 `--eval`, 511 steps:

| set (x86 reference, activations per-token int8 as on the DSP) | PPL / top-1 | × W8 |
|---|---|---|
| W8 `tiled32` | 22.5282 / 186 | 1.000 |
| `w4cx_down8`, all six int4 (RTN `amax/7`) | 46.2279 / 148 | **2.052** |

Torch weight-only fake quant, same tokens (fp32 21.8485, W8 21.7619), the per-tensor sweep the row
asks for: six int4 41.6754 (1.915 ×); one class back to int8 — q 38.65, k 37.13, v 35.57, o 35.95,
gate 34.98, up 33.28 (each still > 1.5 ×); **q/k/v/o int8, gate + up int4: 27.85 = 1.280 ×** (the
"> 1.25 × even with q/k/v/o i8" condition); gate/up int8, q/k/v/o int4 28.74 (1.321 ×); one class
alone: q 1.056, k 1.068, v 1.029, o 1.067, gate 1.033, up 1.186 ×. The errors compound, so no set
of more than ~two of the six stays under 1.10 ×. Per-channel int4 RTN is therefore **not viable
for Qwen3-0.6B** on the HMX u8 × i4 path as planned → `needs-user` (the options the row names:
#51's `w4g128` on HVX only, or HMX u8i8 W8A8 prefill; a third is a better per-channel quantiser
on the producer side with the format unchanged — measured: a per-row weight-MSE clip search gives
six int4 38.65 / 37.70 (`[-7, 7]` / `[-8, 7]`, 1.78 / 1.73 ×) and gate + up only **26.03 = 1.196 ×**,
under 1.25 but not 1.10 and only two tensors — i.e. the format can carry gate/up (≈ 44 % of the six
projections' bytes) at ≈ +20 % PPL, nothing more). S2 must not start on the six-int4 image; the S1
code (format, ABI, reference) is complete and merges on its own so the decision does not block the
tree. Build-machine notes: no docker / SDK 6.4 / HexKL / NDK on the workstation this ran on, so rung 1
ran natively (gcc 13), rung 4 with SDK 6.3.0.0 v79 (`HTP_HMX=0`, skel md5 `aae982ee…` after the review follow-up, no host
harness: no NDK), and the one simulator run (`graph`) fails `graph_partial_attn` with `inf` on the
unmodified `hvx_impl` tree as well (toolchain 8.8 v79, HEXAGON.md §7 rule 1) while `graph_prefill` /
`graph_decode` are 0/0 on both trees.

Device measurements are unavoidable at **S2 (H1)** and **S3 (H2)**; S0 and S1 need none. Handoff
rules: artifacts with md5 + commit, `--eval` columns on every row, reference numbers in the table
(A: 54.6 Mcyc / 41.4947 / 162 at 512), B's `E2E gen` md5 recorded, 4096 rows only as A/B inside one
order (rule 9(e)), battery °C noted.

## 5. Risks

| Risk | Where it bites | How the handoff / gate makes it visible |
|---|---|---|
| **HMX not executed by hexagon-sim** (or executed without the acc-layout permutation the silicon uses) | S0/S2 integer STATs meaningless; the in-place dequant would read a wrong layout on device | `test_hmx` prints `acc_layout base/row_stride`; H1's D probe prints `HEXKL_PROBE_ACC_STRIDE` (0 = vendor-copy fallback) so a device that disagrees with the simulator falls back to `copy_32b_to_submatrix` correctly and *says so*; the `--eval` column catches a silent layout error |
| **HexKL beta1 vs beta2** (installed vs the PR's build) | undocumented behaviour differences in `rm_to_wh_i4` / `acc_read_int32` | version logged at init and copied into the handoff; the WH relocate round trip and the ramp probe are re-run on device through `test_hmx`'s skel twin (`hexagon_rpc_test`-style `n_ops == 0` path extended with a `--hmx-selftest`) before any tok/s row is read |
| **HMX lock thread affinity** (lock on worker 0, acquire on the RPC thread) | `AEE_EFAILED` / hang on the first W4A8 op on silicon only (the simulator does not enforce ownership, rule 7) | H1 step 1 is a 1-layer W4 image `--steps 2`; expected `E2E init ok` + a FARF line `hmx lock ok wid=0`; the fallback variant (lock taken in `init()` on the RPC thread and HMX issued there — the PR's topology) is B′, built only if B fails this step |
| **Bandwidth gap** (simulator does not charge DDR; strips vs the PR's whole-weight buffers) | prefill `mm4` could be DMA-bound rather than issue-bound | C (stream-only) next to B at every context; `mm4(B) − mm4(C)` is the compute that surfaces |
| **qf32 numerics** in the ported epilogue (the PR uses IEEE `Q6_Vsf_*` throughout: 23 `vmpy`, 6 `vadd`) | §7 rule 1 review list bans IEEE paths; #35 closed the v79 question but the rule stands | the kernel is written with the qf helpers of `hvx-base.h`; H1 B's `--eval` against the x86 W4 reference (±2 %) and `find_divergence.py` on the 1-layer image (bound ≤ 1e-4 up to ATTN, as §8.1) are in the table |
| **Per-channel int4 accuracy** on a 0.6B model | the whole format | S1's x86 band *before* any DSP work; the per-tensor i8 sweep is the mitigation, `needs-user` the stop |
| **VTCM bank contention** (PR doc 32 §5: pool workers hitting VTCM lost 30 %) | S4's overlap, and S2's u8 packing writes AH tiles into VTCM from 6 workers | D's `q4` field vs the PR's 22 µs/64 rows; if packing is > 2× that, pack into a DDR staging row block and DMA it in (one extra queue push per row block) |
| **In-place bake idempotency** (a second `init()` re-bakes WH bytes) | app / harness re-init | `g->baked` + rc on re-init; the harness and app always re-pack (documented in §5.3 / §5.4) |
| **Stale artifacts** (two image formats, three layout ids) | wrong image with the right skel | `.hexcfg` `weight_layout` handshake at `init()` (rc 4), `push_if_changed` md5s, the skel md5 the log prints is the one the table repeats |
| **Session state at 4096** (#53) | any 4096 verdict | A/B inside one order, same sitting; H2's 4096 row waits for #58 |

## 6. Docs to update

* `HEXAGON.md`: §1 (architecture diagram gains the HMX op and the compute-res HMX param; §1.4 ABI v5,
  kind 9, layout ids, colsum/scale refs), §2.1 (`w4cx` tile layout, in-place bake, sizes 423 / 307 MB),
  §2.2 (VTCM arena split, 8 MB request), §2.3 (the six ops' kind by layout), §2.4 (`make_w4cx_bin.py`,
  `.hexcfg` ids), §3 (`hmx/` dir, `ops/hmx-matmul.c`, `test_hmx.c`), §4 / §5.1–5.3 (`HEXKL_ADDON_ROOT`,
  `-DHTP_HMX`, the 14th sim test, `profile acc_w4`), §7 new rules: (10) HMX issue only on worker 0 under
  its own lock; (11) simulator HMX coverage and the acc-layout probe as the device check; (12) WH tiles
  are baked in place and an image is packed per session; §8.1 (x86 W4 PPL band per tensor set), §8.2
  (H1 / H2 blocks), §8.3 (`hmx`, `acc_w4` STATs), §9 (S4 items, #26 / #59 re-targeted).
* `HEXAGON_BENCHMARK.md`: the three `NPU, HVX + HMX` TODO rows become `w4a8 (w4cx)` rows from H1 (B)
  and H2 (B); the `w4a8 HVX only` row's note points at S3 (folded #51); Goals: stage 2 "Now" from H1,
  the GENIEX_QAIRT reference cells named per row, the #60 multiple once filled; Log entries for S1's
  band, H1, H2.
* `NOTICE`: `hmx/hexkl_acc_tile.*` and the ported kernel bodies are Apache-2.0 from
  nntrainer/nntrainer PR #4327 (authors kept); HexKL is a Qualcomm addon linked from the user's mount,
  never committed (same footing as the SDK).
* `docs/superpowers/specs/hexagon-hmx/00-overview.md`: one status line pointing here (H1 spike answered
  by the PR's device run; u8i8-everywhere superseded by u8i4 projections + HVX decode).
* Guide: `06-performance.html` after each filled handoff (guide writer).

## 7. Disposition of #51 and ordering

**#51 is folded in as the decode (m=1) path, re-based on the `w4cx` format.** Its three PRs map onto
this plan: PR 1 (producer / reader / lowering / `.hexcfg` / reference / PPL band) **is S1** with the
format changed from `w4g128` to per-channel `w4cx` (HexKL cannot apply group scales, §3.1); PR 2 (tiled
W4A8 / LOGITS / EMBED HVX kernels, unsigned nibbles + `8·Σx` correction) **is S3 (a)+(b)** reading WH
tiles for the six projections and tiled32-nibble tiles for `embed` / `LOGITS`; PR 3 (W4A16 `down`) **is
S3 (a)**. What #51 loses: the group-128 scales (its accuracy insurance) — S1's band is the replacement
gate, and if per-channel fails even with a mixed i8 set, #51's `w4g128` returns as the HVX-only
alternative under `needs-user`. #51's 70.3 / 27.5 cells remain the GENIEX_LLAMACPP reference on the
w4a8 row; the goal multiple is #60's. Label: #51 stays `state:planned` + `needs-user` with a comment
pointing here; it is closed as `completed` when S3 merges or re-planned if S1 stops.

Ordering: **#58 (in progress) merges first** — its FARF instrument and attention share are what H1 /
H2 are read against, and its `htp_graph.c` / `htp_ops.h` hunks are adjacent to S2's. **S0 and S1 can
start now** beside #58 (disjoint files: build scripts, `hmx/`, `test_hmx.c`, the packer / reference;
S1 changes no DSP bytes other than the validator's new kind). #59 (`down` on the DMA ring) and #57
(`MM16_R`) stay valid — `down` stays HVX here — and re-target their gates to the `w4cx` rows after S3;
#26 (prefill attention) becomes the next prefill lever the moment H1's D split lands and should be
planned against that split; #41's clock vote is unchanged (the PR's session `HAP_power_set` block is
the same lever, for its planner); #60 is independent and needed to read every row.

PR sizes: S0 ≈ 400 lines (scripts, graph init, `test_hmx.c`), S1 ≈ 900 (producer, packer, reference,
lowering, tests), S2 ≈ 900 (kernel, bake, sim cases, handoff), S3 two PRs ≈ 600 + 500, each with one
rung-2/3 run right before opening.
