// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file   rope_inplace.cl
 * @brief  FP16 rotary position embedding with cache write offsets.
 * @see    https://github.com/nntrainer/nntrainer
 *
 * @details Extracted from attention_kernels.cpp.
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// 1 (default) = the fp32/fp16 rotation choice is a property of the SEQUENCE,
// so a continuation chunk matches the chunks before it; 0 = the old rule,
// chosen from this chunk's M alone. Set from NNTR_ROPE_TAIL_FIX; see the
// branch below.
#ifndef ROPE_TAIL_CONSISTENT
#define ROPE_TAIL_CONSISTENT 1
#endif
__kernel void rope_inplace_f16(__global const half *in,
                               __global       half *out,
                               __global const half *cos_lut,
                               __global const half *sin_lut,
                               const int M, const int num_heads,
                               const int half_d, const int start_pos,
                               const int write_off) {
  // write_off: when writing the rotated K into a STABLE base
  // (cache_key) instead of an offset-baked slice pointer, out[write_off + ..]
  // addresses the per-token row via a SCALAR (recordable) instead of an SVM
  // pointer. Default 0 == in-place / offset-baked-out behaviour (byte-identical).
  int t = get_global_id(0);
  int h = get_global_id(1);
  int k = get_global_id(2);
  if (t >= M || h >= num_heads || k >= half_d) return;
  long row = (long)t * num_heads * (2 * half_d) + (long)h * (2 * half_d);
  long lut = (long)(start_pos + t) * half_d + k;
  half c = cos_lut[lut];
  half s = sin_lut[lut];
  half lo = in[row + k];
  half hi = in[row + k + half_d];
  // FP32 rotation for SMALL step size (M < 32): decode (M=1) and short prefill,
  // where the half cos/sin multiply-add's precision loss distorts the softmax
  // DISTRIBUTION enough for SAMPLING to degenerate (argmax survives -> greedy
  // coherent). fp32 there is ~free (few rows) and makes it match the host RoPE.
  // For a large prefill (M >= 32) the half rotation is kept: it is ~6% faster on
  // the big RoPE and the model-agnostic threshold avoids a head_dim hack. Store
  // is always FP16 (the activation dtype). See reference_working_run_combos.
  //
  // ...but M is the CHUNK's row count, not the sequence's, and a chunked
  // prefill is one sequence. Choosing the precision from M alone rotates the
  // LAST chunk of a long prompt in fp32 while every earlier chunk was rotated
  // in fp16, whenever that chunk happens to be shorter than 32 rows -- so the
  // K cache the decode steps then read is internally inconsistent across a
  // boundary a few rows before the first generated token. Measured
  // on a 3 595-token prompt, tails of 4, 11 and 18
  // rows all answer wrongly and tails of 32, 395, 445 and 523 are golden --
  // the threshold in the data is this 32, exactly.
  //
  // rope_tail_consistent (NNTR_ROPE_TAIL_FIX=0 restores the old rule) makes
  // the choice a property of the SEQUENCE instead: fp32 for decode (M == 1,
  // unchanged) and for a genuinely short prompt (a first chunk, start_pos == 0,
  // unchanged), fp16 for a CONTINUATION chunk whatever its size -- which is
  // what every earlier chunk of that same prefill used. Large prefill chunks
  // are unchanged either way.
  if (M == 1 || (M < 32 && (ROPE_TAIL_CONSISTENT == 0 || start_pos == 0))) {
    float cf = (float)c, sf = (float)s, lof = (float)lo, hif = (float)hi;
    out[write_off + row + k]          = (half)(lof * cf - hif * sf);
    out[write_off + row + k + half_d] = (half)(hif * cf + lof * sf);
  } else {
    out[write_off + row + k]          = lo * c - hi * s;
    out[write_off + row + k + half_d] = hi * c + lo * s;
  }
}
__kernel void scatter_copy_f16(__global const half *in, __global half *out,
                               const int N) {
  int i = get_global_id(0);
  if (i < N) out[i] = in[i];
}
// Row-offset variant: writes into a STABLE base at out[write_off + i] instead of
// baking the destination offset into the pointer. write_off (= cache_index *
// num_heads_KV * head_dim) is a scalar arg, so a recorded decode KV-write can be
// replayed with write_off overridden per token (cl_qcom_recordable_queues can
// override scalar args but NOT SVM pointers). Same address as the offset-baked
// pointer form -> byte-identical output.
__kernel void scatter_copy_f16_row(__global const half *in, __global half *out,
                                   const int N, const int write_off) {
  int i = get_global_id(0);
  if (i < N) out[write_off + i] = in[i];
}
// Decode-step Q/K/V commit in ONE launch: rope_inplace_f16 over Q (in place),
// rope_inplace_f16 over K into the key cache at k_write_off, and the flat V
// copy into the value cache at v_write_off. One work item per (head lane,
// half_d), heads laid out along get_global_id(1) as [Q heads | K heads | V
// heads]; each lane runs the M == 1 (fp32) rotation of rope_inplace_f16 or the
// scatter_copy_f16_row store verbatim, so the bytes written are the same as
// the three launches this replaces.
__kernel void qkv_rope_commit_f16(__global const half *q_in,
                                  __global       half *q_out,
                                  __global const half *k_in,
                                  __global       half *k_out,
                                  __global const half *v_in,
                                  __global       half *v_out,
                                  __global const half *cos_lut,
                                  __global const half *sin_lut,
                                  const int num_heads_q, const int num_heads_kv,
                                  const int half_d, const int start_pos,
                                  const int k_write_off, const int v_write_off) {
  const int h = get_global_id(1);
  const int k = get_global_id(2);
  if (k >= half_d) return;
  const int d = 2 * half_d;
  if (h < num_heads_q + num_heads_kv) {
    const int hh = h < num_heads_q ? h : h - num_heads_q;
    __global const half *in = h < num_heads_q ? q_in : k_in;
    __global half *out = h < num_heads_q ? q_out : k_out;
    const int write_off = h < num_heads_q ? 0 : k_write_off;
    long row = (long)hh * d;
    long lut = (long)start_pos * half_d + k;
    half c = cos_lut[lut];
    half s = sin_lut[lut];
    half lo = in[row + k];
    half hi = in[row + k + half_d];
    float cf = (float)c, sf = (float)s, lof = (float)lo, hif = (float)hi;
    out[write_off + row + k]          = (half)(lof * cf - hif * sf);
    out[write_off + row + k + half_d] = (half)(hif * cf + lof * sf);
  } else {
    const int hv = h - num_heads_q - num_heads_kv;
    if (hv >= num_heads_kv) return;
    const long row = (long)hv * d;
    v_out[v_write_off + row + k] = v_in[row + k];
    v_out[v_write_off + row + k + half_d] = v_in[row + k + half_d];
  }
}
// OHWI K scatter: src concat [t, hKV, d] -> dst OHWI [hKV, max_S, d] at
// (position+t). Feeds the K image2d view (qk_matmul_f16_ohwi_img). Mirrors
// the K-side scatter into the OHWI mirror.
__kernel void k_scatter_ohwi(__global const half *src, __global half *dst,
                             const int M, const int hKV, const int d,
                             const int max_S, const int position,
                             const int src_off) {
  // src_off: read the current token's rotated K from a STABLE
  // base (cache_key) at a SCALAR row offset instead of a per-token slice
  // pointer. Default 0 == src already points at the token (byte-identical).
  int t = get_global_id(0);
  int h = get_global_id(1);
  int x = get_global_id(2);
  if (t >= M || h >= hKV || x >= d) return;
  dst[(long)h * max_S * d + (long)(position + t) * d + x] =
    src[(long)src_off + (long)t * hKV * d + (long)h * d + x];
}
// OHWI-transposed V scatter: src concat [t, hKV, d] -> dst reversed-OHWI
// [hKV, d, max_S] at (position+t). Feeds the V image2d view
// (sv_matmul_f16_ohwi_img): the transposed V-side scatter.
__kernel void v_scatter_ohwi_t(__global const half *src, __global half *dst,
                               const int M, const int hKV, const int d,
                               const int max_S, const int position,
                               const int src_off) {
  // src_off: read the current token's V from a STABLE base
  // (cache_value) at a SCALAR row offset instead of a per-token slice pointer.
  // `position` still offsets only the DEST column. Default 0 == byte-identical.
  int t = get_global_id(0);
  int h = get_global_id(1);
  int x = get_global_id(2);
  if (t >= M || h >= hKV || x >= d) return;
  dst[(long)h * d * max_S + (long)x * max_S + position + t] =
    src[(long)src_off + (long)t * hKV * d + (long)h * d + x];
}
// Inverse gathers (mirror -> concat SVM cache slice): the NNTR_MHA_CLMEM
// mode promotes the OHWI mirrors to the PRIMARY prefill store (no SVM
// side-fill during the prefill window); the host decode/save paths read the
// concat SVM slab, so the boundary syncs it back with these, once.
__kernel void k_gather_ohwi(__global const half *src, __global half *dst,
                            const int M, const int hKV, const int d,
                            const int max_S, const int position) {
  int t = get_global_id(0);
  int h = get_global_id(1);
  int x = get_global_id(2);
  if (t >= M || h >= hKV || x >= d) return;
  dst[(long)t * hKV * d + (long)h * d + x] =
    src[(long)h * max_S * d + (long)(position + t) * d + x];
}
__kernel void v_gather_ohwi_t(__global const half *src, __global half *dst,
                              const int M, const int hKV, const int d,
                              const int max_S, const int position) {
  int t = get_global_id(0);
  int h = get_global_id(1);
  int x = get_global_id(2);
  if (t >= M || h >= hKV || x >= d) return;
  dst[(long)t * hKV * d + (long)h * d + x] =
    src[(long)h * d * max_S + (long)x * max_S + position + t];
}
