// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   unittest_hexkl_kv_quant.cpp
 * @date   06 Aug 2026
 * @brief  Host gtest for hexkl_kv_quant: KV block quantizer, parametric over
 *         i4 and i8, per MHA_HTP_U8_TASKS.md Task 1.
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs.
 *
 * Cross product (144 combinations):
 *   T             = 32, 64, 256
 *   head_dim      = 64, 128
 *   nch           = 1, 8
 *   head          = 0 and nch-1
 *   n_rows_valid  = 1, T-1, T
 *   width         = HEXKL_W_I4, HEXKL_W_I8
 *
 * Note n_rows_valid == T-1 with T=32 gives 31, exercising the partial tail at
 * a 32-aligned boundary minus one, which is the layout that hides bugs.
 *
 * For each combination, checks (a)-(e):
 *   a) ROUND TRIP within theoretical bound, computed from width + row range.
 *   b) COLSUM equals an independent sum (separate loop).
 *   c) TAIL: every element past n_rows_valid is exactly 0, scale exactly 1.0f.
 *   d) RANGE: no value outside the width's range.
 *   e) CONSISTENCY: i4 and i8 agree in sign and in relative order, elementwise.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#include "fp16.h"
#include "htp_backend/hmx/hexkl_kv_quant.h"

namespace {

/* Width's clamp range and scale denominator. Must agree with the C file. */
struct width_props {
  int lo;
  int hi;
  float denom;
};
width_props props_for(hexkl_w_width w) {
  if (w == HEXKL_W_I4) {
    return {-8, 7, 7.0f};
  }
  return {-127, 127, 127.0f};
}

/** Theoretical round-trip bound for one element: scale * (1 + 0.5*ulp) where
 * ulp is the quant step (= scale). Round-to-nearest-ties-away adds at most
 * half a step, so |x - q*scale| <= scale/2. Return as absolute bound and also
 * as relative bound = (scale/2) / |x| when x != 0. */
float roundtrip_abs_bound(float scale) { return scale * 0.5f; }

/** Range-dynamic bound used in (a): max over the block of
 *   max(scale/2, scale/2 / |x|) -- but the test needs a per-element relative
 *   check, so we compute it inline below. */

/** Build the CPU-cache-layout fp16 block for one head, with deterministic
 * softmax-like non-negative content for V and small signed content for K
 * (K can be signed; softmax-like is irrelevant for K but fine).
 *
 * Layout: [kv_position][nch][head_dim] fp16, row r of head h at
 * (r*nch + h)*head_dim.
 *
 * We fill the whole T*nch*head_dim buffer for every kv position up to
 * n_rows_valid; positions past that are not read by the quantizer (verified
 * by the TAIL check). We DO tag positions >= n_rows_valid with a sentinel
 * pattern (0xBEEF) so that if the quantizer ever reads them, the TAIL or
 * RANGE check catches it. */
std::vector<uint16_t> make_block_fp16(uint32_t T, uint32_t head_dim,
                                      uint32_t nch, uint32_t n_rows_valid,
                                      uint32_t head, bool for_v) {
  std::vector<uint16_t> buf((size_t)T * nch * head_dim, 0);
  for (uint32_t r = 0; r < n_rows_valid; ++r) {
    for (uint32_t h = 0; h < nch; ++h) {
      for (uint32_t d = 0; d < head_dim; ++d) {
        /** Small signed pattern with a per-position offset so different kv
         * positions (K, per-N axis) get different scales. Values in ~[-1,1].
         * For V the column axis is head_dim, so vary d too. */
        float v;
        if (for_v) {
          /* non-negative softmax-like, [0, 1], d-varying */
          v = 0.01f + 0.5f * std::sin((float)(r * 7 + d * 3 + head));
          if (v < 0.0f) {
            v = -v;
          }
        } else {
          /* signed, r-varying (per-N scale for Kt) */
          v = 0.3f * std::cos((float)(r * 5 + d * 2 + head)) +
              0.05f * (float)d / (float)head_dim;
        }
        size_t idx = ((size_t)r * nch + h) * head_dim + d;
        buf[idx] = nntrainer::compute_fp32_to_fp16(v);
      }
    }
  }
  /* poison tail so any read leaks into a RANGE violation */
  const uint16_t poison = 0xC001;
  for (uint32_t r = n_rows_valid; r < T; ++r) {
    for (uint32_t h = 0; h < nch; ++h) {
      for (uint32_t d = 0; d < head_dim; ++d) {
        size_t idx = ((size_t)r * nch + h) * head_dim + d;
        buf[idx] = poison;
      }
    }
  }
  return buf;
}

struct shape_cfg {
  uint32_t T;
  uint32_t head_dim;
  uint32_t nch;
  uint32_t head;
  uint32_t n_rows_valid;
};

std::vector<shape_cfg> all_configs() {
  std::vector<uint32_t> Ts = {32, 64, 256};
  std::vector<uint32_t> head_dims = {64, 128};
  std::vector<uint32_t> nchs = {1, 8};
  std::vector<uint32_t> nrv_factors = {1, 0 /*T-1*/, 2 /*T*/};
  /* nrv_factors: 0 -> T-1, 1 -> 1, 2 -> T (encoded to avoid a 3rd lambda) */
  std::vector<shape_cfg> out;
  for (uint32_t T : Ts) {
    for (uint32_t hd : head_dims) {
      for (uint32_t nch : nchs) {
        for (uint32_t head : {0u, nch - 1u}) {
          for (uint32_t f : nrv_factors) {
            uint32_t nrv = (f == 0) ? (T - 1) : (f == 1 ? 1u : T);
            out.push_back({T, hd, nch, head, nrv});
          }
        }
      }
    }
  }
  return out;
}

/* Run one pack and return the outputs for Kt. */
struct packed_kt {
  std::vector<int8_t> rm;
  std::vector<float> scale;
  std::vector<int32_t> colsum;
};
packed_kt pack_kt(const std::vector<uint16_t> &buf, const shape_cfg &c,
                  hexkl_w_width w) {
  packed_kt pk;
  pk.rm.assign((size_t)c.head_dim * c.T, -1);
  pk.scale.assign(c.T, 0.0f);
  pk.colsum.assign(c.T, 0);
  hexkl_kvq_pack_kt_block(buf.data(), c.n_rows_valid, c.T, c.head_dim, c.nch,
                          c.head, w, pk.rm.data(), pk.scale.data(),
                          pk.colsum.data());
  return pk;
}

packed_kt pack_v(const std::vector<uint16_t> &buf, const shape_cfg &c,
                 hexkl_w_width w) {
  packed_kt pv;
  pv.rm.assign((size_t)c.T * c.head_dim, -1);
  pv.scale.assign(c.head_dim, 0.0f);
  pv.colsum.assign(c.head_dim, 0);
  hexkl_kvq_pack_v_block(buf.data(), c.n_rows_valid, c.T, c.head_dim, c.nch,
                         c.head, w, pv.rm.data(), pv.scale.data(),
                         pv.colsum.data());
  return pv;
}

/** Re-float the input values actually consumed by the quantizer, in pack order,
 * for round-trip comparison. Kt: for r in [0,nrv), d in [0,head_dim). */
std::vector<float> input_kt_in_pack_order(const std::vector<uint16_t> &buf,
                                          const shape_cfg &c) {
  std::vector<float> v;
  v.reserve((size_t)c.n_rows_valid * c.head_dim);
  for (uint32_t r = 0; r < c.n_rows_valid; ++r) {
    for (uint32_t d = 0; d < c.head_dim; ++d) {
      size_t idx = ((size_t)r * c.nch + c.head) * c.head_dim + d;
      v.push_back(nntrainer::compute_fp16_to_fp32(buf[idx]));
    }
  }
  return v;
}
/** V: for d in [0,head_dim), t in [0,nrv) -- column-major so each column is
 * contiguous in pack order. */
std::vector<float> input_v_in_pack_order(const std::vector<uint16_t> &buf,
                                         const shape_cfg &c) {
  std::vector<float> v;
  v.reserve((size_t)c.n_rows_valid * c.head_dim);
  for (uint32_t d = 0; d < c.head_dim; ++d) {
    for (uint32_t t = 0; t < c.n_rows_valid; ++t) {
      size_t idx = ((size_t)t * c.nch + c.head) * c.head_dim + d;
      v.push_back(nntrainer::compute_fp16_to_fp32(buf[idx]));
    }
  }
  return v;
}

void check_round_trip_range_tail_colsum(
  const std::vector<int8_t> &rm, const std::vector<float> &scale,
  const std::vector<int32_t> &colsum, const std::vector<float> &input,
  uint32_t N_axis_len, uint32_t N_inner, uint32_t n_axis_valid,
  uint32_t inner_valid, uint32_t input_outer_stride, hexkl_w_width w,
  const std::string &label) {
  width_props p = props_for(w);
  /** Layout invariant for BOTH Kt and V:
   *   out_rm: per-N axis is `n` (length N_axis_len), other axis `inner`
   *   (length N_inner). Element (inner, n) at `inner * N_axis_len + n`.
   *     Kt: out_rm[d][r] at d*T + r        -> N_axis_len=T, N_inner=head_dim.
   *     V:  out_rm[t][d] at t*head_dim + d  -> N_axis_len=head_dim, N_inner=T.
   *   Valid cells:
   *     Kt: n in [0, n_axis_valid=nrv), inner in [0,
   * inner_valid=N_inner=head_dim). V:  n in [0,
   * n_axis_valid=N_axis_len=head_dim), inner in [0, inner_valid=nrv). Tail: Kt:
   * n in [n_axis_valid, N_axis_len)  (kv-position tail). V:  inner in
   * [inner_valid, N_inner)  (row tail). input: packed (n outer, inner inner):
   * input[n*input_outer_stride + inner]. Kt stride=head_dim. V stride=nrv. */
  /* (d) RANGE */
  for (size_t i = 0; i < rm.size(); ++i) {
    int q = rm[i];
    EXPECT_GE(q, p.lo) << label << ": RANGE underflow at index " << i;
    EXPECT_LE(q, p.hi) << label << ": RANGE overflow at index " << i;
  }
  /** (a) ROUND TRIP over the VALID region only. Theoretical bound: absolute
   *   |x - q*scale| <= scale/2 (round-to-nearest-ties-away). "Relative" is just
   *   abs/|x| -- no separate rel bound; the task's "compute the bound from the
   *   width and the row's dynamic range" is exactly scale = amax/denom, so the
   *   bound is scale/2, derived per (n) from that row's amax. Never hardcoded.
   */
  for (uint32_t n = 0; n < n_axis_valid; ++n) {
    for (uint32_t inner = 0; inner < inner_valid; ++inner) {
      size_t rm_idx = (size_t)inner * N_axis_len + n;
      int q = rm[rm_idx];
      float s = scale[n];
      float x = input[(size_t)n * input_outer_stride + inner];
      float dq = (float)q * s;
      float abs_bound = roundtrip_abs_bound(s);
      EXPECT_LE(std::fabs(x - dq), abs_bound + 1e-6f)
        << label << ": ROUND TRIP at n=" << n << " inner=" << inner
        << " x=" << x << " dq=" << dq << " scale=" << s << " q=" << q
        << " abs_bound=" << abs_bound;
    }
  }
  /** (b) COLSUM: independent sum over the FULL column (all N_inner rows,
   *   tail is zero). colsum[n] = sum_inner rm[inner*N_axis_len+n]. */
  for (uint32_t n = 0; n < N_axis_len; ++n) {
    int32_t independent = 0;
    for (uint32_t inner = 0; inner < N_inner; ++inner) {
      independent += rm[(size_t)inner * N_axis_len + n];
    }
    EXPECT_EQ(colsum[n], independent)
      << label << ": COLSUM mismatch at n=" << n;
  }
  /** (c) TAIL on n axis (Kt): n in [n_axis_valid, N_axis_len) zero, scale 1.0f,
   *   colsum 0. */
  for (uint32_t n = n_axis_valid; n < N_axis_len; ++n) {
    EXPECT_EQ(scale[n], 1.0f) << label << ": TAIL n-scale at n=" << n;
    EXPECT_EQ(colsum[n], 0) << label << ": TAIL n-colsum at n=" << n;
    for (uint32_t inner = 0; inner < N_inner; ++inner) {
      EXPECT_EQ(rm[(size_t)inner * N_axis_len + n], 0)
        << label << ": TAIL n-rm at n=" << n << " inner=" << inner;
    }
  }
  /** (c) TAIL on inner axis (V): inner in [inner_valid, N_inner) zero across
   *   all n. V's scale is per-n and all n are valid, so no scale check here. */
  for (uint32_t inner = inner_valid; inner < N_inner; ++inner) {
    for (uint32_t n = 0; n < N_axis_len; ++n) {
      EXPECT_EQ(rm[(size_t)inner * N_axis_len + n], 0)
        << label << ": TAIL inner-rm at inner=" << inner << " n=" << n;
    }
  }
}

void check_consistency(const std::vector<int8_t> &rm_i4,
                       const std::vector<float> &scale_i4,
                       const std::vector<int8_t> &rm_i8,
                       const std::vector<float> &scale_i8, uint32_t N_axis_len,
                       uint32_t N_inner, uint32_t n_axis_valid,
                       uint32_t inner_valid, const std::string &label) {
  /** (e) i4 and i8 agree in SIGN and in relative ORDER on every element.
   *   "i8 is a refinement of i4": i8 has finer granularity, so where i4 TIES
   *   (0) or near-ties, i8 may distinguish in either direction. The real
   *   check is when BOTH widths produce a nonzero distinction:
   *     - sign:  fail only if s4 != 0 && s8 != 0 && s4 != s8.
   *     - order: fail only if d4 != 0 && d8 != 0 && d4 != d8.
   *   Sampled-stride when inner_valid is large so the test stays fast at T=256.
   *   Indexing: rm[inner*N_axis_len + n] for BOTH Kt and V. */
  const uint32_t FULL_PAIR_CAP = 4096;
  bool full_pairs = (inner_valid * (inner_valid - 1) / 2) <= FULL_PAIR_CAP;
  for (uint32_t n = 0; n < n_axis_valid; ++n) {
    for (uint32_t inner = 0; inner < inner_valid; ++inner) {
      size_t idx = (size_t)inner * N_axis_len + n;
      int q4 = rm_i4[idx];
      int q8 = rm_i8[idx];
      int s4 = (q4 > 0) - (q4 < 0);
      int s8 = (q8 > 0) - (q8 < 0);
      if (s4 != 0 && s8 != 0) {
        EXPECT_EQ(s4, s8) << label << ": CONSISTENCY sign at n=" << n
                          << " inner=" << inner << " q4=" << q4 << " q8=" << q8;
      }
    }
    auto order_check = [&](uint32_t a, uint32_t b, const char *tag) {
      size_t ia = (size_t)a * N_axis_len + n;
      size_t ib = (size_t)b * N_axis_len + n;
      int d4 = (rm_i4[ia] > rm_i4[ib]) - (rm_i4[ia] < rm_i4[ib]);
      int d8 = (rm_i8[ia] > rm_i8[ib]) - (rm_i8[ia] < rm_i8[ib]);
      if (d4 != 0 && d8 != 0) {
        EXPECT_EQ(d4, d8) << label << ": CONSISTENCY order " << tag
                          << " at n=" << n << " a=" << a << " b=" << b
                          << " q4a=" << (int)rm_i4[ia]
                          << " q4b=" << (int)rm_i4[ib]
                          << " q8a=" << (int)rm_i8[ia]
                          << " q8b=" << (int)rm_i8[ib];
      }
    };
    if (full_pairs) {
      for (uint32_t a = 0; a < inner_valid; ++a) {
        for (uint32_t b = a + 1; b < inner_valid; ++b) {
          order_check(a, b, "(full)");
        }
      }
    } else {
      /** strided sampling: cover the column at ~sqrt(FULL_PAIR_CAP) stride so
       *   sampled pairs span the full range, not just the head. */
      uint32_t stride = (uint32_t)std::sqrt((double)inner_valid);
      if (stride < 1) {
        stride = 1;
      }
      for (uint32_t a = 0; a < inner_valid; a += stride) {
        for (uint32_t b = a + stride; b < inner_valid; b += stride) {
          order_check(a, b, "(sampled)");
        }
      }
      /* adjacent pairs catch local order swaps */
      for (uint32_t a = 0; a + 1 < inner_valid; a += 1) {
        order_check(a, a + 1, "(adjacent)");
      }
    }
  }
}
} /* namespace */

TEST(hexkl_kv_quant, kt_block_all_configs) {
  auto cfgs = all_configs();
  uint32_t combos = 0;
  for (const auto &c : cfgs) {
    auto buf = make_block_fp16(c.T, c.head_dim, c.nch, c.n_rows_valid, c.head,
                               /*for_v=*/false);
    auto input = input_kt_in_pack_order(buf, c);
    auto p4 = pack_kt(buf, c, HEXKL_W_I4);
    auto p8 = pack_kt(buf, c, HEXKL_W_I8);
    std::string label =
      "KT T=" + std::to_string(c.T) + " hd=" + std::to_string(c.head_dim) +
      " nch=" + std::to_string(c.nch) + " head=" + std::to_string(c.head) +
      " nrv=" + std::to_string(c.n_rows_valid);
    /* Kt: N axis = kv position (T), tail on n. inner = head_dim (full). */
    check_round_trip_range_tail_colsum(
      p4.rm, p4.scale, p4.colsum, input,
      /*N_axis_len=*/c.T, /*N_inner=*/c.head_dim,
      /*n_axis_valid=*/c.n_rows_valid, /*inner_valid=*/c.head_dim,
      /*input_outer_stride=*/c.head_dim, HEXKL_W_I4, label);
    check_round_trip_range_tail_colsum(p8.rm, p8.scale, p8.colsum, input, c.T,
                                       c.head_dim, c.n_rows_valid, c.head_dim,
                                       c.head_dim, HEXKL_W_I8, label);
    check_consistency(p4.rm, p4.scale, p8.rm, p8.scale, c.T, c.head_dim,
                      c.n_rows_valid, c.head_dim, label);
    combos += 2; /* two widths per config */
  }
  /* Cross product: 3 T x 2 hd x 2 nch x 2 head x 3 nrv x 2 width = 144. */
  EXPECT_EQ(combos, 144u) << "KT combos=" << combos;
  std::cerr << "KT combinations ran = " << combos << " (expected 144)"
            << std::endl;
}

TEST(hexkl_kv_quant, v_block_all_configs) {
  auto cfgs = all_configs();
  uint32_t combos = 0;
  for (const auto &c : cfgs) {
    auto buf = make_block_fp16(c.T, c.head_dim, c.nch, c.n_rows_valid, c.head,
                               /*for_v=*/true);
    auto input = input_v_in_pack_order(buf, c);
    auto p4 = pack_v(buf, c, HEXKL_W_I4);
    auto p8 = pack_v(buf, c, HEXKL_W_I8);
    std::string label =
      "V  T=" + std::to_string(c.T) + " hd=" + std::to_string(c.head_dim) +
      " nch=" + std::to_string(c.nch) + " head=" + std::to_string(c.head) +
      " nrv=" + std::to_string(c.n_rows_valid);
    /** V: N axis = head_dim (full, no tail on n). inner = T (rows), tail on
     *   inner in [nrv, T). input stride = nrv (V input packed d-outer). */
    check_round_trip_range_tail_colsum(
      p4.rm, p4.scale, p4.colsum, input,
      /*N_axis_len=*/c.head_dim, /*N_inner=*/c.T,
      /*n_axis_valid=*/c.head_dim, /*inner_valid=*/c.n_rows_valid,
      /*input_outer_stride=*/c.n_rows_valid, HEXKL_W_I4, label);
    check_round_trip_range_tail_colsum(
      p8.rm, p8.scale, p8.colsum, input, c.head_dim, c.T, c.head_dim,
      c.n_rows_valid, c.n_rows_valid, HEXKL_W_I8, label);
    check_consistency(p4.rm, p4.scale, p8.rm, p8.scale, c.head_dim, c.T,
                      c.head_dim, c.n_rows_valid, label);
    combos += 2; /* two widths per config */
  }
  /* Cross product: 3 T x 2 hd x 2 nch x 2 head x 3 nrv x 2 width = 144. */
  EXPECT_EQ(combos, 144u) << "V combos=" << combos;
  std::cerr << "V combinations ran = " << combos << " (expected 144)"
            << std::endl;
}

int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
