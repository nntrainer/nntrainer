// SPDX-License-Identifier: Apache-2.0
/** @file wh_affine_host_check.c
 * @brief Affine FC and dense FFN must equal independently packed q+offset
 * weights. Nonzero offsets, real column sums, row permutations, row-block
 * boundaries, gate/up before GELU and down after GELU are all exercised.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug See docs/gemma4-tl2-affine-handoff.md for validation limits.
 */
#define main nntr_existing_moe_check_main
#include "moe_layer_host_check.c"
#undef main
#include "fc_wh_det.h"
#include "hexkl_mm_opts.h"
#include "hexkl_wh_affine.h"
#include <assert.h>

/* These optional paths are not requested by this check. */
void hvx_rmsnorm_add_row_acc_f32(float *acc, const float *seg, uint32_t n,
                                 int block_start, int block_end, int first) {
  (void)acc;
  (void)seg;
  (void)n;
  (void)block_start;
  (void)block_end;
  (void)first;
  abort();
}
void hvx_rmsnorm_add_row_finish_f32(float *out, const float *res,
                                    const float *x, const float *gamma,
                                    uint32_t n, float eps, float scale,
                                    const float *acc) {
  (void)out;
  (void)res;
  (void)x;
  (void)gamma;
  (void)acc;
  (void)n;
  (void)eps;
  (void)scale;
  abort();
}
int hexkl_micro_hmx_copy_32b_to_submatrix(uint8_t *vtcm, uint32_t off,
                                          int32_t *dst, uint32_t rb,
                                          uint32_t nt, uint32_t rows,
                                          uint32_t cols) {
  (void)vtcm;
  (void)off;
  (void)dst;
  (void)rb;
  (void)nt;
  (void)rows;
  (void)cols;
  abort();
}

static void put_wh(uint8_t *wh, uint32_t N, uint32_t k, uint32_t n, int q) {
  const size_t at = ((size_t)(k / 32u) * (N / 32u) + n / 32u) * 512u +
                    k % 32u / 8u * 128u + n % 32u * 4u + k % 4u;
  const uint32_t shift = 4u * (k / 4u % 2u);
  wh[at] = (wh[at] & ~(15u << shift)) | (((uint32_t)q & 15u) << shift);
}

/** Independent fixed-grid rounding, including saturation and odd zero points.
 */
static void static_quant_ref(const float *x, uint32_t n, float scale,
                             int32_t zp, uint8_t *q) {
  for (uint32_t k = 0; k < n; ++k) {
    long v = lrintf(x[k] / scale) + zp;
    q[k] = v < 0 ? 0 : v > 255 ? 255 : (uint8_t)v;
  }
}

static void static_linear_cases(uint8_t *vtcm, uint32_t vtcm_size) {
  enum { K = 96, I = 64, N = 32, ROWS = 129 };
  W w[4];
  memset(&g_tbl, 0, sizeof(g_tbl));
  make_weight(0, K, 2 * I, &w[0]);
  make_weight(1, I, N, &w[1]);
  make_weight(2, K, I, &w[2]);
  make_weight(3, K, I, &w[3]);
  const float scales[] = {.125f, .03125f, .125f, .0625f};
  const int32_t zps[] = {171, 142, 171, 93};
  for (unsigned h = 0; h < 4; ++h) {
    g_tbl.slots[h].static_scale = scales[h];
    g_tbl.slots[h].static_zp = zps[h];
    for (uint32_t c = 0; c < w[h].N; ++c) {
      w[h].cs[c] = 0;
      for (uint32_t k = 0; k < w[h].K; ++k) {
        const uint8_t *tile =
          (uint8_t *)w[h].nib +
          ((size_t)(k / 32u) * (w[h].N / 32u) + c / 32u) * 512u;
        w[h].cs[c] += wh_value(tile, k % 32u, c % 32u);
      }
    }
  }
  float x[ROWS * K], got[ROWS * 2 * I], want[ROWS * 2 * I];
  uint32_t rows[ROWS];
  float routing[ROWS];
  for (unsigned j = 0; j < ROWS * K; ++j)
    x[j] = (int)(j % 41u) * .0625f - 1.25f;
  x[0] = -100.f;
  x[K - 1] = 100.f;
  hexkl_moe_scratch scratch = {0};
  const uint32_t sizes[] = {1, 3, 64, 65, 129};
  for (unsigned s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s) {
    const uint32_t M = sizes[s];
    for (unsigned workers = 0; workers <= 3; workers += 3) {
      g_workers = workers;
      const uint32_t handles[] = {2, 3};
      for (unsigned p = 0; p < 2; ++p)
        for (unsigned m = 0; m < M; ++m) {
          uint8_t q[K];
          static_quant_ref(x + m * K, K, scales[p + 2], zps[p + 2], q);
          ref_mm(w + p + 2, q, scales[p + 2], zps[p + 2],
                 want + p * M * I + m * I);
        }
      if (M == 1)
        assert(hexkl_mm_u8i4_fc_m1_run(&g_tbl, vtcm, 0, 0, K, 2, handles, x,
                                       got, NULL, &scratch,
                                       HEXKL_MOE_FLAG_GEMV_FEED_SET) == 0);
      else
        assert(hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, vtcm_size, vtcm_size, M, K,
                                       handles, 2, x, got, NULL) == 0);
      assert(!memcmp(want, got, (size_t)M * 2 * I * sizeof(float)));

      // Dense gate/up have different grids; routed experts share the input
      // grid but quantize their own post-GELU down input.
      for (unsigned dense = 0; dense < 2; ++dense) {
        g_tbl.slots[0].static_gate_plus1 = dense ? 3 : 0;
        g_tbl.slots[0].static_up_plus1 = dense ? 4 : 0;
        for (unsigned m = 0; m < M; ++m) {
          uint8_t q[K];
          float gu[2 * I], mid[I], output[N];
          static_quant_ref(x + m * K, K, scales[0], zps[0], q);
          if (dense) {
            ref_mm(w + 2, q, scales[2], zps[2], gu);
            static_quant_ref(x + m * K, K, scales[3], zps[3], q);
            ref_mm(w + 3, q, scales[3], zps[3], gu + I);
          } else
            ref_mm(w, q, scales[0], zps[0], gu);
          for (unsigned j = 0; j < I; ++j)
            mid[j] = geglu_det_one(gu[j], gu[I + j]);
          static_quant_ref(mid, I, scales[1], zps[1], q);
          ref_mm(w + 1, q, scales[1], zps[1], output);
          rows[m] = dense ? m : M - 1u - m;
          routing[m] = dense ? 1.f : .75f;
          for (unsigned j = 0; j < N; ++j)
            want[m * N + j] = output[j] * routing[m];
        }
        const uint32_t gu = 0, dn = 1;
        assert(hexkl_mm_u8i4_moe_layer_run(
                 &g_tbl, vtcm, M == 1 && dense ? 0 : vtcm_size,
                 M == 1 && dense ? 0 : vtcm_size, M, K, I, N, 1, &gu, &dn, rows,
                 &M, routing, x, got, NULL, &scratch,
                 HEXKL_MOE_FLAG_GELU_TANH | HEXKL_MOE_FLAG_M1_GEMV |
                   HEXKL_MOE_FLAG_GEMV_FEED_SET) == 0);
        assert(!memcmp(want, got, (size_t)M * N * sizeof(float)));
        if (dense) {
          /** Gemma's decode graph uses slot 1 for both the dense input and
           * output. Every chunk must still read the original input. */
          float in_place[ROWS * K];
          uint32_t chunk_rows[2 * ROWS], counts[] = {M, M};
          float chunk_weights[2 * ROWS];
          const uint32_t gu_chunks[] = {gu, gu}, dn_chunks[] = {dn, dn};
          for (unsigned j = 0; j < 2 * M; ++j) {
            chunk_rows[j] = j % M;
            chunk_weights[j] = 1.f;
          }
          /** 28 KiB fits only 64 gate/up rows and 128 down rows beside
           * the weight/accumulator double buffers. Exercise both arena
           * limits, multiple row tiles and a one-row prefill tail. */
          const uint32_t budgets[][2] = {
            {vtcm_size, vtcm_size},
            {28u << 10, vtcm_size},
            {vtcm_size, 28u << 10},
          };
          for (unsigned b = 0; b < sizeof(budgets) / sizeof(budgets[0]); ++b)
            for (unsigned chunks = 1; chunks <= 2; ++chunks) {
              memcpy(in_place, x, (size_t)M * K * sizeof(float));
              assert(hexkl_mm_u8i4_moe_layer_run(
                       &g_tbl, vtcm, M == 1 ? 0 : budgets[b][0],
                       M == 1 ? 0 : budgets[b][1], M, K, I, N, chunks,
                       gu_chunks, dn_chunks, chunk_rows, counts, chunk_weights,
                       in_place, in_place, NULL, &scratch,
                       HEXKL_MOE_FLAG_GELU_TANH | HEXKL_MOE_FLAG_M1_GEMV |
                         HEXKL_MOE_FLAG_GEMV_FEED_SET) == 0);
              for (unsigned j = 0; j < M * N; ++j)
                assert(in_place[j] == chunks * want[j]);
            }
        }
      }
      printf("Static FC/dense/MoE exact M=%u workers=%u\n", M, workers);
    }
  }
  hexkl_moe_scratch_free(&scratch);
  for (unsigned h = 0; h < 4; ++h) {
    free(w[h].nib);
    free(w[h].ws);
    free(w[h].cs);
    free(w[h].bias);
  }
}

int main(void) {
  const uint32_t K = 96u, I = 192u, N = 128u;
  static uint8_t vtcm[8u << 20] __attribute__((aligned(128)));
  {
    float arrays[4u * 32u] __attribute__((aligned(128)));
    float offsets[32] = {0};
    hexkl_weight_u8i4 w = {0};
    w.in_use = 1;
    w.bits = 4;
    w.N = 32;
    w.arrays = arrays;
    w.w_scale = arrays;
    offsets[0] = NAN;
    assert(hexkl_weight_u8i4_set_offsets(&w, offsets) != 0);
    assert(w.w_offset == NULL);
    offsets[0] = .5f;
    assert(hexkl_weight_u8i4_set_offsets(&w, offsets) != 0);
    offsets[0] = 16.f;
    assert(hexkl_weight_u8i4_set_offsets(&w, offsets) != 0);
    offsets[0] = -6.f;
    assert(hexkl_weight_u8i4_set_offsets(&w, offsets) == 0);
    assert(w.w_offset[0] == -6 && w.w_offset[31] == 0);
  }
  W weights[4];
  int32_t offset[2][384];
  for (uint32_t h = 0; h < 2u; ++h) {
    const uint32_t k = h ? I : K, n = h ? N : 2u * I;
    make_weight(h, k, n, &weights[h]);
    make_weight(h + 2u, k, n, &weights[h + 2u]);
    for (uint32_t c = 0; c < n; ++c) {
      offset[h][c] = (int)(c % 11u) - 5;
      weights[h].cs[c] = weights[h + 2u].cs[c] = 0;
      weights[h + 2u].ws[c] = weights[h].ws[c];
      weights[h + 2u].bias[c] = weights[h].bias[c] = 0;
      for (uint32_t r = 0; r < k; ++r) {
        const int q = (int)(rnd() % 5u) - 2;
        put_wh((uint8_t *)weights[h].nib, n, r, c, q);
        put_wh((uint8_t *)weights[h + 2u].nib, n, r, c, q + offset[h][c]);
        weights[h].cs[c] += q;
        weights[h + 2u].cs[c] += q + offset[h][c];
      }
    }
    g_tbl.slots[h].w_offset = offset[h];
  }
  float input[65u * 96u], want[65u * 384u], got[65u * 384u];
  uint32_t rows[65];
  float routing[65];
  for (uint32_t r = 0; r < 65u; ++r) {
    rows[r] = 64u - r;
    routing[r] = .75f;
    for (uint32_t k = 0; k < K; ++k)
      input[r * K + k] = rndf();
  }
  hexkl_moe_scratch scratch = {0};
  for (uint32_t M = 1u; M <= 65u; M = M < 4u    ? M + 1u
                                      : M == 4u ? 64u
                                                : M + 1u) {
    for (uint32_t r = 0; r < M; ++r)
      rows[r] = M - 1u - r;
    for (uint32_t workers = 0; workers <= 3u; workers += 3u) {
      g_workers = workers;
      for (uint32_t mode = 0; mode < 2u; ++mode) {
        uint32_t gu = mode ? 0u : 2u, dn = mode ? 1u : 3u;
        int rc = hexkl_mm_u8i4_moe_layer_run(
          &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K, I, N, 1u, &gu, &dn,
          rows, &M, routing, input, mode ? got : want, NULL, &scratch,
          HEXKL_MOE_FLAG_GELU_TANH);
        assert(rc == 0);
      }
      assert(memcmp(want, got, (size_t)M * N * sizeof(float)) == 0);
      printf("Affine WH dense FFN exact M=%u workers=%u\n", M, workers);
      if (M <= 4u) {
        const uint32_t flags[] = {
          HEXKL_MOE_FLAG_M1_GEMV | HEXKL_MOE_FLAG_GEMV_FEED_SET, 0xcf03e1u};
        for (uint32_t f = 0; f < sizeof(flags) / sizeof(flags[0]); ++f) {
          uint32_t gu = 0u, dn = 1u;
          assert(hexkl_mm_u8i4_moe_layer_run(
                   &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K, I, N, 1u, &gu,
                   &dn, rows, &M, routing, input, got, NULL, &scratch,
                   flags[f] | HEXKL_MOE_FLAG_GELU_TANH) == 0);
          /** Affine decode must use GEMV, including on the DSP queue thread
           * that does not own the FastRPC caller's HMX lock. */
          assert(hexkl_probe_us[HEXKL_PROBE_PATH] == 1u);
          assert(memcmp(want, got, (size_t)M * N * sizeof(float)) == 0);
          printf("Affine WH GEMV equals HMX M=%u workers=%u flags=0x%x\n", M,
                 workers, flags[f]);
        }
      }
      for (uint32_t mode = 0; mode < 2u; ++mode) {
        uint32_t h = mode ? 0u : 2u;
        hexkl_mm_opts opts = {0};
        int rc =
          hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K,
                                  &h, 1u, input, mode ? got : want, &opts);
        assert(rc == 0);
      }
      assert(memcmp(want, got, (size_t)M * 2u * I * sizeof(float)) == 0);
      printf("Affine WH FC prefill exact M=%u workers=%u\n", M, workers);
    }
  }
  {
    /** Shared activation sums span both row blocks and survive across
     * handles. A non-affine handle between two affine ones uses the same
     * activation without an offset correction. */
    static float multi_want[3u * 65u * 384u], multi_got[3u * 65u * 384u];
    const uint32_t ref_handles[] = {2u, 2u, 2u};
    const uint32_t handles[] = {0u, 2u, 0u};
    hexkl_mm_opts opts = {0};
    assert(hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, 65u,
                                   K, ref_handles, 3u, input, multi_want,
                                   &opts) == 0);
    assert(hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, 65u,
                                   K, handles, 3u, input, multi_got,
                                   &opts) == 0);
    assert(!memcmp(multi_want, multi_got, sizeof multi_want));
    printf("Affine WH FC mixed handles exact M=65 handles=3\n");
  }
  for (uint32_t feed = 0; feed < 2u; ++feed) {
    for (uint32_t mode = 0; mode < 2u; ++mode) {
      uint32_t h = mode ? 0u : 2u;
      int rc = hexkl_mm_u8i4_fc_m1_run(
        &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, K, 1u, &h, input,
        mode ? got : want, NULL, &scratch,
        HEXKL_MOE_FLAG_GEMV_FEED_SET | (feed ? HEXKL_MOE_FLAG_GEMV_FEED : 0u));
      assert(rc == 0);
    }
    assert(memcmp(want, got, (size_t)2u * I * sizeof(float)) == 0);
    printf("Affine WH FC decode exact feed=%u\n", feed);
  }
  // The actual issue in the source checkpoint: effective integer values
  // beyond signed int4. Compare against a direct integer dot, without
  // attempting to pack those effective values into a reference int4 tensor.
  for (uint32_t c = 0; c < 2u * I; ++c) {
    offset[0][c] = c % 2u ? -6 : 6;
    weights[0].cs[c] = 0;
    for (uint32_t k = 0; k < K; ++k) {
      const int q = (int)(rnd() % 16u) - 8;
      put_wh((uint8_t *)weights[0].nib, 2u * I, k, c, q);
      weights[0].cs[c] += q;
    }
  }
  uint8_t aq[96];
  float scale;
  int32_t zp;
  fc_wh_quant_row_det(input, K, aq, &scale, &zp);
  for (uint32_t c = 0; c < 2u * I; ++c) {
    int32_t sum = 0;
    for (uint32_t k = 0; k < K; ++k) {
      const uint8_t *tile =
        (uint8_t *)weights[0].nib +
        ((size_t)(k / 32u) * (2u * I / 32u) + c / 32u) * 512u;
      sum += ((int32_t)aq[k] - zp) *
             (wh_value(tile, k % 32u, c % 32u) + offset[0][c]);
    }
    want[c] = (float)sum * scale * weights[0].ws[c];
  }
  uint32_t h = 0;
  assert(hexkl_mm_u8i4_fc_m1_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, K, 1u,
                                 &h, input, got, NULL, &scratch,
                                 HEXKL_MOE_FLAG_GEMV_FEED_SET) == 0);
  assert(memcmp(want, got, (size_t)2u * I * sizeof(float)) == 0);
  printf("Affine WH effective weights outside signed int4: direct integer "
         "reference exact\n");
  hexkl_moe_scratch_free(&scratch);
  for (uint32_t h = 0; h < 4u; ++h) {
    free(weights[h].nib);
    free(weights[h].ws);
    free(weights[h].cs);
    free(weights[h].bias);
  }
  static_linear_cases(vtcm, sizeof(vtcm));
  return 0;
}
