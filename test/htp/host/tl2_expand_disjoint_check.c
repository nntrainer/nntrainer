// SPDX-License-Identifier: Apache-2.0
/**
 * @file tl2_expand_disjoint_check.c
 * @brief tl2 expand disjoint check validation.
 * @author NNTrainer contributors
 * @bug No known bugs.
 */
/** Exact production TL2 expansion and caller selection through libnative HVX.
 */
#include "hvx_expand_i2i4.h"
#include "hvx_gemv_u8tl2.h"
#include <assert.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void reference_hvx_tl2_expand_wh96(const uint8_t *, uint32_t, uint32_t,
                                   uint32_t, int, uint8_t *);

static unsigned original_calls, disjoint_calls;
static void original_expand(const uint8_t *src, uint32_t rows, uint32_t cols,
                            uint32_t stride, int mag, uint8_t *dst) {
  ++original_calls;
  hvx_tl2_expand_wh96(src, rows, cols, stride, mag, dst);
}
static void disjoint_expand(const uint8_t *src, uint32_t rows, uint32_t cols,
                            uint32_t stride, int mag, uint8_t *dst) {
  ++disjoint_calls;
  const uintptr_t s = (uintptr_t)src, d = (uintptr_t)dst;
  const size_t sn = (rows + 23u) / 24u * 640u;
  const size_t dn = rows / 32u * stride;
  assert(s + sn <= d || d + dn <= s);
  hvx_tl2_expand_wh96_disjoint(src, rows, cols, stride, mag, dst);
}

/** The runner extracts these complete types/functions without rewriting their
 * bodies. Only names are changed to link both production versions together. */
#include "candidate-unit.inc"
#include "reference-unit.inc"

static uint8_t *allocate(size_t n) {
  void *p = NULL;
  assert(posix_memalign(&p, 128u, n) == 0);
  memset(p, 0xa5, n);
  return p;
}

static void pattern(uint8_t *p, size_t n, unsigned mode) {
  for (size_t i = 0; i < n; ++i) {
    if (mode == 0u)
      p[i] = 0;
    else if (mode == 1u)
      p[i] = 0xffu; /* Invalid magnitudes must retain the original zero map. */
    else if (mode == 2u)
      p[i] = (uint8_t)((i * 37u + i / 128u * 19u) ^ (i >> 3u));
    else
      p[i] = (i % 640u >= 512u) ? 0xffu : 0u; /* Signed zero codes. */
  }
}

static void expect_equal(const void *a, const void *b, size_t n,
                         const char *why) {
  if (memcmp(a, b, n)) {
    fprintf(stderr, "FAIL: %s\n", why);
    abort();
  }
}

/** An independent nibble-layout oracle. It preserves untouched WH columns and
 * row padding, including zero rows/columns. All 16 magnitude codes are covered.
 */
static void oracle(const uint8_t *src, uint32_t rows, uint32_t cols,
                   uint32_t stride, int mag, uint8_t *dst) {
  static const int8_t trits[16][3] = {
    {0, 0, 0},  {0, 0, 1},  {0, 1, -1}, {0, 1, 0}, {0, 1, 1}, {1, -1, -1},
    {1, -1, 0}, {1, -1, 1}, {1, 0, -1}, {1, 0, 0}, {1, 0, 1}, {1, 1, -1},
    {1, 1, 0},  {1, 1, 1},  {0, 0, 0},  {0, 0, 0}};
  for (uint32_t kt = 0; kt < rows / 32u; ++kt)
    for (uint32_t nt = 0; nt < cols / 32u; ++nt) {
      uint8_t *tile = dst + (size_t)kt * stride + nt * 512u;
      memset(tile, 0, 512u);
      for (uint32_t r = 0; r < 32u; ++r) {
        const uint32_t k = kt * 32u + r, group = k % 24u / 3u;
        const uint8_t *block = src + k / 24u * 640u;
        for (uint32_t c = 0; c < 32u; ++c) {
          const uint32_t j = 4u * c + nt;
          const unsigned m =
            (block[group / 2u * 128u + j] >> (4u * (group % 2u))) & 15u;
          const int sign = (block[512u + j] >> group) & 1u;
          const int value = (sign ? -mag : mag) * trits[m][k % 3u];
          tile[r / 8u * 128u + c * 4u + r % 4u] |= (uint8_t)(value & 15)
                                                   << (4u * (r / 4u % 2u));
        }
      }
    }
}

static unsigned primitive_checks(void) {
  const uint32_t row_cases[] = {0u, 32u, 64u, 96u};
  const uint32_t col_cases[] = {0u, 32u, 64u, 96u, 128u};
  const uint32_t strides[] = {2048u, 49152u};
  const int magnitudes[] = {1, 7, 2};
  const uint32_t offsets[] = {0u, 1u, 31u};
  unsigned cases = 0;
  for (unsigned ri = 0; ri < 4u; ++ri)
    for (unsigned ci = 0; ci < 5u; ++ci)
      for (unsigned si = 0; si < 2u; ++si)
        for (unsigned mi = 0; mi < 3u; ++mi)
          for (unsigned oi = 0; oi < 3u; ++oi)
            for (unsigned mode = 0; mode < 4u; ++mode) {
              const uint32_t rows = row_cases[ri], cols = col_cases[ci];
              const uint32_t stride = strides[si], off = offsets[oi];
              const size_t sn = (rows + 23u) / 24u * 640u;
              const size_t dn = (size_t)(rows / 32u) * stride;
              /** malloc's final red zone directly follows the actual source;
               * the GU32 case must not read a fourth native block. */
              uint8_t *sr = malloc(sn + off + 1u);
              uint8_t *src = sr + off + 1u;
              uint8_t *a = allocate(dn + off + 256u);
              uint8_t *b = allocate(dn + off + 256u);
              uint8_t *c = allocate(dn + off + 256u);
              uint8_t *gold = allocate(dn + off + 256u);
              assert(sr);
              pattern(src, sn, mode);
              const size_t begin = 128u + off;
              oracle(src, rows, cols, stride, magnitudes[mi], gold + begin);
              reference_hvx_tl2_expand_wh96(src, rows, cols, stride,
                                            magnitudes[mi], a + begin);
              hvx_tl2_expand_wh96(src, rows, cols, stride, magnitudes[mi],
                                  b + begin);
              hvx_tl2_expand_wh96_disjoint(src, rows, cols, stride,
                                           magnitudes[mi], c + begin);
              expect_equal(a, gold, dn + off + 256u, "reference/oracle");
              expect_equal(b, gold, dn + off + 256u, "generic/oracle");
              expect_equal(c, gold, dn + off + 256u, "disjoint/oracle");
              free(sr);
              free(a);
              free(b);
              free(c);
              free(gold);
              ++cases;
            }
  return cases;
}

static unsigned popcount(uint64_t x) {
  unsigned n = 0;
  for (; x; x &= x - 1u)
    ++n;
  return n;
}

static void stage_in_place(uint8_t *dst, const uint8_t *src, uint32_t k,
                           uint32_t panels, uint32_t stride) {
  const uint32_t pb = (k + 23u) / 24u * 640u;
  for (uint32_t p = 0; p < panels; ++p)
    for (uint32_t g = 0; g < (k + 95u) / 96u; ++g) {
      const uint32_t rows = k - g * 96u < 96u ? k - g * 96u : 96u;
      const uint32_t count = (rows + 23u) / 24u;
      uint8_t *to = dst + (size_t)g * 3u * stride + p * 2048u;
      const uint8_t *from = src + (size_t)p * pb + g * 2560u;
      memcpy(to, from, (count < 2u ? count : 2u) * 640u);
      if (count > 2u)
        memcpy(to + stride, from + 1280u, (count - 2u) * 640u);
    }
}

static unsigned caller_checks(void) {
  const uint32_t shapes[][3] = {{2048u, 24u, 49152u}, {1536u, 16u, 32768u},
                                {2816u, 11u, 22528u}, {704u, 22u, 45056u},
                                {2048u, 24u, 49664u}, {1536u, 16u, 33280u}};
  unsigned cases = 0;
  for (unsigned shape = 0; shape < 6u; ++shape)
    for (unsigned packed = 0; packed < 2u; ++packed)
      for (unsigned mask_mode = 0; mask_mode < 3u; ++mask_mode)
        for (unsigned mode = 0; mode < 2u; ++mode) {
          const uint32_t k = shapes[shape][0], np = shapes[shape][1];
          const uint32_t stride = shapes[shape][2];
          const size_t wh = (size_t)(k / 32u) * stride;
          const size_t sn = (size_t)np * ((k + 23u) / 24u) * 640u;
          const size_t total = wh + sn + 384u;
          uint8_t *a = allocate(total), *b = allocate(total);
          uint8_t *sa = a + wh + 256u, *sb = b + wh + 256u;
          pattern(sa, sn, mode == 0u ? 2u : 1u);
          memcpy(sb, sa, sn);
          if (!packed) {
            stage_in_place(a + 128u, sa, k, np, stride);
            stage_in_place(b + 128u, sb, k, np, stride);
          }
          const uint64_t mask = mask_mode == 0u ? (UINT64_C(1) << np) - 1u
                                : mask_mode == 1u
                                  ? 0u
                                  : UINT64_C(1) | (UINT64_C(1) << (np / 2u)) |
                                      (UINT64_C(1) << (np - 1u));
          reference_expand_ctx old = {0};
          old.native_base = a + 128u;
          old.native_packed = packed ? sa : NULL;
          old.native_K = k;
          old.native_npanels = np;
          old.native_groups = (k + 95u) / 96u;
          old.native_panel_bytes = (k + 23u) / 24u * 640u;
          old.native_panels = mask;
          old.native_jobs = popcount(mask) * old.native_groups;
          old.native_magnitude = mode == 0u ? 1 : 7;
          old.stride = stride;
          candidate_expand_ctx now;
          _Static_assert(sizeof old == sizeof now, "caller context layout");
          memcpy(&now, &old, sizeof now);
          now.native_base = b + 128u;
          now.native_packed = packed ? sb : NULL;
          original_calls = disjoint_calls = 0;
          /** An odd permutation covers all eight production units. The
           * workers' ranges stay disjoint; the last unit may own GU32. */
          for (uint32_t i = 0; i < 8u; ++i) {
            const uint32_t u = (i * 5u) % 8u;
            reference_expand_unit(8u, u, &old);
            candidate_expand_unit(8u, u, &now);
          }
          expect_equal(a, b, total, "production expansion-unit parity");
          const unsigned special = packed && shape < 2u ? old.native_jobs : 0u;
          if (disjoint_calls != special ||
              original_calls != old.native_jobs - special) {
            fprintf(stderr, "FAIL: caller selection shape=%u packed=%u\n",
                    shape, packed);
            abort();
          }
          free(a);
          free(b);
          ++cases;
        }
  return cases;
}

int main(void) {
  const unsigned primitive = primitive_checks(), callers = caller_checks();
  printf("PASS: %u primitive parity/oracle/canary cases; %u actual caller "
         "selection/snapshot/worker-partition cases\n",
         primitive, callers);
  return 0;
}
