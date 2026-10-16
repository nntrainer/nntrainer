// SPDX-License-Identifier: Apache-2.0
/**
 * @file tl2_lut_check.c
 * @brief Independent ternary GEMV reference, tail/guard checks and DSP probe.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include "hvx_gemv_u8tl2.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef NNTR_TL2_DEVICE
#include "nntr_hvx.h"
#include <AEEStdErr.h>
#include <remote.h>
static remote_handle64 session;
#endif

static uint32_t state = 389123u;
static FILE *fixture;
static uint32_t random_u32(void) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

static uint32_t read_le32(const uint8_t *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
         (uint32_t)p[3] << 24;
}

static void *buffer(size_t bytes) {
  void *p = NULL;
  assert(!posix_memalign(&p, 128u, bytes));
  assert(p);
  memset(p, 0, bytes);
  return p;
}

/** Check every LUT entry independently of GEMV, including unaligned inputs,
 * the bounded vector-load transition, and disjoint worker partitions. */
static void check_prepare(void) {
  unsigned cases = 0;
  for (uint32_t shape = 1; shape <= 148u; ++shape) {
    const uint32_t K = shape <= 145u   ? shape
                       : shape == 146u ? 704u
                       : shape == 147u ? 2816u
                                       : 8192u;
    const uint32_t blocks = (K + 23u) / 24u;
    const size_t bytes = hvx_tl2_table_bytes(K) + 256u;
    uint8_t *source = buffer(K + 1u);
    int16_t *got = buffer(bytes), *want = buffer(bytes);
    for (uint32_t k = 0; k <= K; ++k)
      source[k] = (uint8_t)(k * 37u + 19u);
    for (uint32_t unaligned = 0; unaligned < 2u; ++unaligned)
      for (uint32_t partition = 0; partition < 2u; ++partition) {
        const uint32_t end = partition ? blocks - 1u : blocks;
        const uint32_t begin = partition && end ? 1u : 0u;
        const uint8_t *a = source + unaligned;
        memset(got, 0xa5, bytes);
        memset(want, 0xa5, bytes);
        for (uint32_t kb = begin; kb < end; ++kb) {
          int16_t *table = want + 64u + kb * 256u;
          memset(table, 0, 512u);
          for (uint32_t group = 0; group < 8u; ++group) {
            const uint32_t k = kb * 24u + group * 3u;
            const int a0 = k < K ? a[k] : 0;
            const int a1 = k + 1u < K ? a[k + 1u] : 0;
            const int a2 = k + 2u < K ? a[k + 2u] : 0;
            for (int x = -1; x <= 1; ++x)
              for (int y = -1; y <= 1; ++y)
                for (int z = -1; z <= 1; ++z) {
                  const int magnitude = 9 * x + 3 * y + z;
                  if (magnitude < 0)
                    continue;
                  const uint32_t index = group * 32u + 2u * magnitude;
                  table[index] = table[index + 1u] =
                    (int16_t)(a0 * x + a1 * y + a2 * z);
                }
          }
        }
        hvx_tl2_prepare(a, K, begin, end, got + 64u);
        assert(!memcmp(got, want, bytes));
        ++cases;
      }
    free(source);
    free(got);
    free(want);
  }
  printf("TL2 prepare: %u exact LUT/partition/guard cases passed\n", cases);
}

#ifdef NNTR_TL2_DEVICE
static void check_rpc_validation(void) {
  uint8_t a = 17, packed[640] = {0};
  int32_t out = -1;
  uint32_t stats[6];
  /** AEEStdErr.h uses a different offset under __hexagon__. FastRPC
   * propagates that DSP result unchanged to this ARM client. */
  const uint32_t dsp_badparm = 0x80000400u | AEE_EBADPARM;
  /** Zero weights are canonical even in the padded part of this tiny panel.
   * Exercise direct RPC buffers as well as the aligned benchmark path. */
  for (uint32_t dma = 0; dma <= 256u; dma += 256u) {
    assert((uint32_t)nntr_hvx_tl2_gemv_probe(session, 1, 1, dma, 1, 1, &a, 1,
                                             packed, 640, &out, 1, stats,
                                             6) == AEE_SUCCESS);
    assert(out == 0);
  }
  assert((uint32_t)nntr_hvx_tl2_gemv_probe(session, 1, 1, 4, 1, 1, &a, 1,
                                           packed, 640, &out, 1, stats,
                                           6) == dsp_badparm);
  assert((uint32_t)nntr_hvx_tl2_gemv_probe(session, 1, 1, 0, 0, 1, &a, 1,
                                           packed, 640, &out, 1, stats,
                                           6) == dsp_badparm);
  assert((uint32_t)nntr_hvx_tl2_gemv_probe(session, 1, 1, 0, 1, 1, &a, 1,
                                           packed, 639, &out, 1, stats,
                                           6) == dsp_badparm);
  packed[0] = 14;
  assert((uint32_t)nntr_hvx_tl2_gemv_probe(session, 1, 1, 0, 1, 1, &a, 1,
                                           packed, 640, &out, 1, stats,
                                           6) == dsp_badparm);
  packed[0] = 0;
  packed[512] = 1;
  assert((uint32_t)nntr_hvx_tl2_gemv_probe(session, 1, 1, 0, 1, 1, &a, 1,
                                           packed, 640, &out, 1, stats,
                                           6) == dsp_badparm);
  packed[512] = 0;
  packed[0] = 1;
  assert((uint32_t)nntr_hvx_tl2_gemv_probe(session, 1, 1, 0, 1, 1, &a, 1,
                                           packed, 640, &out, 1, stats,
                                           6) == dsp_badparm);
  puts("TL2 RPC: direct DMA and malformed argument/encoding checks passed");
}

#endif

static void check(uint32_t K, uint32_t N, uint32_t pattern, uint32_t passes) {
  const uint32_t npad = (N + 127u) & ~127u;
  const size_t bytes = hvx_tl2_weight_bytes(K, N);
  int8_t *w = buffer((size_t)K * N), *back = buffer((size_t)K * N);
  uint8_t *a = buffer(K), *packed = buffer(bytes + 128u);
  int16_t *tables = buffer(hvx_tl2_table_bytes(K) + 128u);
  int32_t *ref = buffer(npad * 4u), *out = buffer(npad * 4u + 128u);
  memset(packed + bytes, 0xa5, 128u);
  memset((uint8_t *)tables + hvx_tl2_table_bytes(K), 0xa5, 128u);
  memset(out + npad, 0xa5, 128u);
  if (pattern == 4u)
    assert(fixture && fread(w, 1u, (size_t)K * N, fixture) == (size_t)K * N);
  for (uint32_t k = 0; k < K; ++k) {
    a[k] = pattern && pattern < 4u ? 255u : random_u32() & 255u;
    for (uint32_t n = 0; n < N; ++n) {
      int v = pattern == 4u   ? w[(size_t)k * N + n]
              : pattern == 1u ? 1
              : pattern == 2u ? -1
              : pattern == 3u ? 0
                              : (int)(random_u32() % 3u) - 1;
      w[(size_t)k * N + n] = (int8_t)v;
      ref[n] += (int32_t)a[k] * v;
    }
  }
  assert(hvx_tl2_pack(w, K, N, packed) == 0);
  assert(hvx_tl2_unpack(packed, K, N, back) == 0);
  assert(!memcmp(w, back, (size_t)K * N));
  hvx_tl2_prepare(a, K, 0u, (K + 23u) / 24u, tables);
  for (uint32_t p = 0; p < npad / 128u; ++p)
    hvx_gemv_u8tl2_panel(packed + (size_t)p * ((K + 23u) / 24u) * 640u, K,
                         tables, out + p * 128u);
  assert(!memcmp(ref, out, npad * 4u));
  for (uint32_t i = 0; i < 128u; ++i) {
    assert(packed[bytes + i] == 0xa5);
    assert(((uint8_t *)tables)[hvx_tl2_table_bytes(K) + i] == 0xa5);
    assert(((uint8_t *)(out + npad))[i] == 0xa5);
  }
  uint8_t save = packed[0];
  packed[0] = (save & 240u) | 14u;
  assert(hvx_tl2_unpack(packed, K, N, back) == -1);
  packed[0] = save;
#ifdef NNTR_TL2_DEVICE
  for (uint32_t workers = 1; workers <= 8u; workers += 7u)
    for (uint32_t dma = 0; dma <= 256u; dma += 256u)
      for (uint32_t mode = 0; mode <= 3u; ++mode) {
        uint32_t stats[6];
        int rc = nntr_hvx_tl2_gemv_probe(session, K, N, dma | mode | 512u,
                                         workers, passes, a, K, packed,
                                         (int)bytes, out, N, stats, 6);
        if (rc) {
          fprintf(stderr, "TL2 probe K=%u N=%u rc=%#x\n", K, N, rc);
          abort();
        }
        for (uint32_t n = 0; n < N; ++n)
          if (out[n] != ref[n]) {
            fprintf(stderr, "TL2 mismatch K=%u N=%u n=%u got=%d want=%d\n", K,
                    N, n, out[n], ref[n]);
            abort();
          }
        printf("TL2 K=%u N=%u pattern=%u mode=%u requested_workers=%u "
               "workers=%u dma=%u aligned=1 passes=%u bytes=%zu dma_us=%.3f "
               "prep_us=%.3f "
               "gemv_us=%.3f total_us=%.3f exact=1\n",
               K, N, pattern, mode, workers, stats[5], !!dma, passes, bytes,
               (double)stats[0] / passes, (double)stats[2] / passes,
               (double)stats[3] / passes, (double)stats[4] / passes);
        fflush(stdout);
      }
#else
  (void)passes;
  printf("TL2 exact K=%u N=%u pattern=%u bytes=%zu\n", K, N, pattern, bytes);
#endif
  free(w);
  free(back);
  free(a);
  free(packed);
  free(tables);
  free(ref);
  free(out);
}

int main(int argc, char **argv) {
  const unsigned passes = argc > 1 ? (unsigned)strtoul(argv[1], NULL, 10) : 20u;
  check_prepare();
#ifdef NNTR_TL2_DEVICE
  struct remote_rpc_control_unsigned_module u = {CDSP_DOMAIN_ID, 1};
  assert(remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &u,
                                sizeof(u)) == AEE_SUCCESS);
  assert(nntr_hvx_open(nntr_hvx_URI "&_dom=cdsp", &session) == AEE_SUCCESS);
#endif
  check(1, 1, 0, passes);
  check(25, 129, 0, passes);
  for (unsigned p = 0; p < 4u; ++p)
    check(96, 256, p, passes);
  check(704, 2816, 0, passes);
  check(2816, 1408, 0, passes);
  check(8192, 128, 1, passes);
  check(8192, 128, 2, passes);
  int8_t bad[3] = {-1, 2, 1};
  uint8_t buf[640] = {0};
  assert(hvx_tl2_pack(bad, 3, 1, buf) == -1);
  assert(hvx_tl2_pack(bad, 0, 1, buf) == -1);
  assert(hvx_tl2_weight_bytes(UINT32_MAX, 1) == 0);
  assert(hvx_tl2_weight_bytes(1, UINT32_MAX) == 0);
  assert(hvx_tl2_table_bytes(UINT32_MAX) == 0);
  buf[512] = 1;
  assert(hvx_tl2_unpack(buf, 3, 1, bad) == -1);
  buf[512] = 0;
  buf[0] = 1;
  assert(hvx_tl2_unpack(buf, 1, 1, bad) == -1);
  /** Optional LE fixture: TL2SYM1\0, then repeated u32 K,N + KN int8
   * symmetric weights. This is a test interchange format, not a model. */
  if (argc > 2) {
    fixture = fopen(argv[2], "rb");
    assert(fixture);
    char magic[8];
    assert(fread(magic, 1, 8, fixture) == 8 && !memcmp(magic, "TL2SYM1", 8));
    uint8_t shape_bytes[8];
    size_t count;
    while ((count = fread(shape_bytes, 1, 8, fixture)) != 0) {
      assert(count == 8);
      const uint32_t shape[2] = {read_le32(shape_bytes),
                                 read_le32(shape_bytes + 4)};
      assert(shape[0] && shape[0] <= 8192 && shape[1] && shape[1] <= 8192);
      check(shape[0], shape[1], 4, passes);
    }
    assert(feof(fixture));
    fclose(fixture);
    fixture = NULL;
  }
#ifdef NNTR_TL2_DEVICE
  check_rpc_validation();
  assert(nntr_hvx_close(session) == AEE_SUCCESS);
#endif
  puts("TL2 LUT: all independent arithmetic, layout, padding and guard checks "
       "passed");
  return 0;
}
