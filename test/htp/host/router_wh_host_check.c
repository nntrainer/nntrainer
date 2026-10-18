// SPDX-License-Identifier: Apache-2.0
/**
 * @file router_wh_host_check.c
 * @brief Packed INT4 router against the scalar fused-chain specification.
 * @author Codex <noreply@openai.com>
 * @bug No known bugs.
 */
#include "hvx_m1_ops_f32.h"
#include "m1_ops_det.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t lanes = 1;
void hvx_worker_pool_run(hvx_worker_pool *pool, hvx_worker_pool_func fn,
                         void *ctx, uint32_t units) {
  (void)pool;
  const uint32_t n = lanes < units ? lanes : units;
  // Reverse scheduling also checks that one lane cannot consume another's
  // output.
  for (uint32_t i = n; i > 0; --i)
    fn(n, i - 1, ctx);
}

int main(void) {
  unsigned cases = 0;
  const uint32_t widths[] = {32, 96, 2048};
  for (unsigned wi = 0; wi < 3; ++wi) {
    const uint32_t K = widths[wi];
    for (uint32_t E = 32; E <= 64; E += 32) {
      uint8_t *w = calloc((size_t)K * E / 2 + E * 12, 1);
      float *x = malloc((K + E) * sizeof(float));
      float bias[64], ref[64], sig[64], score[64], rw[4], ow[4], out[66];
      uint32_t rs[4], os[4];
      assert(w && x);
      float *scale = (float *)(w + (size_t)K * E / 2);
      int32_t *offset = (int32_t *)(scale + 2 * E);
      for (unsigned pattern = 0; pattern < 8; ++pattern) {
        for (uint32_t k = 0; k < K; ++k)
          x[k] =
            pattern == 0 ? 0 : sinf((float)(k * 13 + pattern)) * (pattern + 1);
        for (uint32_t e = 0; e < E; ++e) {
          bias[e] = (float)((int)(e % 9) - 4) * .01f;
          scale[e] = .001f * (1 + e % 7);
          offset[e] = (int)(e % 7) - 3;
          float sum = 0;
          for (uint32_t k = 0; k < K; ++k) {
            const int q = (int)((k * 3 + e * 7 + pattern) % 16) - 8;
            const uint32_t r = k % 32, col = e % 32;
            const size_t tile = ((size_t)(k / 32) * (E / 32) + e / 32) * 512;
            const size_t b = tile + (r / 8) * 128 + col * 4 + r % 4;
            const unsigned shift = (r / 4 % 2) * 4;
            w[b] = (w[b] & ~(15u << shift)) | ((q & 15u) << shift);
            sum = fmaf(x[k], (float)(q + offset[e]), sum);
          }
          ref[e] = sum * scale[e];
          m1_router_cpu_sigmoid(ref[e], bias[e], sig + e, score + e);
        }
        m1_router_cpu_pick(sig, score, E, 4, rs, rw);
        for (lanes = 1; lanes <= 8; ++lanes) {
          out[0] = out[E + 1] = 123.5f;
          hvx_router_topk_wh(x, w, bias, K, E, 4, out + 1, os, ow, NULL);
          assert(memcmp(ref, out + 1, E * sizeof(float)) == 0);
          assert(memcmp(rs, os, sizeof(rs)) == 0);
          assert(memcmp(rw, ow, sizeof(rw)) == 0);
          assert(out[0] == 123.5f && out[E + 1] == 123.5f);
          ++cases;
        }
        // Resident graph permits logits to overwrite the router input.
        hvx_router_topk_wh(x, w, bias, K, E, 4, x, os, ow, NULL);
        assert(memcmp(ref, x, E * sizeof(float)) == 0);
        ++cases;
      }
      free(x);
      free(w);
    }
  }
  printf("PACKED ROUTER: %u cases, logits/top-k/weights bit-identical\n",
         cases);
  return 0;
}
