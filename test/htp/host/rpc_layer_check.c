// SPDX-License-Identifier: Apache-2.0
/** @file rpc_layer_check.c
 * @brief Timed/untimed dispatch, validation and profiling without DSP
 * arithmetic.
 * @author NNTrainer contributors
 * @bug Kernels are stand-ins; device tests cover numerical results.
 */
#include "hexkl_mm_u8i8_dma.h"
#include "hexkl_probe.h"
#include "htp_profile_wire.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static unsigned calls;
static int result;
static int compute(uint32_t M, uint32_t K, uint32_t count, const float *in,
                   float *out, const hexkl_mm_opts *opts) {
  assert(M == 1 && K == 32 && count == 1 && opts->pool == NULL);
  ++calls;
  memcpy(out, in, 32 * sizeof(float));
  if (hexkl_probe_on)
    hexkl_probe_us[HEXKL_PROBE_QUANT] = 17;
  return result;
}

int hexkl_mm_u8i4_layer_run(hexkl_weight_u8i4_table *tbl, uint8_t *vtcm,
                            uint32_t bytes, uint32_t off, uint32_t M,
                            uint32_t K, const uint32_t *handles, uint32_t count,
                            const float *in, float *out,
                            const hexkl_mm_opts *opts) {
  (void)tbl;
  (void)vtcm;
  (void)bytes;
  (void)off;
  (void)handles;
  return compute(M, K, count, in, out, opts);
}
int hexkl_mm_u8i8_layer_run(hexkl_weight_u8i8_table *tbl, uint8_t *vtcm,
                            uint32_t bytes, uint32_t off, uint32_t M,
                            uint32_t K, const uint32_t *handles, uint32_t count,
                            const float *in, float *out,
                            const hexkl_mm_opts *opts) {
  (void)tbl;
  (void)vtcm;
  (void)bytes;
  (void)off;
  (void)handles;
  return compute(M, K, count, in, out, opts);
}

int main(void) {
  nntr_hvx_session session = {0};
  session.hmx_locked = 1;
  session.weights_u8i4.slots[0].in_use = 1;
  session.weights_u8i4.slots[0].N = 32;
  session.weights_u8i8.slots[0].in_use = 1;
  session.weights_u8i8.slots[0].N = 32;
  const remote_handle64 h = (remote_handle64)&session;
  const uint32_t handle = 0;
  float in[32], a[32], b[32];
  uint32_t stages[HTP_N_STAGES];
  for (unsigned i = 0; i < 32; ++i)
    in[i] = (float)i - 5.0f;
#define CHECK(bits)                                                            \
  do {                                                                         \
    for (int error = 0; error < 2; ++error) {                                  \
      result = error ? AEE_EFAILED : AEE_SUCCESS;                              \
      hexkl_probe_reset(0);                                                    \
      assert(nntr_hvx_mm_u8i##bits##_layer(h, 1, 32, &handle, 1, in, 32, a,    \
                                           32) == result);                     \
      assert(!hexkl_probe_on);                                                 \
      memset(stages, 0, sizeof(stages));                                       \
      assert(nntr_hvx_mm_u8i##bits##_layer_timed(h, 1, 32, &handle, 1, in, 32, \
                                                 b, 32, stages,                \
                                                 HTP_N_STAGES) == result);     \
      assert(!memcmp(a, b, sizeof(a)) && !hexkl_probe_on);                     \
      assert(stages[HTP_T_QUANT] == 17);                                       \
    }                                                                          \
    unsigned before = calls;                                                   \
    assert(nntr_hvx_mm_u8i##bits##_layer_timed(h, 1, 32, &handle, 1, in, 32,   \
                                               b, 32, NULL,                    \
                                               HTP_N_STAGES) == AEE_EBADPARM); \
    assert(nntr_hvx_mm_u8i##bits##_layer(h, 1, 32, &handle, 1, in, 31, a,      \
                                         32) == AEE_EBADPARM);                 \
    session.hmx_locked = 0;                                                    \
    assert(nntr_hvx_mm_u8i##bits##_layer_timed(h, 1, 32, &handle, 1, in, 32,   \
                                               b, 32, NULL,                    \
                                               0) == AEE_EUNSUPPORTED);        \
    session.hmx_locked = 1;                                                    \
    assert(calls == before);                                                   \
  } while (0)
  CHECK(4);
  CHECK(8);
  puts("HTP timed/untimed RPC dispatch and validation PASS");
}
