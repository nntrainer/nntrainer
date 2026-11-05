// SPDX-License-Identifier: Apache-2.0
/**
 * @file retired_rpc_check.c
 * @brief Retired RPCs refuse calls without accessing session or data.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#include "nntr_hvx.h"
#include <assert.h>
#include <stdio.h>

int main(void) {
  /* Both a missing and an invalid session must be refused before dereference.
   * Output sentinels also catch a stub that accidentally reports a result. */
  for (remote_handle64 handle = 0; handle < 2; ++handle) {
    uint32 output = 0x12345678u;
    float value = 1.25f;
    uint64 touched = 0x123456789abcdef0ULL;
    uint16 position = 0x1234u;
    assert(nntr_hvx_mm_u8i4_from_f32(handle, 1, 32, 32, NULL, 0, NULL, 0, NULL,
                                     0, NULL, 0, NULL, 0, NULL, 0, &value, 1,
                                     NULL, 0, NULL, 0, &value,
                                     1) == AEE_EUNSUPPORTED);
    assert(nntr_hvx_mm_u8i4_layer_fused(handle, 1, 32, NULL, 0, NULL, 0, &value,
                                        1) == AEE_EUNSUPPORTED);
    assert(nntr_hvx_mm_u8i4_layer_fused_timed(handle, 1, 32, NULL, 0, NULL, 0,
                                              &value, 1, &output,
                                              1) == AEE_EUNSUPPORTED);
    assert(nntr_hvx_dspq_bench_start(handle, 0, 0) == AEE_EUNSUPPORTED);
    assert(nntr_hvx_dspq_bench_stop(handle, &output, 1) == AEE_EUNSUPPORTED);
    assert(nntr_hvx_dma_probe(handle, 0, 0, 128, 128, 1, 128, 1, 1, &output,
                              1) == AEE_EUNSUPPORTED);
    assert(nntr_hvx_mailbox_run(handle, -1, 0, 0, 1, 0, 0, &output, 1) ==
           AEE_EUNSUPPORTED);
    assert(nntr_hvx_probe_wh_i8_pos(handle, &position, 1) == AEE_EUNSUPPORTED);
    assert(nntr_hvx_probe_softmax_q(handle, 1, 0, 32, 0, 1, 0, 16384, 8, NULL,
                                    0, NULL, 0, NULL, 0, NULL, 0, &output,
                                    1) == AEE_EUNSUPPORTED);
    assert(nntr_hvx_attn_register(handle, 1, 1, 128, 64, 64, 64, 8, 8,
                                  &output) == AEE_EUNSUPPORTED);
    assert(nntr_hvx_attn_release(handle, 0) == AEE_EUNSUPPORTED);
    assert(nntr_hvx_attn_kv_append(handle, 0, 0, 1, NULL, 0, NULL, 0) ==
           AEE_EUNSUPPORTED);
    assert(nntr_hvx_attn_scores_debug(handle, 0, 0, 1, NULL, 0, &value, 1) ==
           AEE_EUNSUPPORTED);
    assert(nntr_hvx_attn_forward(handle, 0, 0, 1, 1.0f, 1, 0, NULL, 0, NULL, 0,
                                 &value, 1) == AEE_EUNSUPPORTED);
    assert(nntr_hvx_attn_forward_timed(handle, 0, 0, 1, 1.0f, 1, 0, NULL, 0,
                                       NULL, 0, &value, 1, &output,
                                       1) == AEE_EUNSUPPORTED);
    assert(nntr_hvx_softmax_blocked_f32(handle, 1, 64, 1, 1.0f, NULL, 0, NULL,
                                        0, NULL, 0, NULL, 0, &value, 1, &value,
                                        1) == AEE_EUNSUPPORTED);
    assert(nntr_hvx_mem_probe_dsp_heap(handle, 256, 32, &output, 1, &touched,
                                       1) == AEE_EUNSUPPORTED);
    assert(nntr_hvx_mem_probe_touch(handle, NULL, 0, &touched, 1) ==
           AEE_EUNSUPPORTED);
    assert(nntr_hvx_vlut_probe(handle, 0, 1, &output, 1) == AEE_EUNSUPPORTED);
    assert(nntr_hvx_probe_hvx_rate(handle, 1, &output, 1) == AEE_EUNSUPPORTED);
    assert(nntr_hvx_probe_f16_decode_block(handle, 128, 1, NULL, 0, NULL, 0,
                                           &value, 1, NULL, 0, NULL, 0, &value,
                                           1) == AEE_EUNSUPPORTED);
    assert(output == 0x12345678u && value == 1.25f);
    assert(touched == 0x123456789abcdef0ULL);
    assert(position == 0x1234u);
  }
  puts("HTP retired RPCs: unsupported, outputs untouched PASS");
  return 0;
}
