// SPDX-License-Identifier: Apache-2.0
/** @file nntr_hvx_diagnostic_stubs.c
 * @brief Preserve diagnostic RPC ordinals in production DSP builds.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#include "nntr_hvx.h"
#include <AEEStdErr.h>

int nntr_hvx_dma_probe(remote_handle64 _h, uint32 arena, uint32 src_off,
                       uint32 bytes, uint32 row_size, uint32 nrows,
                       uint32 src_stride, uint32 workers, uint32 passes,
                       uint32 *res, int resLen) {
  (void)_h;
  (void)arena;
  (void)src_off;
  (void)bytes;
  (void)row_size;
  (void)nrows;
  (void)src_stride;
  (void)workers;
  (void)passes;
  (void)res;
  (void)resLen;
  return AEE_EUNSUPPORTED;
}

int nntr_hvx_dma_replay(remote_handle64 _h, uint32 arena, uint32 region_bytes,
                        const uint32 *schedule, int scheduleLen, uint32 workers,
                        uint32 load, uint32 pace, uint32 fresh, uint32 gap_us,
                        uint32 calls, uint32 *res, int resLen) {
  (void)_h;
  (void)arena;
  (void)region_bytes;
  (void)schedule;
  (void)scheduleLen;
  (void)workers;
  (void)load;
  (void)pace;
  (void)fresh;
  (void)gap_us;
  (void)calls;
  (void)res;
  (void)resLen;
  return AEE_EUNSUPPORTED;
}

int nntr_hvx_dspq_bench_start(remote_handle64 _h, uint64 queue_id,
                              uint32 mode) {
  (void)_h;
  (void)queue_id;
  (void)mode;
  return AEE_EUNSUPPORTED;
}

int nntr_hvx_dspq_bench_stop(remote_handle64 _h, uint32 *res, int resLen) {
  (void)_h;
  (void)res;
  (void)resLen;
  return AEE_EUNSUPPORTED;
}

int nntr_hvx_attn_m1_probe(remote_handle64 _h, uint32 op, uint32 lanes,
                           uint32 reps, const uint16 *a, int aLen,
                           const uint16 *b, int bLen, const uint16 *c, int cLen,
                           uint16 *y, int yLen, uint32 *prof, int profLen) {
  (void)_h;
  (void)op;
  (void)lanes;
  (void)reps;
  (void)a;
  (void)aLen;
  (void)b;
  (void)bLen;
  (void)c;
  (void)cLen;
  (void)y;
  (void)yLen;
  (void)prof;
  (void)profLen;
  return AEE_EUNSUPPORTED;
}

int nntr_hvx_sf_probe(remote_handle64 _h, uint32 op, const float *a, int aLen,
                      const float *b, int bLen, float *o, int oLen) {
  (void)_h;
  (void)op;
  (void)a;
  (void)aLen;
  (void)b;
  (void)bLen;
  (void)o;
  (void)oLen;
  return AEE_EUNSUPPORTED;
}

int nntr_hvx_vlut_probe(remote_handle64 _h, uint32 mode, uint32 n,
                        uint32 *stats, int statsLen) {
  (void)_h;
  (void)mode;
  (void)n;
  (void)stats;
  (void)statsLen;
  return AEE_EUNSUPPORTED;
}

int nntr_hvx_tl2_gemv_probe(remote_handle64 _h, uint32 K, uint32 N, uint32 mode,
                            uint32 workers, uint32 passes, const uint8 *act,
                            int actLen, const uint8 *weight, int weightLen,
                            int32 *result, int resultLen, uint32 *stats,
                            int statsLen) {
  (void)_h;
  (void)K;
  (void)N;
  (void)mode;
  (void)workers;
  (void)passes;
  (void)act;
  (void)actLen;
  (void)weight;
  (void)weightLen;
  (void)result;
  (void)resultLen;
  (void)stats;
  (void)statsLen;
  return AEE_EUNSUPPORTED;
}

int nntr_hvx_hadamard_probe(remote_handle64 _h, uint32 rows, uint32 width,
                            uint32 mode, uint32 passes, const float *input,
                            int inputLen, float *output, int outputLen,
                            uint32 *stats, int statsLen) {
  (void)_h;
  (void)rows;
  (void)width;
  (void)mode;
  (void)passes;
  (void)input;
  (void)inputLen;
  (void)output;
  (void)outputLen;
  (void)stats;
  (void)statsLen;
  return AEE_EUNSUPPORTED;
}
