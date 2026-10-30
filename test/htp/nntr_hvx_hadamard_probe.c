// SPDX-License-Identifier: Apache-2.0
/**
 * @file nntr_hvx_hadamard_probe.c
 * @brief Validate and time online rotation on the actual DSP.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include "hexkl_hadamard_f16.h"
#include "hexkl_probe.h"
#include "hvx_hadamard_f32.h"
#include "hvx_worker_pool.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"
#include <AEEStdErr.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  float *x;
  uint32_t rows, width, mode;
} hadamard_job;

static void rotate_rows(uint32_t threads, uint32_t id, void *v) {
  const hadamard_job *c = v;
  const uint32_t first = c->rows * id / threads;
  const uint32_t end = c->rows * (id + 1u) / threads;
  (void)hvx_hadamard_rows_f32(c->x + (size_t)first * c->width, end - first,
                              c->width, c->mode);
}

int nntr_hvx_hadamard_probe(remote_handle64 handle, uint32 rows, uint32 width,
                            uint32 mode, uint32 passes, const float *input,
                            int inputLen, float *output, int outputLen,
                            uint32 *stats, int statsLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || !input || !output || !stats || !rows || rows > 1024u ||
      !hvx_hadamard_supported(width) ||
      (mode != 1u && mode != 2u && mode != 3u) || !passes || passes > 1000u ||
      inputLen != (int)(rows * width) || outputLen != inputLen || statsLen != 2)
    return AEE_EBADPARM;
  const uint32_t arena =
    s->config_off < s->vtcm_size ? s->config_off : s->vtcm_size;
  if (mode == 3u && (!s->hmx_locked || !s->hmx_fp16_rate ||
                     arena < HAD_HMX_SPACE || (width != 704 && width != 2816)))
    return AEE_EUNSUPPORTED;
  uint8_t *hv = NULL;
  if (mode == 3u) {
    hv = s->vtcm_base + ((arena - HAD_HMX_SPACE) & ~2047u);
    const int rc = had_hmx_init(hv);
    if (rc != AEE_SUCCESS)
      return rc;
  }
  const size_t bytes = (size_t)inputLen * sizeof(float);
  float *x = memalign(128u, bytes);
  float *source = memalign(128u, bytes);
  if (!x || !source) {
    free(x);
    free(source);
    return AEE_ENOMEMORY;
  }
  memcpy(source, input, bytes);
  hadamard_job job = {x, rows, width, mode};
  uint64_t ticks = 0u;
  for (uint32_t i = 0u; i < passes + 3u; ++i) {
    /* Restore outside the timed region; each pass rotates identical input. */
    memcpy(x, source, bytes);
    const uint64_t start = hexkl_probe_now_ticks();
    if (mode == 3u) {
      if (width == 704u)
        had_hmx_rows(hv, x, rows, 704u, s->quant_pool, NULL, NULL, NULL, NULL,
                     0.0f, x);
      else
        had_hmx_rows(hv, x, rows, 2816u, s->quant_pool, NULL, NULL, NULL, NULL,
                     0.0f, x);
    } else
      hvx_worker_pool_run(s->quant_pool, rotate_rows, &job, rows);
    const uint64_t elapsed = hexkl_probe_now_ticks() - start;
    if (i >= 3u)
      ticks += elapsed;
  }
  memcpy(output, x, bytes);
  stats[0] = (uint32_t)ticks;
  stats[1] = (uint32_t)(ticks >> 32);
  free(source);
  free(x);
  return AEE_SUCCESS;
}
