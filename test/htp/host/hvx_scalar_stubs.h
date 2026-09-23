/* Scalar stand-ins for every HMX/HVX/DMA/pool primitive the block
   kernels call, plus the reference helpers the host checks share. See
   run_host_checks.sh for what this does and does not verify: the
   arithmetic is defined self-consistently on both sides, so what a check
   here catches is loop structure -- which rows, which weights, which
   buffer -- not the hardware's numerics. */
#ifndef HVX_SCALAR_STUBS_H
#define HVX_SCALAR_STUBS_H

#include <stdint.h>

#include "hexkl_mm_u8i4_dma.h"

/* One int4 value of a WH weight tile (htp_wh_layout.h's byte order). */
int wh_value(const uint8_t *tile, uint32_t k, uint32_t c);

/* A reference weight: the same WH bytes the table slot borrows. */
typedef struct {
  uint32_t K, N;
  int8_t *nib;
  float *ws;
  int32_t *cs;
  float *bias;
} W;

/* out[N] = dequant(a_u8 . w) with the same formula the tile stand-in uses. */
void ref_mm(const W *w, const uint8_t *a_u8, float a_scale, int32_t a_zp,
            float *out);
/* Per-row u8 quantization, the scan-and-pack stand-ins' formula. */
void quant_row(const float *x, uint32_t k, uint8_t *q, float *scale,
               int32_t *zp);

uint32_t rnd(void);
float rndf(void);

/* The weight table the checks register into (slot = handle). */
extern hexkl_weight_u8i4_table g_tbl;
void make_weight(uint32_t slot, uint32_t K, uint32_t N, W *w);

#endif /* HVX_SCALAR_STUBS_H */
