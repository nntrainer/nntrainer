#ifndef HEXKL_MICRO_H
#define HEXKL_MICRO_H
#include <stdint.h>
#define HEXKL_HMX_INT8_BLOCK_N_INNER 32u
#define HEXKL_HMX_INT8_BLOCK_N_COL 32u
#define HEXKL_HMX_INT8_BLOCK_N_ROW 64u
#define HEXKL_HMX_ACTIVATION_ALIGNMENT 2048u
int hexkl_micro_hmx_acc_clear_int32(void);
int hexkl_micro_hmx_mm_u8i4(uint8_t *b, uint32_t a, uint32_t w);
int hexkl_micro_hmx_acc_read_int32(uint8_t *b, uint32_t cfg, uint32_t off);
/* Link-only for hexkl_mm_u8i4_dma.c (the bake and the no-acc-layout
   fallback): the stand-ins abort if a check ever reaches them. */
int hexkl_micro_hmx_rm_to_wh_i4(uint8_t *b, uint32_t off, const int8_t *rm,
                                uint32_t tr, uint32_t tc, uint32_t N);
int hexkl_micro_hmx_copy_32b_to_submatrix(uint8_t *b, uint32_t off,
                                          int32_t *dst, uint32_t rb,
                                          uint32_t nt, uint32_t m_pad,
                                          uint32_t N);
#endif
