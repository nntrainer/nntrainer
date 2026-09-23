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
#endif
