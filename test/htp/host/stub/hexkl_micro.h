#ifndef HEXKL_MICRO_H
#define HEXKL_MICRO_H
/* Host stand-in for HexKL's hexkl_micro.h: the constants the kernels lay
   VTCM out with and the prototypes of every micro call the skel subset
   makes (hexkl_addon beta.2's signatures; hw_init is the three-argument
   form). The host checks define the tile calls (standin/hvx_scalar.c) and
   abort on the rest; the in-process build defines them all
   (inproc/hexkl_micro_standin.c). */
#include <stdint.h>
#define HEXKL_HMX_INT8_BLOCK_N_INNER 32u
#define HEXKL_HMX_INT8_BLOCK_N_COL 32u
#define HEXKL_HMX_INT8_BLOCK_N_ROW 64u
#define HEXKL_HMX_ACTIVATION_ALIGNMENT 2048u
#define HEXKL_HMX_WEIGHTS_ALIGNMENT 128u
#define HEXKL_HMX_CONFIG_ALIGNMENT 256u
int hexkl_micro_hw_init(uint8_t **vtcm_base, uint32_t *vtcm_size,
                        uint32_t *hmx_fp16_rate);
int hexkl_micro_hmx_lock(void);
int hexkl_micro_hmx_unlock(void);
uint32_t hexkl_micro_hmx_config_size(void);
int hexkl_micro_hmx_setup_acc_read_int32(uint8_t *b, uint32_t cfg);
int hexkl_micro_hmx_acc_clear_int32(void);
int hexkl_micro_hmx_mm_u8i4(uint8_t *b, uint32_t a, uint32_t w);
int hexkl_micro_hmx_mm_u8i8(uint8_t *b, uint32_t a, uint32_t w);
int hexkl_micro_hmx_acc_read_int32(uint8_t *b, uint32_t cfg, uint32_t off);
int hexkl_micro_hmx_rm_to_wh_i4(uint8_t *b, uint32_t off, const int8_t *rm,
                                uint32_t tr, uint32_t tc, uint32_t N);
int hexkl_micro_hmx_rm_to_wh_i8(uint8_t *b, uint32_t off, const int8_t *rm,
                                uint32_t tr, uint32_t tc, uint32_t N);
int hexkl_micro_hmx_copy_32b_to_submatrix(uint8_t *b, uint32_t off,
                                          int32_t *dst, uint32_t rb,
                                          uint32_t nt, uint32_t m_pad,
                                          uint32_t N);
#endif
