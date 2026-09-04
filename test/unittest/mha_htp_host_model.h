// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file mha_htp_host_model.h
 * @date 07 Aug 2026
 * @brief Host-side executable specification of the fused HTP attention loop
 * @see https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug No known bugs.
 *
 * "Flash attention" on this part is fusion + KV streaming + band tiling, NOT
 * textbook online softmax: there is exactly one HMX accumulator and both
 * matmuls use it, so keeping O in the accumulator across KV tiles and
 * rescaling it by exp(m_old - m_new) is not expressible and at our context
 * lengths the whole [M_band][kv_len] score band fits in VTCM anyway, so a
 * two-pass exact softmax is simpler, exact, and lets P.V accumulate over the
 * entire kv_len.
 *
 * This is a SPECIFICATION, not a second implementation: the order of
 * operations is the order the DSP kernel will use, so that a later device
 * kernel can be checked bit-exactly against it. Clarity over speed.
 *
 * Five properties of the loop are load-bearing. Each one is listed here with
 * the single assertion a later DEVICE test should use to check the kernel
 * reproduces it none of them fails as a tolerance miss, which is why they
 * need naming:
 *
 * 1. BLOCK-MAJOR intermediates, no repack. hexkl_mm_u8iX_layer_run returns
 * one contiguous M x N block per handle in call order; row m of block j
 * is at off_j + m*T. The block-major layout IS the KV tiling.
 * ponytail: device assert attn_scores_debug's S for a shape with
 * kv_len > T must differ from the same S read as a flat [M][kv_len]
 * matrix. Reading it flat is the bug this catches, and a flat read still
 * produces plausible numbers.
 *
 * 2. P is quantized PER BLOCK. Each block's P rows get their own scale.
 * ponytail: device assert for kv_len > T, the act_scale the kernel
 * used for block 0 and block 1 must differ on data whose blocks have
 * different maxima. One shared scale still passes a loose tolerance.
 *
 * 3. O accumulates in f32 ACROSS blocks, because each block's dequant
 * constants differ.
 * ponytail: device assert O for (kv_len = 2T) must equal O for the
 * same scores computed as two n_handles=1 calls summed in f32. An i32
 * accumulation across blocks would silently apply block 0's scale.
 *
 * 4. The 1/l normalization happens ONCE, at the very end. Phase B emits
 * unnormalized values plus l.
 * ponytail: device assert max over kv of P (pre-normalization,
 * attn_forward_debug's P) is exactly 1.0f for every row with a valid
 * position. A third softmax pass makes it 1/l instead, which still
 * yields a correct O and hides the wasted u8 codes.
 *
 * 5. M_band <= T, so a band's rows never span more than one extra block
 * once masking restricts them.
 * ponytail: device assert attn_register must reject M_band > T with
 * AEE_EBADPARM. Silently accepting it only misbehaves once block
 * skipping lands, i.e. one task after the one that introduced it. */

#ifndef __NNTRAINER_MHA_HTP_HOST_MODEL_H__
#define __NNTRAINER_MHA_HTP_HOST_MODEL_H__

#include <cstdint>
#include <vector>

#include "htp_backend/hmx/hexkl_kv_quant.h"

/**
 * @brief Host reference for the u8-in/u8-out HVX RoPE contract.
 *
 * cos_u8 and sin_u8 contain only the first half of each row, each quantized
 * with its own per-tensor (scale, zero-point) matching the QNN RoPE
 * lowering where cos and sin carry distinct scale/zp. Rounding is
 * round-to-nearest-even, matching hvx_sf_to_w_rne on the DSP. */
void rope_u8_ref(const uint8_t *x, uint8_t *y, uint32_t n_rows, uint32_t width,
                 uint32_t dim, const float *s_in, const int32_t *zp_in,
                 const uint8_t *cos_u8, const uint8_t *sin_u8, float cos_scale,
                 int32_t cos_zp, float sin_scale, int32_t sin_zp, float s_out,
                 int32_t zp_out, uint32_t *n_saturated);

/**
 * @brief Unnormalized softmax output plus the per-row denominator.
 *
 * @a p is block-major and mirrors the input segmentation exactly: element
 * (segment j, row m, column t) is at p[j*M*T + m*T + t]. */
struct MhaSoftmaxOut {
  std::vector<float> p; /**< n_seg * M * T, block-major, UNNORMALIZED */
  std::vector<float> l; /**< M, the row denominator incl. the sink term */
};

/**
 * @brief Blocked masked softmax over a block-major score band (PHASE B).
 *
 * Block-major scores: segment j is s[off_j + m*T + t], t in [0, T). Softmax
 * runs along the kv axis, ACROSS segments, for each row m; kv position
 * j*T + t is valid iff it lies in [begin[m], end[m]).
 *
 * p holds UNNORMALIZED exp values the row maximum is exactly 1.0f and
 * l[m] is the row denominator, including the sink term when present. The
 * 1/l normalization is deliberately left to the caller so that P.V can
 * accumulate over the entire kv_len first.
 *
 * @param seg n_seg pointers, one per block, each M*T floats
 * @param n_seg number of blocks
 * @param T kv positions per block
 * @param M rows in the band
 * @param scale folded into the scores before the max (1/sqrt(head_dim))
 * @param begin M entries, first valid kv position of each row
 * @param end M entries, one past the last valid kv position of each row
 * @param sink nullptr, or one logit per row; enters l only, never p */
MhaSoftmaxOut mha_softmax_blocked_ref(const std::vector<const float *> &seg,
                                      uint32_t n_seg, uint32_t T, uint32_t M,
                                      float scale,
                                      const std::vector<uint32_t> &begin,
                                      const std::vector<uint32_t> &end,
                                      const float *sink);

/**
 * @brief PHASE A alone: the S band for one kv_head, block-major.
 *
 * Split out so a device kernel can be gated against this stage on its own *
 * without a per-stage oracle a wrong O says nothing about which of the five
 * stages produced it. mha_htp_host_forward calls this, so there is exactly one
 * implementation of PHASE A rather than a second opinion beside it.
 *
 * @param M rows in the band, n_query * gqa folded
 * @param q_band [M][head_dim] f32, already gathered for this head
 * @param[out] out_s [n_blocks][M][T] f32, block-major; n_blocks is
 * ceil(kv_len / T) */
void mha_htp_host_scores(uint32_t kv_len, uint32_t nch, uint32_t head_dim,
                         uint32_t T, uint32_t M, uint32_t head,
                         hexkl_w_width w_k, const float *q_band,
                         const uint16_t *k_cache, float *out_s);

/**
 * @brief Scalar model of the whole fused attention kernel.
 *
 * Same band order, same block order, same quantization points and same mask
 * arithmetic as the DSP kernel, with plain arithmetic instead of HMX/HVX/DMA.
 * Calls hexkl_kvq_pack_kt_block / hexkl_kvq_pack_v_block and
 * mha_softmax_blocked_ref; it does not reimplement either.
 *
 * kv_len is kv_from + n_query: the caches already carry the appended step
 * rows, and query row i sits at absolute kv position kv_from + i.
 *
 * Masking, for query row i at absolute position p = kv_from + i:
 * end = is_causal ? p + 1 : kv_len
 * begin = (window && window < end) ? end - window : 0
 * so @a window means "the last @a window allowed positions" in both causal
 * modes, and 0 means no windowing.
 *
 * @param n_query new query rows in this call
 * @param kv_from kv positions already in the cache (= cache_index)
 * @param nch cache heads
 * @param gqa query heads per cache head
 * @param head_dim per-head dimension
 * @param T kv positions per block; must be a multiple of 32
 * @param M_band query rows per band; must be <= T
 * @param sink nullptr, or one logit per query head, [nch*gqa]
 * @param w_k Kt weight width, independent of @a w_v
 * @param w_v V weight width
 * @param q [n_query][nch*gqa*head_dim] f32, post-RoPE
 * @param k_cache [kv_len][nch][head_dim] fp16, post-RoPE
 * @param v_cache [kv_len][nch][head_dim] fp16
 * @param[out] out [n_query][nch*gqa*head_dim] f32 */
void mha_htp_host_forward(uint32_t n_query, uint32_t kv_from, uint32_t nch,
                          uint32_t gqa, uint32_t head_dim, uint32_t T,
                          uint32_t M_band, bool is_causal, uint32_t window,
                          const float *sink, hexkl_w_width w_k,
                          hexkl_w_width w_v, const float *q,
                          const uint16_t *k_cache, const uint16_t *v_cache,
                          float *out);

#endif /* __NNTRAINER_MHA_HTP_HOST_MODEL_H__ */
