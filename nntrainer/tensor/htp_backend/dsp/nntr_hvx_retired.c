// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd.
 *
 * @file   nntr_hvx_retired.c
 * @brief  Reserved FastRPC entries for retired kernels and standalone probes.
 * @see    https://github.com/nntrainer/nntrainer
 * @author NNTrainer contributors
 * @bug    No known bugs except for NYI items
 *
 * Keep these IDL slots in both production and diagnostic builds so later
 * methods retain their wire ordinals. Current models use attn_q2, attn_f16
 * and attn_m1; the old per-block KV re-quantization path is unsupported.
 * The historical memory, instruction-rate, layout-dump and copied
 * decode-block probes, two-PD mailbox and dspqueue echo experiments are
 * also retired. The single-call FFN prototype was replaced by separate
 * gate/up and down calls after diverging on real model weights.
 */

#include "nntr_hvx.h"
#include <AEEStdErr.h>

int nntr_hvx_mm_u8i4_layer_fused(remote_handle64 handle, uint32 M, uint32 K,
                                 const uint32 *w_handles, int w_handlesLen,
                                 const float *act_f32, int act_f32Len,
                                 float *out_f32, int out_f32Len) {
  (void)handle;
  (void)M;
  (void)K;
  (void)w_handles;
  (void)w_handlesLen;
  (void)act_f32;
  (void)act_f32Len;
  (void)out_f32;
  (void)out_f32Len;
  return AEE_EUNSUPPORTED;
}

int nntr_hvx_mm_u8i4_layer_fused_timed(remote_handle64 handle, uint32 M,
                                       uint32 K, const uint32 *w_handles,
                                       int w_handlesLen, const float *act_f32,
                                       int act_f32Len, float *out_f32,
                                       int out_f32Len, uint32 *stage_us,
                                       int stage_usLen) {
  (void)handle;
  (void)M;
  (void)K;
  (void)w_handles;
  (void)w_handlesLen;
  (void)act_f32;
  (void)act_f32Len;
  (void)out_f32;
  (void)out_f32Len;
  (void)stage_us;
  (void)stage_usLen;
  return AEE_EUNSUPPORTED;
}

int nntr_hvx_dspq_bench_start(remote_handle64 handle, uint64 queue_id,
                              uint32 mode) {
  (void)handle;
  (void)queue_id;
  (void)mode;
  return AEE_EUNSUPPORTED;
}

int nntr_hvx_dspq_bench_stop(remote_handle64 handle, uint32 *res, int resLen) {
  (void)handle;
  (void)res;
  (void)resLen;
  return AEE_EUNSUPPORTED;
}

int nntr_hvx_mailbox_run(remote_handle64 handle, int32 fd, uint32 bytes,
                         uint32 role, uint32 n, uint32 payload, uint32 spin_us,
                         uint32 *res, int resLen) {
  (void)handle;
  (void)fd;
  (void)bytes;
  (void)role;
  (void)n;
  (void)payload;
  (void)spin_us;
  (void)res;
  (void)resLen;
  return AEE_EUNSUPPORTED;
}

int nntr_hvx_probe_wh_i8_pos(remote_handle64 handle, uint16 *pos, int posLen) {
  (void)handle;
  (void)pos;
  (void)posLen;
  return AEE_EUNSUPPORTED;
}

int nntr_hvx_attn_register(remote_handle64 handle, uint32 nch, uint32 gqa,
                           uint32 head_dim, uint32 max_kv, uint32 T,
                           uint32 M_band, uint32 w_k, uint32 w_v, uint32 *h) {
  (void)handle;
  (void)nch;
  (void)gqa;
  (void)head_dim;
  (void)max_kv;
  (void)T;
  (void)M_band;
  (void)w_k;
  (void)w_v;
  (void)h;
  return AEE_EUNSUPPORTED;
}

int nntr_hvx_attn_release(remote_handle64 handle, uint32 h) {
  (void)handle;
  (void)h;
  return AEE_EUNSUPPORTED;
}

int nntr_hvx_attn_kv_append(remote_handle64 handle, uint32 h, uint32 kv_from,
                            uint32 n_rows, const uint16 *k_rows, int k_rowsLen,
                            const uint16 *v_rows, int v_rowsLen) {
  (void)handle;
  (void)h;
  (void)kv_from;
  (void)n_rows;
  (void)k_rows;
  (void)k_rowsLen;
  (void)v_rows;
  (void)v_rowsLen;
  return AEE_EUNSUPPORTED;
}

int nntr_hvx_attn_scores_debug(remote_handle64 handle, uint32 h, uint32 head,
                               uint32 M, const float *q_band, int q_bandLen,
                               float *out_s, int out_sLen) {
  (void)handle;
  (void)h;
  (void)head;
  (void)M;
  (void)q_band;
  (void)q_bandLen;
  (void)out_s;
  (void)out_sLen;
  return AEE_EUNSUPPORTED;
}

int nntr_hvx_attn_forward(remote_handle64 handle, uint32 h, uint32 kv_from,
                          uint32 n_query, float scale, uint32 is_causal,
                          uint32 window, const float *sink, int sinkLen,
                          const float *q, int qLen, float *out, int outLen) {
  (void)handle;
  (void)h;
  (void)kv_from;
  (void)n_query;
  (void)scale;
  (void)is_causal;
  (void)window;
  (void)sink;
  (void)sinkLen;
  (void)q;
  (void)qLen;
  (void)out;
  (void)outLen;
  return AEE_EUNSUPPORTED;
}

int nntr_hvx_attn_forward_timed(remote_handle64 handle, uint32 h,
                                uint32 kv_from, uint32 n_query, float scale,
                                uint32 is_causal, uint32 window,
                                const float *sink, int sinkLen, const float *q,
                                int qLen, float *attn_out, int attn_outLen,
                                uint32 *stage_us, int stage_usLen) {
  (void)handle;
  (void)h;
  (void)kv_from;
  (void)n_query;
  (void)scale;
  (void)is_causal;
  (void)window;
  (void)sink;
  (void)sinkLen;
  (void)q;
  (void)qLen;
  (void)attn_out;
  (void)attn_outLen;
  (void)stage_us;
  (void)stage_usLen;
  return AEE_EUNSUPPORTED;
}

int nntr_hvx_softmax_blocked_f32(remote_handle64 handle, uint32 n_seg, uint32 T,
                                 uint32 M, float scale, const float *band_in,
                                 int band_inLen, const uint32 *begin,
                                 int beginLen, const uint32 *end, int endLen,
                                 const float *sink, int sinkLen,
                                 float *band_out, int band_outLen, float *l_out,
                                 int l_outLen) {
  (void)handle;
  (void)n_seg;
  (void)T;
  (void)M;
  (void)scale;
  (void)band_in;
  (void)band_inLen;
  (void)begin;
  (void)beginLen;
  (void)end;
  (void)endLen;
  (void)sink;
  (void)sinkLen;
  (void)band_out;
  (void)band_outLen;
  (void)l_out;
  (void)l_outLen;
  return AEE_EUNSUPPORTED;
}

int nntr_hvx_mem_probe_dsp_heap(remote_handle64 handle, uint32 chunk_mb,
                                uint32 max_chunks, uint32 *chunks_ok,
                                int chunks_okLen, uint64 *touched_sum,
                                int touched_sumLen) {
  (void)handle;
  (void)chunk_mb;
  (void)max_chunks;
  (void)chunks_ok;
  (void)chunks_okLen;
  (void)touched_sum;
  (void)touched_sumLen;
  return AEE_EUNSUPPORTED;
}

int nntr_hvx_mem_probe_touch(remote_handle64 handle, const uint8 *buf,
                             int bufLen, uint64 *touched_sum,
                             int touched_sumLen) {
  (void)handle;
  (void)buf;
  (void)bufLen;
  (void)touched_sum;
  (void)touched_sumLen;
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

int nntr_hvx_probe_hvx_rate(remote_handle64 handle, uint32 n, uint32 *stats,
                            int statsLen) {
  (void)handle;
  (void)n;
  (void)stats;
  (void)statsLen;
  return AEE_EUNSUPPORTED;
}

int nntr_hvx_probe_f16_decode_block(remote_handle64 handle, uint32 head_dim,
                                    uint32 kv, const float *q, int qLen,
                                    const uint16 *k, int kLen, float *sc_f32,
                                    int sc_f32Len, uint16 *sv_hf, int sv_hfLen,
                                    uint16 *p_hf, int p_hfLen, float *misc,
                                    int miscLen) {
  (void)handle;
  (void)head_dim;
  (void)kv;
  (void)q;
  (void)qLen;
  (void)k;
  (void)kLen;
  (void)sc_f32;
  (void)sc_f32Len;
  (void)sv_hf;
  (void)sv_hfLen;
  (void)p_hf;
  (void)p_hfLen;
  (void)misc;
  (void)miscLen;
  return AEE_EUNSUPPORTED;
}

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

int nntr_hvx_mm_u8i4_from_f32(
  remote_handle64 handle, uint32 M, uint32 K, uint32 N, const float *act_f32,
  int act_f32Len, const int8 *w_i4_rm, int w_i4_rmLen, const float *w_scale,
  int w_scaleLen, const int32 *colsum_w, int colsum_wLen, const float *bias,
  int biasLen, uint8 *act_u8_ah, int act_u8_ahLen, float *act_scale,
  int act_scaleLen, int32 *act_zp, int act_zpLen, int32 *acc_i32,
  int acc_i32Len, float *out_f32, int out_f32Len) {
  (void)handle;
  (void)M;
  (void)K;
  (void)N;
  (void)act_f32;
  (void)act_f32Len;
  (void)w_i4_rm;
  (void)w_i4_rmLen;
  (void)w_scale;
  (void)w_scaleLen;
  (void)colsum_w;
  (void)colsum_wLen;
  (void)bias;
  (void)biasLen;
  (void)act_u8_ah;
  (void)act_u8_ahLen;
  (void)act_scale;
  (void)act_scaleLen;
  (void)act_zp;
  (void)act_zpLen;
  (void)acc_i32;
  (void)acc_i32Len;
  (void)out_f32;
  (void)out_f32Len;
  return AEE_EUNSUPPORTED;
}

int nntr_hvx_probe_softmax_q(remote_handle64 handle, uint32 n_col_tiles,
                             uint32 col0, uint32 n_cols, uint32 row0,
                             uint32 n_rows, uint32 window, uint32 rho_q15,
                             uint32 frac_bits, const int16 *s_tiles,
                             int s_tilesLen, const int16 *corr, int corrLen,
                             uint8 *p_tiles, int p_tilesLen, int32 *rowsum,
                             int rowsumLen, uint32 *stats, int statsLen) {
  (void)handle;
  (void)n_col_tiles;
  (void)col0;
  (void)n_cols;
  (void)row0;
  (void)n_rows;
  (void)window;
  (void)rho_q15;
  (void)frac_bits;
  (void)s_tiles;
  (void)s_tilesLen;
  (void)corr;
  (void)corrLen;
  (void)p_tiles;
  (void)p_tilesLen;
  (void)rowsum;
  (void)rowsumLen;
  (void)stats;
  (void)statsLen;
  return AEE_EUNSUPPORTED;
}
