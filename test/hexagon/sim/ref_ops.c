// SPDX-License-Identifier: Apache-2.0
/**
 * @file	ref_ops.c
 * @date	18 August 2026
 * @brief	Scalar C reference implementations for hexagon-sim primitive tests
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include "ref_ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "nntr_htp_common.h"
#include "nntr_htp_rope.h"

float ref_quant_row(const __fp16 *x, int8_t *q, uint32_t k) {
  float amax = 0.f;
  for (uint32_t i = 0; i < k; ++i) {
    float v = fabsf((float)x[i]);
    if (v > amax)
      amax = v;
  }
  float inv = amax > 0.f ? 127.f / amax : 0.f;
  for (uint32_t i = 0; i < k; ++i)
    q[i] = (int8_t)lrintf((float)x[i] * inv);
  return amax / 127.f;
}

float ref_quant_row_i16(const __fp16 *x, int16_t *q, uint32_t k) {
  float amax = 0.f;
  for (uint32_t i = 0; i < k; ++i) {
    float v = fabsf((float)x[i]);
    if (v > amax)
      amax = v;
  }
  float inv = amax > 0.f ? 32767.f / amax : 0.f;
  for (uint32_t i = 0; i < k; ++i)
    q[i] = (int16_t)lrintf((float)x[i] * inv);
  return amax / 32767.f;
}

int32_t ref_dot_i8(const int8_t *w, const int8_t *x, uint32_t k) {
  int32_t acc = 0;
  for (uint32_t i = 0; i < k; ++i)
    acc += (int32_t)w[i] * (int32_t)x[i];
  return acc;
}

int32_t ref_dot_i8_tiled(const int8_t *w_base, uint32_t n, const int8_t *x,
                         uint32_t K) {
  int32_t acc = 0;
  /** Four consecutive k share one tile lane, so index once per 4 (K % 128
   * == 0 is validated). */
  for (uint32_t k = 0; k < K; k += 4u) {
    const int8_t *w4 = w_base + nntr_htp_tile_off(n, k, K);
    for (uint32_t j = 0; j < 4u; ++j)
      acc += (int32_t)w4[j] * (int32_t)x[k + j];
  }
  return acc;
}

void ref_matmul_w8a8(const __fp16 *x, const int8_t *w, const float *sw,
                     __fp16 *y, uint32_t m, uint32_t k, uint32_t n) {
  int8_t *xq = (int8_t *)malloc((size_t)k);
  for (uint32_t t = 0; t < m; ++t) {
    float sx = ref_quant_row(x + (size_t)t * k, xq, k);
    for (uint32_t j = 0; j < n; ++j) {
      int32_t dot = ref_dot_i8_tiled(w, j, xq, k);
      y[(size_t)t * n + j] = (__fp16)((float)dot * sw[j] * sx);
    }
  }
  free(xq);
}

void ref_matmul_w4a8(const __fp16 *x, const uint8_t *w, const float *sw,
                     const int32_t *colsum, __fp16 *y, uint32_t m, uint32_t k,
                     uint32_t n) {
  int8_t *xq = (int8_t *)malloc((size_t)k);
  for (uint32_t t = 0; t < m; ++t) {
    float sx = ref_quant_row(x + (size_t)t * k, xq, k);
    for (uint32_t j = 0; j < n; ++j) {
      /** u8 activation (x_i8 + 128) against the int4 weight, as HMX sees it;
       * exact in int32 for k <= 16384 (255 * 8 * 16384 < 2^31). */
      int32_t acc = 0;
      for (uint32_t i = 0; i < k; ++i)
        acc += ((int32_t)xq[i] + 128) * nntr_htp_w4_get(w, j, i, k);
      const int32_t dot = acc - 128 * colsum[j];
      y[(size_t)t * n + j] = (__fp16)((float)dot * sw[j] * sx);
    }
  }
  free(xq);
}

void ref_matmul_w8a16(const __fp16 *x, const int8_t *w, const float *sw,
                      __fp16 *y, uint32_t m, uint32_t k, uint32_t n) {
  int16_t *xq = (int16_t *)malloc((size_t)k * sizeof(int16_t));
  for (uint32_t t = 0; t < m; ++t) {
    float sx = ref_quant_row_i16(x + (size_t)t * k, xq, k);
    for (uint32_t j = 0; j < n; ++j) {
      int64_t dot = 0;
      for (uint32_t i = 0; i < k; ++i)
        dot += (int64_t)w[(size_t)j * k + i] * (int64_t)xq[i];
      y[(size_t)t * n + j] = (__fp16)((float)dot * sw[j] * sx);
    }
  }
  free(xq);
}

void ref_matmul_logits(const __fp16 *x_last, const int8_t *w, const float *sw,
                       float *out, uint32_t k, uint32_t n) {
  int8_t *xq = (int8_t *)malloc((size_t)k);
  float sx = ref_quant_row(x_last, xq, k);
  for (uint32_t j = 0; j < n; ++j)
    out[j] = (float)ref_dot_i8_tiled(w, j, xq, k) * sw[j] * sx;
  free(xq);
}

void ref_rmsnorm(const __fp16 *x, const __fp16 *gamma, __fp16 *y, uint32_t m,
                 uint32_t n, uint32_t chunk, float eps) {
  for (uint32_t t = 0; t < m; ++t) {
    const __fp16 *xrow = x + (size_t)t * n;
    __fp16 *yrow = y + (size_t)t * n;
    for (uint32_t c0 = 0; c0 < n; c0 += chunk) {
      float sumsq = 0.f;
      for (uint32_t i = 0; i < chunk; ++i) {
        float v = (float)xrow[c0 + i];
        sumsq += v * v;
      }
      float r = 1.0f / sqrtf(sumsq / (float)chunk + eps);
      for (uint32_t i = 0; i < chunk; ++i)
        yrow[c0 + i] = (__fp16)((float)xrow[c0 + i] * r * (float)gamma[i]);
    }
  }
}

void ref_rope_table_fill(__fp16 *table, uint32_t max_seq, float theta) {
  for (uint32_t p = 0; p < max_seq; ++p) {
    __fp16 *row = table + (size_t)p * 128;
    float f[128];
    nntr_htp_rope_row_f32(f, p, theta);
    for (uint32_t i = 0; i < 128; ++i)
      row[i] = (__fp16)f[i];
  }
}

void ref_rope(__fp16 *x, const __fp16 *table, uint32_t m, uint32_t heads,
              uint32_t pos) {
  for (uint32_t t = 0; t < m; ++t) {
    const __fp16 *row = table + (size_t)(pos + t) * 128;
    for (uint32_t h = 0; h < heads; ++h) {
      __fp16 *xh = x + ((size_t)t * heads + h) * 128;
      for (uint32_t i = 0; i < 64; ++i) {
        float x0 = (float)xh[i], x1 = (float)xh[64 + i];
        float cs = (float)row[i], sn = (float)row[64 + i];
        xh[i] = (__fp16)(x0 * cs - x1 * sn);
        xh[64 + i] = (__fp16)(x1 * cs + x0 * sn);
      }
    }
  }
}

void ref_add(const __fp16 *a, const __fp16 *b, __fp16 *y, uint32_t count) {
  for (uint32_t i = 0; i < count; ++i)
    y[i] = (__fp16)((float)a[i] + (float)b[i]);
}

void ref_silu_mul(const __fp16 *g, const __fp16 *u, __fp16 *y, uint32_t count) {
  for (uint32_t i = 0; i < count; ++i) {
    float gf = (float)g[i];
    float silu = gf / (1.0f + expf(-gf));
    y[i] = (__fp16)(silu * (float)u[i]);
  }
}

void ref_attn(const __fp16 *q, const __fp16 *k, const __fp16 *v, __fp16 *kv,
              __fp16 *out, uint32_t m, uint32_t pos, uint32_t layer,
              uint32_t n_layers, uint32_t n_heads, uint32_t n_kv_heads,
              uint32_t hd, uint32_t max_seq, float scale) {
  const size_t v_off = (size_t)n_layers * n_kv_heads * max_seq * hd;
  const uint32_t group = n_heads / n_kv_heads;
  float *scores = (float *)malloc((size_t)max_seq * sizeof(float));

  for (uint32_t h = 0; h < n_kv_heads; ++h) { /* KV append */
    __fp16 *kh = kv + ((size_t)layer * n_kv_heads + h) * max_seq * hd;
    __fp16 *vh = kh + v_off;
    for (uint32_t t = 0; t < m; ++t) {
      const size_t src = ((size_t)t * n_kv_heads + h) * hd;
      const size_t dst = (size_t)(pos + t) * hd;
      for (uint32_t i = 0; i < hd; ++i) {
        kh[dst + i] = k[src + i];
        vh[dst + i] = v[src + i];
      }
    }
  }

  for (uint32_t hq = 0; hq < n_heads; ++hq) {
    const uint32_t h = hq / group;
    const __fp16 *kh = kv + ((size_t)layer * n_kv_heads + h) * max_seq * hd;
    const __fp16 *vh = kh + v_off;
    for (uint32_t t = 0; t < m; ++t) {
      const __fp16 *qrow = q + ((size_t)t * n_heads + hq) * hd;
      const uint32_t len = pos + t + 1; /* causal */
      float mx = -INFINITY;
      for (uint32_t p = 0; p < len; ++p) {
        float s = 0.f;
        for (uint32_t i = 0; i < hd; ++i)
          s += (float)qrow[i] * (float)kh[(size_t)p * hd + i];
        scores[p] = s * scale;
        if (scores[p] > mx)
          mx = scores[p];
      }
      float sum = 0.f;
      for (uint32_t p = 0; p < len; ++p) {
        scores[p] = expf(scores[p] - mx);
        sum += scores[p];
      }
      __fp16 *orow = out + ((size_t)t * n_heads + hq) * hd;
      for (uint32_t i = 0; i < hd; ++i) {
        float acc = 0.f;
        for (uint32_t p = 0; p < len; ++p)
          acc += scores[p] * (float)vh[(size_t)p * hd + i];
        orow[i] = (__fp16)(acc / sum);
      }
    }
  }
  free(scores);
}

void ref_embed(const int32_t *tokens, const int8_t *w, const float *scale,
               __fp16 *y, uint32_t m, uint32_t k) {
  for (uint32_t t = 0; t < m; ++t) {
    uint32_t row = (uint32_t)tokens[t];
    __fp16 *yrow = y + (size_t)t * k;
    for (uint32_t i = 0; i < k; i += 4u) {
      const int8_t *w4 = w + nntr_htp_tile_off(row, i, k);
      for (uint32_t j = 0; j < 4u; ++j)
        yrow[i + j] = (__fp16)((float)w4[j] * scale[row]);
    }
  }
}

void ref_graph_forward_upto(const uint8_t *oplist, uint8_t *weights,
                            uint8_t *kv, uint8_t *act, const int32_t *tokens,
                            uint32_t n_tokens, uint32_t pos, float *logits,
                            uint32_t n_ops_limit) {
  struct nntr_htp_oplist_header h;
  const struct nntr_htp_op_desc *ops;
  uint8_t *bufs[NNTR_HTP_BUF_COUNT];

  memcpy(&h, oplist, sizeof(h));
  ops = (const struct nntr_htp_op_desc *)(const void *)(oplist + sizeof(h));
  bufs[NNTR_HTP_BUF_WEIGHTS] = weights;
  bufs[NNTR_HTP_BUF_KV] = kv;
  bufs[NNTR_HTP_BUF_ACT] = act;
  bufs[NNTR_HTP_BUF_TOKENS] = (uint8_t *)(uintptr_t)tokens;
  bufs[NNTR_HTP_BUF_LOGITS] = (uint8_t *)logits;

  const uint32_t n_run = h.n_ops < n_ops_limit ? h.n_ops : n_ops_limit;
  for (uint32_t i = 0; i < n_run; ++i) {
    const struct nntr_htp_op_desc *d = &ops[i];
    const uint32_t m = nntr_htp_op_rows(d, n_tokens);
    uint8_t *p0 = bufs[d->in0.buf] + d->in0.offset;
    uint8_t *p1 = bufs[d->in1.buf] + d->in1.offset;
    uint8_t *p2 = bufs[d->in2.buf] + d->in2.offset;
    uint8_t *po = bufs[d->out.buf] + d->out.offset;
    float pf;
    memcpy(&pf, &d->param0, sizeof(pf));

    switch (d->kind) {
    case NNTR_HTP_OP_EMBED:
      ref_embed((const int32_t *)(const void *)p0,
                (const int8_t *)(const void *)p1,
                (const float *)(const void *)p2, (__fp16 *)(void *)po, m, d->k);
      break;
    case NNTR_HTP_OP_RMSNORM: {
      const uint32_t chunk =
        (d->flags & NNTR_HTP_FLAG_PER_HEAD) ? h.head_dim : d->n;
      ref_rmsnorm((const __fp16 *)(const void *)p0,
                  (const __fp16 *)(const void *)p1, (__fp16 *)(void *)po, m,
                  d->n, chunk, pf);
      break;
    }
    case NNTR_HTP_OP_MATMUL_W8A8:
      ref_matmul_w8a8(
        (const __fp16 *)(const void *)p0, (const int8_t *)(const void *)p1,
        (const float *)(const void *)p2, (__fp16 *)(void *)po, m, d->k, d->n);
      break;
    case NNTR_HTP_OP_MATMUL_W8A16:
      ref_matmul_w8a16(
        (const __fp16 *)(const void *)p0, (const int8_t *)(const void *)p1,
        (const float *)(const void *)p2, (__fp16 *)(void *)po, m, d->k, d->n);
      break;
    case NNTR_HTP_OP_MATMUL_W4A8: /* colsum: the fifth operand at param0 */
      ref_matmul_w4a8(
        (const __fp16 *)(const void *)p0, p1, (const float *)(const void *)p2,
        (const int32_t *)(const void *)(bufs[NNTR_HTP_BUF_WEIGHTS] + d->param0),
        (__fp16 *)(void *)po, m, d->k, d->n);
      break;
    case NNTR_HTP_OP_ROPE:
      ref_rope((__fp16 *)(void *)p0, (const __fp16 *)(const void *)p2, m,
               h.n_heads, pos);
      ref_rope((__fp16 *)(void *)p1, (const __fp16 *)(const void *)p2, m,
               h.n_kv_heads, pos);
      break;
    case NNTR_HTP_OP_ATTN:
      ref_attn((const __fp16 *)(const void *)p0,
               (const __fp16 *)(const void *)p1,
               (const __fp16 *)(const void *)p2, (__fp16 *)(void *)kv,
               (__fp16 *)(void *)po, m, pos, d->layer, h.n_layers, h.n_heads,
               h.n_kv_heads, h.head_dim, h.max_seq, pf);
      break;
    case NNTR_HTP_OP_SILU_MUL:
      ref_silu_mul((const __fp16 *)(const void *)p0,
                   (const __fp16 *)(const void *)p1, (__fp16 *)(void *)po,
                   m * d->n);
      break;
    case NNTR_HTP_OP_ADD:
      ref_add((const __fp16 *)(const void *)p0,
              (const __fp16 *)(const void *)p1, (__fp16 *)(void *)po, m * d->n);
      break;
    case NNTR_HTP_OP_MATMUL_LOGITS:
      ref_matmul_logits(
        (const __fp16 *)(const void *)p0 + (size_t)(n_tokens - 1u) * d->k,
        (const int8_t *)(const void *)p1, (const float *)(const void *)p2,
        (float *)(void *)po, d->k, d->n);
      break;
    default:
      break;
    }
  }
}

void ref_graph_forward(const uint8_t *oplist, uint8_t *weights, uint8_t *kv,
                       uint8_t *act, const int32_t *tokens, uint32_t n_tokens,
                       uint32_t pos, float *logits) {
  ref_graph_forward_upto(oplist, weights, kv, act, tokens, n_tokens, pos,
                         logits, 0xffffffffu);
}
