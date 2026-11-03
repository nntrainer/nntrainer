// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd.
 * @file integer_fc_chain_bench.h
 * @brief Two-FC row-layout versus directly tiled activation benchmark
 * @bug No known bugs except for NYI items
 * Included after the common device benchmark's helpers.
 */
#ifndef NNTR_INTEGER_FC_CHAIN_BENCH_H
#define NNTR_INTEGER_FC_CHAIN_BENCH_H

typedef struct {
  const uint16_t *src;
  uint16_t *dst;
  uint32_t m, n, slice;
} chain_join_job;

static void chain_join_worker(uint32_t threads, uint32_t lane, void *ctx) {
  const chain_join_job *j = ctx;
  const uint32_t begin = j->m * lane / threads;
  const uint32_t end = j->m * (lane + 1u) / threads;
  for (uint32_t r = begin; r < end; ++r)
    for (uint32_t h = 0; h < j->n / j->slice; ++h) {
      const uint16_t *src = j->src + (size_t)h * j->m * j->slice + r * j->slice;
      uint16_t *dst = j->dst + (size_t)r * j->n + h * j->slice;
      for (uint32_t c = 0; c < j->slice; c += 64u) {
        const HVX_Vector v = *(const HVX_Vector *)(src + c);
        Q6_vmem_QnRIV_nt(Q6_Q_vcmp_gt_VwVw(v, v), (HVX_Vector *)(dst + c), v);
      }
    }
}

static int chain_register(nntr_hvx_session *s, uint32_t k, uint32_t n,
                          uint32_t *handle) {
  int8_t *w = malloc((size_t)k * n);
  float *scale = malloc(n * sizeof(float)), *bias = calloc(n, sizeof(float));
  int32_t *cs = calloc(n, sizeof(int32_t));
  REQUIRE(w && scale && bias && cs);
  for (uint32_t i = 0; i < k; ++i)
    for (uint32_t c = 0; c < n; ++c) {
      const int v = (int)((rnd() >> 16) & 15u) - 8;
      w[(size_t)i * n + c] = (int8_t)v;
      cs[c] += v;
    }
  for (uint32_t c = 0; c < n; ++c)
    scale[c] = 1.0f;
  const int rc =
    hexkl_weight_u8i4_register(&s->weights_u8i4, s->vtcm_base, s->vtcm_size, k,
                               n, w, scale, cs, bias, s->quant_pool, handle);
  free(w);
  free(scale);
  free(bias);
  free(cs);
  return rc;
}

typedef struct {
  nntr_hvx_session *s;
  uint32_t m, k, n, consumer_n, nh;
  uint32_t handles[32], consumer;
  uint16_t *input, *row_codes, *joined;
  uint8_t *packed;
  size_t packed_bytes;
  int32_t *acc[2], *out[2];
  const hexkl_requant_u16 *rp[32];
  uint32_t nr[32];
} chain_case;

static int chain_producer(chain_case *c, uint32_t packed) {
  nntr_hvx_session *s = c->s;
#ifndef HEXKL_BENCH_ROW_ONLY
  if (packed) {
    hexkl_u16_ah_io io = {NULL, 0, c->packed, c->packed_bytes};
    return hexkl_mm_u16i4_layer_run_i32_ah(
      &s->weights_u8i4, s->vtcm_base, s->vtcm_size, s->config_off, c->m, c->k,
      c->handles, c->nh, c->input, 30000, NULL, c->acc[1], s->quant_pool, c->rp,
      c->nr, &io);
  }
#else
  (void)packed;
#endif
  return hexkl_mm_u16i4_layer_run_i32_requant(
    &s->weights_u8i4, s->vtcm_base, s->vtcm_size, s->config_off, c->m, c->k,
    c->handles, c->nh, c->input, 30000, NULL, c->acc[0], s->quant_pool, c->rp,
    c->nr, c->row_codes);
}

static void chain_join(chain_case *c) {
  if (c->nh == 1u)
    return;
  chain_join_job j = {c->row_codes, c->joined, c->m, c->n, 256};
  hvx_worker_pool_run(c->s->quant_pool, chain_join_worker, &j, c->m);
}

static int chain_consumer(chain_case *c, uint32_t packed) {
  nntr_hvx_session *s = c->s;
#ifndef HEXKL_BENCH_ROW_ONLY
  if (packed) {
    hexkl_u16_ah_io io = {c->packed, c->packed_bytes, NULL, 0};
    return hexkl_mm_u16i4_layer_run_i32_ah(
      &s->weights_u8i4, s->vtcm_base, s->vtcm_size, s->config_off, c->m, c->n,
      &c->consumer, 1, NULL, 30000, NULL, c->out[1], s->quant_pool, NULL, NULL,
      &io);
  }
#else
  (void)packed;
#endif
  return hexkl_mm_u16i4_layer_run_i32(
    &s->weights_u8i4, s->vtcm_base, s->vtcm_size, s->config_off, c->m, c->n,
    &c->consumer, 1, c->nh == 1u ? c->row_codes : c->joined, 30000, NULL,
    c->out[0], s->quant_pool);
}

static int chain_run(chain_case *c, uint32_t packed) {
  REQUIRE(chain_producer(c, packed) == 0);
  if (!packed)
    chain_join(c);
  REQUIRE(chain_consumer(c, packed) == 0);
  return 0;
}

#ifndef HEXKL_BENCH_ROW_ONLY
static int chain_equal(chain_case *c) {
  REQUIRE(memcmp(c->acc[0], c->acc[1], (size_t)c->m * c->n * 4u) == 0);
  REQUIRE(memcmp(c->out[0], c->out[1], (size_t)c->m * c->consumer_n * 4u) == 0);
  const uint32_t mp = (c->m + 63u) / 64u * 64u;
  const size_t plane = (size_t)mp * c->n;
  for (uint32_t r = 0; r < mp; ++r)
    for (uint32_t col = 0; col < c->n; ++col) {
      uint16_t expected = 0;
      if (r < c->m) {
        const uint32_t h = col / 256u;
        const size_t index = (size_t)h * c->m * 256u + r * 256u + col % 256u;
        expected = rqref(c->acc[0][index],
                         c->rp[h] + (c->nr[h] == 1u ? 0u : col % 256u));
        REQUIRE(c->row_codes[index] == expected);
      }
      const size_t index =
        ((size_t)(r / 64u) * (c->n / 32u) + col / 32u) * 2048u +
        (r % 64u) * 32u + col % 32u;
      REQUIRE((uint16_t)(c->packed[index] | ((uint16_t)c->packed[plane + index]
                                             << 8)) == expected);
    }
  return 0;
}
#endif

static int chain_bench(nntr_hvx_session *s, uint32_t m, uint32_t k,
                       uint32_t n) {
  chain_case c = {0};
  c.s = s;
  c.m = m;
  c.k = k;
  c.n = n;
  c.nh = n / 256u;
  c.consumer_n = 256;
  c.input = memalign(128, (size_t)m * k * 2u);
  c.row_codes = memalign(128, (size_t)m * n * 2u);
  c.joined = memalign(128, (size_t)m * n * 2u);
  c.packed_bytes = (size_t)((m + 63u) / 64u * 64u) * n * 2u;
  c.packed = memalign(128, c.packed_bytes);
  REQUIRE(c.input && c.row_codes && c.joined && c.packed && c.nh <= 32u);
  for (uint32_t i = 0; i < 2u; ++i) {
    c.acc[i] = memalign(128, (size_t)m * n * 4u);
    c.out[i] = memalign(128, (size_t)m * c.consumer_n * 4u);
    REQUIRE(c.acc[i] && c.out[i]);
  }
  for (size_t i = 0; i < (size_t)m * k; ++i)
    c.input[i] = (uint16_t)(rnd() >> 8);
  hexkl_requant_u16 p = {3, 8, 30000};
  for (uint32_t h = 0; h < c.nh; ++h) {
    REQUIRE(chain_register(s, k, 256, c.handles + h) == 0);
    c.rp[h] = &p;
    c.nr[h] = 1;
  }
  REQUIRE(chain_register(s, n, c.consumer_n, &c.consumer) == 0);
#ifndef HEXKL_BENCH_ROW_ONLY
  // Exercise generic per-channel shifts, then the optimized common encoding.
  hexkl_requant_u16 pp[256];
  for (uint32_t j = 0; j < 256u; ++j)
    pp[j] =
      (hexkl_requant_u16){j & 1u ? INT32_MAX : 123456789u, j % 63u, 30000};
  for (uint32_t h = 0; h < c.nh; ++h) {
    c.rp[h] = pp;
    c.nr[h] = 256;
  }
  REQUIRE(chain_run(&c, 0) == 0);
  REQUIRE(chain_run(&c, 1) == 0);
  REQUIRE(chain_equal(&c) == 0);
  for (uint32_t h = 0; h < c.nh; ++h) {
    c.rp[h] = &p;
    c.nr[h] = 1;
  }
  const uint32_t modes = 2;
#else
  const uint32_t modes = 1;
#endif
  for (uint32_t rep = 0; rep < 8u; ++rep) {
    double times[2] = {0};
    for (uint32_t step = 0; step < modes; ++step) {
      const uint32_t mode = (step + rep) % modes;
      REQUIRE(chain_run(&c, mode) == 0);
      const uint64_t t = HAP_perf_get_qtimer_count();
      for (uint32_t b = 0; b < 3u; ++b)
        REQUIRE(chain_run(&c, mode) == 0);
      times[mode] = us(t) / 3.;
    }
#ifndef HEXKL_BENCH_ROW_ONLY
    REQUIRE(chain_equal(&c) == 0);
#endif
    if (rep >= 2u)
      printf("CHAIN M=%u K=%u N=%u O=%u rep=%u row_us=%.3f packed_us=%.3f\n", m,
             k, n, c.consumer_n, rep - 2u, times[0], times[1]);
  }
  // Separate component timings are diagnostic, not summed into chain time.
  for (uint32_t mode = 0; mode < modes; ++mode) {
    double producer = 0, join = 0, consumer = 0;
    for (uint32_t rep = 0; rep < 6u; ++rep) {
      uint64_t t = HAP_perf_get_qtimer_count();
      REQUIRE(chain_producer(&c, mode) == 0);
      producer += us(t);
      t = HAP_perf_get_qtimer_count();
      if (!mode)
        chain_join(&c);
      join += us(t);
      t = HAP_perf_get_qtimer_count();
      REQUIRE(chain_consumer(&c, mode) == 0);
      consumer += us(t);
    }
    printf("COMPONENT M=%u K=%u N=%u mode=%u producer_us=%.3f join_us=%.3f "
           "consumer_us=%.3f\n",
           m, k, n, mode, producer / 6., join / 6., consumer / 6.);
  }
  uint32_t hash = 2166136261u;
  for (size_t i = 0; i < (size_t)m * c.consumer_n; ++i)
    hash = (hash ^ (uint32_t)c.out[0][i]) * 16777619u;
  printf("CHAIN_CHECK M=%u K=%u N=%u hash=%08x\n", m, k, n, hash);
  for (uint32_t h = 0; h < c.nh; ++h)
    REQUIRE(hexkl_weight_u8i4_release(&s->weights_u8i4, c.handles[h]) == 0);
  REQUIRE(hexkl_weight_u8i4_release(&s->weights_u8i4, c.consumer) == 0);
  free(c.input);
  free(c.row_codes);
  free(c.joined);
  free(c.packed);
  for (uint32_t i = 0; i < 2u; ++i) {
    free(c.acc[i]);
    free(c.out[i]);
  }
  return 0;
}
#endif
