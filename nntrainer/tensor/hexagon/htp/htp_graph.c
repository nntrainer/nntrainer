// SPDX-License-Identifier: Apache-2.0
/**
 * @file	htp_graph.c
 * @date	19 August 2026
 * @brief	Op-list graph executor: validates once at init, then forward is
 *		a plain sequential dispatch loop over the op table (each op
 *		runs its own parallel section and returns at the barrier)
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <HAP_compute_res.h>
#include <HAP_perf.h>
#if defined(HTP_PROF_FARF) || HTP_HMX
#include <HAP_farf.h>
#endif
#include <stdlib.h>
#include <string.h>

#include "dma-queue.h"
#include "htp_graph.h"

/** A NULL slot is a kind the wire format defines but this build cannot
 * execute; init rejects any op-list that uses it (rc 4), so an image is
 * never run silently on the wrong kernel. MATMUL_W4A8 (v5, #65 S1) has its
 * HMX kernel in S2, for HTP_HMX builds only. */
const htp_op_fn htp_op_table[NNTR_HTP_OP_KIND_COUNT] = {
  hvx_op_embed, hvx_op_rmsnorm,       hvx_op_matmul_w8a8,
  hvx_op_rope,  hvx_op_attn,          hvx_op_silu_mul,
  hvx_op_add,   hvx_op_matmul_logits, hvx_op_matmul_w8a16,
  NULL, /* NNTR_HTP_OP_MATMUL_W4A8 */
};

#define HTP_GRAPH_VTCM_BYTES (4u * 1024u * 1024u)
#if HTP_HMX
/** With HexKL linked the session asks for 8 MB (v79 has 8 MB) with the
 * HMX unit, keeps the first HTP_GRAPH_VTCM_BYTES for the HVX slabs exactly
 * as before and lays the HMX arena over the rest; a 4 MB grant carves only
 * the arena's fixed part (htp_hmx_arena_min_bytes) off the HVX region. */
#define HTP_HMX_VTCM_BYTES (8u * 1024u * 1024u)
#endif

int htp_graph_dma_init(struct htp_exec_ctx *c, int n_workers) {
  const size_t align = dma_queue_alignof();
  const size_t one =
    (dma_queue_sizeof(HTP_MM_DMA_QUEUE_CAP) + align - 1u) & ~(align - 1u);
  int i;

  if (!c || n_workers <= 0)
    return 1;
  c->dmaq = calloc((size_t)n_workers, sizeof(*c->dmaq));
  c->pf = calloc((size_t)n_workers, sizeof(*c->pf));
  c->dmaq_mem = memalign(align, one * (size_t)n_workers);
  if (!c->dmaq || !c->pf || !c->dmaq_mem) {
    htp_graph_dma_destroy(c);
    return 1;
  }
  /** The queue's VTCM range decides the bypass bit of every descriptor
   * whose end is in VTCM; with an HMX arena above the HVX slabs the range
   * covers both (the arena is DMA'd into by S2's strip streaming). */
  for (i = 0; i < n_workers; ++i)
    c->dmaq[i] =
      dma_queue_init((uint8_t *)c->dmaq_mem + one * (size_t)i,
                     HTP_MM_DMA_QUEUE_CAP, (uintptr_t)c->vtcm,
#if HTP_HMX
                     c->hmx.base ? (size_t)(c->hmx.base - c->vtcm) + c->hmx.size
                                 : c->vtcm_size,
#else
                     c->vtcm_size,
#endif
                     NULL);
  return 0;
}

static void dma_flush_job(void *arg, int wid, int nw) {
  struct htp_exec_ctx *c = arg;
  (void)nw;
  dma_queue_flush(c->dmaq[wid]);
  c->pf[wid].desc = NULL;
}

void htp_graph_dma_flush(struct htp_exec_ctx *c) {
  if (c && c->dmaq && c->pool)
    wp_run(c->pool, dma_flush_job, c);
}

void htp_graph_dma_destroy(struct htp_exec_ctx *c) {
  if (!c)
    return;
  free(c->dmaq);
  free(c->pf);
  free(c->dmaq_mem);
  c->dmaq = NULL;
  c->pf = NULL;
  c->dmaq_mem = NULL;
}

int htp_graph_init_ex(struct htp_graph *g, const uint8_t *oplist, uint32_t len,
                      uint8_t *weights, uint32_t wsize, uint8_t *kv,
                      uint32_t kvsize, uint8_t *act, uint32_t actsize,
                      int n_workers) {
  uint32_t buf_size[NNTR_HTP_BUF_COUNT];
  uint32_t k_max = 0, i;
  int rc;

  if (!g)
    return 1;
  memset(g, 0, sizeof(*g));
  if (nntr_htp_oplist_check(oplist, len))
    return 1;
  memcpy(&g->cfg, oplist, sizeof(g->cfg));

  /** TOKENS/LOGITS sizes are the forward() contract: at most max_chunk
   * int32 token ids and exactly vocab fp32 logits. */
  buf_size[NNTR_HTP_BUF_WEIGHTS] = wsize;
  buf_size[NNTR_HTP_BUF_KV] = kvsize;
  buf_size[NNTR_HTP_BUF_ACT] = actsize;
  buf_size[NNTR_HTP_BUF_TOKENS] = g->cfg.max_chunk * 4u;
  buf_size[NNTR_HTP_BUF_LOGITS] = g->cfg.vocab * 4u;
  rc = nntr_htp_oplist_validate(oplist, len, buf_size);
  if (rc)
    return rc;

  g->ops =
    (const struct nntr_htp_op_desc *)(const void *)(oplist + sizeof(g->cfg));
  /* A valid kind without a kernel in this build (see htp_op_table). */
  for (i = 0; i < g->cfg.n_ops; ++i)
    if (!htp_op_table[g->ops[i].kind])
      return 4;
  memcpy(g->ctx.buf_size, buf_size, sizeof(buf_size));
  g->ctx.buf[NNTR_HTP_BUF_WEIGHTS] = weights;
  g->ctx.buf[NNTR_HTP_BUF_KV] = kv;
  g->ctx.buf[NNTR_HTP_BUF_ACT] = act;
  g->ctx.cfg = &g->cfg;

  g->ctx.pool = wp_create(n_workers);
  if (!g->ctx.pool)
    return 1;

  /** Quant scratch sized by the widest matmul k in this op-list; x2 for the
   * int16 rows of MATMUL_W8A16. MATMUL_W4A8 (v5) quantises its rows into
   * the same scratch and streams n*k/2 bytes of nibble tiles; counted here
   * already so S2's kernel cannot inherit an undersized xq. Its cross-op
   * prefetch (next_mm) is S2's: the kick geometry is per kind. */
  for (i = 0; i < g->cfg.n_ops; ++i)
    if (g->ops[i].kind == (uint32_t)NNTR_HTP_OP_MATMUL_W8A8 ||
        g->ops[i].kind == (uint32_t)NNTR_HTP_OP_MATMUL_LOGITS ||
        g->ops[i].kind == (uint32_t)NNTR_HTP_OP_MATMUL_W8A16 ||
        g->ops[i].kind == (uint32_t)NNTR_HTP_OP_MATMUL_W4A8) {
      if (g->ops[i].k > k_max)
        k_max = g->ops[i].k;
      g->stream_bytes += g->ops[i].kind == (uint32_t)NNTR_HTP_OP_MATMUL_W4A8
                           ? (uint64_t)g->ops[i].n * g->ops[i].k / 2u
                           : (uint64_t)g->ops[i].n * g->ops[i].k;
    }
  if (k_max) {
    g->ctx.xq = memalign(128, (size_t)g->cfg.max_chunk * k_max * 2u);
    g->ctx.xq_scale = malloc((size_t)g->cfg.max_chunk * sizeof(float));
  }
  g->ctx.prof_op_cycles =
    calloc(g->cfg.n_ops ? g->cfg.n_ops : 1u, sizeof(uint64_t));
  /** next_mm[i]: the first tiled matmul after op i, walked once backwards.
   * The forward loop clips it to n_ops_limit so a partial run never kicks
   * a chunk for an op it will not execute. */
  g->next_mm = malloc((g->cfg.n_ops ? g->cfg.n_ops : 1u) * sizeof(uint32_t));
  if (g->next_mm) {
    uint32_t nx = UINT32_MAX;
    for (i = g->cfg.n_ops; i-- > 0u;) {
      g->next_mm[i] = nx;
      if (g->ops[i].kind == (uint32_t)NNTR_HTP_OP_MATMUL_W8A8 ||
          g->ops[i].kind == (uint32_t)NNTR_HTP_OP_MATMUL_LOGITS)
        nx = i;
    }
  }
  /** [n_workers][max_seq] fp32 scores, +128B pad: hvx_exp_f32's tail path
   * reads one whole unaligned vector starting at the last elements. */
  g->ctx.attn_scratch = memalign(
    128, (size_t)wp_size(g->ctx.pool) * g->cfg.max_seq * sizeof(float) + 128u);
  if ((k_max && (!g->ctx.xq || !g->ctx.xq_scale)) || !g->ctx.attn_scratch ||
      !g->ctx.prof_op_cycles || !g->next_mm) {
    htp_graph_destroy(g);
    return 1;
  }

  /** Best-effort VTCM: NULL keeps the matmul DDR direct-read fallback and
   * is not an error. */
#if HTP_HMX
  /** First try: VTCM 8 MB (4 MB floor) plus the HMX unit in one context;
   * the HVX kernels keep exactly their HTP_GRAPH_VTCM_BYTES at the bottom
   * and HexKL's arena takes what is above it. If the resource manager
   * refuses the HMX attribute (no HMX on the part, or held elsewhere) the
   * plain 4 MB acquire below runs and the session has no HMX. */
  (void)htp_hmx_version(g->ctx.hmx.version, sizeof(g->ctx.hmx.version));
  {
    compute_res_attr_t rattr;
    unsigned id;
    void *p = NULL;
    unsigned sz = 0;
    HAP_compute_res_attr_init(&rattr);
    if (HAP_compute_res_attr_set_vtcm_param_v2(&rattr, HTP_HMX_VTCM_BYTES, 0,
                                               HTP_GRAPH_VTCM_BYTES) == 0 &&
        HAP_compute_res_attr_set_hmx_param(&rattr, 1) == 0) {
      id = HAP_compute_res_acquire(&rattr, 10000 /*us*/);
      if (id) {
        if (HAP_compute_res_attr_get_vtcm_ptr_v2(&rattr, &p, &sz) == 0 && p &&
            sz >= HTP_GRAPH_VTCM_BYTES) {
          uint32_t arena = sz - HTP_GRAPH_VTCM_BYTES;
          const uint32_t min_arena = htp_hmx_arena_min_bytes();
          if (arena < min_arena)
            arena = min_arena; /* 4 MB grant: the fixed part comes off HVX */
          arena = (arena + HTP_HMX_ACT_ALIGN - 1u) & ~(HTP_HMX_ACT_ALIGN - 1u);
          g->vtcm_ctx_id = id;
          g->ctx.vtcm = (uint8_t *)p;
          g->ctx.vtcm_size = sz - arena;
          g->ctx.hmx.ctx_id = id;
          if (htp_hmx_arena_init(&g->ctx.hmx, (uint8_t *)p + (sz - arena),
                                 arena)) {
            g->ctx.hmx.ctx_id = 0; /* arena too small: HVX keeps it all */
            g->ctx.vtcm_size = sz;
          }
        } else {
          HAP_compute_res_release(id);
        }
      }
    }
  }
  if (!g->ctx.vtcm)
#endif
  {
    compute_res_attr_t rattr;
    unsigned id;
    HAP_compute_res_attr_init(&rattr);
    HAP_compute_res_attr_set_vtcm_param(&rattr, HTP_GRAPH_VTCM_BYTES, 1);
    id = HAP_compute_res_acquire(&rattr, 10000 /*us*/);
    if (id) {
      void *p = HAP_compute_res_attr_get_vtcm_ptr(&rattr);
      if (p) {
        g->vtcm_ctx_id = id;
        g->ctx.vtcm = (uint8_t *)p;
        g->ctx.vtcm_size = HTP_GRAPH_VTCM_BYTES;
      } else {
        HAP_compute_res_release(id);
      }
    }
  }
#if HTP_HMX
  /** HexKL build (issue #65 S0): the library version this skel linked and
   * whether the session got the HMX unit with its arena, so a handoff can
   * copy both off logcat (plan 65 section 5, "beta1 vs beta2"). Logged
   * here rather than in executor.c so the HTP_HMX=0 build keeps
   * executor.o byte-identical (FARF embeds __LINE__). */
  FARF(ALWAYS, "nntr_htp: hexkl %s hmx=%d vtcm_hvx=%u hmx_arena=%u",
       g->ctx.hmx.version, g->ctx.hmx.base ? 1 : 0, (unsigned)g->ctx.vtcm_size,
       (unsigned)g->ctx.hmx.size);
#endif
  /** The matmul streaming path needs one DMA queue per worker for the
   * session (chunks are kicked ahead across ops, so the queue cannot live
   * inside an op call any more). Without VTCM the queues are not needed. */
  if (g->ctx.vtcm && htp_graph_dma_init(&g->ctx, wp_size(g->ctx.pool))) {
    htp_graph_destroy(g);
    return 1;
  }
  return 0;
}

int htp_graph_init(struct htp_graph *g, const uint8_t *oplist, uint32_t len,
                   uint8_t *weights, uint32_t wsize, uint8_t *kv,
                   uint32_t kvsize, uint8_t *act, uint32_t actsize) {
  return htp_graph_init_ex(g, oplist, len, weights, wsize, kv, kvsize, act,
                           actsize, 0);
}

int htp_graph_forward_upto(struct htp_graph *g, const int32_t *tokens,
                           uint32_t n_tokens, uint32_t pos, float *logits,
                           uint32_t n_logits, uint32_t n_ops_limit,
                           uint64_t *pcycles) {
  uint32_t i;
  uint64_t t0;

  if (!g || !tokens || !logits)
    return 1;
  /** Runtime-argument gate: init cannot validate these, and a violation
   * reads past the rope table / writes past the KV cache. */
  if (n_tokens == 0u || n_tokens > g->cfg.max_chunk ||
      (uint64_t)pos + n_tokens > (uint64_t)g->cfg.max_seq)
    return 1;
  if (n_logits != g->cfg.vocab)
    return 1;
  if (n_ops_limit > g->cfg.n_ops)
    return 1;
  /** Token ids exist only per call: EMBED gathers row (uint32_t)tokens[t]
   * of the vocab-row table unchecked, so reject here, before any buffer
   * pointer or KV row is touched. */
  if (!nntr_htp_token_ids_ok(tokens, n_tokens, g->cfg.vocab))
    return 1;

  g->ctx.buf[NNTR_HTP_BUF_TOKENS] = (uint8_t *)(uintptr_t)tokens;
  g->ctx.buf[NNTR_HTP_BUF_LOGITS] = (uint8_t *)logits;
  g->ctx.n_tokens = n_tokens;
  g->ctx.pos = pos;

#ifdef HTP_PROF_FARF
  uint64_t prof0[NNTR_HTP_OP_KIND_COUNT];
  memcpy(prof0, g->ctx.prof_cycles, sizeof(prof0));
#endif

  t0 = HAP_perf_get_pcycles();
  for (i = 0; i < n_ops_limit; ++i) {
    const uint32_t kind = g->ops[i].kind;
    const uint64_t s = HAP_perf_get_pcycles();
    g->ctx.next_mm =
      g->next_mm[i] < n_ops_limit ? &g->ops[g->next_mm[i]] : NULL;
    htp_op_table[kind](&g->ctx, &g->ops[i]);
    const uint64_t dt = HAP_perf_get_pcycles() - s;
    g->ctx.prof_cycles[kind] += dt;
    g->ctx.prof_calls[kind] += 1u;
    g->ctx.prof_op_cycles[i] += dt;
  }
  if (pcycles)
    *pcycles = HAP_perf_get_pcycles() - t0;
#ifdef HTP_PROF_FARF
  /** Device profile (measurement builds only, HEXAGON.md section 5.3): this
   * call's per-kind split in kilo-pcycles, so a handoff can read the
   * matmul / non-matmul share off logcat without an IDL change. "rest" is
   * every kind that is not one of the four named. */
  {
    const uint64_t loop = HAP_perf_get_pcycles() - t0;
    uint64_t d[NNTR_HTP_OP_KIND_COUNT], rest = 0;
    uint32_t kk;
    for (kk = 0; kk < (uint32_t)NNTR_HTP_OP_KIND_COUNT; ++kk) {
      d[kk] = g->ctx.prof_cycles[kk] - prof0[kk];
      rest += d[kk];
    }
    rest -= d[NNTR_HTP_OP_MATMUL_W8A8] + d[NNTR_HTP_OP_MATMUL_W8A16] +
            d[NNTR_HTP_OP_MATMUL_LOGITS] + d[NNTR_HTP_OP_ATTN];
    FARF(ALWAYS,
         "nntr_htp: prof n=%u pos=%u ops=%u kcyc=%u mm8=%u mm16=%u lg=%u "
         "attn=%u rest=%u",
         (unsigned)n_tokens, (unsigned)pos, (unsigned)n_ops_limit,
         (unsigned)(loop / 1000u),
         (unsigned)(d[NNTR_HTP_OP_MATMUL_W8A8] / 1000u),
         (unsigned)(d[NNTR_HTP_OP_MATMUL_W8A16] / 1000u),
         (unsigned)(d[NNTR_HTP_OP_MATMUL_LOGITS] / 1000u),
         (unsigned)(d[NNTR_HTP_OP_ATTN] / 1000u), (unsigned)(rest / 1000u));
  }
#endif
  return 0;
}

void htp_graph_profile_reset(struct htp_graph *g) {
  if (!g)
    return;
  memset(g->ctx.prof_cycles, 0, sizeof(g->ctx.prof_cycles));
  memset(g->ctx.prof_calls, 0, sizeof(g->ctx.prof_calls));
  memset(g->ctx.prof_op_cycles, 0, (size_t)g->cfg.n_ops * sizeof(uint64_t));
}

void htp_graph_profile_get(const struct htp_graph *g,
                           uint64_t cycles[NNTR_HTP_OP_KIND_COUNT],
                           uint32_t calls[NNTR_HTP_OP_KIND_COUNT]) {
  if (!g)
    return;
  memcpy(cycles, g->ctx.prof_cycles, sizeof(g->ctx.prof_cycles));
  memcpy(calls, g->ctx.prof_calls, sizeof(g->ctx.prof_calls));
}

int htp_graph_profile_get_ops(const struct htp_graph *g, uint64_t *cycles,
                              uint32_t n) {
  if (!g || !cycles || n < g->cfg.n_ops)
    return 1;
  memcpy(cycles, g->ctx.prof_op_cycles,
         (size_t)g->cfg.n_ops * sizeof(uint64_t));
  return 0;
}

int htp_graph_forward(struct htp_graph *g, const int32_t *tokens,
                      uint32_t n_tokens, uint32_t pos, float *logits,
                      uint32_t n_logits) {
  if (!g)
    return 1;
  return htp_graph_forward_upto(g, tokens, n_tokens, pos, logits, n_logits,
                                g->cfg.n_ops, 0);
}

const uint8_t *htp_graph_buf_ref(const struct htp_graph *g, uint32_t buf,
                                 uint32_t offset, uint32_t bytes) {
  if (!g || buf > (uint32_t)NNTR_HTP_BUF_ACT)
    return 0;
  if ((uint64_t)offset + bytes > (uint64_t)g->ctx.buf_size[buf])
    return 0;
  return g->ctx.buf[buf] + offset;
}

void htp_graph_destroy(struct htp_graph *g) {
  if (!g)
    return;
  /** Drain every worker's DMA queue on the workers before the VTCM the
   * descriptors write into goes away. */
  htp_graph_dma_flush(&g->ctx);
#if HTP_HMX
  /** The HMX lock is per thread: worker 0 took it, worker 0 gives it
   * back, before the context that granted it is released. */
  if (g->ctx.hmx.locked && g->ctx.pool)
    wp_run(g->ctx.pool, htp_hmx_release_job, &g->ctx.hmx);
#endif
  if (g->vtcm_ctx_id)
    HAP_compute_res_release(g->vtcm_ctx_id);
  htp_graph_dma_destroy(&g->ctx);
  free(g->ctx.xq);
  free(g->ctx.xq_scale);
  free(g->ctx.attn_scratch);
  free(g->ctx.prof_op_cycles);
  free(g->next_mm);
  if (g->ctx.pool)
    wp_destroy(g->ctx.pool);
  memset(g, 0, sizeof(*g));
}
