// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   bert_decoder.h
 * @date   18 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  General-purpose BERT decoder — embeddings + causal self-attention
 *         + cross-attention + FFN + LM head.
 *
 * Graph inputs (in order):
 *   input0          [1,1,1,1]     token id
 *   position_ids    [1,1,1,1]
 *   token_type_ids  [1,1,1,1]
 *   encoder_hidden  [1,1,196,256] encoder hidden states for cross-attention
 *   cache_k_l0..N-1 [1,1,MAX_SEQ,DIM]  per-layer KV cache
 *   cache_v_l0..N-1 [1,1,MAX_SEQ,DIM]
 *
 * Weight converter layer name contract (collect_decoder order):
 *   word_emb, pos_emb, type_emb, emb_ln
 *   per layer: dec_layer{i}_self_q, _self_k, _self_v, _self_out, _self_ln
 *              dec_layer{i}_cross_q, _cross_k, _cross_v, _cross_out, _cross_ln
 *              dec_layer{i}_ffn_inter, _ffn_out, _ffn_ln
 *   LM head: lmhead_dense, lmhead_ln, lm_head_bias/weights
 */

#ifndef __BERT_DECODER_H__
#define __BERT_DECODER_H__

#include <kv_cache_manager.h>
#include <mha_core.h>
#include <transformer.h>

namespace causallm {

/**
 * @brief BertDecoder class
 *
 * General-purpose BERT-style cross-attention decoder. No feature-specific
 * dependency. Architecture (this task — embeddings + causal self-attention
 * only):
 *
 *   [input0] [position_ids] [token_type_ids]
 *       |           |               |
 *  [word_emb]  [pos_emb]      [type_emb]
 *       \___________|_______________/
 *                   |
 *              [Addition]
 *                   |
 *             [LayerNorm eps=1e-12]   (emb_ln)
 *                   |
 *       [SelfAttentionBlock] x NUM_LAYERS
 *         Q/K/V FC (bias on) + mha_core(causal,no-rope)
 *         + O FC (bias on) + residual + LayerNorm eps=1e-12
 */
class BertDecoder : virtual public Transformer {

public:
  static constexpr const char *architectures = "BertDecoder";

  // Hardcoded decoder constants matching the source BERT decoder.
  static constexpr int BD_DIM = 256;
  static constexpr int BD_NUM_LAYERS = 4;
  static constexpr int BD_NUM_HEADS = 4;
  static constexpr int BD_HEAD_DIM = 64;
  static constexpr unsigned int BD_NUM_VOCAB = 30522;
  static constexpr unsigned int BD_MAX_POSITION_EMBEDDINGS = 512;
  static constexpr unsigned int BD_TYPE_VOCAB_SIZE = 2;
  static constexpr float BD_NORM_EPS = 1e-12f;
  static constexpr int BD_GQA_SIZE = 1; // NUM_HEADS == NUM_KEY_VALUE_HEADS
  static constexpr unsigned int BD_BATCH_SIZE = 1;
  static constexpr unsigned int BD_INIT_SEQ_LEN = 1;
  static constexpr unsigned int BD_MAX_SEQ_LEN = 512;
  // Encoder (SigLIP2) output sequence length fed to cross-attention.
  static constexpr unsigned int BD_ENC_LEN = 196;
  // NUM_TO_GENERATE must be set for Transformer; use a reasonable default.
  static constexpr unsigned int BD_NUM_TO_GENERATE = 1;

  // FFN intermediate size (4 * DIM = 1024 for BERT-small)
  static constexpr int BD_INTERMEDIATE_SIZE = 1024;

  /**
   * @brief Construct a BertDecoder with hardcoded parameters.
   *        Populates Transformer base fields directly (no JSON config
   * required).
   */
  BertDecoder() : Transformer() {
    DIM = BD_DIM;
    NUM_LAYERS = BD_NUM_LAYERS;
    NUM_HEADS = BD_NUM_HEADS;
    HEAD_DIM = BD_HEAD_DIM;
    NUM_KEY_VALUE_HEADS = BD_NUM_HEADS; // no GQA
    NUM_VOCAB = BD_NUM_VOCAB;
    MAX_POSITION_EMBEDDINGS = BD_MAX_POSITION_EMBEDDINGS;
    MAX_SEQ_LEN = BD_MAX_SEQ_LEN;
    BATCH_SIZE = BD_BATCH_SIZE;
    INIT_SEQ_LEN = BD_INIT_SEQ_LEN;
    GQA_SIZE = BD_GQA_SIZE;
    NORM_EPS = BD_NORM_EPS;
    ROPE_THETA = 0; // CRITICAL: use_rope=false; rope_theta=0 + use_rope=true
                    // caused NaN in encoder — always false here
    IS_CAUSAL = true;
    TIE_WORD_EMBEDDINGS = false; // LM-head projection is tied via shared_from
    USE_VOCAB_SELECTION = false; // not used in decoder path
    INTERMEDIATE_SIZE = BD_INTERMEDIATE_SIZE;
    MODEL_TENSOR_TYPE = "FP32-FP32";
    EMBEDDING_DTYPE = "FP32";
    FC_LAYER_DTYPE = "FP32";
    MEMORY_SWAP = false;
    FSU_LOOKAHEAD = 1;
    NUM_TO_GENERATE = static_cast<int>(BD_NUM_TO_GENERATE);
  }

  virtual ~BertDecoder() = default;

  /**
   * @brief Override the (otherwise hardcoded FP32) weight tensor types so the
   *        decoder graph can be built to load quantized weights.
   *
   * Must be called BEFORE initialize() (the dtypes are baked into the graph at
   * compile time). The orchestrating caller invokes this with the values
   * from nntr_config when running a quantized model.
   *
   * @param model_tensor_type "<weight>-<activation>" (e.g. "Q4_0-FP32").
   * @param fc_layer_dtype FC weight dtype string (e.g. "Q4_0").
   * @param embedding_dtype embedding lookup-table weight dtype string
   *        (e.g. "Q4_0", "Q6_K", or "FP32"). Defaults to "FP32" so callers that
   *        do not quantize embeddings keep the original behaviour. The
   *        word/pos/type embedding_layer createLayer calls pass EMBEDDING_DTYPE
   *        as their per-layer weight_dtype, so the decoder graph is built to
   *        load embeddings at this dtype.
   */
  void setTensorTypes(const std::string &model_tensor_type,
                      const std::string &fc_layer_dtype,
                      const std::string &embedding_dtype = "FP32") {
    MODEL_TENSOR_TYPE = model_tensor_type;
    FC_LAYER_DTYPE = fc_layer_dtype;
    EMBEDDING_DTYPE = embedding_dtype;
  }

  /**
   * @brief Initialize the decoder model (register layers, build graph,
   * compile).
   */
  void initialize() override;

  /**
   * @brief Smoke test: allocate KV cache, bind it, run one incremental step.
   * @return true on success, false on failure.
   */
  bool smokeTest();

  /**
   * @brief Allocate the host-owned KV cache and bind it to mha_core
   * placeholders. Idempotent — only the first call allocates; subsequent calls
   * are no-ops unless the cache was already freed.
   */
  void allocateAndBindKVCache();

  /**
   * @brief Run a single decode step with the given encoder_hidden and token_id,
   *        save output logits to @p out_npy, print argmax to stdout.
   *
   * Intended for --decoder-init-parity parity testing. Assumes weights are
   * already loaded. Allocates + binds KV cache if not done yet.
   *
   * @param enc_hidden  Pointer to BD_ENC_LEN * BD_DIM floats (encoder hidden)
   * @param token_id    Input token id (default: 101 = [CLS])
   * @param out_npy     Output .npy path for logits [1,1,1,BD_NUM_VOCAB]
   * @return argmax token index, or -1 on failure
   */
  int decodeStep(const float *enc_hidden, int token_id,
                 const std::string &out_npy);

  /**
   * @brief Populate the per-layer cross-attention K/V cache with the real
   *        encoder K/V (encoder_hidden @ Wk_cross / @ Wv_cross), so the decode
   *        step's cross-attn (cross_attention=true, read-only) attends the
   *        actual encoder signal instead of zeros.
   *
   * Mechanism: compute K = encoder_hidden @ Wk_cross(+bk) and
   * V = encoder_hidden @ Wv_cross(+bv) directly from each layer's persistent
   * _cross_k / _cross_v FC weights (same dot()+add_i() math as
   * FullyConnectedLayer::forwarding), then store the FP32 result into the
   * cross-cache buffers via Tensor::copyData — the EXACT same FP32->fp16/UINT16
   * encoding and per-row head layout (num_heads * head_dim contiguous) that the
   * self-attn KV-cache write path uses. No hand-encoded fp16 bits.
   *
   * Computing from persistent weights (rather than reading the cross_k/_cross_v
   * FC output tensors after a forward) is required: under INFERENCE memory
   * planning those intermediate output buffers are aliased across layers, so a
   * post-forward read returns the wrong layer's data. Loaded weights never
   * alias.
   *
   * Idempotent w.r.t. allocation; recomputes K/V on every call (Task 7's
   * orchestrator calls this once per image). Assumes weights are loaded.
   *
   * @param enc_hidden  Pointer to BD_ENC_LEN * BD_DIM floats (encoder hidden)
   * @param token_id    Token id driving the throwaway forward (default 101=CLS)
   */
  void prefillCrossCache(const float *enc_hidden, int token_id = 101);

  /**
   * @brief Run ONE autoregressive decode step at the given position.
   *
   * Reads the already-prefilled cross-attention K/V cache (call
   * prefillCrossCache() once per image first) and the self-attention KV cache,
   * writing the new step's self-attn K/V at cache slot @p pos so the cache
   * grows across the generation loop. The cross-cache is fixed (read-only).
   *
   * Feeds token=@p token_id, position_ids=[pos], token_type_ids=[0]. Does NOT
   * reset the KV cache to position 0 (unlike decodeStep) — the caller drives
   * the loop: pos=0,1,2,... The encoder_hidden graph input is bound to @p
   * enc_hidden (only the cross-cache it feeds is actually consumed by
   * cross-attn at runtime, but the input slot must still be supplied).
   *
   * @param enc_hidden  Pointer to BD_ENC_LEN * BD_DIM floats (encoder hidden)
   * @param token_id    Input token id for this step
   * @param pos         Sequence position (0-based) for this step
   * @return owned logits buffer [BD_NUM_VOCAB] (caller takes ownership via the
   *         returned vector copy; the internal incremental_inference buffer is
   *         freed before returning)
   */
  std::vector<float> decodeToken(const float *enc_hidden, int token_id,
                                 unsigned int pos);

  /**
   * @brief Reset all mha_core layers' cache_index to @p pos.
   */
  void setKVCachePosition(unsigned int pos);

protected:
  /**
   * @brief Construct the full decoder graph.
   * @return {inputs, output} where inputs = [input0, position_ids,
   *         token_type_ids, encoder_hidden, cache_k_l0..3, cache_v_l0..3]
   *
   * KV cache input layers cache_k_l{i} / cache_v_l{i} are leaf nodes created
   * inside createSelfAttentionBlock via createKVCachePlaceholders.  They are
   * reachable during backward graph traversal from the output without being
   * listed in the primary inputs vector.
   */
  std::pair<std::vector<Tensor>, Tensor> constructDecoderGraph();

  /**
   * @brief Create one causal self-attention block with residual + post
   * LayerNorm.
   *
   * Layer names follow the weight converter's collect_decoder order:
   *   dec_layer{i}_self_q  / _self_k  / _self_v
   *   dec_layer{i}_self_attn
   *   dec_layer{i}_self_out
   *   dec_layer{i}_self_res
   *   dec_layer{i}_self_ln
   *
   * @param layer_id  decoder layer index (0..NUM_LAYERS-1)
   * @param h         input hidden state tensor
   * @return output tensor after residual + LayerNorm
   */
  Tensor createSelfAttentionBlock(int layer_id, Tensor h);

  /**
   * @brief Create the per-layer cross-attention K/V cache placeholders.
   *
   * These are static read-only buffers of @c BD_ENC_LEN rows that are filled
   * once from the encoder hidden states (mechanism finalized in Task 7) and
   * read by mha_core in cross_attention mode. Declared as input leaf nodes
   * named cross_cache_k_l{layer_id} / cross_cache_v_l{layer_id}.
   *
   * @param layer_id  decoder layer index
   * @param enc_len   number of encoder rows (BD_ENC_LEN)
   * @return {cross_cache_k, cross_cache_v} placeholder tensors
   */
  std::pair<Tensor, Tensor> createCrossCachePlaceholders(int layer_id,
                                                         unsigned int enc_len);

  /**
   * @brief Create one cross-attention block with residual + post LayerNorm.
   *
   * Q-proj (bias) runs on the post-self-attn-norm decoder hidden; K/V-proj
   * (bias) run on @p encoder_hidden; mha_core runs in external-cache mode with
   * cross_attention=true (num_heads=4, num_heads_kv=4, use_rope=false);
   * followed by output dense (bias) + residual + output LayerNorm (eps 1e-12).
   *
   * Layer names follow the weight converter's collect_decoder order, emitted
   * immediately after dec_layer{i}_self_ln:
   *   dec_layer{i}_cross_q / _cross_k / _cross_v
   *   dec_layer{i}_cross_attn
   *   dec_layer{i}_cross_out  (crossattention.output.dense)
   *   dec_layer{i}_cross_res
   *   dec_layer{i}_cross_ln   (crossattention.output.LayerNorm)
   *
   * @param layer_id        decoder layer index
   * @param h               post-self-attn-norm decoder hidden state
   * @param encoder_hidden  encoder hidden states [1,1,BD_ENC_LEN,BD_DIM]
   * @return output tensor after residual + LayerNorm
   */
  Tensor createCrossAttentionBlock(int layer_id, Tensor h,
                                   Tensor encoder_hidden);

  /**
   * @brief Create one FFN sub-block with GELU + residual + post LayerNorm.
   *
   * Layer names follow the weight converter's collect_decoder order, emitted
   * immediately after dec_layer{i}_cross_ln:
   *   dec_layer{i}_ffn_inter  (intermediate.dense, unit=1024, bias)
   *   dec_layer{i}_ffn_gelu   (activation, gelu)
   *   dec_layer{i}_ffn_out    (output.dense, unit=256, bias)
   *   dec_layer{i}_ffn_res    (addition: h + ffn_out)
   *   dec_layer{i}_ffn_ln     (output.LayerNorm, eps=1e-12, axis=3)
   *
   * @param layer_id  decoder layer index
   * @param h         post-cross-attn-norm decoder hidden state
   * @return output tensor after residual + LayerNorm
   */
  Tensor createFFNBlock(int layer_id, Tensor h);

  /**
   * @brief Lazily allocate the host-owned cross-attention K/V cache buffers
   *        (UINT16 storage holding fp16 bits — same dtype as the self-attn
   *        KV cache; fixes the prior UINT16-declared / float-bound mismatch).
   */
  void allocateCrossCacheBuffers();

  /**
   * @brief Run one decoder forward, binding the given (prefilled) cross-cache
   *        buffers. Used by decodeStep(); mha_core reads the cross-cache in
   *        cross_attention=true (read-only) mode.
   * @return owning pointers from incremental_inference (caller frees).
   */
  std::vector<float *> runDecoderForward(const float *enc_hidden, int token_id,
                                         std::vector<uint16_t> *cross_k_bufs,
                                         std::vector<uint16_t> *cross_v_bufs,
                                         unsigned int pos = 0);

  /**
   * @brief Externalized KV cache (host-owned).  Allocated lazily by
   *        allocateAndBindKVCache() after initialize().
   */
  KVCacheManager kv_cache_;
  bool kv_cache_bound_ = false; /**< True once KV cache tensors are bound */

  /**
   * @brief Host-owned cross-attention K/V cache, one UINT16 buffer per layer
   *        of BD_ENC_LEN * kv_width fp16-encoded elements. Populated by
   *        prefillCrossCache(); read by the decode-step cross-attn in
   *        cross_attention=true (read-only) mode.
   */
  std::vector<std::vector<uint16_t>> cross_cache_k_;
  std::vector<std::vector<uint16_t>> cross_cache_v_;
  /**
   * @brief True once prefillCrossCache() has populated the cross caches for the
   *        current image. decodeToken() checks this to guard against running a
   *        decode step before the cross-attention K/V have been computed.
   */
  bool cross_cache_prefilled_ = false;
};

} // namespace causallm

#endif /* __BERT_DECODER_H__ */
