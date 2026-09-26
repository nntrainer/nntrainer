// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   transformer.h
 * @brief  Base Transformer class shared by CausalLM and encoder models.
 * @date   31 Dec 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This transformer.h constructs a class for Transformer model which can
 * be a parent of CausalLM and Encoder models with transformer structure.
 * @note   This transformer assumes the following structure :
 *
 *           [Input]
 *              |
 *         [Embedding]
 *              |
 *        [Decoder Block] (repeated N times)
 *              |
 *          [RMSNorm]
 *
 */
#ifndef __TRANSFORMER_H__
#define __TRANSFORMER_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#define WSTR std::string
#define WCHAR_P std::string &
#else
#define WIN_EXPORT
#define WSTR std::string
#define WCHAR_P std::string &
#endif

#include <algorithm> // std::min (prefillChunk clamp)
#include <context.h> // nntrainer::ModelFeatures (T11)
#include <future>    // async tokenizer load (round-13 init overlap)
#include <layer.h>
#include <map>
#include <model.h>
#include <mutex>
#include <random>
#include <stdexcept>
#include <tensor_api.h>
#include <utility>
#include <vector>

#include <limits.h>

#include "json.hpp"
#include "performance_metrics.h"
#include <fstream>
#include <tokenizers_c.h>
#include <tokenizers_cpp.h>

namespace causallm {

/*** ALIAS ****/
using LayerHandle = ml::train::LayerHandle;
using Tensor = ml::train::Tensor;
using ModelHandle = std::unique_ptr<ml::train::Model>;

using json = nlohmann::json;

/**
 * @brief Model Type Enum
 */
enum class ModelType { MODEL, CAUSALLM, EMBEDDING, UNKNOWN };

/**
 * @brief {data, size} pointer pair produced/consumed by multimodal vision
 *        models. The buffer is heap-allocated by the producer (run_image) and
 *        ownership transfers to the caller.
 */
using multimodal_pointer = std::pair<void *, size_t>;

/**
 * @brief Whether the sliding-window KV ring may turn itself ON without an
 *        explicit request.
 * @details The ring is only correct where the attention kernels modulo-map the
 * cache row, which today means the GPU attention paths. The host CPU attention
 * fallback walks absolute rows, so a CPU run must keep the linear cache.
 * NNTR_KV_WINDOW_RING=1 forces the ring on regardless (kernel bring-up).
 */
inline bool kvRingAutoEligible() {
  const char *e = std::getenv("NNTR_ENGINE");
  if (e != nullptr && std::string(e) == "cpu")
    return false;
  if (e != nullptr && std::string(e) == "cuda")
    return true;
#if defined(ENABLE_OPENCL)
  return true; // no explicit engine + an OpenCL build == the gpu engine
#else
  return false;
#endif
}

/**
 * @brief Whether the KV ring is enabled for this process.
 * @details NNTR_KV_WINDOW_RING: '0' disables, '1' forces on, unset means on
 * wherever it is eligible (kvRingAutoEligible).
 */
inline bool kvRingEnabled() {
  const char *g = std::getenv("NNTR_KV_WINDOW_RING");
  if (g && g[0] == '0')
    return false;
  if (g && g[0] == '1')
    return true;
  return kvRingAutoEligible();
}

/**
 * @brief Requested prefill chunk size (0 = no chunking / single-block prefill).
 * @details An explicit NNTR_PREFILL_CHUNK always wins (user override, per-GPU
 * tuning). Otherwise, when the KV ring is enabled, chunking is what bounds a
 * launch's live key span, so the ring picks the chunk: 4096 for every backend.
 * The equal-thermal ring-on sweep is monotone in the chunk but with a poor
 * marginal ratio past 4096 (the next step up buys under a percent of prefill
 * for another GB of working set), and the CUDA tensor-core GEMMs want a large
 * chunk anyway -- so one constant, no backend branch.
 *
 * This is the REQUEST, not what the prefill actually runs: a chunk cannot
 * exceed the activation-plane height it has to fit in, so the number every
 * consumer must use is Transformer::prefillChunk(), which clamps this to
 * INIT_SEQ_LEN. Use that accessor, not this function, anywhere the answer
 * feeds sizing or control flow.
 */
inline unsigned int requestedPrefillChunk() {
  const char *pc = std::getenv("NNTR_PREFILL_CHUNK");
  if (pc && pc[0])
    return static_cast<unsigned int>(std::atoi(pc)); // explicit override wins
  if (!kvRingEnabled())
    return 0u; // chunking is auto-enabled only by the ring
  return 4096u;
}

/**
 * @brief Sliding-window KV ring capacity (NNTR_KV_WINDOW_RING, default off).
 * @details A sliding-window attention layer with local window W only ever
 * attends to the last W keys, so with chunked prefill -- which bounds one
 * launch's live key span to W+C -- its KV storage can be a ring of Wcap rows
 * instead of the full max_seq.
 *
 * The ring is the memory half of the sliding-window design contract: a
 * W-bounded attention deserves W-bounded storage, and the rows past W are dead
 * weight the mask can never read again. It is therefore ON wherever it is
 * correct (see kvRingEnabled); NNTR_KV_WINDOW_RING=0 opts out.
 *
 * Returns Wcap (the physical row capacity to allocate and modulo-index) for a
 * sliding layer, or 0 meaning "no ring, keep full max_seq" (full-attention
 * layer, ring disabled, no chunking, or no benefit). Every site -- placeholder
 * shape, KV allocation, cache write, attention kernel dispatch -- computes Wcap
 * from THIS one function so they stay consistent; a disagreement is an
 * out-of-bounds write, not a wrong answer.
 *
 * Wcap is a multiple of C and >= W + C: a multiple of C means a C-aligned chunk
 * write never straddles the wrap seam (it stays one contiguous slice), and
 * >= W + C means the live window [pos-W+1, pos+C) never self-collides mod Wcap.
 * Returning 0 keeps the exact pre-ring behaviour, so ring-off is bit-identical.
 *
 * @param chunk C, the chunk the prefill ACTUALLY runs -- Transformer::
 * prefillChunk(), not requestedPrefillChunk(). Sizing the ring off the raw
 * request while the prefill runs the clamped one buys nothing (the live span
 * is bounded by the chunk that runs) and costs a Wcap up to 4x too large.
 * It is a parameter rather than a call so that the caller's chunk and this
 * cap cannot drift apart.
 */
inline unsigned int kvRingCap(unsigned int local_window, unsigned int max_seq,
                              unsigned int chunk) {
  if (!kvRingEnabled())
    return 0; // ring off -> full max_seq (bit-identical legacy)
  if (local_window == 0 || local_window >= max_seq)
    return 0; // full-attention layer -> no ring
  const unsigned int C = chunk;
  if (C == 0)
    return 0; // the ring requires chunked prefill to bound the live span
  // multiple of C, >= W + C (headroom so the window never wraps onto itself).
  const unsigned int cap = (local_window / C + 2u) * C;
  return (cap < max_seq) ? cap : 0u; // no benefit if it would not shrink
}

/**
 * @brief Non-owning logits processor hook for token generation
 */
class LogitsProcessor {
public:
  /**
   * @brief Destroy the LogitsProcessor object
   */
  virtual ~LogitsProcessor() = default;

  /**
   * @brief Mutate one batch row of logits before token selection
   * @param logits FP32 logits for a single batch row
   * @param vocab_size Number of logits in the row
   * @param batch_index Batch row index
   */
  virtual void process(float *logits, unsigned int vocab_size,
                       unsigned int batch_index) = 0;

  /**
   * @brief Receive the selected token after token selection
   * @param token_id Selected token id
   * @param batch_index Batch row index
   */
  virtual void acceptToken(unsigned int token_id, unsigned int batch_index) = 0;

  /**
   * @brief Reset processor state when requested by the caller
   */
  virtual void reset() {}
};

/**
 * @brief Transformer Class
 */
WIN_EXPORT class Transformer {

public:
  /**
   * @brief Construct a new Transformer object
   * @param cfg Configuration for the model (config.json)
   * @param generation_cfg Configuration for the generation (generation.json)
   * @param nntr_cfg Configuration for nntrainer (nntrainer_config.json)
   * @param model_type Type of the model (default: ModelType::MODEL)
   */
  Transformer(json &cfg, json &generation_cfg, json &nntr_cfg,
              ModelType model_type = ModelType::MODEL);

  /**
   * @brief Empty constructor for Transformer.
   * @brief Child Class Needs to implement all features of the original
   * Transformer constructor
   */
  Transformer() {}

  /**
   * @brief Destroy the Transformer object
   */
  virtual ~Transformer() {}

  /**
   * @brief Initialize and Construct the Transformer model
   */
  virtual void initialize();

  /**
   * @brief Load the model weights from a file
   */
  virtual void load_weight(const std::string &weight_path);

  /**
   * @brief Repack all QS4CX weights after loading
   * @note Must be called after load_weight() for QS4CX quantized tensors
   * @note Prepares weights for efficient computation by eagerly packing them
   */
  virtual void repack_weight();

  /**
   * @brief Save the weight to a file
   */
  virtual void save_weight(const std::string &weight_path);
  /**
   * @brief Save the weight to a file with type conversion
   * @param weight_path Path to save the weight file
   * @param dtype Global target data type for all layers (NONE = keep original)
   * @param layer_dtype_map Per-layer data type overrides (layer_name -> dtype)
   * @param target_isa Target ISA for quantization (default: DEFAULT)
   */
  virtual void
  save_weight(const std::string &weight_path,
              ml::train::TensorDim::DataType dtype,
              const std::map<std::string, ml::train::TensorDim::DataType>
                &layer_dtype_map = {},
              ml::train::ISA target_isa = ml::train::ISA::DEFAULT);

  /**
   * @brief run the Transformer model
   */
  virtual void run(const WSTR prompt, bool do_sample = false,
                   const WSTR system_prompt = WSTR(),
                   const WSTR tail_prompt = WSTR(), bool log_output = true);

  // ── Multimodal composition interface (model-agnostic) ──────────────────
  // Lets a generic composer drive any [vision producer, LLM consumer] pair
  // through base pointers, without knowing the concrete model type.
  // Default implementations mean "this role is not supported by this model".

  /** Embedding-CONSUMER (LLM): bytes of one token embedding (0 ⇒ no table). */
  virtual size_t embeddingBytesPerToken() const { return 0; }

  /** Embedding-CONSUMER (LLM): embedding of @p token_id, or nullptr. */
  virtual const void *lookupEmbedding(int token_id) const {
    (void)token_id;
    return nullptr;
  }

  /** Embedding-CONSUMER (LLM): (scale, offset) of the embedding quant space. */
  virtual std::pair<float, int> get_embedding_info() { return {1.0f, 0}; }

  /** Embedding-CONSUMER (LLM): run generation from precomputed embeddings. */
  virtual void run_with_embeddings(const void *prefill_embeds, size_t n_tokens,
                                   std::vector<int> seed_tokens, bool do_sample,
                                   bool log_output) {
    (void)prefill_embeds;
    (void)n_tokens;
    (void)seed_tokens;
    (void)do_sample;
    (void)log_output;
    throw std::runtime_error("run_with_embeddings not supported by this model");
  }

  /** Embedding-PRODUCER (vision): set the (scale, offset) it should emit in. */
  virtual void set_quant_param(float scale, int offset) {
    (void)scale;
    (void)offset;
  }

  /** Embedding-PRODUCER (vision): encode an image into LLM-space embeddings.
   *  Returns a heap buffer (caller frees) of size {bytes}; the default
   *  {nullptr,0} means "this model is not a vision producer". */
  virtual multimodal_pointer
  run_image(const WSTR prompt, multimodal_pointer image, int image_height,
            int image_width, bool do_sample = false,
            const WSTR system_prompt = WSTR(), const WSTR tail_prompt = WSTR(),
            bool log_output = true) {
    (void)prompt;
    (void)image;
    (void)image_height;
    (void)image_width;
    (void)do_sample;
    (void)system_prompt;
    (void)tail_prompt;
    (void)log_output;
    return {nullptr, 0};
  }

  /** Current KV-cache length (0 if the model has no persistent KV cache). */
  virtual int getKvLen() const { return 0; }

  /**
   * @brief Per-layer KV-cache width (num_kv_heads * head_dim) for the layer.
   * @details The single per-model degree of freedom the generic
   * allocateAndBindKVCache needs: uniform-geometry models use this default,
   * variable-geometry models (whose sliding and global layers carry different
   * head dims / KV head counts) override it. Lets one base allocate/bind serve
   * every model instead of a per-model copy of the whole routine.
   */
  virtual unsigned int getKVCacheWidth(int layer_id) const {
    (void)layer_id;
    return static_cast<unsigned int>(NUM_KEY_VALUE_HEADS) *
           static_cast<unsigned int>(HEAD_DIM);
  }

  /**
   * @brief The prefill chunk this model actually runs, in query rows
   *        (0 = no chunking / single-block prefill).
   * @details requestedPrefillChunk() is only the request; one chunk is fed at
   * input row 0 of the activation plane, which is built INIT_SEQ_LEN rows tall
   * (transformer.cpp constructModel), so a larger chunk would overflow it. The
   * clamp therefore belongs to the model, which is the only thing that knows
   * its plane -- and every consumer must read the SAME clamped number:
   *
   *   - the prompt budget (a chunked prefill is bounded by the KV budget, an
   *     unchunked one by the plane),
   *   - the prefill drive loop (chunk length per forward),
   *   - the KV ring capacity (Wcap is a multiple of the chunk).
   *
   * They used to disagree: the drive loop and the budget gate keyed off the
   * NNTR_PREFILL_CHUNK env var being SET, so the ring's auto-chunk turned on
   * the ring but not the chunking, and a long prompt was silently truncated to
   * INIT_SEQ_LEN; the ring cap meanwhile used the unclamped request and
   * over-allocated. One accessor, one answer.
   */
  unsigned int prefillChunk() const {
    const unsigned int c = requestedPrefillChunk();
    return c ? std::min<unsigned int>(c,
                                      static_cast<unsigned int>(INIT_SEQ_LEN))
             : 0u;
  }

  /**
   * @brief Per-layer physical KV-cache row count.
   * @details The row counterpart of getKVCacheWidth(). Every KV sizing site --
   * each model's placeholder factory and the KVCacheManager allocation -- must
   * agree on how many rows a layer's cache has, and hard-coding MAX_SEQ_LEN in
   * each of them made "how many rows" un-answerable from one place.
   *
   * A sliding-window layer under the KV ring stores only kvRingCap() rows;
   * every other layer keeps the full context window.
   */
  unsigned int getKVCacheRows(int layer_id) const {
    const unsigned int cap =
      kvRingSupported()
        ? kvRingCap(getLayerSlidingWindow(layer_id),
                    static_cast<unsigned int>(MAX_SEQ_LEN), prefillChunk())
        : 0u;
    return cap ? cap : static_cast<unsigned int>(MAX_SEQ_LEN);
  }

  /**
   * @brief Whether this model's attention can read a ringed KV cache.
   * @details The ring needs an attention path that modulo-maps the cache row.
   * A model whose attention variant does not (the attention-sink variant reads
   * the cache through the host compute path) must keep the linear cache;
   * mha_core makes the same call for the same layers, so the two agree.
   */
  virtual bool kvRingSupported() const { return true; }

  /**
   * @brief Per-layer attention window: the value this model feeds mha_core's
   *        `sliding_window` property for the layer, or UINT_MAX for a
   *        full-attention layer.
   * @details Every model already decided this inside its createAttention
   * override, where nothing outside could see it. Host-side sizing decisions
   * that must agree with the layer's window (KV-cache capacity being the one
   * that matters) need the same answer, so the choice lives here and the
   * createAttention overrides read it. The default is the base pattern rule
   * (every SLIDING_WINDOW_PATTERN-th layer is full attention); models with a
   * layer_types table or a fixed window override it.
   */
  virtual unsigned int getLayerSlidingWindow(int layer_id) const {
    return ((layer_id + 1) % SLIDING_WINDOW_PATTERN) ? SLIDING_WINDOW
                                                     : UINT_MAX;
  }

  /**
   * @brief Get TransformerPerformanceMetrics
   */
  TransformerPerformanceMetrics getPerformanceMetrics() const {
    return performance_metrics;
  }

  /**
   * @brief get the status of run
   */
  bool hasRun() const { return has_run_; }

  /**
   * @brief Get configured vocabulary size
   * @return Vocabulary size
   */
  unsigned int getVocabSize() const { return NUM_VOCAB; }

  /**
   * @brief Override the max number of new tokens for subsequent run() calls
   * @param num_to_generate Max new tokens per run; must leave room within
   *        MAX_SEQ_LEN (run() caps the prompt to MAX_SEQ_LEN - NUM_TO_GENERATE)
   */
  void setNumToGenerate(unsigned int num_to_generate) {
    NUM_TO_GENERATE = static_cast<int>(num_to_generate);
  }

  /**
   * @brief Get the configured max number of new tokens per run
   */
  unsigned int getNumToGenerate() const {
    return static_cast<unsigned int>(NUM_TO_GENERATE);
  }

  /**
   * @brief Get tokenizer owned by this model, or nullptr if no tokenizer exists
   */
  tokenizers::Tokenizer *getTokenizer() {
    ensureTokenizer();
    return tokenizer.get();
  }

  /**
   * @brief Join the async tokenizer load (round-13 init overlap: the ~30MB
   *        tokenizer.json parse runs on a side thread concurrent with graph
   *        compile + weight load; call this before any direct `tokenizer`
   *        member access). Idempotent and cheap after the first call.
   */
  void ensureTokenizer();

  /**
   * @brief Attach a non-owning logits processor
   * @param processor Processor pointer, or nullptr to detach
   */
  virtual void setLogitsProcessor(LogitsProcessor *) {}

  /**
   * @brief Reset attached logits processor state
   */
  virtual void resetLogitsProcessor() {}

protected:
  /**
   * @brief Setup the parameters for the Transformer model
   */
  virtual void setupParameters(json &cfg, json &generation_cfg, json &nntr_cfg);

  /**
   * @brief Construct Model
   * @return {input_tensor, output_tensor} pair representing the symbolic
   *         tensor graph. Derived classes can extend by taking the output
   *         and feeding additional layers before returning.
   */
  virtual std::pair<Tensor, Tensor> constructModel();

  /**
   * @brief Build common CausalLM embedding layer properties
   * @param name Layer name
   * @param in_dim Vocabulary/input dimension
   * @param out_dim Embedding output dimension
   * @param weight_dtype Layer weight dtype
   * @param scale Embedding scale
   * @param quantized_lut_path Optional sidecar LUT path
   * @return Layer property strings
   */
  std::vector<std::string>
  buildEmbeddingLayerProperties(const std::string &name, unsigned int in_dim,
                                unsigned int out_dim,
                                const std::string &weight_dtype, float scale,
                                const std::string &quantized_lut_path) const;

  /**
   * @brief Create one Transformer decoder block (norm + attention + residual +
   *        norm + ffn + residual)
   * @param layer_id index of the decoder block
   * @param input    symbolic input tensor for this block
   * @return symbolic output tensor of the block
   */
  virtual Tensor createTransformerDecoderBlock(const int layer_id,
                                               Tensor input);

  /**
   * @brief Create the attention sub-graph (Q/K/V projections + mha_core +
   *        output projection)
   * @return symbolic output tensor of the attention sub-graph
   */
  virtual Tensor createAttention(const int layer_id, int seq_len, int n_heads,
                                 int head_dim, Tensor query, Tensor key,
                                 Tensor value);

  /**
   * @brief Create the feed-forward sub-graph
   * @return symbolic output tensor of the FFN sub-graph
   */
  virtual Tensor createMlp(const int layer_id, int dim, int hidden_dim,
                           Tensor input);

  /**
   * @brief Declare WHAT THIS MODEL IS as a flat ModelFeatures struct (mlp kind,
   *        q/k/v-norm, sliding window, KV-share, PLE, soft-caps, lm_head,
   * decode path, ...). The resolver pairs it with the backend's DeviceCaps to
   *        produce an ExecPlan, replacing per-model-identity branching. Base
   *        returns the defaults; each {Model}Transformer overrides it.
   */
  virtual nntrainer::ModelFeatures getModelFeatures() const {
    return nntrainer::ModelFeatures{};
  }

  /**
   * @brief Create the per-layer external KV-cache placeholder Tensors that
   *        feed mha_core's input slots 3 and 4. The actual storage is owned
   *        by the host (e.g. KVCacheManager) and is bound at runtime via
   *        Model::setExternalTensors using the names
   *          "cache_k_l<layer_id>" and "cache_v_l<layer_id>".
   * @param layer_id  attention layer index
   * @param n_heads   total query heads (used together with GQA_SIZE to derive
   *                  the KV head count)
   * @return {cache_k, cache_v} symbolic placeholder tensors
   */
  virtual std::pair<Tensor, Tensor>
  createKVCachePlaceholders(const int layer_id, int n_heads);

  /**
   * @brief Wire the attention core's KV cache: 3-input internal-int8 mode
   *        when use_int8, else 5-input external-fp16 mode with per-layer
   *        placeholders. The placeholder factory is virtual
   *        (createKVCachePlaceholders) so a model with a non-uniform cache
   *        width overrides it instead of re-implementing the wiring.
   *        Consolidates the if/else previously duplicated across every
   *        per-model createAttention override.
   * @param layer_id  attention layer index
   * @param n_heads   total query heads (passed through to
   *                  createKVCachePlaceholders)
   * @param mha       the mha_core layer handle to invoke
   * @param q,k,v     query/key/value tensors to feed mha_core
   * @param use_int8  true selects the 3-input internal-int8 cache mode
   * @return attention output tensor
   */
  Tensor wireAttentionKVCache(const int layer_id, int n_heads, LayerHandle mha,
                              Tensor q, Tensor k, Tensor v, bool use_int8);

  /**
   * @brief register CustomLayers
   */
  virtual void registerCustomLayers();

  /**
   * @brief Get model format from weight file extension.
   * @param weight_path Path to the weight file.
   * @return Model format for the given file extension.
   */
  virtual ml::train::ModelFormat
  formatFromExtension(const std::string &weight_path);

  /**
   * @brief register Outputs
   */
  bool is_initialized = false; /**< Flag to check if the model is initialized */
  ModelHandle model;

  /** tokenizer */
  std::unique_ptr<tokenizers::Tokenizer> tokenizer;
  std::future<std::unique_ptr<tokenizers::Tokenizer>>
    tokenizer_future_;            /**< async load; joined by ensureTokenizer */
  std::mutex tokenizer_join_mtx_; /**< ensureTokenizer idempotence guard */

  unsigned int NUM_VOCAB;
  int DIM;
  int HEAD_DIM;
  int INTERMEDIATE_SIZE;
  int NUM_LAYERS;
  bool USE_VOCAB_SELECTION;
  bool TIE_WORD_EMBEDDINGS;
  unsigned int MAX_SEQ_LEN;
  int NUM_HEADS;
  int NUM_KEY_VALUE_HEADS;
  int NUM_TO_GENERATE;
  std::string MODEL_TENSOR_TYPE;
  std::string EMBEDDING_DTYPE; /** embedding dtype */
  std::string FC_LAYER_DTYPE;  /** custom_fc_lora */
  std::string EMBEDDING_FILE_NAME;
  std::string PLE_FILE_NAME;
  /** nntr_quantize --ple_sidecar / --embd_sidecar: save() writes the table to
   *  these paths instead of the model file (extraction-time keys, not runtime
   *  ones) */
  std::string PLE_SIDECAR_EXPORT;
  std::string EMBD_SIDECAR_EXPORT;
  /** untie lm_head from the input embedding (separate FC weight, not shared
   *  with the embedding table). Lives here (not CausalLM) because embedding0's
   *  layer-type choice — tied TieWordEmbedding vs untied embedding_layer —
   *  happens in <model>Transformer::constructModel scope. A dedicated flag
   *  (not derived from LMHEAD_DTYPE) so the quantizer builds the same untied
   *  graph from FP32 source weights while the dtype map quantizes
   *  output_of_causallm on save. */
  bool LMHEAD_UNTIE = false;

  /** [lmhead-tie-lut] re-tie the UNTIED lm_head onto the embedding sidecar
   *  LUT at dispatch time (CUDA): the LUT payload+scales go device-resident
   *  once at load and the head FC reads them instead of building its own
   *  dp4a weight cache, so the head's QS4CX record stops costing VRAM (and
   *  its host payload is droppable outright). Graph shape is untouched --
   *  this is a routing flag, meaningful only with lmhead_untie and an
   *  embedding_file_name sidecar; anything missing at load falls back to the
   *  fp-act route with a warning. */
  bool LMHEAD_TIE_LUT = false;

  unsigned int SLIDING_WINDOW = UINT_MAX;
  unsigned int SLIDING_WINDOW_PATTERN = 5;
  unsigned int ROPE_THETA = 10000; /**< RoPE theta value */
  float NORM_EPS = 1e-5;           /**< RMSNorm epsilon value */
  float EMBEDDING_SCALE = 1.0f;
  int GQA_SIZE;

  unsigned int BATCH_SIZE;              /**< Batch size for the model */
  unsigned int INIT_SEQ_LEN;            /**< Initial sequence length */
  unsigned int MAX_POSITION_EMBEDDINGS; /**< max_position embeddings */
  bool MEMORY_SWAP;                     /**< memory swap option */
  unsigned int FSU_LOOKAHEAD;
  float ATTN_LOGIT_SOFTCAPPING = 0.0f; /**< attention logit softcapping */
  bool IS_CAUSAL = true;

  // Performance metrics
  TransformerPerformanceMetrics performance_metrics;

  bool has_run_ = false;
};
/**
 * Loads JSON data from a file with detailed error handling
 * @param file_path Path to JSON file
 * @return JSON object
 * @throws std::runtime_error on file open or parse failure
 */
inline json LoadJsonFile(const std::string &file_path) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + file_path +
                             " | Reason: " + std::strerror(errno));
  }

  try {
    json data;
    file >> data;
    return data;
  } catch (const json::parse_error &e) {
    throw std::runtime_error("JSON parse error in " + file_path +
                             " | Details: " + e.what());
  }
}
} // namespace causallm

#endif
