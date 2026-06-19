// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   screenai_caption.h
 * @date   19 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  ScreenAI caption orchestrator: SigLIP2 vision encoder + BERT decoder.
 *
 * Composes the (already-proven) Siglip2VisionEncoder and BertDecoder sub-models
 * into an image-captioning pipeline. The encoder runs once per image producing
 * [1,196,256] projected hidden states; those prefill the decoder's per-layer
 * cross-attention K/V cache (fixed for the whole generation). The decoder then
 * greedily generates caption tokens autoregressively, growing its self-attn KV
 * cache one step at a time, until EOS or NUM_TO_GENERATE. Token IDs are decoded
 * with the WordPiece tokenizer.
 *
 * Generation loop (greedy, do_sample=false):
 *   enc = encoder.encode(image)              // [196,256], run ONCE
 *   decoder.prefillCrossCache(enc)           // fill cross-cache ONCE
 *   cur = decoder_start_token_id (101); pos = 0
 *   loop up to NUM_TO_GENERATE:
 *     logits = decoder.decodeToken(enc, cur, pos)   // self-KV grows; cross
 * fixed next = argmax(logits); append; if next==eos(102) break cur = next; pos
 * += 1 caption = tokenizer.decode(token_ids)
 */

#ifndef __SCREENAI_CAPTION_H__
#define __SCREENAI_CAPTION_H__

#include <memory>
#include <string>
#include <vector>

#include "bert_decoder/bert_decoder.h"
#include "json.hpp"
#include "siglip2/siglip2_vision_encoder.h"
#include <transformer.h>

namespace causallm {

/**
 * @brief ScreenAICaption class — encoder/decoder caption orchestrator.
 */
class ScreenAICaption : public Transformer {

public:
  static constexpr const char *architectures = "ScreenAICaption";

  // BERT-decoder special token ids (WordPiece / bert-base-uncased vocab).
  static constexpr int DECODER_START_TOKEN_ID = 101; // [CLS]
  static constexpr int EOS_TOKEN_ID = 102;           // [SEP]

  /**
   * @brief Construct a ScreenAICaption from the model configs.
   *
   * Stores the configs and reads the orchestrator-level nntr_config fields
   * (encoder/decoder weight file names, num_to_generate, tokenizer_file). The
   * sub-models are not built until initialize(). The base Transformer is
   * constructed with skip_tokenizer semantics handled here (we load the
   * tokenizer ourselves from tokenizer_file).
   */
  ScreenAICaption(json &cfg, json &generation_cfg, json &nntr_cfg);

  /**
   * @brief Destroy the ScreenAICaption object.
   */
  ~ScreenAICaption() override = default;

  /**
   * @brief Build + compile both sub-models and allocate caches.
   *
   * Constructs and initializes the Siglip2VisionEncoder and BertDecoder,
   * allocates + binds the decoder self-attn KV cache, and (since the cross
   * caches are filled per-image) leaves cross-cache prefill to run().
   */
  void initialize() override;

  /**
   * @brief Load both sub-model weight files.
   *
   * @param weight_path Ignored — the encoder/decoder weight paths come from
   *        encoder_model_file_name / decoder_model_file_name in nntr_config,
   *        resolved relative to the model directory. Kept to match the base
   *        Transformer signature (main.cpp passes model_file_name).
   */
  void load_weight(const std::string &weight_path) override;

  /**
   * @brief Save both sub-model weights with optional dtype conversion.
   *
   * The base Transformer::save_weight operates on a single `model` member that
   * ScreenAICaption does not populate (it owns encoder_/decoder_ sub-models,
   * each a Transformer with its own `model` + save_weight). This override
   * DELEGATES: it derives encoder/decoder output paths (mirroring the
   * encoder/decoder input naming with a dtype suffix from @p weight_path) and
   * calls encoder_->save_weight() / decoder_->save_weight() with the same
   * dtype / layer_dtype_map / isa. Used by nntr_quantize.
   *
   * @param weight_path Output path supplied by the caller (e.g. the quantize
   *        tool's dst path). Its dtype suffix / extension is reused to name the
   *        two derived encoder/decoder output files.
   * @param dtype Global target dtype (NONE = keep original per-tensor dtype).
   * @param layer_dtype_map Per-layer dtype overrides (the real quant driver).
   * @param target_isa Target instruction set for the quantized weights.
   */
  void save_weight(const std::string &weight_path,
                   ml::train::TensorDim::DataType dtype,
                   const std::map<std::string, ml::train::TensorDim::DataType>
                     &layer_dtype_map,
                   ml::train::ISA target_isa) override;

  /**
   * @brief Run captioning on an image path (encode via C++ resize) and emit.
   *
   * @param prompt        Image file path (reusing the prompt arg as the path).
   * @param do_sample     Ignored (greedy only).
   * @param system_prompt Ignored.
   * @param tail_prompt   Ignored.
   * @param log_output    If true, print the caption to stdout.
   */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = WSTR(), const WSTR tail_prompt = WSTR(),
           bool log_output = true) override;

  /**
   * @brief Run captioning starting from a pre-loaded pixel tensor (bypass the
   *        C++ resize), for parity Stage A (PyTorch-preprocessed pixels).
   * @param pixel_data  [1,3,224,224] CHW float32 (PIL/PyTorch preprocessed).
   * @param pixel_count Number of floats (must equal 3*224*224).
   * @return generated caption string.
   */
  std::string runFromPixels(const float *pixel_data, size_t pixel_count,
                            bool log_output = true);

  /**
   * @brief Generated token IDs from the most recent run (for verification).
   *        Includes the leading start token (101) and trailing EOS (102),
   *        matching the PyTorch golden token_ids layout.
   */
  const std::vector<int> &token_ids() const { return token_ids_; }

  /**
   * @brief Decoded caption text from the most recent run.
   */
  const std::string &caption() const { return caption_; }

  /**
   * @brief Resolve the model directory (used to find weight files / tokenizer).
   */
  void setModelDir(const std::string &dir) { model_dir_ = dir; }

private:
  /**
   * @brief Single shared decode path: cross-cache prefill + full autoregressive
   *        generation loop + caption decoding + banner + metrics population.
   *
   * Both run() and runFromPixels() delegate here after producing the encoder
   * hidden states so there is ONE argmax, ONE EOS check, and ONE place where
   * performance_metrics and has_run_ are updated.
   *
   * Token-counting convention:
   *   prefill phase  = encoder forward pass (BD_ENC_LEN patch tokens),
   *                    timed by the caller and passed in as @p prefill_us.
   *   generation phase = all decoded tokens in the autoregressive loop
   *                    (pos 0 .. EOS/max), timed inside this function.
   *   prefill_tokens + generation_tokens = total tokens produced.
   *
   * @param enc_hidden  BD_ENC_LEN * BD_DIM floats (projected encoder hidden).
   * @param log_output  If true, print token_ids, caption, and timing banner.
   * @param prefill_us  Encoder forward-pass duration in microseconds (from
   *                    caller's high_resolution_clock measurement).
   */
  void generateFromEncoderHidden(const std::vector<float> &enc_hidden,
                                 bool log_output, long long prefill_us);

  json cfg_;
  json generation_cfg_;
  json nntr_cfg_;

  std::string model_dir_;           /**< Directory holding weights/tokenizer. */
  std::string encoder_weight_file_; /**< Resolved encoder weight path. */
  std::string decoder_weight_file_; /**< Resolved decoder weight path. */

  std::unique_ptr<Siglip2VisionEncoder> encoder_;
  std::unique_ptr<BertDecoder> decoder_;

  std::vector<int> token_ids_; /**< Last run's token ids (incl. 101 ... 102). */
  std::string caption_;        /**< Last run's decoded caption. */
};

} // namespace causallm

#endif /* __SCREENAI_CAPTION_H__ */
