// SPDX-License-Identifier: Apache-2.0
/** @file htp_qkv_layer.h
 * @brief Compatibility type for graphs saved before QKV consolidation.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#ifndef CAUSALLM_HTP_QKV_LAYER_H
#define CAUSALLM_HTP_QKV_LAYER_H
#include <qkv_layer.h>
namespace causallm {
class HtpQKVLayer : public nntrainer::QKVLayer {
public:
  HtpQKVLayer() : QKVLayer(true) {}
  const std::string getType() const override { return type; }
  inline static const std::string type = "htp_qkv_layer";
};
} // namespace causallm
#endif
