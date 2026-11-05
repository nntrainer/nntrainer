// SPDX-License-Identifier: Apache-2.0
/** @file htp_mha_core.h
 * @brief Compatibility type selecting the explicit FP16 cache contract.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#ifndef CAUSALLM_HTP_MHA_CORE_H
#define CAUSALLM_HTP_MHA_CORE_H
#include <mha_core.h>
namespace causallm {
namespace htp_mha_props = nntrainer::attention_props;
class HtpMHACoreLayer : public MHACoreLayer {
public:
  HtpMHACoreLayer() : MHACoreLayer(true) {}
  const std::string getType() const override { return type; }
  inline static const std::string type = "htp_mha_core";
};
} // namespace causallm
#endif
