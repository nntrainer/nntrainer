// SPDX-License-Identifier: Apache-2.0
/** @file htp_weight_preparation.h
 * @brief HTP preparation service shared by independent model classes.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#ifndef CAUSALLM_HTP_WEIGHT_PREPARATION_H
#define CAUSALLM_HTP_WEIGHT_PREPARATION_H
#include <string>
namespace ml {
namespace train {
class Model;
}
} // namespace ml
namespace causallm {
struct HtpWeightPreparation {
  std::string static_activation_file, weight_dir, fc_wh_file, fc_wh_format;
  unsigned num_experts, prefill_rows;
  bool all_linear_htp;
};
void prepareHtpWeights(ml::train::Model &model,
                       const HtpWeightPreparation &options);
} // namespace causallm
#endif
