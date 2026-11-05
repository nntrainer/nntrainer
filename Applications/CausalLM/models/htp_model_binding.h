// SPDX-License-Identifier: Apache-2.0
/** @file htp_model_binding.h
 * @brief Non-owning typed access to weights in the existing model graph.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#ifndef CAUSALLM_HTP_MODEL_BINDING_H
#define CAUSALLM_HTP_MODEL_BINDING_H
#include <layer_context.h>
#include <map>
#include <model.h>
#include <stdexcept>
#include <string>
namespace causallm {
class HtpModelBinding {
public:
  explicit HtpModelBinding(ml::train::Model &model) {
    model.forEachLayer(
      [this](ml::train::Layer &layer, nntrainer::RunLayerContext &context,
             void *) {
        for (auto *weight : context.getWeights()) {
          const auto &name = weight->getName();
          weights_[name] = &weight->getVariableRef();
          const auto colon = name.rfind(':');
          if (colon != std::string::npos)
            weights_.emplace(layer.getName() + name.substr(colon),
                             &weight->getVariableRef());
        }
      },
      nullptr);
  }
  bool contains(const std::string &name) const { return weights_.count(name); }
  std::string choose(const std::string &folded, const std::string &own) const {
    return contains(folded) ? folded : own;
  }
  nntrainer::Tensor &weight(const std::string &name) const {
    auto it = weights_.find(name);
    if (it == weights_.end())
      throw std::invalid_argument("HTP binding: missing weight " + name);
    return *it->second;
  }
  const float *f32(const std::string &name, size_t count) const {
    auto &tensor = weight(name);
    if (tensor.getDataType() != ml::train::TensorDim::DataType::FP32 ||
        tensor.size() != count)
      throw std::invalid_argument("HTP binding: invalid FP32 parameter " +
                                  name);
    return tensor.getData<float>();
  }

private:
  std::map<std::string, nntrainer::Tensor *> weights_;
};
} // namespace causallm
#endif
