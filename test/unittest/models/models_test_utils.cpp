// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   models_test_utils.cpp
 * @date   15 Oct 2020
 * @brief  Models test utils
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <gtest/gtest.h>

#include <memory>

#include <identity_layer.h>
#include <models_test_utils.h>
#include <multiout_layer.h>
#include <nntrainer_test_util.h>
#include <tensor.h>

/**
 * @brief Get the Models Path object
 *
 * @param file_name file name
 * @return const std::string model path
 */
static const std::string getModelsPath(const std::string &file_name) {
  return getResPath(file_name, {"test", "unittest_models"});
}

using namespace nntrainer;

static sharedConstTensors toSharedTensors(const std::vector<Tensor> &ts) {
  sharedConstTensors sts;
  sts.reserve(ts.size());
  std::transform(ts.begin(), ts.end(), std::back_inserter(sts),
                 [](const auto &ts) { return MAKE_SHARED_TENSOR(ts); });
  return sts;
}

/**
 * @brief verify tensor to the reference and throw if not match to stop
 *
 * @param actual actual tensor
 * @param expected reference tensor
 * @param error_msg error msg to print if not match
 */
static void verify(const nntrainer::Tensor &actual,
                   const nntrainer::Tensor &expected,
                   const std::string &error_msg) {
  bool equal = false;

  if (actual.getDataType() == ml::train::TensorDim::DataType::FP32 &&
      expected.getDataType() == ml::train::TensorDim::DataType::FP32) {
    equal = (actual == expected);
    if (!equal) {
      float mseError = mse<float>(actual.getData<float>(),
                                  expected.getData<float>(), actual.size());
      if (mseError > 10 - 4) {
        equal = false;
      } else {
        equal = true;
      }
    }
  }

#ifdef ENABLE_FP16
  if (!equal) {
    if (actual.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        expected.getDataType() == ml::train::TensorDim::DataType::FP16) {
      float mseError = mse<_FP16>(actual.getData<_FP16>(),
                                  expected.getData<_FP16>(), actual.size());
      if (mseError > 10 - 2) {
        equal = false;
      } else {
        equal = true;
      }
    }
  }
#endif

  if (!equal) {
    nntrainer::Tensor diff = actual.subtract(expected);
    const float *diff_data = diff.getData();

    std::cout
      << "============================================================\n";
    std::cout << "\033[1;33m" << error_msg << "\033[0m\n";
    std::cout << "\033[1;33mcurrent\033[0m  " << actual.getName() << " - "
              << actual << "\033[1;33mexpected\033[0m  " << expected.getName()
              << " - " << expected;

    if (actual.getDim() == expected.getDim()) {
      std::cout << "\033[1;33mdifference\033[0m " << diff;
      std::cout << "number of data: " << diff.size() << std::endl;
      std::cout << "\033[4;33mMAX DIFF: "
                << *std::max_element(diff_data, diff_data + diff.size(),
                                     [](auto v1, auto v2) {
                                       return std::fabs(v1) < std::fabs(v2);
                                     })
                << "\033[0m\n";
    }
    std::stringstream ss;
    ss << "\033[4;33m" << error_msg << "\033[0m";
    throw std::invalid_argument(ss.str().c_str());
  }
}

/**
 * @brief verify tensor to the reference and throw if not match to stop
 *
 * @param actual actual tensor vector
 * @param expected reference tensor vector
 * @param error_msg error msg to print if not match
 */
static void verify(const std::vector<nntrainer::Tensor> &actual,
                   const std::vector<nntrainer::Tensor> &expected,
                   const std::string &error_msg) {
  NNTR_THROW_IF(actual.size() != expected.size(), std::invalid_argument)
    << error_msg;

  for (size_t i = 0; i < actual.size(); ++i) {
    verify(actual[i], expected[i], error_msg);
  }
}

/**
 * @brief Iteration abstraction for golden test v2
 *
 */
class IterationForGolden {
public:
  IterationForGolden(NeuralNetwork *nn) : nn(nn) {
    auto in_dims = nn->getInputDimension();
    auto out_dims = nn->getOutputDimension();

    // The golden-file recorder (recorder_v2.py) always writes inputs, labels,
    // and outputs as float32 with a 4-byte (int32) element-count prefix.  When
    // the model's input/output dtype is FP16 the tensors constructed directly
    // from the model dims would be read with a 2-byte prefix + 2 bytes/element,
    // causing a stream offset error and an uncaught exception.  Use FP32
    // read-buffers for these tensors so the binary layout matches the file.
    auto to_fp32_dims = [](const std::vector<TensorDim> &dims) {
      std::vector<TensorDim> fp32_dims;
      fp32_dims.reserve(dims.size());
      for (auto d : dims) {
        d.setDataType(ml::train::TensorDim::DataType::FP32);
        fp32_dims.push_back(d);
      }
      return fp32_dims;
    };

    auto in_fp32_dims = to_fp32_dims(in_dims);
    auto out_fp32_dims = to_fp32_dims(out_dims);

    // Read-buffers: always FP32 to match the golden-file binary layout.
    inputs_fp32 = std::vector<Tensor>(in_fp32_dims.begin(), in_fp32_dims.end());
    labels_fp32 =
      std::vector<Tensor>(out_fp32_dims.begin(), out_fp32_dims.end());
    expected_outputs =
      std::vector<Tensor>(out_fp32_dims.begin(), out_fp32_dims.end());

    // Forwarding buffers: match the model's declared input/output dtype so
    // that fillPlaceholder's raw memory-sharing does not misinterpret bytes.
    inputs = std::vector<Tensor>(in_dims.begin(), in_dims.end());
    labels = std::vector<Tensor>(out_dims.begin(), out_dims.end());

    NetworkGraphType model_graph = nn->getNetworkGraph();
    for (auto it = model_graph.cbegin(); it != model_graph.cend(); ++it) {
      auto const &lnode = *it;
      auto &rc = lnode->getRunContext();
      for (unsigned int i = 0; i < rc.getNumWeights(); ++i) {
        if (!rc.isGradientLastAccess(i)) {
          continue;
        }

        Tensor &t = rc.getWeight(i);

        if (t.getDataType() != ml::train::TensorDim::DataType::FP32) {
          Tensor &t32 = rc.getWeightFP32(i);
          weights32.push_back(t32);
        }

        weights.push_back(t);
        expected_weights.push_back(t.clone());
      }
    }
  }

  /**
   * @brief test and match with the golden
   *
   * @param iteration iteration number
   * @param f  file stream which contains golden data
   * @param test_weights test weights if true else fill weights from read
   */
  void test(unsigned iteration, std::ifstream &f, bool test_weights) {
    auto read_tensors = [&f](std::vector<Tensor> &ts) {
      std::for_each(ts.begin(), ts.end(), [&f](Tensor &t) {
        sizeCheckedReadTensor(t, f, "read_failed for " + t.getName());
      });
    };

    auto to_tensors = [](sharedConstTensors &sts) {
      std::vector<Tensor> ts;
      ts.reserve(sts.size());
      std::transform(
        sts.begin(), sts.end(), std::back_inserter(ts),
        [](const auto &ts) -> const auto & { return *ts; });
      return ts;
    };

    // Read inputs/labels from file into FP32 buffers (golden format is always
    // float32 with int32 element-count prefix).  Then copy into model-dtype
    // forwarding tensors so fillPlaceholder's memory-sharing stays type-safe.
    read_tensors(inputs_fp32);
    for (size_t i = 0; i < inputs.size(); ++i)
      inputs.at(i).copyData(inputs_fp32.at(i));

    read_tensors(labels_fp32);
    for (size_t i = 0; i < labels.size(); ++i)
      labels.at(i).copyData(labels_fp32.at(i));

    read_tensors(expected_weights);
    read_tensors(expected_outputs);

    if (test_weights) {
      verify(weights, expected_weights, " weights");
    } else {
      for (unsigned int i = 0; i < weights.size(); ++i) {
        weights.at(i).fill(expected_weights.at(i));
        if (iteration == 0 &&
            weights.at(i).getDataType() != ml::train::TensorDim::DataType::FP32)
          weights32.at(i).fill(
            weights.at(i).clone(ml::train::TensorDim::DataType::FP32));
      }
    }

    auto shared_inputs = toSharedTensors(inputs);
    auto shared_labels = toSharedTensors(labels);

    auto out = nn->forwarding(shared_inputs, shared_labels);

    // expected_outputs are read as FP32 from the golden file.  Convert actual
    // outputs to FP32 before comparing so the verify() dtype checks pass when
    // the model runs at FP16 activation precision.
    auto out_tensors = to_tensors(out);
    std::vector<Tensor> out_fp32;
    out_fp32.reserve(out_tensors.size());
    for (const auto &t : out_tensors) {
      if (t.getDataType() != ml::train::TensorDim::DataType::FP32) {
        out_fp32.push_back(t.clone(ml::train::TensorDim::DataType::FP32));
      } else {
        out_fp32.push_back(t);
      }
    }
    verify(out_fp32, expected_outputs, " output");
    nn->backwarding(iteration);
  }

private:
  NeuralNetwork *nn;
  std::vector<Tensor> inputs;      /**< model-dtype forwarding buffer */
  std::vector<Tensor> labels;      /**< model-dtype forwarding buffer */
  std::vector<Tensor> inputs_fp32; /**< FP32 read-buffer (golden layout) */
  std::vector<Tensor> labels_fp32; /**< FP32 read-buffer (golden layout) */
  std::vector<Tensor> weights;
  std::vector<Tensor> weights32;
  std::vector<Tensor> expected_weights;
  std::vector<Tensor> expected_outputs;
};

NodeWatcher::NodeWatcher(const NodeType &node) : node(node) {
  unsigned int num_weights = node->getNumWeights();
  try {
    node->setProperty({"trainable=true"});
  } catch (...) {
    std::cout << "Cannot set layer " << node->getType() << " trainable";
  }

  auto &rc = node->getRunContext();
  for (unsigned int i = 0; i < num_weights; ++i) {
    // const nntrainer::Weight &w = node->getWeightObject(i);
    // expected_weights.push_back(w.clone());
    if (rc.isGradientLastAccess(i)) {
      expected_weights.push_back(node->getWeightWrapper(i).clone());
    }
  }

  for (auto &out_dim : node->getOutputDimensions()) {
    expected_output.emplace_back(out_dim);
  }

  for (auto &in_dim : node->getInputDimensions()) {
    expected_dx.emplace_back(in_dim);
  }
}

NodeWatcher::NodeWatcher() : node(nullptr) {}

void NodeWatcher::readLayerWeight(std::ifstream &f) {
  auto &rc = node->getRunContext();
  for (unsigned int i = 0; i < node->getNumWeights(); ++i) {
    if (rc.isGradientLastAccess(i)) {
      node->getWeight(i).read(f);
    }
  }
}

void NodeWatcher::read(std::ifstream &in) {
  // log prints are commented on purpose
  // std::cout << "[=======" << node->getName() << "==========]\n";

  auto read_ = [&in](auto &target) {
    // std::cout << target.getDim();
    target.read(in);
  };

  // std::cout << "expected_output " << expected_output.size() << "\n";
  std::for_each(expected_output.begin(), expected_output.end(), read_);
  // std::cout << "expected_dx " << expected_dx.size() << "\n";
  std::for_each(expected_dx.begin(), expected_dx.end(), read_);

  for (auto &i : expected_weights) {
    if (i.hasGradient()) {
      i.getGradientRef().read(in);
    }
  }

  for (auto &i : expected_weights) {
    i.getVariableRef().read(in);
  }
}

void NodeWatcher::verifyWeight(const std::string &error_msg) {
  for (unsigned int i = 0; i < expected_weights.size(); ++i) {
    verify(node->getWeight(i), expected_weights[i].getVariable(),
           error_msg + " at weight " + std::to_string(i));
  }
}

void NodeWatcher::verifyGrad(const std::string &error_msg) {
  for (unsigned int i = 0; i < expected_weights.size(); ++i) {
    auto weight = node->getWeightWrapper(i);
    if (weight.hasGradient()) {
      verify(node->getWeightGrad(i), expected_weights[i].getGradient(),
             error_msg + " at grad " + std::to_string(i));
    }
  }
}

void NodeWatcher::forward(int iteration, bool verify_forward) {
  std::stringstream ss;
  ss << "forward failed at " << node->getName() << ", " << node->getType()
     << " at iteration " << iteration;
  std::string err_msg = ss.str();

  std::vector<nntrainer::Tensor> out;
  for (unsigned int idx = 0; idx < node->getNumOutputs(); idx++) {
    out.push_back(node->getOutput(idx));
  }

  if (verify_forward && getType() != nntrainer::MultiOutLayer::type &&
      getType() != nntrainer::IdentityLayer::type)
    verify(out, expected_output, err_msg + " at output");
}

void NodeWatcher::backward(int iteration, bool verify_deriv, bool verify_grad) {

  if (getType() == nntrainer::MultiOutLayer::type ||
      getType() == nntrainer::IdentityLayer::type) {
    return;
  }

  std::stringstream ss;
  ss << "backward failed at " << node->getName() << ", " << node->getType()
     << " at iteration " << iteration;
  std::string err_msg = ss.str();

  std::vector<nntrainer::Tensor> out;
  for (unsigned int idx = 0; idx < node->getNumInputs(); idx++) {
    out.push_back(node->getInputGrad(idx));
  }

  if (verify_grad) {
    verifyGrad(err_msg + " grad");
  }

  if (verify_deriv) {
    verify(out, expected_dx, err_msg + " derivative");
  }

  verifyWeight(err_msg);
}

GraphWatcher::GraphWatcher(const std::string &config, const bool opt) :
  nn(new nntrainer::NeuralNetwork()), expected_losses{}, optimize(opt) {
  nn->loadFromConfig(config);
  initialize();
}

GraphWatcher::GraphWatcher(std::unique_ptr<nntrainer::NeuralNetwork> &&net,
                           const bool opt) :
  nn(std::move(net)), optimize(opt) {
  initialize();
}

void GraphWatcher::initialize() {
  /** Disable memory optimization as memory being matched for each layer
   */
  if (!optimize) {
    nn->setProperty({"memory_optimization=false"});
  }

  if (nn->compile()) {
    throw std::invalid_argument("compilation failed");
  };

  if (nn->initialize()) {
    throw std::invalid_argument("initiation failed");
  };

  if (nn->allocate()) {
    throw std::invalid_argument("assign Memory failed");
  };

  NetworkGraphType model_graph = nn->getNetworkGraph();

  for (auto it = model_graph.cbegin(); it != model_graph.cend(); ++it) {
    auto const &lnode = *it;
    if (it->requireLabel()) {
      loss_nodes.push_back(NodeWatcher(lnode));
      expected_losses.push_back(0);
    } else {
      nodes.push_back(NodeWatcher(lnode));
    }
  }
}

void GraphWatcher::compareFor(const std::string &reference,
                              const nntrainer::TensorDim &label_shape,
                              const unsigned int iterations) {
  std::ifstream ref(getModelsPath(reference),
                    std::ios_base::in | std::ios_base::binary);

  if (!ref.good()) {
    std::stringstream ss;
    ss << "ref is bad! ref path: " << reference;
    throw std::runtime_error(ss.str().c_str());
  }

  auto data = prepareData(ref, label_shape);
  nntrainer::sharedConstTensors input;

  for (unsigned int iteration = 0; iteration < iterations; ++iteration) {
    input = {MAKE_SHARED_TENSOR(std::get<0>(data).clone())};
    nntrainer::sharedConstTensors label = {
      MAKE_SHARED_TENSOR(std::get<1>(data).clone())};

    readIteration(ref);

    nn->forwarding(input, label);
    for (unsigned int i = 0; i < loss_nodes.size(); ++i) {
      EXPECT_NEAR(expected_losses[i], loss_nodes[i].getLoss(),
                  nntrainer::Tensor::epsilon);
    }

    auto it = nodes.begin();
    for (; it != nodes.end() - 1; ++it) {
      it->forward(iteration, it->isOutputNode() | !optimize);
    }
    it->forward(iteration, true);

    if (loss_nodes.size()) {
      nn->backwarding(iteration);

      for (auto it = nodes.rbegin(); it != nodes.rend(); it++) {
        if (it->needsCalcDerivative())
          it->backward(iteration, !optimize, !optimize);
      }
    } else {
      EXPECT_THROW(nn->backwarding(iteration), std::runtime_error);
    }
  }

  /**
   * This inference is to ensure that inference runs with/without
   * optimizations for various kinds of models
   */
  EXPECT_NO_THROW(nn->inference(input, false));
}

void GraphWatcher::validateFor(const nntrainer::TensorDim &label_shape) {
  sharedConstTensors input;
  for (auto const &dim : nn->getInputDimension()) {
    auto in_tensor = MAKE_SHARED_TENSOR(dim);
    in_tensor->setRandNormal();
    input.push_back(in_tensor);
  }

  // nntrainer::sharedConstTensors input = {in_tensor};

  auto label_tensor = MAKE_SHARED_TENSOR(label_shape);
  label_tensor->setRandNormal();
  nntrainer::sharedConstTensors label = {label_tensor};

  if (loss_nodes.size()) {
    EXPECT_NO_THROW(nn->forwarding(input, label));
  } else {
    EXPECT_NO_THROW(nn->forwarding(input, {}));
  }

  if (loss_nodes.size()) {
    EXPECT_NO_THROW(nn->backwarding(0));
  }

  /**
   * This inference is to ensure that inference runs with/without
   * optimizations for various kinds of models
   */
  EXPECT_NO_THROW(nn->inference(input, false));
  /** run inference again which frees the memory */
  EXPECT_NO_THROW(nn->inference(input, true));
  /** run inference again which will force to allocate memory again */
  EXPECT_NO_THROW(nn->inference(input, true));
}

void GraphWatcher::compareFor_V2(const std::string &reference) {

  auto file = nntrainer::checkedOpenStream<std::ifstream>(
    getModelsPath(reference), std::ios::in | std::ios::binary);

  unsigned int num_iter;
  nntrainer::checkedRead(file, (char *)&num_iter, sizeof(unsigned));

  IterationForGolden ifg(nn.get());

  for (unsigned int i = 0; i < num_iter; ++i) {
    ifg.test(i, file, i != 0);
  }
}

void GraphWatcher::validateFor_V2() {
  auto in_dims = nn->getInputDimension();
  auto out_dims = nn->getOutputDimension();

  std::vector<Tensor> inputs;
  std::vector<Tensor> labels;

  for (const auto &dim : in_dims) {
    inputs.emplace_back(dim, true, Initializer::ZEROS);
  }

  for (const auto &dim : out_dims) {
    labels.emplace_back(dim, true, Initializer::ZEROS);
  }

  auto shared_inputs = toSharedTensors(inputs);
  auto shared_labels = toSharedTensors(labels);

  if (loss_nodes.size()) {
    EXPECT_NO_THROW(nn->forwarding(shared_inputs, shared_labels));
  } else {
    EXPECT_NO_THROW(nn->forwarding(shared_inputs, {}));
  }

  if (loss_nodes.size()) {
    EXPECT_NO_THROW(nn->backwarding(0));
  }

  /**
   * This inference is to ensure that inference runs with/without
   * optimizations for various kinds of models
   */
  EXPECT_NO_THROW(nn->inference(shared_inputs, false));
  /** run inference again which frees the memory */
  EXPECT_NO_THROW(nn->inference(shared_inputs, true));
  /** run inference again which will force to allocate memory again */
  EXPECT_NO_THROW(nn->inference(shared_inputs, true));
}

std::array<nntrainer::Tensor, 2>
GraphWatcher::prepareData(std::ifstream &f,
                          const nntrainer::TensorDim &label_dim) {
  nntrainer::Tensor in(nn->getInputDimension()[0]);
  nntrainer::Tensor lb(label_dim);

  in.read(f);
  lb.read(f);
  in.setName("input");
  lb.setName("label");

  for (auto &i : nodes) {
    i.readLayerWeight(f);
  }

  return {in, lb};
}

void GraphWatcher::readIteration(std::ifstream &f) {
  for (auto &i : nodes) {
    if (i.getType() == nntrainer::MultiOutLayer::type ||
        i.getType() == nntrainer::IdentityLayer::type) {
      continue;
    }

    i.read(f);
  }

  for (unsigned int i = 0; i < expected_losses.size(); ++i) {
    f.read((char *)&expected_losses[i], sizeof(float));
  }
}
