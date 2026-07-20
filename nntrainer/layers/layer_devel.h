/**
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file	layer_devel.h
 * @date	10 June 2021
 * @brief	This is Layer classes of Neural Network
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#ifndef __LAYER_DEVEL_H__
#define __LAYER_DEVEL_H__
#ifdef __cplusplus

#include <memory>
#include <string>
#include <vector>

#include <base_properties.h>
#include <common.h>
#include <cpu_backend.h>
#include <layer_context.h>
#include <tensor_dim.h>

namespace ml::train {
class Layer;
}

namespace nntrainer {

class InitLayerContext;
class Exporter;

/**
 * @brief Enum class for the various types of inplace modes supported by layer
 *
 */
enum class InPlaceType {
  NONE,           /**< layer is not inplace */
  RESTRICTING,    /**< layer is in-place and does place restriction on layers
                    ahead of it to be in-place */
  NON_RESTRICTING /**< layer is in-place and does NOT place restriction on the
                    layers ahead of it to be in-place */
};

/**
 * @brief Emum class for the direction of inplace
 *
 * @details When the In-Place option is enabled and the layer has binary inputs,
 * you can specify the direction of the in-place operation using this
 * enumeration. For instance, if a layer is in-place with the direction set to
 * LEFT, the output of the layer will be written directly into the 'input[0]' of
 * the preceding layer. Conversely, if the direction is set to RIGHT, the output
 * of the layer will be written directly into the 'input[1]' of the preceding
 * layer.
 */
enum class InPlaceDirection {
  NONE,  /**< default. It will be set to LEFT or RIGHT only when the type of the
            operation  layer is binary and the is_inplace setting is true */
  LEFT,  /**< left side of the layer is in-place */
  RIGHT, /**< right side of the layer is in-place */
};

/**
 * @class   Layer Base class for layers
 * @brief   Base class for all layers
 *
 * @details nntrainer::Layer inherits ml::train::Layer but has been omitted to
 * disallow static_cast between nntrainer::Layer and ml::train::Layer objects.
 */
class Layer {

public:
  /**
   * @brief     Property Enumeration
   *            0. input shape : string
   *            1. normalization : bool
   *            2. standardization : bool
   *            3. activation : string (type)
   *            4. epsilon : float
   *            5. weight_regularizer : string (type)
   *            6. weight_regularizer_constant : float
   *            7. unit : int
   *            8. weight_initializer : string (type)
   *            9. bias initializer : string (type)
   *            10. filter_size : int
   *            11. kernel_size : ( n , m )
   *            12. stride : ( n, m )
   *            13. padding : ( n, m )
   *            14. pool_size : ( n,m )
   *            15. pooling : max, average, global_max, global_average
   *            16. flatten : bool
   *            17. name : string (type)
   *            18. momentum : float,
   *            19. moving_mean_initializer : string (type),
   *            20. moving_variance_initializer : string (type),
   *            21. gamma_initializer : string (type),
   *            22. beta_initializer" : string (type)
   *            23. modelfile : model file for loading config for backbone layer
   *            24. input_layers : string (type)
   *            25. output_layers : string (type)
   *            26. trainable :
   *            27. flip_direction
   *            28. random_translate
   *            29. in_dim : int ( input dimension for embedding layer )
   *            30. out_dim : int ( output dimesion for embedding layer )
   *            31. recurrent_activation :  string (type) - lstm
   *            32. distribute : bool
   *            33. axis : string (type)
   *            34. return_sequences :  bool (type) - lstm
   *            35. hidden_state_activation :  string (type) - lstm
   *            36. dropout : bool
   */
  enum class PropertyType {
    input_shape = 0,
    normalization = 1,
    standardization = 2,
    activation = 3,
    epsilon = 4,
    weight_regularizer = 5,
    weight_regularizer_constant = 6,
    unit = 7,
    weight_initializer = 8,
    bias_initializer = 9,
    filters = 10,
    kernel_size = 11,
    stride = 12,
    padding = 13,
    pool_size = 14,
    pooling = 15,
    flatten = 16,
    name = 17,
    momentum = 18,
    moving_mean_initializer = 19,
    moving_variance_initializer = 20,
    gamma_initializer = 21,
    beta_initializer = 22,
    modelfile = 23, /** model file for loading config for backbone layer */
    input_layers = 24,
    output_layers = 25,
    trainable = 26,
    flip_direction = 27,
    random_translate = 28,
    in_dim = 29,
    out_dim = 30,
    recurrent_activation = 31,
    distribute = 32,
    axis = 33,
    return_sequences = 34,
    hidden_state_activation = 35,
    dropout = 36,
    num_inputs = 37,
    unknown
  };

  /**
   * @brief     Destructor of Layer Class
   */
  virtual ~Layer() = default;

  /**
   * @brief Get the layer type
   * @return const std::string type representation
   */
  virtual const std::string getType() const = 0;

  /**
   * @brief     Finalize creating the layer
   * @param     context Context of the layer
   *
   * @details   Input dimensions will be provided set in the context. This
   * function must set output dimensions in the given context. Further, context
   * can be used to request weights for the layer, and any extra tensor required
   * for the operation of the layer.
   * @note      After calling this it is not allowed to
   * change properties.
   * @note      No memory allocation must be performed in the initialization
   * step. Any tensor memory required must be requested to the context which
   * will be made available during execution of the layer with the context.
   */
  virtual void finalize(InitLayerContext &context) = 0;

  /**
   * @brief    Initialize the layer
   */
  virtual void initialize(RunLayerContext &context){};

  /**
   * @brief     Forward Propagation of a layer
   * @param     context Context of the layer
   * @param     training true if training, false if inference
   *
   * @note      Output must be set in the output tensors.
   * @details   context provides access to the weights (if any), inputs,
   * outputs, and tensors (if any) for the layer. Input and output dimensions
   * can be access from the inputs/outputs tensors themselves.
   */
  virtual void forwarding(RunLayerContext &context, bool training) = 0;

  /**
   * @brief     Incremental forward Propagation of a layer
   * @param     context Context of the layer
   * @param     from start step
   * @param     to end step
   * @param     training true if training, false if inference
   *
   * @note      Output must be set in the output tensors.
   * @details   context provides access to the weights (if any), inputs,
   * outputs, and tensors (if any) for the layer. Input and output dimensions
   * can be access from the inputs/outputs tensors themselves.
   */
  virtual void incremental_forwarding(RunLayerContext &context,
                                      unsigned int from, unsigned int to,
                                      bool training) {
    forwarding(context, training);
  };

  /**
   * @brief     calc the derivative to be passed to the previous layer
   * @param     context Context of the layer
   * @note      Return derivatives must be set in input gradient tensors.
   * @details   context provides access to the weights (if any), inputs,
   * outputs, and tensors (if any) for the layer. Input and output dimensions
   * can be access from the inputs/outputs tensors themselves.
   */
  virtual void calcDerivative(RunLayerContext &context) = 0;

  /**
   * @brief     Calculate the derivative of a layer
   * @details   context provides access to the weights (if any), inputs,
   * outputs, and tensors (if any) for the layer. Input and output dimensions
   * can be access from the inputs/outputs tensors themselves.
   * @note      Gradients must be set in weight gradient tensors.
   */
  virtual void calcGradient(RunLayerContext &context) {}

  /**
   * @brief     set Property of layer
   * @param     values values of property
   * @throw std::invalid_argument invalid parameter.
   */
  virtual void setProperty(const std::vector<std::string> &values) = 0;

  /**
   * @brief     Get property value of layer
   * @param     key Property key to retrieve
   * @return    Property value as string (empty string if not found)
   */
  virtual std::string getProperty(const std::string &key) { return ""; }

  /**
   * @brief this function helps exporting the layer in a predefined format,
   * while workarounding issue caused by templated function type eraser
   *
   * @param     exporter exporter that conatins exporting logic
   * @param     method enum value to identify how it should be exported to
   */
  virtual void exportTo(Exporter &exporter,
                        const ml::train::ExportMethods &method) const {}

  /**
   * @brief Set the batch for the layer
   * @param     context Context of the layer
   * @param     batch Batch value to be set
   * @details Update the run context based on the updated batch size if required
   */
  virtual void setBatch(RunLayerContext &context, unsigned int batch) {}

  /**
   * @brief Update the tensor dimensions of layer by input dimensions
   * @param     context Context of the layer
   * @param     input_dimensions input dimensions of layer
   * @details Update the dimensions of inputs, outputs, weights and tensors
   * based on the input dimensions
   */
  virtual void
  updateTensorsByInputDimensions(RunLayerContext &context,
                                 std::vector<TensorDim> input_dimensions) {
    throw std::invalid_argument("updateTensorsByInputDimensions() is currently "
                                "not supported for layer type " +
                                getType());
  }

  /**
   * @brief   If the current layer can support in-place
   *
   * @return  true if inplace, else false
   * @details all layers default to out of place execution
   * @note all layers default to out of place execution
   */
  virtual bool supportInPlace() const { return is_inplace; }

  /**
   * @brief Get the inplace direction for the tensor operation layer
   *
   * @return InPlaceDirection
   */
  virtual InPlaceDirection getInPlaceDirection() {
    return InPlaceDirection::NONE;
  };

  /**
   * @brief Initialize the in-place settings of the layer
   * @details If it is a layer that supports in-place, the default in-place type
   * is NONE_RESTRICTING, but if there is a RESTRICTING type among the input
   * layers, it is set to NONE in the network_graph.cpp.
   * Layers with exceptional behavior such as No-Operation layers should
   * override this function.
   * @return InPlaceType
   */
  virtual InPlaceType initializeInPlace() {
    if (!supportInPlace())
      return InPlaceType::NONE;
    else
      return InPlaceType::NON_RESTRICTING;
  }

  /**
   * @brief  check if this layer requires label to be passed
   * @note   if requireLabel() == true means, for now, that it is endpoint of a
   * graph(numOutlayers == 0). label will be fed to the gradient of hidden if
   * requireLabel is true
   * @return true if requires a label when training, else false
   */
  virtual bool requireLabel() const { return false; }

  /**
   * @brief  check if this layer supports backwarding
   * @note   support backwarding primarily means that the layer can process the
   * derivatives and return back the gradients to the previous layer.
   * @return true if supports backwarding, else false
   */
  virtual bool supportBackwarding() const = 0;

  /**
   * @brief     save layer Weight & Bias data from file
   * @param file output file stream
   * @param run_context run context for the layer
   * @param opt_var boolean variable whether saving optimizer variables
   * @param mode Execution mode
   * @param trainable is there trainable weight
   * @param dtype data type to save this layer
   * @param target_isa target ISA (Instruction Set Architecture) format for
   * quantization (DEFAULT/X86/ARM)
   */
  virtual void save(std::ofstream &file, RunLayerContext &run_context,
                    bool opt_var, ml::train::ExecutionMode mode, bool trainable,
                    TensorDim::DataType dtype = TensorDim::DataType::NONE,
                    ml::train::ISA target_isa = ml::train::ISA::DEFAULT) const {

    if (opt_var) {
      for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
        if (run_context.isGradientFirstAccess(i) && trainable) {
          // @note save optimizer variables
          if (run_context.weightHasGradient(i)) {
            for (unsigned int j = 0; j < run_context.getNumWeightOptVar(i);
                 ++j) {
              run_context.getWeightOptVar(i, j).save(file);
            }
          }
        }
      }
    } else {
      // @note shared weights are only be saved at the first access
      for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
        if (run_context.isGradientFirstAccess(i)) {
          auto &weight = run_context.getWeight(i);
          if (dtype == TensorDim::DataType::NONE ||
              weight.getDataType() == dtype)
            weight.save(file);
          else {
            if (dtype == TensorDim::DataType::Q4_0) {
              NNTR_THROW_IF(weight.getDataType() != TensorDim::DataType::FP32,
                            std::runtime_error)
                << "Save with quantization only supports for FP32 weight.";
              ///@note The codelines below can be replaced with quantizer's
              /// quantize()
              TensorDim dim = weight.getDim();
              unsigned int K = dim.height();
              unsigned int N = dim.width();

              // Skip quantization for bias-like tensors (1D with height == 1)
              // as they are not suitable for Q4_0 block quantization
              if (K == 1) {
                weight.save(file);
              } else {
                NNTR_THROW_IF(N % 32 != 0 || K % 32 != 0, std::invalid_argument)
                  << "Q4_0 quantization requires both width and height to be "
                     "divisible by 32, but got height="
                  << K << ", width=" << N;

                Tensor weight_t = weight.transpose("0:2:1");
                Tensor quant_weight(dim.batch(), dim.channel(), K, N,
                                    {Tformat::NCHW, dtype});
                std::vector<char> tmp(quant_weight.size());

                quantize_q4_0(weight_t.getData<float>(), tmp.data(), N, K,
                              nullptr);
                repack_q4_0(quant_weight.getData<uint8_t>(), tmp.data(),
                            quant_weight.size(), N, K, target_isa);
                quant_weight.save(file);
              }
            } else if (dtype == TensorDim::DataType::QS4CX) {
              NNTR_THROW_IF(weight.getDataType() != TensorDim::DataType::FP32,
                            std::runtime_error)
                << "Save with quantization only supports for FP32 weight.";
              ///@note The codelines below can be replaced with quantizer's
              /// quantize()
              TensorDim dim = weight.getDim();
              unsigned int K = dim.height();
              unsigned int N = dim.width();
              Tensor weight_t = weight.transpose("0:2:1");

              size_t q_size = N * (K + 1) / 2;
              size_t scale_size = N * sizeof(float);

              // allocate packed size, not an unpacked size
              std::vector<uint8_t> rhs_q(q_size + scale_size);
              uint8_t *data = rhs_q.data();

              uint8_t *scale = data + q_size;

              nntrainer::quant_qs4cx_f32(N, K, weight_t.getData(), data, scale,
                                         true);
              file.write((const char *)data, q_size + scale_size);
            } else if (dtype == TensorDim::DataType::Q8_0) {
              NNTR_THROW_IF(weight.getDataType() != TensorDim::DataType::FP32,
                            std::runtime_error)
                << "Save with quantization only supports for FP32 weight.";
              TensorDim dim = weight.getDim();
              unsigned int K = dim.height();
              unsigned int N = dim.width();

              // Skip quantization for bias-like tensors (1D with height == 1)
              // as they are not suitable for Q8_0 block quantization (mirrors
              // the Q4_0 path above).
              if (K == 1) {
                weight.save(file);
              } else {
                NNTR_THROW_IF(N % 32 != 0 || K % 32 != 0, std::invalid_argument)
                  << "Q8_0 weight quantization requires dimensions to be "
                     "divisible by 32, but got height="
                  << K << ", width=" << N;

                Tensor weight_t = weight.transpose("0:2:1");
                Tensor quant_weight(dim.batch(), dim.channel(), K, N,
                                    {Tformat::NCHW, dtype});

                quantize_q8_0(weight_t.getData<float>(),
                              quant_weight.getData<uint8_t>(), N, K, nullptr);
                quant_weight.save(file);
              }
            } else {
              NNTR_THROW_IF(true, std::runtime_error)
                << "This dtype is not supported in save with quantization";
            }
          }
        }
      }
    }
  }

  /**
   * @brief     read layer Weight & Bias data from file
   * @param file input file stream
   * @param run context for layer
   * @param bool read optimizer variables
   * @param mode execution mode
   * @param bool trainable
   * @param type Required Weight Tensor Type from Network
   * @param bool fsu flag
   *
   */
  virtual void read(std::ifstream &file, RunLayerContext &run_context,
                    bool opt_var, ml::train::ExecutionMode mode, bool trainable,
                    TensorDim::DataType defineWeightDataType, bool fsu,
                    size_t start_offset = 0, bool read_from_offset = false,
                    int file_fd = -1) {
    if (fsu) {
      for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
        if (run_context.getWeight(i).getDataType() ==
            TensorDim::DataType::BCQ) {
          run_context.getWeight(i).readFSU();
        }
      }
    } else {
      if (opt_var) {
        for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
          if (run_context.isGradientLastAccess(i) && trainable) {
            /// @note read optimizer variables
            for (unsigned int j = 0; j < run_context.getNumWeightOptVar(i);
                 ++j) {
              run_context.getWeightOptVar(i, j).read(file, start_offset);
            }
          }
        }
      } else {
        for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
          /// @note shared weights are only be read at the first acecss
          if (run_context.isGradientFirstAccess(i)) {
            run_context.getWeight(i).read(file, start_offset, read_from_offset,
                                          file_fd);
            if (run_context.isMixedPrecision(i) && trainable &&
                !run_context.getWeightFP32(i).empty()) {
              run_context.getWeightFP32(i).copyData(run_context.getWeight(i));
            }
          }
        }
      }
    }
  }

  /**
   * @brief     read layer Weight & Bias data from file
   * @param ReadSource input file stream
   * @param run context for layer
   * @param bool read optimizer variables
   * @param mode execution mode
   * @param bool trainable
   * @param type Required Weight Tensor Type from Network
   * @param bool fsu flag
   *
   */
  virtual void read(ReadSource src, RunLayerContext &run_context, bool opt_var,
                    ml::train::ExecutionMode mode, bool trainable,
                    TensorDim::DataType defineWeightDataType, bool fsu,
                    size_t start_offset = 0, bool read_from_offset = false,
                    int file_fd = -1) {
    if (fsu) {
      for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
        if (run_context.getWeight(i).getDataType() ==
            TensorDim::DataType::BCQ) {
          run_context.getWeight(i).readFSU();
        }
      }
    } else {
      if (opt_var) {
        for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
          if (run_context.isGradientLastAccess(i) && trainable) {
            /// @note read optimizer variables
            for (unsigned int j = 0; j < run_context.getNumWeightOptVar(i);
                 ++j) {
              run_context.getWeightOptVar(i, j).read(src, start_offset);
            }
          }
        }
      } else {
        for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
          /// @note shared weights are only be read at the first acecss
          if (run_context.isGradientFirstAccess(i)) {
            // file_fd is forwarded so virtual weights (e.g. SlimMoE expert
            // tensors) can capture a long-lived fd for later mmap-on-demand
            // in activate(); non-virtual weights ignore it.
            run_context.getWeight(i).read(src, start_offset, read_from_offset,
                                          file_fd);
            if (run_context.isMixedPrecision(i) && trainable &&
                !run_context.getWeightFP32(i).empty()) {
              run_context.getWeightFP32(i).copyData(run_context.getWeight(i));
            }
          }
        }
      }
    }
  }

protected:
  bool is_inplace = false; /**< whether this layer is in-place or not */
};

/// @todo Decide where to put and how to implement(#986)
// /**
//  * @brief   Overriding output stream for layers and it's derived class
//  */
// template <typename T, typename std::enable_if_t<
//                         std::is_base_of<Layer, T>::value, T> * = nullptr>
// std::ostream &operator<<(std::ostream &out, T &l) {
//   // l.printPreset(out, Layer::PrintPreset::PRINT_SUMMARY);
//   return out;
// }

using CreateLayerFunc = nntrainer::Layer *(*)();
using DestroyLayerFunc = void (*)(nntrainer::Layer *);

/**
 * @brief General Layer Factory function to register Layer
 *
 * @param props property representation
 * @return std::unique_ptr<nntrainer::Layer> created object
 */
template <typename T,
          std::enable_if_t<std::is_base_of<Layer, T>::value, T> * = nullptr>
std::unique_ptr<Layer> createLayer(const std::vector<std::string> &props = {}) {
  std::unique_ptr<Layer> ptr = std::make_unique<T>();
  ptr->setProperty(props);
  return ptr;
}

/**
 * @brief  Layer Pluggable struct that enables pluggable layer
 *
 */
typedef struct {
  CreateLayerFunc createfunc;   /**< create layer function */
  DestroyLayerFunc destroyfunc; /**< destory function */
} LayerPluggable;

/**
 * @brief pluggable layer must have this structure defined
 */
extern "C" LayerPluggable ml_train_layer_pluggable;

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LAYER_DEVEL_H__ */
