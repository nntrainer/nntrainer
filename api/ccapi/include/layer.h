// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   layer.h
 * @date   14 October 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @author Debadri Samaddar <s.debadri@samsung.com>
 * @author Niket Agarwal <niket.a@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is layers interface for c++ API
 *
 * @note This is experimental API and not stable.
 */

#ifndef __ML_TRAIN_LAYER_H__
#define __ML_TRAIN_LAYER_H__

#if __cplusplus < MIN_CPP_VERSION
#error "CPP versions c++17 or over are only supported"
#endif // __cpluscplus

#include <memory>
#include <string>
#include <tensor_dim.h>
#include <vector>

#include <common.h>

namespace ml {
namespace train {

/**
 * @brief     Enumeration of layer type
 */
enum LayerType {
  LAYER_IN = ML_TRAIN_LAYER_TYPE_INPUT,          /**< Input Layer type */
  LAYER_WEIGHT = ML_TRAIN_LAYER_TYPE_WEIGHT,     /**< Weight Layer type */
  LAYER_TENSOR = ML_TRAIN_LAYER_TYPE_TENSOR,     /**< Weight Layer type */
  LAYER_ADD = ML_TRAIN_LAYER_TYPE_ADD,           /**< Add Layer type */
  LAYER_SUBTRACT = ML_TRAIN_LAYER_TYPE_SUBTRACT, /**< Subtract Layer type */
  LAYER_MULTIPLY = ML_TRAIN_LAYER_TYPE_MULTIPLY, /**< Multiply Layer type */
  LAYER_DIVIDE = ML_TRAIN_LAYER_TYPE_DIVIDE,     /**< Divide Layer type */
  LAYER_POW = ML_TRAIN_LAYER_TYPE_POW,           /**< Pow Layer type */
  LAYER_SQRT = ML_TRAIN_LAYER_TYPE_SQRT,         /**< SQRT Layer type */
  LAYER_SINE = ML_TRAIN_LAYER_TYPE_SINE,         /**< Sine Layer type */
  LAYER_COSINE = ML_TRAIN_LAYER_TYPE_COSINE,     /**< Cosine Layer type */
  LAYER_TANGENT = ML_TRAIN_LAYER_TYPE_TANGENT,   /**< Tangent Layer type */
  LAYER_MATMUL =
    ML_TRAIN_LAYER_TYPE_MATMUL, /**< Matrix multiplication Layer type */
  LAYER_CAST = ML_TRAIN_LAYER_TYPE_CAST,     /**< Cast Layer type */
  LAYER_GATHER = ML_TRAIN_LAYER_TYPE_GATHER, /**< Gather Layer type */
  LAYER_SLICE = ML_TRAIN_LAYER_TYPE_SLICE,   /**< Slice Layer type */
  LAYER_NEG = ML_TRAIN_LAYER_TYPE_NEG,       /**< Negative Layer type */
  LAYER_FC = ML_TRAIN_LAYER_TYPE_FC,         /**< Fully Connected Layer type */
  LAYER_SWIGLU = ML_TRAIN_LAYER_TYPE_SWIGLU, /**< Swiglu Layer type */
  LAYER_BN = ML_TRAIN_LAYER_TYPE_BN, /**< Batch Normalization Layer type */
  LAYER_CONV2D = ML_TRAIN_LAYER_TYPE_CONV2D, /**< Convolution 2D Layer type */
  LAYER_CONV2D_TRANSPOSE =
    ML_TRAIN_LAYER_TYPE_CONV2D_TRANSPOSE, /**< Convolution 2D Transpose Layer
                                             type */
  LAYER_POOLING2D = ML_TRAIN_LAYER_TYPE_POOLING2D, /**< Pooling 2D Layer type */
  LAYER_FLATTEN = ML_TRAIN_LAYER_TYPE_FLATTEN,     /**< Flatten Layer type */
  LAYER_ACTIVATION =
    ML_TRAIN_LAYER_TYPE_ACTIVATION,              /**< Activation Layer type */
  LAYER_ADDITION = ML_TRAIN_LAYER_TYPE_ADDITION, /**< Addition Layer type */
  LAYER_CONCAT = ML_TRAIN_LAYER_TYPE_CONCAT,     /**< Concat Layer type */
  LAYER_MULTIOUT = ML_TRAIN_LAYER_TYPE_MULTIOUT, /**< Multi Output Layer type */
  LAYER_EMBEDDING = ML_TRAIN_LAYER_TYPE_EMBEDDING, /**< Embedding Layer type */
  LAYER_RNN = ML_TRAIN_LAYER_TYPE_RNN,             /**< RNN Layer type */
  LAYER_LSTM = ML_TRAIN_LAYER_TYPE_LSTM,           /**< LSTM Layer type */
  LAYER_SPLIT = ML_TRAIN_LAYER_TYPE_SPLIT,         /**< Splite Layer type */
  LAYER_GRU = ML_TRAIN_LAYER_TYPE_GRU,             /**< GRU Layer type */
  LAYER_PERMUTE = ML_TRAIN_LAYER_TYPE_PERMUTE,     /**< Permute layer */
  LAYER_DROPOUT = ML_TRAIN_LAYER_TYPE_DROPOUT,     /**< DropOut Layer type */
  LAYER_BACKBONE_NNSTREAMER =
    ML_TRAIN_LAYER_TYPE_BACKBONE_NNSTREAMER, /**< Backbone using NNStreamer */
  LAYER_CENTROID_KNN =
    ML_TRAIN_LAYER_TYPE_CENTROID_KNN,        /**< Centroid KNN Layer */
  LAYER_CONV1D = ML_TRAIN_LAYER_TYPE_CONV1D, /**< Convolution 1D Layer type */
  LAYER_LSTMCELL = ML_TRAIN_LAYER_TYPE_LSTMCELL, /**< LSTM Cell Layer type */
  LAYER_GRUCELL = ML_TRAIN_LAYER_TYPE_GRUCELL,   /**< GRU Cell Layer type */
  LAYER_RNNCELL = ML_TRAIN_LAYER_TYPE_RNNCELL,   /**< RNN Cell Layer type */
  LAYER_ZONEOUT_LSTMCELL =
    ML_TRAIN_LAYER_TYPE_ZONEOUTLSTMCELL, /**< Zoneout LSTM Cell Layer type */
  LAYER_ATTENTION = ML_TRAIN_LAYER_TYPE_ATTENTION, /**< Attention Layer type */
  LAYER_MOL_ATTENTION =
    ML_TRAIN_LAYER_TYPE_MOL_ATTENTION, /**< MoL Attention Layer type */
  LAYER_MULTI_HEAD_ATTENTION =
    ML_TRAIN_LAYER_TYPE_MULTI_HEAD_ATTENTION, /**< Multi Head Attention Layer
                                                 type */
  LAYER_LAYER_NORMALIZATION =
    ML_TRAIN_LAYER_TYPE_LAYER_NORMALIZATION, /**< Layer Normalization Layer type
                                              */
  LAYER_POSITIONAL_ENCODING =
    ML_TRAIN_LAYER_TYPE_POSITIONAL_ENCODING, /**< Positional Encoding Layer type
                                              */
  LAYER_IDENTITY = ML_TRAIN_LAYER_TYPE_IDENTITY, /**< Identity Layer type */
  LAYER_PREPROCESS_FLIP =
    ML_TRAIN_LAYER_TYPE_PREPROCESS_FLIP, /**< Preprocess flip Layer type */
  LAYER_PREPROCESS_TRANSLATE =
    ML_TRAIN_LAYER_TYPE_PREPROCESS_TRANSLATE, /**< Preprocess translate Layer
                                                 type */
  LAYER_PREPROCESS_L2NORM =
    ML_TRAIN_LAYER_TYPE_PREPROCESS_L2NORM, /**< Preprocess l2norm Layer type */
  LAYER_LOSS_MSE =
    ML_TRAIN_LAYER_TYPE_LOSS_MSE, /**< Mean Squared Error Loss Layer type */
  LAYER_LOSS_CROSS_ENTROPY_SIGMOID =
    ML_TRAIN_LAYER_TYPE_LOSS_CROSS_ENTROPY_SIGMOID, /**< Cross Entropy with
                                                       Sigmoid Loss Layer type
                                                     */
  LAYER_LOSS_CROSS_ENTROPY_SOFTMAX =
    ML_TRAIN_LAYER_TYPE_LOSS_CROSS_ENTROPY_SOFTMAX, /**< Cross Entropy with
                                                       Softmax Loss Layer type
                                                     */
  LAYER_RMSNORM = ML_TRAIN_LAYER_TYPE_RMSNORM,      /**<RMS NORM Layer */
  LAYER_TRANSPOSE = ML_TRAIN_LAYER_TYPE_TRANSPOSE,  /**< Transpose Layer type */
  LAYER_CHANNEL_SHUFFLE =
    ML_TRAIN_LAYER_TYPE_CHANNEL_SHUFFLE, /**< Channel Shuffle Layer type */
  LAYER_REDUCE_SUM =
    ML_TRAIN_LAYER_TYPE_REDUCE_SUM, /**< Reduce sum Layer type */
  LAYER_REDUCE_MEAN =
    ML_TRAIN_LAYER_TYPE_REDUCE_MEAN,           /**< Reduce mean Layer type */
  LAYER_RESHAPE = ML_TRAIN_LAYER_TYPE_RESHAPE, /**< Reshape Layer type */
  LAYER_UPSAMPLE2D =
    ML_TRAIN_LAYER_TYPE_UPSAMPLE2D, /**< Upsample 2D Layer type */

  LAYER_UNKNOWN = ML_TRAIN_LAYER_TYPE_UNKNOWN, /**< Unknown */

  LAYER_TIME_DIST =
    ML_TRAIN_LAYER_EXPERIMENTAL, /**< Time Distributed Layer type */
  LAYER_BACKBONE_TFLITE =
    ML_TRAIN_LAYER_EXPERIMENTAL + 1, /**< Backbone using TFLite */
  LAYER_LOSS_CONSTANT_DERIVATIVE =
    ML_TRAIN_LAYER_EXPERIMENTAL + 2, /**< Synthetic loss layer to feed
        constant derivative */
};

/**
 * @class   Layer Base class for layers
 * @brief   Base class for all layers
 */
class Layer {
public:
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
   * @brief Get the layer type
   */
  virtual void initialize() = 0;

  /**
   * @brief     Default allowed properties
   * - input shape : string
   * - bias zero : bool
   * - normalization : bool
   * - standardization : bool
   * - activation : string (type)
   * - epsilon : float
   * - weight_regularizer : string (type)
   * - weight_regularizer_constant : float
   * - unit : int
   * - weight_initializer : string (type)
   * - filter_size : int
   * - kernel_size : ( n , m )
   * - stride : ( n, m )
   * - padding : ( n, m )
   * - pool_size : ( n,m )
   * - pooling : max, average, global_max, global_average
   * - flatten : bool
   * - name : string (type)
   * - momentum : float,
   * - moving_mean_initializer : string (type),
   * - moving_variance_initializer : string (type),
   * - gamma_initializer : string (type),
   * - beta_initializer" : string (type)
   * - modelfile : model file for loading config for backbone layer
   * - input_layers : string (type)
   * - output_layers : string (type)
   * - trainable :
   * - flip_direction
   * - random_translate
   * - in_dim : int ( input dimension for embedding layer )
   * - out_dim : int ( output dimesion for embedding layer )
   * - recurrent_activation : string (type) - used only in lstm
   * - return_sequences : bool (type) - used only in lstm
   * - distribute : bool
   * - hidden_state_activation : string (type) - used only in lstm
   * - drop_out : float (type) - drop out rate
   */
  /**
   * @brief     set Property of layer
   * @todo      change the function signature
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   * @details   This function accepts vector of properties in the format -
   *  { std::string property_name, void * property_val, ...}
   */
  virtual void setProperty(const std::vector<std::string> &values) = 0;

  /**
   * @brief     Get property value of layer
   * @param[in] key Property key to retrieve
   * @retval    std::string Property value as string
   * @retval    Empty string if property not found
   * @details   For layers derived from layer_impl: Property lookup is handled
   * automatically.
   * @details For layers derived from layer_devel: The getProperty() function
   * must be overridden to enable property retrieval.
   */
  virtual std::string getProperty(const std::string &key) = 0;

  /**
   * @brief     Get name of the layer
   * @retval    name of the layer
   * @note      This name is unique to this layer in a model
   * @note      This name might be changed once this layer is added to the model
   * to keep the name unique to the model
   */
  virtual const std::string getName() const = 0;

  /**
   * @brief Get the Weight object name
   *
   * @param idx Identifier of the weight
   * @return const std::string &Name of the weight
   */
  virtual const std::string &getWeightName(unsigned int idx) = 0;

  /**
   * @brief     Get weight data of the layer
   * @retval    weight data of the layer
   * @note      nntrainer assign the vector and if there is no weights, the size
   * of vector is zero
   * @note      layer needs to be finalized before called.
   */
  virtual const std::vector<float *> getWeights() = 0;

  /**
   * @brief     Get weight data of the layer
   * @retval    weights : float * arrary to store weight data
   * @retval    weights_dim : TensorDim for each weights
   * @note      nntrainer assign the vector and if there is no weights, the size
   * of vector is zero
   * @note      layer needs to be finalized before called.
   */
  virtual void getWeights(std::vector<float *> &weights,
                          std::vector<ml::train::TensorDim> &weights_dim) = 0;

#ifdef ENABLE_FP16
  /**
   * @brief     Get weight data of the layer
   * @retval    weight data of the layer
   * @note      nntrainer assign the vector and if there is no weights, the size
   * of vector is zero
   * @note      layer needs to be finalized before called.
   */
  virtual const std::vector<_FP16 *> getFP16Weights() = 0;

  /**
   * @brief     Get weight data of the layer
   * @retval    weights : float * arrary to store weight data
   * @retval    weights_dim : TensorDim for each weights
   * @note      nntrainer assign the vector and if there is no weights, the size
   * of vector is zero
   * @note      layer needs to be finalized before called.
   */
  virtual void
  getFP16Weights(std::vector<_FP16 *> &weights,
                 std::vector<ml::train::TensorDim> &weights_dim) = 0;
#endif
  /**
   * @brief     Set weight data of the layer
   * @note      Size of vector must be the same with number of weights.
   * @note      layer needs to be finalized before called.
   */
  virtual void setWeights(const std::vector<float *>) = 0;
};

/**
 * @brief Factory creator with constructor for layer type
 */
std::unique_ptr<Layer>
createLayer(const LayerType &type,
            const std::vector<std::string> &properties = {});

/**
 * @brief Factory creator with constructor for layer
 */
std::unique_ptr<Layer>
createLayer(const std::string &type,
            const std::vector<std::string> &properties = {});

/**
 * @brief General Layer Factory function to register Layer
 *
 * @param props property representation
 * @return std::unique_ptr<ml::train::Layer> created object
 */
template <typename T,
          std::enable_if_t<std::is_base_of<Layer, T>::value, T> * = nullptr>
std::unique_ptr<Layer> createLayer(const std::vector<std::string> &props = {}) {
  std::unique_ptr<Layer> ptr = std::make_unique<T>();

  ptr->setProperty(props);
  return ptr;
}

/**
 * Aliases for various layers and losses
 */
namespace layer {

/**
 * @brief Helper function to create input layer
 */
inline std::unique_ptr<Layer>
Input(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_IN, properties);
}

/**
 * @brief Helper function to create weight layer
 */
inline std::unique_ptr<Layer>
WeightLayer(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_WEIGHT, properties);
}

/**
 * @brief Helper function to create add layer
 */
inline std::unique_ptr<Layer>
AddLayer(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_ADD, properties);
}

/**
 * @brief Helper function to create subtract layer
 */
inline std::unique_ptr<Layer>
SubtractLayer(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_SUBTRACT, properties);
}

/**
 * @brief Helper function to create multiply layer
 */
inline std::unique_ptr<Layer>
MultiplyLayer(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_MULTIPLY, properties);
}

/**
 * @brief Helper function to create divide layer
 */
inline std::unique_ptr<Layer>
DivideLayer(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_DIVIDE, properties);
}

/**
 * @brief Helper function to create pow layer
 */
inline std::unique_ptr<Layer>
PowLayer(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_POW, properties);
}

/**
 * @brief Helper function to create SQRT layer
 */
inline std::unique_ptr<Layer>
SQRTLayer(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_SQRT, properties);
}

/**
 * @brief Helper function to create sine layer
 */
inline std::unique_ptr<Layer>
SineLayer(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_SINE, properties);
}

/**
 * @brief Helper function to create cosine layer
 */
inline std::unique_ptr<Layer>
CosineLayer(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_COSINE, properties);
}

/**
 * @brief Helper function to create tangent layer
 */
inline std::unique_ptr<Layer>
TangentLayer(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_TANGENT, properties);
}

/**
 * @brief Helper function to create matmul layer
 */
inline std::unique_ptr<Layer>
MatMulLayer(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_MATMUL, properties);
}

/**
 * @brief Helper function to create cast layer
 */
inline std::unique_ptr<Layer>
CastLayer(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_CAST, properties);
}

/**
 * @brief Helper function to create gather layer
 */
inline std::unique_ptr<Layer>
GatherLayer(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_GATHER, properties);
}

/**
 * @brief Helper function to create Slice layer
 */
inline std::unique_ptr<Layer>
SliceLayer(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_SLICE, properties);
}

/**
 * @brief Helper function to create Negative layer
 */
inline std::unique_ptr<Layer>
NegativeLayer(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_NEG, properties);
}

/**
 * @brief Helper function to create fully connected layer
 */
inline std::unique_ptr<Layer>
FullyConnected(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_FC, properties);
}

/**
 * @brief Helper function to create Swiglu layer
 */
inline std::unique_ptr<Layer>
Swiglu(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_SWIGLU, properties);
}

// /**
//  * @brief Helper function to create RMS normalization layer for GPU
//  */
// inline std::unique_ptr<Layer>
// RMSNormCl(const std::vector<std::string> &properties = {},
//           const LayerComputeEngine &compute_engine =
//           LayerComputeEngine::GPU)
//           {
//   return createLayer(LayerType::LAYER_RMSNORM, properties, compute_engine);
// }

/**
 * @brief Helper function to create Transpose layer
 */
inline std::unique_ptr<Layer>
Transpose(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_TRANSPOSE, properties);
}

/**
 * @brief Helper function to create Channel Shuffle layer
 */
inline std::unique_ptr<Layer>
ChannelShuffle(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_CHANNEL_SHUFFLE, properties);
}

/**
 * @brief Helper function to create batch normalization layer
 */
inline std::unique_ptr<Layer>
BatchNormalization(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_BN, properties);
}

/**
 * @brief Helper function to create layer normalization layer
 */
inline std::unique_ptr<Layer>
LayerNormalization(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_LAYER_NORMALIZATION, properties);
}

/**
 * @brief Helper function to create convolution 2d layer
 */
inline std::unique_ptr<Layer>
Convolution2D(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_CONV2D, properties);
}

/**
 * @brief Helper function to create convolution 1d layer
 */
inline std::unique_ptr<Layer>
Convolution1D(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_CONV1D, properties);
}

/**
 * @brief Helper function to create pooling 2d layer
 */
inline std::unique_ptr<Layer>
Pooling2D(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_POOLING2D, properties);
}

/**
 * @brief Helper function to create flatten layer
 */
inline std::unique_ptr<Layer>
Flatten(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_FLATTEN, properties);
}

/**
 * @brief Helper function to create reshape layer
 */
inline std::unique_ptr<Layer>
Reshape(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_RESHAPE, properties);
}

/**
 * @brief Helper function to create addition layer
 */
inline std::unique_ptr<Layer>
Addition(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_ADDITION, properties);
}

/**
 * @brief Helper function to create concat layer
 */
inline std::unique_ptr<Layer>
Concat(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_CONCAT, properties);
}

/**
 * @brief Helper function to create multi-out layer
 */
inline std::unique_ptr<Layer>
MultiOut(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_MULTIOUT, properties);
}

/**
 * @brief Helper function to create nnstreamer backbone layer
 */
inline std::unique_ptr<Layer>
BackboneNNStreamer(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_BACKBONE_NNSTREAMER, properties);
}

/**
 * @brief Helper function to create tflite backbone layer
 */
inline std::unique_ptr<Layer>
BackboneTFLite(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_BACKBONE_TFLITE, properties);
}

/**
 * @brief Helper function to create Embedding layer
 */
inline std::unique_ptr<Layer>
Embedding(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_EMBEDDING, properties);
}

/**
 * @brief Helper function to create RNN layer
 */
inline std::unique_ptr<Layer>
RNN(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_RNN, properties);
}

/**
 * @brief Helper function to create RNNCell layer
 */
inline std::unique_ptr<Layer>
RNNCell(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_RNNCELL, properties);
}

/**
 * @brief Helper function to create LSTM layer
 */
inline std::unique_ptr<Layer>
LSTM(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_LSTM, properties);
}

/**
 * @brief Helper function to create LSTMCell layer
 */
inline std::unique_ptr<Layer>
LSTMCell(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_LSTMCELL, properties);
}

/**
 * @brief Helper function to create ZoneoutLSTMCell layer
 */
inline std::unique_ptr<Layer>
ZoneoutLSTMCell(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_ZONEOUT_LSTMCELL, properties);
}

/**
 * @brief Helper function to create GRU layer
 */
inline std::unique_ptr<Layer>
GRU(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_GRU, properties);
}

/**
 * @brief Helper function to create GRUCell layer
 */
inline std::unique_ptr<Layer>
GRUCell(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_GRUCELL, properties);
}

/**
 * @brief Helper function to create DropOut layer
 */
inline std::unique_ptr<Layer>
DropOut(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_DROPOUT, properties);
}

/**
 * @brief Helper function to create Time Distributed layer
 */
inline std::unique_ptr<Layer>
TimeDistLayer(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_TIME_DIST, properties);
}

/**
 * @brief Helper function to create Centroid KNN Layer
 */
inline std::unique_ptr<Layer>
CentroidKNN(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_CENTROID_KNN, properties);
}

/**
 * @brief Helper function to create Attention Layer
 */
inline std::unique_ptr<Layer>
Attention(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_ATTENTION, properties);
}

/**
 * @brief Helper function to create MoL Attention Layer
 */
inline std::unique_ptr<Layer>
MoLAttention(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_MOL_ATTENTION, properties);
}

/**
 * @brief Helper function to create Multi Head Attention Layer
 */
inline std::unique_ptr<Layer>
MultiHeadAttention(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_MULTI_HEAD_ATTENTION, properties);
}

/**
 * @brief Helper function to create Positional Encoding Layer
 */
inline std::unique_ptr<Layer>
PositionalEncoding(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_POSITIONAL_ENCODING, properties);
}

/**
 * @brief Helper function to create Permute Layer
 */
inline std::unique_ptr<Layer>
Permute(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_PERMUTE, properties);
}

/**
 * @brief Helper function to create Reduce Mean Layer
 */
inline std::unique_ptr<Layer>
ReduceMean(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_REDUCE_MEAN, properties);
}

/**
 * @brief Helper function to create Reduce Sum Layer
 */
inline std::unique_ptr<Layer>
ReduceSum(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_REDUCE_SUM, properties);
}

/**
 * @brief Helper function to create Identity layer
 */
inline std::unique_ptr<Layer>
Identity(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_IDENTITY, properties);
}

/**
 * @brief Helper function to create Upsample2d layer
 */
inline std::unique_ptr<Layer>
Upsample2D(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_UPSAMPLE2D, properties);
}

/**
 * @brief Helper function to create activation layer
 */
inline std::unique_ptr<Layer>
Activation(const std::string &act,
           const std::vector<std::string> &properties = {}) {
  std::vector<std::string> props(properties);
  props.push_back(act);
  return createLayer(LayerType::LAYER_ACTIVATION, props);
}

/**
 * @brief Helper function to create ReLU activation layer
 */
inline std::unique_ptr<Layer>
ReLU(const std::vector<std::string> &properties = {}) {
  return Activation("Activation=relu", properties);
}

/**
 * @brief Helper function to create swish activation layer
 */
inline std::unique_ptr<Layer>
Swish(const std::vector<std::string> &properties = {}) {
  return Activation("Activation=swish", properties);
}

/**
 * @brief Helper function to create gelu activation layer
 */
inline std::unique_ptr<Layer>
GeLU(const std::vector<std::string> &properties = {}) {
  return Activation("Activation=gelu", properties);
}

/**
 * @brief Helper function to create Tanh layer
 */
inline std::unique_ptr<Layer>
Tanh(const std::vector<std::string> &properties = {}) {
  return Activation("Activation=tanh", properties);
}

/**
 * @brief Helper function to create sigmoid layer
 */
inline std::unique_ptr<Layer>
Sigmoid(const std::vector<std::string> &properties = {}) {
  return Activation("Activation=sigmoid", properties);
}

/**
 * @brief Helper function to create softmax layer
 */
inline std::unique_ptr<Layer>
Softmax(const std::vector<std::string> &properties = {}) {
  return Activation("Activation=softmax", properties);
}

/**
 * @brief Helper function to create elu activation layer
 */
inline std::unique_ptr<Layer>
ELU(const std::vector<std::string> &properties = {}) {
  return Activation("Activation=elu", properties);
}

/**
 * @brief Helper function to create selu activation layer
 */
inline std::unique_ptr<Layer>
SELU(const std::vector<std::string> &properties = {}) {
  return Activation("Activation=selu", properties);
}

/**
 * @brief Helper function to create mish activation layer
 */
inline std::unique_ptr<Layer>
Mish(const std::vector<std::string> &properties = {}) {
  return Activation("Activation=mish", properties);
}

} // namespace layer

namespace loss {
/**
 * @brief Helper function to create mse layer
 */
inline std::unique_ptr<Layer>
MSE(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_LOSS_MSE, properties);
}

/**
 * @brief Helper function to create cross entropy with sigmoid layer
 */
inline std::unique_ptr<Layer>
CrossEntropySigmoid(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_LOSS_CROSS_ENTROPY_SIGMOID, properties);
}

/**
 * @brief Helper function to create cross entropy with softmax layer
 */
inline std::unique_ptr<Layer>
CrossEntropySoftmax(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_LOSS_CROSS_ENTROPY_SOFTMAX, properties);
}

} // namespace loss

} // namespace train
} // namespace ml
#endif // __ML_TRAIN_LAYER_H__
