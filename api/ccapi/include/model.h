// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   model.h
 * @date   14 October 2020
 * @see	   https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is model interface for c++ API
 *
 * @note This is experimental API and not stable.
 */

#ifndef __ML_TRAIN_MODEL_H__
#define __ML_TRAIN_MODEL_H__

#if __cplusplus >= MIN_CPP_VERSION

#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include <common.h>

#include <dataset.h>
#include <layer.h>
#include <optimizer.h>
#include <tensor_dim.h>

namespace nntrainer {
class RunLayerContext;
}
/** Define more aliases for the model in the API */
namespace ml {
namespace train {

class Tensor; // Forward declaration for graph-based compile

/**
 * @brief     Statistics from running or training a model
 */
struct RunStats {
  float accuracy;     /** accuracy of the model */
  float loss;         /** loss of the model */
  int num_iterations; /** number of iterations done on this stat */
  int max_epoch;
  int epoch_idx;
  unsigned int
    num_correct_predictions; /** number of right sample on this run */

  /**
   * @brief     Initializer of RunStats
   */
  RunStats() :
    accuracy(0),
    loss(0),
    num_iterations(0),
    max_epoch(0),
    epoch_idx(0),
    num_correct_predictions(0) {}
};

/**
 * @brief     Enumeration of Network Type
 */
enum class ModelType {
  KNN,        /** k Nearest Neighbor */
  NEURAL_NET, /** Neural Network */
  UNKNOWN     /** Unknown */
};

/**
 * @brief     Enumeration to state type of reference of layers
 */
enum class ReferenceLayersType {
  BACKBONE,  /** backbone */
  RECURRENT, /** recurrent */
};

/**
 * @brief Model saving options
 *
 */
enum class ModelFormat {
  MODEL_FORMAT_BIN =
    ML_TRAIN_MODEL_FORMAT_BIN, /**< raw bin file saves model weights required
for inference and training without any configurations*/
  MODEL_FORMAT_INI = ML_TRAIN_MODEL_FORMAT_INI, /**< ini file */
  MODEL_FORMAT_INI_WITH_BIN =
    ML_TRAIN_MODEL_FORMAT_INI_WITH_BIN, /**< ini file with save_path defined
                                           where the binary will be saved */
  MODEL_FORMAT_FLATBUFFER =
    ML_TRAIN_MODEL_FORMAT_FLATBUFFER,             /**< flatbuffer file */
  MODEL_FORMAT_ONNX = ML_TRAIN_MODEL_FORMAT_ONNX, /**< ONNX file */

  MODEL_FORMAT_QNN = ML_TRAIN_MODEL_FORMAT_QNN /**< qnn binary file */
};

/**
 * @class   Model Class
 * @brief   Model Class containing configuration, layers, optimizer and dataset
 */
class Model {
public:
  /**
   * @brief     Destructor of Model Class
   */
  virtual ~Model() = default;

  /**
   * @brief     Create and load the Network with ini configuration file.
   * @param[in] config config file path
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int loadFromConfig(const std::string &config) = 0;

  /**
   * @brief     Minimal set of properties that must be supported by the model
   * @details   The minimal properties:
                - loss_type
                - batch_size
                - epochs
                - save_path
                - continue_train
   */
  /**
   * @brief     set Property of Network
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   * @details   This function accepts vector of properties in the format -
   *  { std::string property_name, void * property_val, ...}
   */
  virtual void setProperty(const std::vector<std::string> &values) = 0;

  /**
   * @brief     Compile Network. This should be called before initialize
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int compile(ExecutionMode exec_mode_ = ExecutionMode::TRAIN) = 0;

  /**
   * @brief     Compile from symbolic tensor graph.
   *
   * Extracts the computation graph by walking backwards from output to input,
   * then adds all discovered layers (with input_layers connections) and
   * compiles. This replaces manual addLayer() + input_layers strings.
   *
   * @param input  Leaf symbolic tensor (model input)
   * @param output Output symbolic tensor (model output)
   * @param mode   Execution mode (default: TRAIN)
   * @retval #ML_ERROR_NONE Successful.
   */
  int compile(Tensor &input, Tensor &output,
              ExecutionMode mode = ExecutionMode::INFERENCE);

  /**
   * @brief     Compile from symbolic tensor graph with multiple outputs.
   *
   * Extracts the computation graph by walking backwards from each output to
   * the input, then adds all discovered layers and compiles. This supports
   * models with multiple output heads (e.g., YOLOv3 with 3 loss layers).
   *
   * @param input   Leaf symbolic tensor (model input)
   * @param outputs Vector of output symbolic tensors
   * @param mode    Execution mode (default: TRAIN)
   * @retval #ML_ERROR_NONE Successful.
   */
  int compile(Tensor &input, std::vector<Tensor> &outputs,
              ExecutionMode mode = ExecutionMode::INFERENCE);

  /**
   * @brief     Compile from symbolic tensor graph with multiple inputs and
   *            multiple outputs.
   *
   * Extracts the computation graph by walking backwards from each output to
   * discover all layers. Supports models with multiple input heads and
   * multiple output heads.
   *
   * @param inputs  Vector of leaf symbolic tensors (model inputs)
   * @param outputs Vector of output symbolic tensors
   * @param mode    Execution mode (default: TRAIN)
   * @retval #ML_ERROR_NONE Successful.
   */
  int compile(std::vector<Tensor> &inputs, std::vector<Tensor> &outputs,
              ExecutionMode mode = ExecutionMode::INFERENCE);

  /**
   * @brief     Initialize Network. This should be called after setting the
   * property and compiling.
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int initialize(ExecutionMode exec_mode_ = ExecutionMode::TRAIN) = 0;

  /**
   * @brief     Reinitialize Network. This should be called after initialize
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int reinitialize() = 0;

  /**
   * @brief  save model states and training parameters from a file
   * @param[in] file_path file_path to save the model, if full path is not
   * given, it should be saved inside working directory
   * @param[in] format format to save parameters
   * @param[in] dtype default data type to save all weights. Only Q4_0 is
   *            currently supported as the target type. Conversion is only
   *            supported when the current weight type is FP32.
   * @param[in] layer_dtype_map a map of layer name to data type. If a layer
   * name is found in this map, its weights are saved with the specific data
   *            type instead of @a dtype.
   * @note When @a dtype equals the current weight type of a layer and the layer
   *       is not in @a layer_dtype_map , the weights are saved as-is without
   * any conversion.
   * @note save-with-dtype only supports the `MODEL_FORMAT_BIN` model format
   */
  virtual void save(
    const std::string &file_path,
    ModelFormat format = ModelFormat::MODEL_FORMAT_BIN,
    TensorDim::DataType dtype = TensorDim::DataType::NONE,
    const std::map<std::string, TensorDim::DataType> &layer_dtype_map = {}) = 0;

  /**
   * @brief  load model with regard to the format
   * @param file_path file_path to save the model, if full path is not
   * given, it should be saved inside working directory
   * @param format format to save parameters
   */
  virtual void load(const std::string &file_path,
                    ModelFormat format = ModelFormat::MODEL_FORMAT_BIN) = 0;

  /**
   * @brief     Run Model training and validation
   * @param[in] values hyper parameters
   * @param[in] stop_cb callback function to decide stop training or not
   * ~~~~~
   * @a stop_user_data user_data to be used in stop_cb
   * @a bool true if stop the training
   * ~~~~~
   * @param[in] epoch_complete_cb Called the end of an epoch
   * ~~~~~
   * @a epoch_user_data user_data to be used in epoch_complete_cb
   * ~~~~~
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   * @details   This function accepts vector of properties in the format -
   *  { std::string property_name, void * property_val, ...}
   */
  virtual int train(
    const std::vector<std::string> &values = {},
    std::function<bool(void *)> stop_cb =
      [](void *stop_user_data) { return false; },
    void *stop_user_data = nullptr,
    std::function<void(void *)> epoch_complete_cb =
      [](void *epoch_user_data) { return false; },
    void *epoch_user_data = nullptr) = 0;

  /**
   * @brief     Run Model train with callback function by user
   * @param[in] mode mode of the dataset
   * @param[in] dataset set the dataset
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int setDataset(const ml::train::DatasetModeType &mode,
                         std::shared_ptr<Dataset> dataset) = 0;

  /**
   * @brief     add layer into neural network model
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int addLayer(std::shared_ptr<Layer> layer) = 0;

  /**
   * @brief add referring to reference layers.
   * @note This method does add the provided layers itself but adds a deep copy
   * of the passed layers to the model. The layers passed to this function can
   * be reused later.
   * @note @a reference is a set of layers connected each other to form a part
   * or whole graph which can be loaded to a model and can be run.
   * More specifically, the graph with a non cyclic, directed graph with all
   * node has either incoming or outgoing connection defined when considering
   * non-input layer is directly connected to the previous layer. 'input
   * layer' is defined as a layer that has no incoming connection and
   * (identified as @a start_layers or input shape is specified explicitly).
   *
   * @param reference a group of layers being referred to.
   * @param type type of reference layers
   * @param scope scope of added layers, identifier will be added
   * @param input_layers input layers which will be used to connect input layers
   * @param start_layers start layers which will be used to specify start of the
   * layers inside @a reference
   * @param end_layers end layers which will be used to specify end of the
   * layers inside @a reference
   * @param type_properties type dependent properties
   */
  virtual void addWithReferenceLayers(
    const std::vector<std::shared_ptr<Layer>> &reference,
    const std::string &scope, const std::vector<std::string> &input_layers,
    const std::vector<std::string> &start_layers,
    const std::vector<std::string> &end_layers, ReferenceLayersType type,
    const std::vector<std::string> &type_properties = {}) = 0;

  /**
   * @brief Visit each layer inside model
   * @param fn function to be called for each layer
   * - call param
   *   @param layer layer the call back is visiting
   *   @param rc run layer context reference clubbed with layer
   *   @param user_data user data passed along the function
   * @param user_data user data to pass along the callback
   */
  virtual void
  forEachLayer(std::function<void(Layer & /**< layer */,
                                  nntrainer::RunLayerContext & /**< rc */,
                                  void * /**< user_data */)>
                 fn,
               void *user_data = nullptr) = 0;

  /**
   * @brief     set optimizer for the neural network model
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int setOptimizer(std::shared_ptr<Optimizer> optimizer) = 0;

  /**
   * @brief     get layer by name from neural network model
   * @param[in] name name of the layer to get
   * @param[out] layer shared_ptr to hold the layer to get
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int getLayer(const char *name, std::shared_ptr<Layer> *layer) = 0;

  /**
   * @brief     get input dimension of a model
   * @retval    std::vector<ml::train::TensorDim> input dimension
   */
  virtual std::vector<ml::train::TensorDim> getInputDimension() = 0;

  /**
   * @brief     get output dimension of a model
   * @retval    std::vector<ml::train::TensorDim> output dimension
   */
  virtual std::vector<ml::train::TensorDim> getOutputDimension() = 0;

  /**
   * @brief     Run the inference of the model
   * @param[in] batch batch size of current input
   * @param[in] input inputs as a list of each input data
   * @param[in] label labels as a list of each label data
   * @retval list of output as float *
   * @note The output memory must not be freed by the caller
   */
  virtual std::vector<float *>
  inference(unsigned int batch, const std::vector<float *> &input,
            const std::vector<float *> &label = std::vector<float *>()) = 0;

  /**
   * @brief     Run the incremental inference of the model
   * @param[in] batch batch size of current input
   * @param[in] input inputs as a list of each input data
   * @param[in] label labels as a list of each label data
   * @param[in] init_seq_len initial sequence length
   * @param[in] from current working step index
   * @param[in] to next working step index
   * @param[in] output_hidden_state (NYI) true to return all hidden state,
   * false to return last hidden state
   * @retval list of output as float *
   * @note If output_hidden_state is false, the output memory must be freed by
   * the caller after use. Otherwise, the output memory must not be freed by the
   * caller.
   */
  virtual std::vector<float *>
  incremental_inference(unsigned int batch, const std::vector<float *> &input,
                        const std::vector<float *> &label,
                        unsigned int init_seq_len, unsigned int from,
                        unsigned int to, bool output_hidden_state = false) = 0;

  /**
   * @brief     reset input dimensions of a model
   * @param[in] dims input dimensions
   * @note Similar to reinitialize, the resetInputDimension API is used for
   * modifying input dimensions after model initialization. The reinitialize
   * function should be the officially called API when changing input
   * dimensions, as it properly recalculates weights, tensors, and outputs for
   * each layer. On the other hand, resetInputDimension is a specialized API
   * created to modify only specific dimensions (specifically height values)
   * within input/output dimensions. Since this API uniformly adjusts the height
   * across all model layers, developers must verify that every layer in their
   * model architecture can safely accommodate such height modifications.
   */
  virtual void resetInputDimension(std::vector<ml::train::TensorDim> dims) = 0;

  /**
   * @brief     Summarize the model
   * @param out std::ostream to get the model summary
   * @param verbosity verbosity of the summary
   */
  virtual void summarize(std::ostream &out,
                         ml_train_summary_type_e verbosity) = 0;

  /**
   * @brief     Get current epoch_idx
   * @retval    epoch_idx
   */
  virtual unsigned int getCurrentEpoch() = 0;

  /**
   * @brief     Get Loss
   * @retval    loss value
   */
  virtual float getLoss() = 0;

  /**
   * @brief returns compilation state of a network
   * @retval initialized value
   */
  virtual bool getCompiled() const = 0;

  /**
   * @brief returns initialization state of a network
   * @retval initialized value
   */
  virtual bool getInitialized() const = 0;

  /**
   * @brief returns loadedFromConfig state of a network
   * @retval loadedFromConfig value
   */
  virtual bool getLoadedFromConfig() const = 0;

  /**
   * @brief     Get Loss from the previous epoch of training data
   * @retval    loss value
   */
  virtual float getTrainingLoss() = 0;

  /**
   * @brief     Get Loss from the previous epoch of validation data
   * @retval    loss value
   */
  virtual float getValidationLoss() = 0;

  /**
   * @brief     Get training statstics
   * @retval    training statstics
   */
  virtual RunStats getTrainingStats() = 0;

  /**
   * @brief     Get validation statstics
   * @retval    validation statstics
   */
  virtual RunStats getValidStats() = 0;

  /**
   * @brief     Get test statstics
   * @retval    test statstics
   */
  virtual RunStats getTestStats() = 0;

  /**
   * @brief     allocate tensor according to execution mode
   */
  virtual int allocate(ExecutionMode mode = ExecutionMode::TRAIN) = 0;

  /**
   * @brief export the model according to given export method
   * @param method export method
   * @param file_path path to be serialized
   */
  virtual void exports(const ExportMethods &method,
                       const std::string file_path) = 0;
};

/**
 * @brief Factory creator with constructor for optimizer
 */
std::unique_ptr<Model>
createModel(ModelType type = ml::train::ModelType::NEURAL_NET,
            const std::vector<std::string> &properties = {});

/**
 * @brief creator by copying the configuration of other model
 */
std::unique_ptr<Model> copyConfiguration(Model &from);

} // namespace train
} // namespace ml

#else
#error "CPP versions c++17 or over are only supported"
#endif // __cpluscplus
#endif // __ML_TRAIN_MODEL_H__
