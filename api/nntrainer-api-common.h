// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file nntrainer-api-common.h
 * @date 02 April 2020
 * @brief NNTrainer Common-API Header.
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __TIZEN_MACHINELEARNING_NNTRAINER_API_COMMON_H__
#define __TIZEN_MACHINELEARNING_NNTRAINER_API_COMMON_H__

#define ML_TRAIN_LAYER_EXPERIMENTAL (10000)

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @addtogroup CAPI_ML_NNTRAINER_TRAIN_MODULE
 * @{
 */

/**
 * @brief Enumeration for the neural network layer type of NNTrainer.
 * @since_tizen 6.0
 */
typedef enum {
  ML_TRAIN_LAYER_TYPE_INPUT = 0,  /**< Input Layer */
  ML_TRAIN_LAYER_TYPE_FC = 1,     /**< Fully Connected Layer */
  ML_TRAIN_LAYER_TYPE_BN = 2,     /**< Batch Normalization Layer (Since 6.5) */
  ML_TRAIN_LAYER_TYPE_CONV2D = 3, /**< Convolution 2D Layer (Since 6.5) */
  ML_TRAIN_LAYER_TYPE_POOLING2D = 4,  /**< Pooling 2D Layer (Since 6.5) */
  ML_TRAIN_LAYER_TYPE_FLATTEN = 5,    /**< Flatten Layer (Since 6.5) */
  ML_TRAIN_LAYER_TYPE_ACTIVATION = 6, /**< Activation Layer (Since 6.5) */
  ML_TRAIN_LAYER_TYPE_ADDITION = 7,   /**< Addition Layer (Since 6.5) */
  ML_TRAIN_LAYER_TYPE_CONCAT = 8,     /**< Concat Layer (Since 6.5) */
  ML_TRAIN_LAYER_TYPE_MULTIOUT = 9,   /**< MultiOut Layer (Since 6.5) */
  ML_TRAIN_LAYER_TYPE_EMBEDDING = 10, /**< Embedding Layer (Since 6.5) */
  ML_TRAIN_LAYER_TYPE_RNN = 11,       /**< RNN Layer (Since 6.5) */
  ML_TRAIN_LAYER_TYPE_LSTM = 12,      /**< LSTM Layer (Since 6.5) */
  ML_TRAIN_LAYER_TYPE_SPLIT = 13,     /**< Split Layer (Since 6.5) */
  ML_TRAIN_LAYER_TYPE_GRU = 14,       /**< GRU Layer (Since 6.5) */
  ML_TRAIN_LAYER_TYPE_PERMUTE = 15,   /**< Permute layer (Since 6.5) */
  ML_TRAIN_LAYER_TYPE_DROPOUT = 16,   /**< Dropout Layer (Since 6.5) */
  ML_TRAIN_LAYER_TYPE_BACKBONE_NNSTREAMER = 17, /**< Backbone using NNStreamer
                                              (Since 6.5) */
  ML_TRAIN_LAYER_TYPE_CENTROID_KNN = 18, /**< Centroid KNN Layer (Since 6.5) */
  ML_TRAIN_LAYER_TYPE_CONV1D = 19, /**< Convolution 1D Layer type (Since 7.0) */
  ML_TRAIN_LAYER_TYPE_LSTMCELL = 20, /**< LSTM Cell Layer type (Since 7.0) */
  ML_TRAIN_LAYER_TYPE_GRUCELL = 21,  /**< GRU Cell Layer type (Since 7.0) */
  ML_TRAIN_LAYER_TYPE_RNNCELL = 22,  /**< RNN Cell Layer type (Since 7.0) */
  ML_TRAIN_LAYER_TYPE_ZONEOUTLSTMCELL =
    23, /**< ZoneoutLSTM Cell Layer type (Since 7.0) */
  ML_TRAIN_LAYER_TYPE_ATTENTION = 24, /**< Attention Layer type (Since 7.0) */
  ML_TRAIN_LAYER_TYPE_MOL_ATTENTION =
    25, /**< MoL Attention Layer type (Since 7.0) */
  ML_TRAIN_LAYER_TYPE_MULTI_HEAD_ATTENTION =
    26, /**< Multi Head Attention Layer type (Since 7.0) */
  ML_TRAIN_LAYER_TYPE_LAYER_NORMALIZATION =
    27, /**< Layer Normalization Layer type (Since 7.0) */
  ML_TRAIN_LAYER_TYPE_POSITIONAL_ENCODING =
    28, /**< Positional Encoding Layer type (Since 7.0) */
  ML_TRAIN_LAYER_TYPE_IDENTITY = 29,  /**< Identity Layer type (Since 8.0) */
  ML_TRAIN_LAYER_TYPE_SWIGLU = 30,    /**< Swiglu Layer type */
  ML_TRAIN_LAYER_TYPE_WEIGHT = 31,    /**< Weight Layer type (Since 9.0)*/
  ML_TRAIN_LAYER_TYPE_ADD = 32,       /**< Add Layer type (Since 9.0)*/
  ML_TRAIN_LAYER_TYPE_SUBTRACT = 33,  /**< Subtract Layer type (Since 9.0)*/
  ML_TRAIN_LAYER_TYPE_MULTIPLY = 34,  /**< Multiply Layer type (Since 9.0)*/
  ML_TRAIN_LAYER_TYPE_DIVIDE = 35,    /**< Divide Layer type (Since 9.0)*/
  ML_TRAIN_LAYER_TYPE_TRANSPOSE = 36, /**< Transpose Layer type */
  ML_TRAIN_LAYER_TYPE_CONV2D_TRANSPOSE =
    37, /**< Convolution 2D Transpose Layer (Since 9.0) */
  ML_TRAIN_LAYER_TYPE_POW = 38,     /**< Pow Layer type (Since 9.0)*/
  ML_TRAIN_LAYER_TYPE_TENSOR = 39,  /**< Tensor Layer type (Since 9.0)*/
  ML_TRAIN_LAYER_TYPE_SQRT = 40,    /**< SQRT Layer type (Since 9.0)*/
  ML_TRAIN_LAYER_TYPE_SINE = 41,    /**< Sine Layer type (Since 9.0)*/
  ML_TRAIN_LAYER_TYPE_COSINE = 42,  /**< Cosine Layer type (Since 9.0)*/
  ML_TRAIN_LAYER_TYPE_TANGENT = 43, /**< Tangent Layer type (Since 9.0)*/
  ML_TRAIN_LAYER_TYPE_MATMUL = 44,  /**< MatMul Layer type (Since 9.0)*/
  ML_TRAIN_LAYER_TYPE_CHANNEL_SHUFFLE =
    45,                            /**< Channel Shuffle Layer type (Since 9.0)*/
  ML_TRAIN_LAYER_TYPE_NEG = 46,    /**< Negative Layer type (Since 9.0)*/
  ML_TRAIN_LAYER_TYPE_GATHER = 47, /**< Gather Layer type (Since 9.0)*/
  ML_TRAIN_LAYER_TYPE_REDUCE_SUM = 48, /**< ReduceSum Layer type (Since 9.0) */
  ML_TRAIN_LAYER_TYPE_SLICE = 49,      /**< SQRT Layer type (Since 9.0)*/
  ML_TRAIN_LAYER_TYPE_CAST = 50,       /**< Cast Layer type (Since 9.0)*/
  ML_TRAIN_LAYER_TYPE_REDUCE_MEAN =
    51,                             /**< ReduceMean Layer type (Since 10.0) */
  ML_TRAIN_LAYER_TYPE_RESHAPE = 52, /**< Reshape Layer type (Since 10.0) */
  ML_TRAIN_LAYER_TYPE_UPSAMPLE2D =
    53, /**< Upsample2D Layer type (Since 10.0) */
  ML_TRAIN_LAYER_TYPE_PREPROCESS_FLIP =
    300, /**< Preprocess flip Layer (Since 6.5) */
  ML_TRAIN_LAYER_TYPE_PREPROCESS_TRANSLATE =
    301, /**< Preprocess translate Layer (Since 6.5) */
  ML_TRAIN_LAYER_TYPE_PREPROCESS_L2NORM =
    302, /**< Preprocess L2Normalization Layer (Since 6.5) */
  ML_TRAIN_LAYER_TYPE_LOSS_MSE =
    500, /**< Mean Squared Error Loss Layer type (Since 6.5) */
  ML_TRAIN_LAYER_TYPE_LOSS_CROSS_ENTROPY_SIGMOID = 501, /**< Cross Entropy with
                                       Sigmoid Loss Layer type (Since 6.5) */
  ML_TRAIN_LAYER_TYPE_LOSS_CROSS_ENTROPY_SOFTMAX = 502, /**< Cross Entropy with
                                       Softmax Loss Layer type (Since 6.5) */
  ML_TRAIN_LAYER_TYPE_RMSNORM = 503, /**< Cross Entropy with */
  ML_TRAIN_LAYER_TYPE_UNKNOWN = 999  /**< Unknown Layer */
} ml_train_layer_type_e;

/**
 * @brief Enumeration for the neural network optimizer type of NNTrainer.
 * @since_tizen 6.0
 */
typedef enum {
  ML_TRAIN_OPTIMIZER_TYPE_ADAM = 0,  /**< Adam Optimizer */
  ML_TRAIN_OPTIMIZER_TYPE_ADAMW = 2, /**< AdamW Optimizer */
  ML_TRAIN_OPTIMIZER_TYPE_LION = 3,  /**< Lion Optimizer */
  ML_TRAIN_OPTIMIZER_TYPE_SGD = 1, /**< Stochastic Gradient Descent Optimizer */
  ML_TRAIN_OPTIMIZER_TYPE_UNKNOWN = 999 /**< Unknown Optimizer */
} ml_train_optimizer_type_e;

/**
 * @brief Enumeration for the learning rate scheduler type of NNTrainer.
 * @since_tizen 8.0
 */
typedef enum {
  ML_TRAIN_LR_SCHEDULER_TYPE_CONSTANT = 0,    /**< Constant lr scheduler */
  ML_TRAIN_LR_SCHEDULER_TYPE_EXPONENTIAL = 1, /**< Exponentially lr scheduler */
  ML_TRAIN_LR_SCHEDULER_TYPE_STEP = 2,        /**< Step lr scheduler */
  ML_TRAIN_LR_SCHEDULER_TYPE_COSINE = 3,      /**< Cosine lr scheduler */
  ML_TRAIN_LR_SCHEDULER_TYPE_UNKNOWN = 999    /**< Unknown lr scheduler */
} ml_train_lr_scheduler_type_e;

/**
 * @brief Dataset generator callback function for train/valid/test data.
 *
 * @details The user of the API must provide this callback function to supply
 * data to the model and register the callback with
 * ml_train_dataset_add_generator(). The model will call this callback
 * whenever it needs more data. This function should provide a single element of
 * input and label data in the passed containers. The containers passed by the
 * caller will already be allocated with sufficient space to contain the data.
 * This function callback should fill the data row-wise in the containers
 * provided. The containers represent array of memory to hold inputs for the
 * model. If the model contains two inputs, then input[0] will hold the first
 * input, and input[1] will hold the second input. The same applies for labels
 * as well. The number of inputs and labels, and the size of each input and
 * label should match with the shape of each input and label set in the model.
 * The order of the inputs/labels, in case of multiple of inputs/labels, will be
 * determined based on the sequence of addition of the input layers to the
 * model.
 * @since_tizen 6.0
 * @note This function can be called multiple times in parallel when total
 * number of samples are set as a property for this dataset. In this case, last
 * is only used for verification purposes. If total number of samples for the
 * dataset is unknown, this function will be called in sequence.
 * @note @a last has to be set true when filling the last @a input, @a label.
 * @param[out] input Container to hold all the input data. Should not be freed
 * by the user.
 * @param[out] label Container to hold corresponding label data. Should not be
 * freed by the user.
 * @param[out] last Container to notify if data is finished. Set true if no more
 * data to provide, else set false. Should not be freed by the user.
 * @param[in] user_data User application's private data passed along with
 * ml_train_dataset_add_generator().
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 *
 * A sample implementation of this function is below:
 * Note : input data information available from outside this function.
 * num_samples : total number of data samples in the dataset.
 * count : number of samples already given.
 * num_inputs : number of inputs.
 * num_labels : number of labels.
 * input_length[num_inputs] : length of the input. With (batch, c, h, w) as
 *                            input shape, the length will be c * h * w.
 * label_length[num_labels] : length of the label. With (batch, l) as label
 *                            shape, then length will be l.
 * @code
 * // function signature :
 * // int rand_dataset_generator (float **input, float **label, bool *last,
 * // void *user_data).
 *
 * // This sample fills inputs and labels with random data.
 * srand(0);
 * if (count >= num_samples) {
 *   *last = true;
 *   // handle preparation for start of next epoch
 * } else {
 *   *last = false;
 * }
 *
 * // Fill input data
 * for (int idx = 0; idx < num_inputs; ++ idx) {
 *   for (int len = 0; len < input_length[idx]; ++ len) {
 *     input[idx][len] = rand();
 *   }
 * }
 *
 * // Fill label data
 * for (int idx = 0; idx < num_labels; ++ idx) {
 *   for (int len = 0; len < label_length[idx]; ++ len) {
 *     label[idx][len] = rand();
 *   }
 * }
 *
 * // Update the helper variables
 * count += 1;
 *
 * return ML_ERROR_NONE;
 * @endcode
 *
 *
 * Below is an example of the usage of this sample:
 * @code
 * int status;
 * void * user_data;
 * ml_train_dataset_h handle;
 * status = ml_train_dataset_create(&handle);
 * if (status != ML_ERROR_NONE) {
 *    // handle error case.
 *    return status;
 * }
 *
 * status = ml_train_dataset_add_generator(dataset,
 *      ML_TRAIN_DATASET_MODE_TRAIN, getBatch_train, user_data);
 * if (status != ML_ERROR_NONE) {
 *    // handle error case.
 *    return status;
 * }
 *
 * // Destroy the handle if not added to a model.
 * status = ml_train_dataset_destroy(handle);
 * if (status != ML_ERROR_NONE) {
 *    // handle error case
 *    return status;
 * }
 * @endcode
 */
typedef int (*ml_train_datagen_cb)(float **input, float **label, bool *last,
                                   void *user_data);

/**
 * @brief Enumeration for the dataset data type of NNTrainer.
 * @since_tizen 6.5
 */
typedef enum {
  ML_TRAIN_DATASET_MODE_TRAIN =
    0, /**< The given data is for used when training */
  ML_TRAIN_DATASET_MODE_VALID =
    1, /**< The given data is for used when validating */
  ML_TRAIN_DATASET_MODE_TEST =
    2, /**< The given data is for used when testing */
} ml_train_dataset_mode_e;

/**
 * @brief Enumeration for the neural network summary verbosity of NNTrainer.
 * @since_tizen 6.0
 */
typedef enum {
  ML_TRAIN_SUMMARY_MODEL = 0, /**< Overview of model
                                   summary with one-line layer information */
  ML_TRAIN_SUMMARY_LAYER =
    1, /**< Detailed model summary with layer properties */
  ML_TRAIN_SUMMARY_TENSOR = 2 /**< Model summary layer's including weight
                             information */
} ml_train_summary_type_e;

/**
 * @brief Enumeration for the neural network.
 * @since_tizen 6.5
 *
 */
typedef enum {
  ML_TRAIN_MODEL_FORMAT_BIN =
    0, /**< Raw bin file saves model weights required for inference and training
          without any configurations*/
  ML_TRAIN_MODEL_FORMAT_INI =
    1, /**< Ini format file saves model configurations. */
  ML_TRAIN_MODEL_FORMAT_INI_WITH_BIN =
    2, /**< Ini with bin format file saves configurations with parameters
         required for inference and training. */
  ML_TRAIN_MODEL_FORMAT_FLATBUFFER =
    3, /**< Flatbuffer format file saves model configurations and weights. */
  ML_TRAIN_MODEL_FORMAT_ONNX =
    4, /**< QNNX binary format file saves model configurations and weights. */
  ML_TRAIN_MODEL_FORMAT_QNN =
    5 /**< QNN binary format file saves model configurations and weights. */
} ml_train_model_format_e;

/**
 * @}
 */
#ifdef __cplusplus
}

#endif /* __cplusplus */
#endif /* __TIZEN_MACHINELEARNING_NNTRAINER_API_COMMON_H__ */
