// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   multi_loader.h
 * @date   5 July 2023
 * @brief  multi data loader
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include <tensor_dim.h>

#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace nntrainer::util {

using TensorDim = ml::train::TensorDim;

/**
 * @brief DataLoader interface used to load cifar data
 */
class DataLoader {
public:
  /**
   * @brief Destroy the Data Loader object
   */
  virtual ~DataLoader() {}

  /**
   * @brief create an iteration to fed to the generator callback
   *
   * @param[out] input list of inputs that is already allocated by nntrainer,
   * and this function is obliged to fill
   * @param[out] label list of label that is already allocated by nntrainer, and
   * this function is obliged to fill
   * @param[out] last  optional property to set when the epoch has finished
   */
  virtual void next(float **input, float **label, bool *last) = 0;

protected:
  std::mt19937 rng;
};

/**
 * @brief MultiData Generator
 *
 */
class MultiInoutDataLoader final : public DataLoader {
public:
  /**
   * @brief Construct a new Random Data Loader object
   *
   * @param input_shapes input_shapes with appropriate batch
   * @param output_shapes label_shapes with appropriate batch
   * @param iteration     iteration per epoch
   */
  MultiInoutDataLoader(const std::vector<TensorDim> &input_shapes,
                       const std::vector<TensorDim> &output_shapes,
                       int iteration);

  /**
   * @brief Destroy the Random Data Loader object
   */
  ~MultiInoutDataLoader() {}

  /**
   * @copydoc void DataLoader::next(float **input, float**label, bool *last)
   */
  void next(float **input, float **label, bool *last);

private:
  unsigned int iteration;
  unsigned int data_size;
  unsigned int count;
  std::vector<unsigned int> indicies;

  std::vector<TensorDim> input_shapes;
  std::vector<TensorDim> output_shapes;

  std::uniform_real_distribution<float> input_dist;
  std::uniform_int_distribution<int> label_dist;
};

} // namespace nntrainer::util
