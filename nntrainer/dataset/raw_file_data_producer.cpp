
// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   raw_file_data_producer.cpp
 * @date   12 July 2021
 * @brief  This file contains raw file data producers, reading from a file
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <raw_file_data_producer.h>

#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <common_properties.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

RawFileDataProducer::RawFileDataProducer() : raw_file_props(new PropTypes()) {}

RawFileDataProducer::RawFileDataProducer(const std::string &path) :
  raw_file_props(new PropTypes(props::FilePath(path))) {}
RawFileDataProducer::~RawFileDataProducer() {}

const std::string RawFileDataProducer::getType() const {
  return RawFileDataProducer::type;
}

void RawFileDataProducer::setProperty(
  const std::vector<std::string> &properties) {
  auto left = loadProperties(properties, *raw_file_props);
  NNTR_THROW_IF(!left.empty(), std::invalid_argument)
    << "There is unparsed properties, size: " << left.size();
}

DataProducer::Generator
RawFileDataProducer::finalize(const std::vector<TensorDim> &input_dims,
                              const std::vector<TensorDim> &label_dims,
                              void *user_data) {
  auto sz = size(input_dims, label_dims);
  auto path_prop = std::get<props::FilePath>(*raw_file_props);

  auto size_accumulator = [](const size_t &a, const TensorDim &b) {
    return a + b.getFeatureLen();
  };

  auto sample_size = std::accumulate(input_dims.begin(), input_dims.end(),
                                     (size_t)0, size_accumulator);
  sample_size = std::accumulate(label_dims.begin(), label_dims.end(),
                                sample_size, size_accumulator);

  /// as we are passing the reference of file, this means created lamabda is
  /// tightly couple with the file, this is not desirable but working fine for
  /// now...
  file = std::ifstream(path_prop.get(), std::ios::binary);
  return [sample_size, sz, this](unsigned int idx, std::vector<Tensor> &inputs,
                                 std::vector<Tensor> &labels) {
    NNTR_THROW_IF(idx >= sz, std::range_error)
      << "given index is out of bound, index: " << idx << " size: " << sz;
    std::streamoff offset = static_cast<std::streamoff>(idx) *
                            static_cast<std::streamoff>(sample_size) *
                            RawFileDataProducer::pixel_size;
    file.seekg(offset, std::ios_base::beg);
    for (auto &input : inputs) {
      input.read(file);
    }
    for (auto &label : labels) {
      label.read(file);
    }

    return idx == sz - 1;
  };
}

unsigned int
RawFileDataProducer::size(const std::vector<TensorDim> &input_dims,
                          const std::vector<TensorDim> &label_dims) const {
  auto size_accumulator = [](const size_t &a, const TensorDim &b) {
    return a + b.getFeatureLen();
  };

  auto sample_size = std::accumulate(input_dims.begin(), input_dims.end(),
                                     (size_t)0, size_accumulator);
  sample_size = std::accumulate(label_dims.begin(), label_dims.end(),
                                sample_size, size_accumulator);
  NNTR_THROW_IF(sample_size == 0, std::invalid_argument)
    << "The feature size of input_dims and label_dims are zeros";

  auto path_prop = std::get<props::FilePath>(*raw_file_props);
  auto file_size = path_prop.file_size();

  /// checking alignment is a good way to make check if a file is valid,
  /// unfortunately, our test dataset does not have this property
  /// (trainingSet.dat, valSet.dat, testSet.dat) after checking, we can
  /// uncomment below line.
  // NNTR_THROW_IF((file_size % sample_size * RawFileDataProducer::pixel_size !=
  // 0),
  //               std::invalid_argument)
  //   << " Given file does not align with the given sample size, sample size: "
  //   << sample_size << " file_size: " << file_size;

  return static_cast<unsigned int>(file_size) /
         (sample_size * RawFileDataProducer::pixel_size);
}

void RawFileDataProducer::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  exporter.saveResult(*raw_file_props, method, this);
}
} // namespace nntrainer
