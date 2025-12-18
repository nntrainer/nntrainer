// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   dir_data_producers.h
 * @date   24 Feb 2023
 * @brief  This file contains dir data producers, reading from the files in
 * directory
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <dir_data_producers.h>
#include <filesystem>

#include <memory>
#include <numeric>
#include <random>
#include <sys/stat.h>
#include <vector>

#include <common_properties.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <util_func.h>

/**
 * @brief this function helps to read the image
 * currently only bmp image is supported and extend the image type will be
 * remain as TODO. ( BGR --> RGB )
 * @param     path file path
 * @param     inputs float data for image pixel
 * @param     width width
 * @param     height height
 */
static void readImage(const std::string path, float *input, unsigned int width,
                      unsigned int height) {
  FILE *f = fopen(path.c_str(), "rb");

  if (f == nullptr)
    throw std::invalid_argument("Cannot open file: " + path);

  unsigned char info[54];
  size_t result = fread(info, sizeof(unsigned char), 54, f);
  NNTR_THROW_IF(result != 54, std::invalid_argument)
    << "Cannot read bmp header";

  size_t row_padded = (width * 3 + 3) & (~3);
  unsigned char *data = new unsigned char[row_padded];

  for (unsigned int i = 0; i < height; i++) {
    result = fread(data, sizeof(unsigned char), row_padded, f);
    NNTR_THROW_IF(result != row_padded, std::invalid_argument)
      << "Cannot read bmp pixel data";

    for (unsigned int j = 0; j < width; j++) {

      input[height * i + j] = (float)data[j * 3 + 2];

      input[(height * width) + height * i + j] = (float)data[j * 3 + 1];

      input[(height * width) * 2 + height * i + j] = (float)data[j * 3];
    }
  }

  delete[] data;
  fclose(f);
}

namespace nntrainer {

DirDataProducer::DirDataProducer() :
  dir_data_props(new Props()), num_class(0), num_data_total(0) {}

DirDataProducer::DirDataProducer(const std::string &dir_path) :
  dir_data_props(new Props(props::DirPath(dir_path))),
  num_class(0),
  num_data_total(0) {}

DirDataProducer::~DirDataProducer() {}

const std::string DirDataProducer::getType() const {
  return DirDataProducer::type;
}

bool DirDataProducer::isMultiThreadSafe() const {
  /// @todo make this true, it is needed to test multiple worker scenario
  return false;
}

void DirDataProducer::setProperty(const std::vector<std::string> &properties) {
  auto left = loadProperties(properties, *dir_data_props);
  NNTR_THROW_IF(!left.empty(), std::invalid_argument)
    << "There are unparsed properties, size: " << left.size();
}

DataProducer::Generator
DirDataProducer::finalize(const std::vector<TensorDim> &input_dims,
                          const std::vector<TensorDim> &label_dims,
                          void *user_data) {

  const auto &dir_path = std::get<props::DirPath>(*dir_data_props).get();

  for (const auto &entry : std::filesystem::directory_iterator(dir_path))
    class_names.push_back(entry.path().string());

  num_class = class_names.size();

  size_t id = 0;
  size_t num_data = 0;
  for (auto &c_name : class_names) {
    num_data = 0;
    std::filesystem::directory_iterator itr(c_name);
    while (itr != std::filesystem::end(itr)) {
      const std::filesystem::directory_entry &entry = *itr;
      std::string p = std::filesystem::absolute(entry.path()).string();
      if (p.compare(".") && p.compare("..")) {
        num_data++;
        data_list.push_back(std::make_pair((unsigned int)id, p));
      }
      itr++;
    }

    id++;
    num_data_total += num_data;
  }

  /// @todo expand this to non onehot case
  NNTR_THROW_IF(std::any_of(label_dims.begin(), label_dims.end(),
                            [](const TensorDim &dim) {
                              return dim.channel() != 1 || dim.height() != 1;
                            }),
                std::invalid_argument)
    << "Label dimension containing channel or height not allowed";

  auto sz = size(input_dims, label_dims);

  NNTR_THROW_IF(sz == 0, std::invalid_argument)
    << "size is zero, data producer does not provide anything";

  return [sz, input_dims, this](unsigned int idx, std::vector<Tensor> &inputs,
                                std::vector<Tensor> &labels) {
    NNTR_THROW_IF(idx >= sz, std::range_error)
      << "given index is out of bound, index: " << idx << " size: " << sz;

    std::string file_name = data_list[idx].second;

    readImage(file_name, inputs[0].getData(), input_dims[0].width(),
              input_dims[0].height());

    unsigned int c_id = data_list[idx].first;

    std::memset(labels[0].getData(), 0, num_class * sizeof(float));

    labels[0].getData()[c_id] = 1.0;

    return idx == sz - 1;
  };
}

unsigned int
DirDataProducer::size(const std::vector<TensorDim> &input_dims,
                      const std::vector<TensorDim> &label_dims) const {

  return num_data_total;
}

} // namespace nntrainer
