/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
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
 * @file    mnist_loader.h
 * @date    April 2026
 * @brief   Interface of the MNIST IDX-format loader used by LoRA examples
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Sumon Nath <sumon.nath@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#pragma once

#include <string>
#include <vector>

namespace lora {

/**
 * @brief Load MNIST dataset from IDX files into flattened float vectors with
 * normalization
 * @param images_path Path to MNIST images IDX file (train-images-idx3-ubyte)
 * @param labels_path Path to MNIST labels IDX file (train-labels-idx1-ubyte)
 * @param images Output vector of images (N x H x W flattened, normalized)
 * @param labels Output vector of labels (N class indices)
 * @param num_classes Number of label classes (default 10)
 * @return bool True on success
 *
 * Images are normalized using: (x / 255.0 - 0.1307) / 0.3081
 * These are MNIST dataset statistics (mean=0.1307, std=0.3081)
 * Labels are stored as class indices (0-9 for MNIST)
 */
bool loadMNIST(const std::string &images_path, const std::string &labels_path,
               std::vector<float> &images, std::vector<float> &labels,
               unsigned int num_classes = 10);

} // namespace lora