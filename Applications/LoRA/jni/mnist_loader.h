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