#include "mnist_loader.h"

#include <arpa/inet.h>
#include <fstream>
#include <nntrainer_log.h>
#include <stdexcept>

namespace lora {

// MNIST normalization constants (from PyTorch torchvision.transforms.Normalize)
static constexpr float MNIST_MEAN = 0.1307f;
static constexpr float MNIST_STD = 0.3081f;

static uint32_t read_u32_be(std::ifstream &ifs) {
  uint32_t val = 0;
  ifs.read(reinterpret_cast<char *>(&val), sizeof(val));
  return ntohl(val);
}

bool loadMNIST(const std::string &images_path, const std::string &labels_path,
               std::vector<float> &images, std::vector<float> &labels,
               unsigned int num_classes) {
  images.clear();
  labels.clear();

  std::ifstream ifs_images(images_path, std::ios::binary);
  if (!ifs_images.is_open()) {
    ml_loge("Failed to open MNIST images file: %s", images_path.c_str());
    return false;
  }

  uint32_t magic_images = read_u32_be(ifs_images);
  uint32_t num_images = read_u32_be(ifs_images);
  uint32_t rows = read_u32_be(ifs_images);
  uint32_t cols = read_u32_be(ifs_images);

  if (magic_images != 2051) {
    ml_loge("Invalid MNIST images magic: %u", magic_images);
    return false;
  }

  std::ifstream ifs_labels(labels_path, std::ios::binary);
  if (!ifs_labels.is_open()) {
    ml_loge("Failed to open MNIST labels file: %s", labels_path.c_str());
    return false;
  }

  uint32_t magic_labels = read_u32_be(ifs_labels);
  uint32_t num_labels = read_u32_be(ifs_labels);

  if (magic_labels != 2049) {
    ml_loge("Invalid MNIST labels magic: %u", magic_labels);
    return false;
  }

  if (num_images != num_labels) {
    ml_loge("Images/labels count mismatch: %u vs %u", num_images, num_labels);
    return false;
  }

  size_t image_size = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  images.reserve((size_t)num_images * image_size);
  labels.reserve((size_t)num_images);

  // Read images with normalization
  for (uint32_t i = 0; i < num_images; ++i) {
    std::vector<unsigned char> buf(image_size);
    ifs_images.read(reinterpret_cast<char *>(buf.data()), image_size);
    if (!ifs_images) {
      ml_loge("Error reading image %u", i);
      return false;
    }
    for (size_t p = 0; p < image_size; ++p) {
      // Normalize: (pixel / 255.0 - mean) / std
      float pixel = static_cast<float>(buf[p]) / 255.0f;
      float normalized = (pixel - MNIST_MEAN) / MNIST_STD;
      images.push_back(normalized);
    }
  }

  // Read labels as class indices
  for (uint32_t i = 0; i < num_labels; ++i) {
    unsigned char lab = 0;
    ifs_labels.read(reinterpret_cast<char *>(&lab), 1);
    if (!ifs_labels) {
      ml_loge("Error reading label %u", i);
      return false;
    }
    labels.push_back(static_cast<float>(lab));
  }

  ml_logi("Loaded MNIST: %u samples, image=%ux%u, classes=%u", num_images, rows,
          cols, num_classes);
  return true;
}

} // namespace lora