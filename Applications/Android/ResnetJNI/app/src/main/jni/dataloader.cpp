// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    dataloader.cpp
 * @date    08 Sept 2021
 * @see     https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is simple nntrainer implementaiton with JNI
 *
 */

#include "dataloader.h"

#include <android/log.h>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <filesystem>
#include <image.h>
#include <nntrainer_error.h>
#include <random>
#include <unistd.h>
#include <vector>

#define LOG_TAG "nntrainer"

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

/**
 * @brief Bitmap image struct
 *
 */
struct bmpimage {
  bmpimage(int height, int width) : w(width), h(height), rgb(w * h * 3) {}
  uint8_t &r(int y, int x) { return rgb[(x + y * w) * 3 + 2]; }
  uint8_t &g(int y, int x) { return rgb[(x + y * w) * 3 + 1]; }
  uint8_t &b(int y, int x) { return rgb[(x + y * w) * 3 + 0]; }

  int w, h;
  std::vector<uint8_t> rgb;
};

template <class Stream> Stream &operator<<(Stream &out, bmpimage const &img) {
  uint32_t w = img.w, h = img.h;
  uint32_t pad = w * -3 & 3;
  uint32_t total = 54 + 3 * w * h + pad * h;
  uint32_t head[13] = {total, 0, 54, 40, w, h, (24 << 16) | 1};
  char const *rgb = (char const *)img.rgb.data();

  out.write("BM", 2);
  out.write((char *)head, 52);
  for (uint32_t i = 0; i < h; i++) {
    out.write(rgb + (3 * w * i), 3 * w);
    out.write((char *)&pad, pad);
  }
  return out;
}

bool SaveBitmap24bitColor(const char *szPathName, float *img, int w, int h) {

  bmpimage bmpimg(h, w);
  int fs = w * h;
  int fs2 = fs * 2;
  for (int y = 0; y < h; y++) {
    int hh = y * w;
    for (int x = 0; x < w; x++) {
      bmpimg.r(y, x) = (unsigned char)img[hh + x];
      bmpimg.g(y, x) = (unsigned char)img[fs + hh + x];
      bmpimg.b(y, x) = (unsigned char)img[fs2 + hh + x];
    }
  }

  std::ofstream(szPathName) << bmpimg;
  return true;
}

namespace nntrainer::resnet {

namespace {
/**
 * @brief fill label to the given memory
 *
 * @param data data to fill
 * @param length size of the data
 * @param label label
 */
void fillLabel(float *data, unsigned int length, unsigned int label) {
  if (length == 1) {
    *data = label;
    return;
  }

  memset(data, 0, length * sizeof(float));
  *(data + label) = 1;
}

/**
 * @brief fill last to the given memory
 * @note this function increases iteration value, if last is set to true,
 * iteration resets to 0
 *
 * @param[in/out] iteration current iteration
 * @param data_size Data size
 * @return bool true if iteration has finished
 */
bool updateIteration(unsigned int &iteration, unsigned int data_size) {
  if (iteration++ == data_size) {
    iteration = 0;
    return true;
  }
  return false;
};

} // namespace

RandomDataLoader::RandomDataLoader(const std::vector<TensorDim> &input_shapes,
                                   const std::vector<TensorDim> &output_shapes,
                                   int data_size_) :
  iteration(0),
  data_size(data_size_),
  input_shapes(input_shapes),
  output_shapes(output_shapes),
  input_dist(0, 255),
  label_dist(0, output_shapes.front().width() - 1) {
  NNTR_THROW_IF(output_shapes.empty(), std::invalid_argument)
    << "output_shape size empty not supported";
  NNTR_THROW_IF(output_shapes.size() > 1, std::invalid_argument)
    << "output_shape size > 1 is not supported";
}

void RandomDataLoader::next(float **input, float **label, bool *last) {
  auto fill_input = [this](float *input, unsigned int length) {
    for (unsigned int i = 0; i < length; ++i) {
      *input = input_dist(rng);
      input++;
    }
  };

  auto fill_label = [this](float *label, unsigned int batch,
                           unsigned int length) {
    unsigned int generated_label = label_dist(rng);
    fillLabel(label, length, generated_label);
    label += length;
  };

  if (updateIteration(iteration, data_size)) {
    *last = true;
    return;
  }

  float **cur_input_tensor = input;
  for (unsigned int i = 0; i < input_shapes.size(); ++i) {
    fill_input(*cur_input_tensor, input_shapes.at(i).getFeatureLen());
    cur_input_tensor++;
  }

  float **cur_label_tensor = label;
  for (unsigned int i = 0; i < output_shapes.size(); ++i) {
    fill_label(*label, output_shapes.at(i).batch(),
               output_shapes.at(i).getFeatureLen());
    cur_label_tensor++;
  }
}

DirDataLoader::DirDataLoader(const char *directory_, float split_ratio,
                             int label_len_, int c, int w, int h,
                             bool is_train_) :
  label_len(label_len_), width(w), height(h), is_train(is_train_),
  channel(c), total_image_num(0) {

  dir_path.assign(directory_);
  LOGI("Dir : %s", dir_path.c_str());

  std::vector<std::string> class_names;

  for (const auto &entry : std::filesystem::directory_iterator(dir_path))
    class_names.push_back(entry.path());

  unsigned int num_class = class_names.size();

  unsigned int id = 0;
  unsigned int num_data_total = 0;
  unsigned int num_data = 1;

  for (auto &c_name : class_names) {
    num_data = 0;
    std::filesystem::directory_iterator itr(c_name);
    while (itr != std::filesystem::end(itr)) {
      const std::filesystem::directory_entry &entry = *itr;
      std::string p = std::filesystem::absolute(entry.path()).string();
      if (p.compare(".") && p.compare("..")) {
        num_data++;
        data_list.push_back(std::make_pair(id, p));
      }
      itr++;
    }

    if (is_train) {
      unsigned int remove = num_data * split_ratio;
      for (unsigned int i = 0; i < remove; i++) {
        data_list.pop_back();
        num_data = num_data - 1;
      }
    } else {
      unsigned int remove = (num_data - (int)(num_data * split_ratio));
      for (unsigned int i = 0; i < remove; i++) {
        data_list.erase(data_list.end() - num_data);
        num_data = num_data - 1;
      }
    }

    id++;
    num_data_total += num_data;
  }

  idxes = std::vector<unsigned int>(data_list.size());

  std::iota(idxes.begin(), idxes.end(), 0);
  std::shuffle(idxes.begin(), idxes.end(), rng);

  data_size = data_list.size();

  if (label_len != num_class)
    throw ::std::invalid_argument(
      "label length is not equal to data directory");

  count = 0;
}

void read_image(const std::string path, float *input, unsigned int &width,
                unsigned int &height) {

  std::unique_ptr<Image> image = nullptr;

  int fd = open(path.c_str(), O_RDONLY);
  image = ImageFactory::FromFd(fd);

  int ia, ib, ic, id, x, y, index;

  float x_ratio = ((float)(image->width() - 1) / width);
  float y_ratio = ((float)(image->height() - 1) / height);
  float x_diff, y_diff, blue, red, green;
  int offset = 0;

  int fs = width * height;
  int fs2 = fs * 2;

  int r = 0;
  int g = 1;
  int b = 2;

  int ii = 0;

  if (height != image->height() || width != image->width()) {
    for (int i = 0; i < height; i++) {
      int hh = (height - i - 1) * width;
      for (int j = 0; j < width; j++) {
        x = (int)(x_ratio * j);
        y = (int)(y_ratio * i);
        x_diff = (x_ratio * j) - x;
        y_diff = (y_ratio * i) - y;

        blue = image->get_pixel(x, y, b) * (1 - x_diff) * (1 - y_diff) +
               image->get_pixel(x + 1, y, b) * (x_diff) * (1 - y_diff) +
               image->get_pixel(x, y + 1, b) * (y_diff) * (1 - x_diff) +
               image->get_pixel(x + 1, y + 1, b) * (x_diff * y_diff);

        green = image->get_pixel(x, y, g) * (1 - x_diff) * (1 - y_diff) +
                image->get_pixel(x + 1, y, g) * (x_diff) * (1 - y_diff) +
                image->get_pixel(x, y + 1, g) * (y_diff) * (1 - x_diff) +
                image->get_pixel(x + 1, y + 1, g) * (x_diff * y_diff);

        red = image->get_pixel(x, y, r) * (1 - x_diff) * (1 - y_diff) +
              image->get_pixel(x + 1, y, r) * (x_diff) * (1 - y_diff) +
              image->get_pixel(x, y + 1, r) * (y_diff) * (1 - x_diff) +
              image->get_pixel(x + 1, y + 1, r) * (x_diff * y_diff);

        input[hh + j] = red / 255.0;

        input[fs + hh + j] = green / 255.0;

        input[fs2 + hh + j] = blue / 255.0;
      }
    }
  } else {

    for (unsigned int i = 0; i < height; i++) {
      int hh = (height - i - 1) * width;
      for (unsigned int j = 0; j < width; j++) {
        unsigned int c = j * 4;
        input[hh + j] = image->get_pixel(j, i, 0) / 255.0;       // R
        input[fs + hh + j] = image->get_pixel(j, i, 1) / 255.0;  // G
        input[fs2 + hh + j] = image->get_pixel(j, i, 2) / 255.0; // B
      }
    }
  }

  // #ifdef DEBUG
  // std::string n = path+"_resized.bmp";
  // SaveBitmap24bitColor(n.c_str(), input, width, height);
  // #endif

  close(fd);
}

void DirDataLoader::next(float **input, float **label, bool *last) {
  auto fill_one_sample = [this](float *input_, float *label_, int index) {
    std::string file_name = data_list[index].second;

    setCurFileName(file_name);

    read_image(file_name, input_, width, height);
    LOGI("%d %d: %s", width, height, file_name.c_str());

    unsigned int c_id = data_list[index].first;

    std::memset(label_, 0.0, label_len * sizeof(float));

    label_[c_id] = 1.0;
  };

  fill_one_sample(*input, *label, idxes[count]);

  count++;

  if (count < data_size) {
    *last = false;
  } else {
    *last = true;
    count = 0;
    std::shuffle(idxes.begin(), idxes.end(), rng);
  }
}

std::string DirDataLoader::getCurFileName() { return cur_file_name; }

} // namespace nntrainer::resnet
