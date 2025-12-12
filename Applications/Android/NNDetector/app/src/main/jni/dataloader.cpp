// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright 2023. Umberto Michieli <u.michieli@samsung.com>.
 * Copyright 2023. Kirill Paramonov <k.paramonov@samsung.com>.
 * Copyright 2023. Mete Ozay <m.ozay@samsung.com>.
 * Copyright 2023. JIJOONG MOON <jijoong.moon@samsung.com>.
 *
 * @file   dataloader.cpp
 * @date   24 Oct 2023
 * @brief  data handler for image
 * @author Umberto Michieli(u.michieli@samsung.com)
 * @author Kirill Paramonov(k.paramonov@samsung.com)
 * @author Mete Ozay(m.ozay@samsung.com)
 * @author JIJOONG MOON(jijoong.moon@samsung.com)
 * @bug    No known bugs
 */
#include "dataloader.h"

#include <android/log.h>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <filesystem>
#include <image.h>
#include <map>
#include <nntrainer_error.h>
#include <random>
#include <set>
#include <unistd.h>
#include <vector>

namespace nntrainer {
namespace simpleshot {

/**
 * @brief resize input with width, height, dimension
 *
 * @param input input image data
 * @param resized_input outpot image data resized
 * @param input_w width of input
 * @param input_h height of input
 * @param new_dim dimension
 */
void resize_input(float *input, float *resized_input, unsigned int &input_w,
                  unsigned int &input_h, unsigned int &new_dim) {

  int old_j, old_i;

  float j_ratio = ((float)(input_w - 1) / new_dim);
  float i_ratio = ((float)(input_h - 1) / new_dim);
  float j_diff, i_diff, blue, red, green;

  int r = 0;
  int g = 1;
  int b = 2;

  auto get_pixel = [input, input_w, input_h](int i, int j, int ch) {
    return input[3 * (i * input_w + j) + ch];
  };

  for (int i = 0; i < new_dim; i++) {
    for (int j = 0; j < new_dim; j++) {
      old_j = (int)(j_ratio * j);
      old_i = (int)(i_ratio * i);
      j_diff = (j_ratio * j) - old_j;
      i_diff = (i_ratio * i) - old_i;

      blue = get_pixel(old_i, old_j, b) * (1 - j_diff) * (1 - i_diff) +
             get_pixel(old_i, old_j + 1, b) * (j_diff) * (1 - i_diff) +
             get_pixel(old_i + 1, old_j, b) * (i_diff) * (1 - j_diff) +
             get_pixel(old_i + 1, old_j + 1, b) * (j_diff * i_diff);

      green = get_pixel(old_i, old_j, g) * (1 - j_diff) * (1 - i_diff) +
              get_pixel(old_i, old_j + 1, g) * (j_diff) * (1 - i_diff) +
              get_pixel(old_i + 1, old_j, g) * (i_diff) * (1 - j_diff) +
              get_pixel(old_i + 1, old_j + 1, g) * (j_diff * i_diff);

      red = get_pixel(old_i, old_j, r) * (1 - j_diff) * (1 - i_diff) +
            get_pixel(old_i, old_j + 1, r) * (j_diff) * (1 - i_diff) +
            get_pixel(old_i + 1, old_j, r) * (i_diff) * (1 - j_diff) +
            get_pixel(old_i + 1, old_j + 1, r) * (j_diff * i_diff);

      int new_pix_ind = 3 * (i * new_dim + j);
      resized_input[new_pix_ind] = red;
      resized_input[new_pix_ind + 1] = green;
      resized_input[new_pix_ind + 2] = blue;
    }
  }
}

/**
 * @brief normalize input with mean and standard deviation
 *
 * @param input input image data
 * @param input_dim dimension of input
 * @param ch_mean mean
 * @param ch_std standard deviation
 */
void normalize_input(float *input, unsigned int &input_dim,
                     std::vector<float> ch_mean, std::vector<float> ch_std) {

  int fs = input_dim * input_dim;
  for (unsigned int i = 0; i < input_dim; i++) {
    for (unsigned int j = 0; j < input_dim; j++) {
      int pix_ind = 3 * (i * input_dim + j);
      input[pix_ind] = (input[pix_ind] - ch_mean[0]) / ch_std[0];         // R
      input[pix_ind + 1] = (input[pix_ind + 1] - ch_mean[1]) / ch_std[1]; // G
      input[pix_ind + 2] = (input[pix_ind + 2] - ch_mean[2]) / ch_std[2]; // B
    }
  }
}

/**
 * @brief crop the image data
 *
 * @param input input image data
 * @param cropped_input output of cropped data
 * @param xyxy coordinate to be cropped
 * @param old_dim dimension of input
 */
void crop_input(float *input, float *cropped_input, std::vector<int> xyxy,
                unsigned int &old_dim) {
  int x1 = std::max(xyxy[0], (int)(0));
  int y1 = std::max(xyxy[1], (int)(0));
  int x2 = std::min(xyxy[2], (int)(old_dim));
  int y2 = std::min(xyxy[3], (int)(old_dim));

  unsigned int crop_width = x2 - x1;
  unsigned int crop_height = y2 - y1;
  int crop_fs = crop_width * crop_height;
  int old_fs = old_dim * old_dim;

  for (int i = y1; i < y2; i++) {
    for (int j = x1; j < x2; j++) {
      int old_pix_ind = 3 * (i * old_dim + j);
      int new_pix_ind = 3 * ((i - y1) * crop_width + (j - x1));
      cropped_input[new_pix_ind] = input[old_pix_ind];         // R
      cropped_input[new_pix_ind + 1] = input[old_pix_ind + 1]; // G
      cropped_input[new_pix_ind + 2] = input[old_pix_ind + 2]; // B
    }
  }
}

/**
 * @brief read image from path
 *
 * @param path path the image exist
 * @param input output of image
 * @param new_img_dim dimension of image
 */
void read_image_from_path(const std::string path, float *input,
                          unsigned int &new_img_dim) {

  std::unique_ptr<Image> image = nullptr;

  int fd = open(path.c_str(), O_RDONLY);
  image = ImageFactory::FromFd(fd);
  unsigned int cur_height = image->height();
  unsigned int cur_width = image->width();

  float *tmp_input = new float[3 * cur_height * cur_width];
  for (unsigned int i = 0; i < cur_height; i++) {
    for (unsigned int j = 0; j < cur_width; j++) {
      int pix_ind = 3 * (i * cur_width + j);
      tmp_input[pix_ind] = image->get_pixel(j, i, 0) / 255.0;     // R
      tmp_input[pix_ind + 1] = image->get_pixel(j, i, 1) / 255.0; // G
      tmp_input[pix_ind + 2] = image->get_pixel(j, i, 2) / 255.0; // B
    }
  }

  resize_input(tmp_input, input, cur_width, cur_height, new_img_dim);

  delete[] tmp_input;
  close(fd);
}

/**
 * @brief get index of max value in vector
 *
 * @param vec vector of values
 * @param num_class number of class
 * @return unsigned int index of maximum value
 */
unsigned int argmax(float *vec, unsigned int num_class) {
  unsigned int ret = 0;
  float val = vec[0];
  for (unsigned int i = 1; i < num_class; i++) {
    if (val < vec[i]) {
      val = vec[i];
      ret = i;
    }
  }
  return ret;
}

/**
 * @brief get index of max value in vector
 *
 * @param vec vector of values
 * @param num_class number of class
 * @param candidate_inds candicated index of vector
 * @return unsigned int index of maximum value
 */
unsigned int argmax(float *vec, unsigned int num_class,
                    std::set<unsigned int> candidate_inds) {
  unsigned int ret;
  float val;
  bool set_val = false;
  for (const auto &i : candidate_inds) {
    if (!set_val || val < vec[i]) {
      val = vec[i];
      ret = i;
      set_val = true;
    }
  }
  return ret;
}

/**
 * @brief detecting object in input image
 *
 * @param input input image
 * @param det_model model of detection
 * @param input_img_dim dimension of input image
 * @param anchor_num anchor number
 * @param labels label vector
 * @param score_thr score threshold
 * @param iou_thr threshold of IoU
 * @param max_bb_num maximum number of bounding box
 * @return std::vector vector of bounding box
 */
std::vector<boundingBoxInfo>
detect_objects(float *input, ml::train::Model *det_model,
               unsigned int &input_img_dim, unsigned int &anchor_num,
               std::vector<unsigned int> &labels, float &score_thr,
               float &iou_thr, int &max_bb_num) {
  std::vector<boundingBoxInfo> bounding_boxes;

  std::vector<float *> in;
  std::vector<float *> result;
  std::vector<float *> label;
  in.push_back(input);
  result = det_model->inference(1, in, label);

  auto get_xyxy = [result, anchor_num,
                   input_img_dim](const unsigned int &anchor_ind_) {
    std::vector<float> xywh(4);
    for (unsigned int i = 0; i < 4; i++)
      xywh[i] = result[0][i * anchor_num + anchor_ind_];
    std::vector<int> xyxy(4);
    xyxy[0] = std::max((int)(xywh[0] - xywh[2] * 0.5), (int)(0));
    xyxy[1] = std::max((int)(xywh[1] - xywh[3] * 0.5), (int)(0));
    xyxy[2] = std::min((int)(xywh[0] + xywh[2] * 0.5), (int)(input_img_dim));
    xyxy[3] = std::min((int)(xywh[1] + xywh[3] * 0.5), (int)(input_img_dim));
    return xyxy;
  };

  auto bb_iou = [](std::vector<int> xyxy1, std::vector<int> xyxy2) {
    auto bb_area = [](std::vector<int> xyxy_) {
      return (xyxy_[2] - xyxy_[0]) * (xyxy_[3] - xyxy_[1]);
    };
    int inter_x1 = std::max(xyxy1[0], xyxy2[0]);
    int inter_y1 = std::max(xyxy1[1], xyxy2[1]);
    int inter_x2 = std::min(xyxy1[2], xyxy2[2]);
    int inter_y2 = std::min(xyxy1[3], xyxy2[3]);
    if (inter_x1 >= inter_x2 || inter_y1 >= inter_y2)
      return (float)0;
    int inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
    int union_area = bb_area(xyxy1) + bb_area(xyxy2) - inter_area;
    return (float)(inter_area) / (float)(union_area);
  };

  int bb_count = 0;
  for (const auto &label_ind : labels) {
    std::vector<std::vector<int>> bbs_for_label;
    float *label_scores = result[0] + anchor_num * (label_ind + 4);

    std::set<unsigned int> candidate_inds;
    for (unsigned int i = 0; i < anchor_num; i++) {
      if (label_scores[i] >= score_thr) {
        candidate_inds.insert(i);
      }
    }
    while (!candidate_inds.empty() && bb_count < max_bb_num) {
      unsigned int best_anchor_ind =
        argmax(label_scores, anchor_num, candidate_inds);
      std::vector<int> xyxy = get_xyxy(best_anchor_ind);

      boundingBoxInfo ibb = {
        0,
      };
      ibb.detected_macro_label = label_ind;
      ibb.bounding_box = xyxy;
      ibb.detection_score = label_scores[best_anchor_ind];
      bounding_boxes.push_back(ibb);

      bb_count++;
      if (bb_count >= max_bb_num) {
        break;
      }

      // Update mask for non-maximum supression
      bool non_zero_mask = false;
      std::vector<unsigned int> inds_to_erase;
      for (const unsigned int &cand_i : candidate_inds) {
        std::vector<int> cand_xyxy = get_xyxy(cand_i);
        float iou = bb_iou(xyxy, cand_xyxy);

        if (iou >= iou_thr) {
          inds_to_erase.push_back(cand_i);
        }
      }
      for (const unsigned int &cand_i : inds_to_erase) {
        candidate_inds.erase(cand_i);
      }
    }
  }

  return bounding_boxes;
}

/**
 * @brief constructor of DirDataLoader
 *
 * @param directory_ path of directory have images
 * @param label_len_ length of label
 * @param new_img_dim_ dimension of input
 * @param ch_mean_ vector of mean
 * @param ch_std_ vector of std
 */
DirDataLoader::DirDataLoader(const char *directory_, int label_len_,
                             int new_img_dim_, std::vector<float> ch_mean_,
                             std::vector<float> ch_std_) :
  det_input_img_dim(0),
  curr_bb_info(NULL),
  label_len(label_len_),
  new_img_dim(new_img_dim_),
  ch_mean(ch_mean_),
  ch_std(ch_std_) {

  dir_path.assign(directory_);
  ANDROID_LOG_I("Dir : %s", dir_path.c_str());

  for (const auto &entry : std::filesystem::directory_iterator(dir_path))
    class_names.push_back(entry.path());

  unsigned int num_class = class_names.size();
  unsigned int id = 0;

  for (auto c_name : class_names) {
    std::filesystem::directory_iterator itr(c_name);
    while (itr != std::filesystem::end(itr)) {
      const std::filesystem::directory_entry &entry = *itr;
      std::string p = std::filesystem::absolute(entry.path()).string();
      if (p.compare(".") && p.compare("..")) {
        data_list.push_back(std::make_pair(id, p));
      }
      itr++;
    }
    id++;
  }

  if (label_len > 0 && label_len < num_class) {
    ANDROID_LOG_D("Passed label length: %d", label_len);
    ANDROID_LOG_D("Number of classes: %d", num_class);
    throw ::std::invalid_argument("too many classes");
  }

  count = 0;
  detector_set = false;
  data_size = data_list.size();
}

/**
 * @brief run object detection
 *
 * @param det_model object detection model
 * @param det_input_img_dim_ dimension of input
 * @param det_output_dim dimension of output
 * @param det_anchor_num number of anchor
 * @param det_labels vector of detected labels
 * @param det_score_thr threshold of score
 * @param det_iou_thr threshold of IoU
 * @param det_max_bb_num maxium number of detection bounding box
 */
void DirDataLoader::runDetector(ml::train::Model *det_model,
                                unsigned int &det_input_img_dim_,
                                unsigned int &det_output_dim,
                                unsigned int &det_anchor_num,
                                std::vector<unsigned int> &det_labels,
                                float &det_score_thr, float &det_iou_thr,
                                int &det_max_bb_num) {

  detector_set = true;
  det_input_img_dim = det_input_img_dim_;

  for (const auto &id_file_pair : data_list) {
    std::string file_name = id_file_pair.second;
    float *tmp_input =
      new float[det_input_img_dim * det_input_img_dim * 3]; // HWC
    read_image_from_path(file_name, tmp_input, det_input_img_dim);

    std::vector<boundingBoxInfo> img_bbs =
      detect_objects(tmp_input, det_model, det_input_img_dim, det_anchor_num,
                     det_labels, det_score_thr, det_iou_thr, det_max_bb_num);

    for (auto const &img_bb : img_bbs) {
      boundingBoxInfo ibb;
      ibb.micro_label = id_file_pair.first;
      ibb.file_name = file_name;
      ibb.detected_macro_label = img_bb.detected_macro_label;
      ibb.bounding_box = img_bb.bounding_box;
      ibb.detection_score = img_bb.detection_score;
      bounding_box_list.push_back(ibb);
    }

    delete[] tmp_input;
  }
  data_size = bounding_box_list.size();
}

/**
 * @brief fill next sample
 *
 * @param input pointer of input list
 * @param label pointer of label list
 * @param last last of input
 */
void DirDataLoader::next(float **input, float **label, bool *last) {

  auto fill_one_sample = [this](float *input_, float *label_, int index) {
    if (detector_set) {

      boundingBoxInfo bb_info = bounding_box_list[index];
      curr_bb_info = bb_info;
      std::vector<int> xyxy = bb_info.bounding_box;

      std::string xyxy_string = std::to_string(xyxy[0]);
      for (unsigned int i = 1; i < 4; i++)
        xyxy_string += " " + std::to_string(xyxy[i]);

      std::string file_name = bb_info.file_name;
      cur_file_name = file_name;

      float *tmp_input = new float[det_input_img_dim * det_input_img_dim * 3];
      read_image_from_path(file_name, tmp_input, det_input_img_dim);

      unsigned int crop_width = xyxy[2] - xyxy[0];
      unsigned int crop_height = xyxy[3] - xyxy[1];
      float *cropped_input = new float[crop_height * crop_width * 3]; // HWC
      crop_input(tmp_input, cropped_input, xyxy, det_input_img_dim);

      resize_input(cropped_input, input_, crop_width, crop_height, new_img_dim);
      normalize_input(input_, new_img_dim, ch_mean, ch_std);

      std::memset(label_, 0.0, label_len * sizeof(float));
      label_[bb_info.micro_label] = 1.0;

      delete[] cropped_input;
      delete[] tmp_input;

    } else {

      std::string file_name = data_list[index].second;

      cur_file_name = file_name;
      read_image_from_path(file_name, input_, new_img_dim);
      normalize_input(input_, new_img_dim, ch_mean, ch_std);

      unsigned int c_id = data_list[index].first;
      std::memset(label_, 0.0, label_len * sizeof(float));
      label_[c_id] = 1.0;
    }
  };

  fill_one_sample(*input, *label, count);
  count++;

  if (count < data_size) {
    *last = false;
  } else {
    *last = true;
    count = 0;
  }
}

/**
 * @brief returning current file name
 *
 * @return std::string file name
 */
std::string DirDataLoader::getCurFileName() { return cur_file_name; }

/**
 * @brief returning class name
 *
 * @return std::vector vector of class name
 */
std::vector<std::string> DirDataLoader::getClassNames() { return class_names; }

/**
 * @brief returning current boundingbox information
 *
 * @return boundingBoxInfo bounding box information
 */
boundingBoxInfo DirDataLoader::getCurrBBInfo() { return curr_bb_info; }

/**
 * @brief returning boundingbox list
 *
 * @return std::vector vector of bounding box
 */
std::vector<boundingBoxInfo> DirDataLoader::getBBList() {
  return bounding_box_list;
}

} // namespace simpleshot
} // namespace nntrainer
