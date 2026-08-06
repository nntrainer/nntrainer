// SPDX-License-Identifier: Apache-2.0
/**
 * @file   graph_parser.cpp
 * @brief  Parses QNN graph-info JSON into tensor/graph metadata.
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#include "graph_parser.h"
#include <algorithm>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "nntrainer_error.h"

GraphParser::GraphParser() {}

GraphParser::~GraphParser() {}

int GraphParser::find_tensor_index(const TensorInfoList &tensor_infos,
                                   const std::string &tensor_name) {
  int index = 0;
  for (const auto &info : tensor_infos) {
    if (info.name == tensor_name) {
      return index;
    }
    index++;
  }
  std::cerr << "Error: Failed to find tensor index for: " << tensor_name
            << std::endl;
  throw std::invalid_argument("Failed to find tensor index for: " +
                              tensor_name);
}

const TensorInfo &
GraphParser::get_tensor_info_or_throw(const TensorInfoList &tensor_infos,
                                      const std::string &tensor_name) {
  return tensor_infos[find_tensor_index(tensor_infos, tensor_name)];
}

int GraphParser::get_named_tensor_elements_or_throw(
  const TensorInfoList &tensor_infos, const std::string &tensor_name) {
  return get_tensor_count(get_tensor_info_or_throw(tensor_infos, tensor_name));
}

std::map<std::string, GraphInfo>
GraphParser::parseJsonFile(const std::string &file_path) {
  std::map<std::string, GraphInfo> graphs_info;
  try {
    std::ifstream file(file_path);
    NNTR_THROW_IF(!file.is_open(), std::invalid_argument)
      << "Failed to open file: " << file_path;

    auto json_data_ = json::parse(file);
    file.close();

    // Parse graphs info
    auto &graphs = json_data_["info"]["graphs"];
    for (const auto &graph_object : graphs) {
      GraphInfo graph_info = extractGraphInfo(graph_object);
      graphs_info[graph_info.graph_name] = graph_info;
    }

    return graphs_info;
  } catch (const std::exception &e) {
    std::cerr << "Error parsing JSON file: " << e.what() << std::endl;
    return graphs_info;
  }
}

GraphInfo GraphParser::extractGraphInfo(const json &graph_object) {
  GraphInfo graph_info;

  auto graph_info_json = graph_object["info"];
  graph_info.graph_name = graph_info_json["graphName"];

  auto graph_inputs = graph_info_json["graphInputs"];
  auto graph_outputs = graph_info_json["graphOutputs"];

  for (const auto &element : graph_inputs) {
    graph_info.raw_inputs.push_back(extractTensorInfo(element));
  }

  for (const auto &element : graph_outputs) {
    graph_info.raw_outputs.push_back(extractTensorInfo(element));
  }

  return graph_info;
}

TensorInfo GraphParser::extractTensorInfo(const json &tensor_object) {
  TensorInfo tensor_info;

  auto tensor_info_json = tensor_object["info"];

  tensor_info.name = tensor_info_json["name"];
  tensor_info.dimensions =
    tensor_info_json["dimensions"].get<std::vector<int>>();
  tensor_info.data_type = tensor_info_json["dataType"];
  if (tensor_info_json.contains("quantizeParams") &&
      tensor_info_json["quantizeParams"].contains("scaleOffset") &&
      tensor_info_json["quantizeParams"]["scaleOffset"].is_object()) {
    auto scale_offset = tensor_info_json["quantizeParams"]["scaleOffset"];
    tensor_info.scale = scale_offset.value("scale", 0.0f);
    tensor_info.offset = scale_offset.value("offset", 0);
  } else {
    tensor_info.scale = 0.0f;
    tensor_info.offset = 0;
  }

  return tensor_info;
}

int GraphParser::get_tensor_count(const TensorInfo &tensor_info) {
  auto &tensor_shape = tensor_info.dimensions;

  int tensor_size = 1;
  for (const auto &tensor_dim : tensor_shape) {
    tensor_size *= tensor_dim;
  }
  return tensor_size;
}

int GraphParser::get_tensor_bit_width(const TensorInfo &tensor_info) {
  int bit_width;
  if (tensor_info.data_type == "QNN_DATATYPE_UFIXED_POINT_16" ||
      tensor_info.data_type == "QNN_DATATYPE_FLOAT_16") {
    bit_width = 2;
  } else if (tensor_info.data_type == "QNN_DATATYPE_UFIXED_POINT_8") {
    bit_width = 1;
  } else {
    throw std::invalid_argument("qnn tensor dtype is " + tensor_info.data_type);
  }
  return bit_width;
}

int GraphParser::get_tensor_size(const TensorInfo &tensor_info) {
  return get_tensor_bit_width(tensor_info) * get_tensor_count(tensor_info);
}
