#ifndef GRAPH_PARSER_H
#define GRAPH_PARSER_H

#include "../../../nntrainer/Applications/CausalLM/json.hpp"
#include <map>
#include <string>
#include <vector>

using json = nlohmann::json;

struct TensorInfo {
  std::string name;
  std::vector<int> dimensions;
  std::string data_type;
  float scale;
  int offset;
};

using TensorInfoList = std::vector<TensorInfo>;

struct GraphInfo {
  std::string graph_name;
  TensorInfoList raw_inputs;
  TensorInfoList raw_outputs;
};

class GraphParser {
public:
  GraphParser();
  ~GraphParser();

  std::map<std::string, GraphInfo> parseJsonFile(const std::string &file_path);
  static int find_tensor_index(const TensorInfoList &tensor_infos,
                               const std::string &tensor_name);
  static const TensorInfo &
  get_tensor_info_or_throw(const TensorInfoList &tensor_infos,
                           const std::string &tensor_name);
  static int
  get_named_tensor_elements_or_throw(const TensorInfoList &tensor_infos,
                                     const std::string &tensor_name);
  static int get_tensor_count(const TensorInfo &tensor_info);
  static int get_tensor_bit_width(const TensorInfo &tensor_info);
  static int get_tensor_size(const TensorInfo &tensor_info);

private:
  GraphInfo extractGraphInfo(const json &graph_object);
  TensorInfo extractTensorInfo(const json &tensor_object);
};

#endif // GRAPH_PARSER_H
