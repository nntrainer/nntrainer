#include "qnn_properties.h"

namespace nntrainer {

template <>
std::string str_converter<props::quant_param_prop_tag,
                          std::pair<std::string, std::pair<float, int>>>::
  to_string(const std::pair<std::string, std::pair<float, int>> &quant_param) {
  std::stringstream ss;
  ss << quant_param.first << ':' << quant_param.second.first << ':'
     << quant_param.second.second;
  return ss.str();
}

template <>
std::pair<std::string, std::pair<float, int>>
str_converter<props::quant_param_prop_tag,
              std::pair<std::string, std::pair<float, int>>>::
  from_string(const std::string &value) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream iss(value);

  while (std::getline(iss, token, ':')) {
    tokens.push_back(token);
  }

  NNTR_THROW_IF(tokens.size() != 3, std::invalid_argument)
    << "String is wrong format, got: " << value;

  return std::make_pair(
    tokens[0], std::make_pair(std::stof(tokens[1]), std::stoi(tokens[2])));
}
} // namespace nntrainer