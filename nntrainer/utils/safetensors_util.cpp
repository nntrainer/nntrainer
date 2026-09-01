// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @brief  safetensors dtype mapping + JSON header build/parse helpers.
 * @file   safetensors_util.cpp
 * @date   18 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "safetensors_util.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace nntrainer::safetensors {

using DT = ml::train::TensorDim::DataType;

bool isQuantized(DT dtype) {
  switch (dtype) {
  case DT::Q4_0:
  case DT::Q8_0:
  case DT::Q4_K:
  case DT::Q6_K:
    return true;
  default:
    return false;
  }
}

const char *dtypeToString(DT dtype) {
  switch (dtype) {
  case DT::FP32:
    return "F32";
  case DT::FP16:
    return "F16";
  case DT::QINT4:
    return "I4";
  case DT::QINT8:
    return "I8";
  case DT::QINT16:
    return "I16";
  case DT::UINT4:
    return "U4";
  case DT::UINT8:
    return "U8";
  case DT::UINT16:
    return "U16";
  case DT::UINT32:
    return "U32";
  // Block-quantized payloads are stored as opaque byte blobs so that standard
  // safetensors parsers can still read them; the native type is recorded
  // separately via the nntr_dtype extension field.
  case DT::Q4_0:
  case DT::Q8_0:
  case DT::Q4_K:
  case DT::Q6_K:
    return "U8";
  default:
    return "F32";
  }
}

const char *nntrDtypeName(DT dtype) {
  switch (dtype) {
  case DT::FP32:
    return "FP32";
  case DT::FP16:
    return "FP16";
  case DT::QINT4:
    return "QINT4";
  case DT::QINT8:
    return "QINT8";
  case DT::QINT16:
    return "QINT16";
  case DT::UINT4:
    return "UINT4";
  case DT::UINT8:
    return "UINT8";
  case DT::UINT16:
    return "UINT16";
  case DT::UINT32:
    return "UINT32";
  case DT::Q4_0:
    return "Q4_0";
  case DT::Q8_0:
    return "Q8_0";
  case DT::Q4_K:
    return "Q4_K";
  case DT::Q6_K:
    return "Q6_K";
  default:
    return "FP32";
  }
}

DT nntrDtypeFromName(const std::string &name) {
  std::string up = name;
  std::transform(up.begin(), up.end(), up.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  if (up == "FP32" || up == "F32")
    return DT::FP32;
  if (up == "FP16" || up == "F16")
    return DT::FP16;
  if (up == "QINT4" || up == "I4")
    return DT::QINT4;
  if (up == "QINT8" || up == "I8")
    return DT::QINT8;
  if (up == "QINT16" || up == "I16")
    return DT::QINT16;
  if (up == "UINT4" || up == "U4")
    return DT::UINT4;
  if (up == "UINT8" || up == "U8")
    return DT::UINT8;
  if (up == "UINT16" || up == "U16")
    return DT::UINT16;
  if (up == "UINT32" || up == "U32")
    return DT::UINT32;
  if (up == "Q4_0")
    return DT::Q4_0;
  if (up == "Q8_0")
    return DT::Q8_0;
  if (up == "Q4_K")
    return DT::Q4_K;
  if (up == "Q6_K")
    return DT::Q6_K;
  throw std::invalid_argument("safetensors: unknown nntr_dtype name: " + name);
}

namespace {

void appendShape(std::ostringstream &out, const std::vector<size_t> &shape) {
  out << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0)
      out << ",";
    out << shape[i];
  }
  out << "]";
}

std::string buildHeaderImpl(const std::vector<TensorEntry> &entries,
                            const std::map<std::string, std::string> &meta) {
  std::ostringstream out;
  out << "{\"__metadata__\":{\"format\":\"nntrainer\"";
  for (const auto &[k, v] : meta) {
    if (k == "format")
      continue;
    out << ",\"" << k << "\":\"" << v << "\"";
  }
  out << "}";
  for (const auto &e : entries) {
    out << ",\"" << e.name << "\":{\"dtype\":\"" << e.dtype << "\",\"shape\":";
    appendShape(out, e.shape);
    if (!e.nntr_dtype.empty())
      out << ",\"nntr_dtype\":\"" << e.nntr_dtype << "\"";
    if (!e.nntr_shape.empty()) {
      out << ",\"nntr_shape\":";
      appendShape(out, e.nntr_shape);
    }
    out << ",\"data_offsets\":[" << e.offset_start << "," << e.offset_end
        << "]}";
  }
  out << "}";
  std::string s = out.str();
  const size_t pad = (8 - (s.size() % 8)) % 8;
  s.append(pad, ' ');
  return s;
}

} // namespace

std::string buildHeader(const std::vector<TensorEntry> &entries) {
  return buildHeaderImpl(entries, {});
}

std::string buildHeader(const std::vector<TensorEntry> &entries,
                        const std::map<std::string, std::string> &metadata) {
  return buildHeaderImpl(entries, metadata);
}

namespace {

/**
 * @brief Lightweight JSON scanner for parsing safetensors headers.
 */
class Scanner {
public:
  explicit Scanner(const std::string &s) : src(s), pos(0) {}

  void skipWs() {
    while (pos < src.size() &&
           std::isspace(static_cast<unsigned char>(src[pos])))
      ++pos;
  }

  bool peek(char c) {
    skipWs();
    return pos < src.size() && src[pos] == c;
  }

  void expect(char c) {
    if (!peek(c))
      throw std::runtime_error(std::string("safetensors: expected '") + c +
                               "'");
    ++pos;
  }

  std::string readString() {
    expect('"');
    std::string out;
    while (pos < src.size() && src[pos] != '"') {
      if (src[pos] == '\\' && pos + 1 < src.size())
        ++pos;
      out += src[pos++];
    }
    expect('"');
    return out;
  }

  size_t readNumber() {
    skipWs();
    size_t v = 0;
    while (pos < src.size() && src[pos] >= '0' && src[pos] <= '9')
      v = v * 10 + (src[pos++] - '0');
    return v;
  }

  std::vector<size_t> readNumberArray() {
    std::vector<size_t> out;
    expect('[');
    for (bool first = true; !peek(']'); first = false) {
      if (!first)
        expect(',');
      out.push_back(readNumber());
    }
    expect(']');
    return out;
  }

  void skipValue() {
    skipWs();
    if (pos >= src.size())
      return;
    char c = src[pos];
    if (c == '"') {
      readString();
      return;
    }
    if (c == '{' || c == '[') {
      const char close = (c == '{') ? '}' : ']';
      int depth = 0;
      while (pos < src.size()) {
        if (src[pos] == '"') {
          readString();
          continue;
        }
        if (src[pos] == c)
          ++depth;
        else if (src[pos] == close && --depth == 0) {
          ++pos;
          return;
        }
        ++pos;
      }
      return;
    }
    while (pos < src.size() && src[pos] != ',' && src[pos] != '}' &&
           src[pos] != ']')
      ++pos;
  }

private:
  const std::string &src;
  size_t pos;
};

} // namespace

std::vector<TensorEntry> parseHeaderEntries(const std::string &json) {
  std::vector<TensorEntry> out;
  Scanner s(json);
  s.expect('{');
  for (bool first = true; !s.peek('}'); first = false) {
    if (!first)
      s.expect(',');
    const std::string key = s.readString();
    s.expect(':');
    if (key == "__metadata__") {
      s.skipValue();
      continue;
    }
    s.expect('{');
    TensorEntry e;
    e.name = key;
    e.offset_start = 0;
    e.offset_end = 0;
    for (bool inner_first = true; !s.peek('}'); inner_first = false) {
      if (!inner_first)
        s.expect(',');
      const std::string field = s.readString();
      s.expect(':');
      if (field == "dtype") {
        e.dtype = s.readString();
      } else if (field == "shape") {
        e.shape = s.readNumberArray();
      } else if (field == "nntr_dtype") {
        e.nntr_dtype = s.readString();
      } else if (field == "nntr_shape") {
        e.nntr_shape = s.readNumberArray();
      } else if (field == "data_offsets") {
        s.expect('[');
        e.offset_start = s.readNumber();
        s.expect(',');
        e.offset_end = s.readNumber();
        s.expect(']');
      } else {
        s.skipValue();
      }
    }
    s.expect('}');
    out.push_back(std::move(e));
  }
  s.expect('}');
  return out;
}

std::unordered_map<std::string, std::pair<size_t, size_t>>
parseHeader(const std::string &json) {
  std::unordered_map<std::string, std::pair<size_t, size_t>> out;
  for (const auto &e : parseHeaderEntries(json))
    out.emplace(e.name,
                std::make_pair(e.offset_start, e.offset_end - e.offset_start));
  return out;
}

std::map<std::string, std::string> parseMetadata(const std::string &json) {
  std::map<std::string, std::string> out;
  Scanner s(json);
  s.expect('{');
  for (bool first = true; !s.peek('}'); first = false) {
    if (!first)
      s.expect(',');
    const std::string key = s.readString();
    s.expect(':');
    if (key != "__metadata__") {
      s.skipValue();
      continue;
    }
    s.expect('{');
    for (bool inner_first = true; !s.peek('}'); inner_first = false) {
      if (!inner_first)
        s.expect(',');
      const std::string mk = s.readString();
      s.expect(':');
      out[mk] = s.readString();
    }
    s.expect('}');
  }
  s.expect('}');
  return out;
}

std::string inspect(const std::string &json) {
  std::ostringstream out;
  const auto meta = parseMetadata(json);
  out << "metadata:\n";
  if (meta.empty()) {
    out << "  (none)\n";
  } else {
    for (const auto &[k, v] : meta)
      out << "  " << k << " = " << v << "\n";
  }

  auto entries = parseHeaderEntries(json);
  std::sort(entries.begin(), entries.end(),
            [](const TensorEntry &a, const TensorEntry &b) {
              return a.offset_start < b.offset_start;
            });

  size_t name_w = 4;
  for (const auto &e : entries)
    name_w = std::max(name_w, e.name.size());

  out << "\ntensors: " << entries.size() << "\n";
  out << "  " << std::left << std::setw(static_cast<int>(name_w)) << "name"
      << "  " << std::setw(8) << "dtype"
      << "  " << std::setw(12) << "bytes"
      << "  shape\n";
  for (const auto &e : entries) {
    const std::string dt = e.nntr_dtype.empty() ? e.dtype : e.nntr_dtype;
    const auto &shape = e.nntr_shape.empty() ? e.shape : e.nntr_shape;
    out << "  " << std::left << std::setw(static_cast<int>(name_w)) << e.name
        << "  " << std::setw(8) << dt << "  " << std::setw(12)
        << (e.offset_end - e.offset_start) << "  [";
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i > 0)
        out << ",";
      out << shape[i];
    }
    out << "]\n";
  }
  return out.str();
}

} // namespace nntrainer::safetensors
