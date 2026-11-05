// SPDX-License-Identifier: Apache-2.0
/** @file baked_prefix.cpp
 * @brief Checked native prefill assets with transactional publication.
 */
#include <baked_prefix.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <htp_backend/htp_attention_state.h>
#include <iomanip>
#include <sstream>
#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif
namespace causallm {
namespace {
using json = BakedPrefix::json;
using Snapshot = PrefixFlashStore::Snapshot;
constexpr size_t metadata_limit = 64u << 20;
uint64_t hashBytes(uint64_t h, const char *p, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    h ^= static_cast<unsigned char>(p[i]);
    h *= 1099511628211ULL;
  }
  return h;
}
std::string hex(uint64_t value) {
  std::ostringstream s;
  s << std::hex << std::setw(16) << std::setfill('0') << value;
  return s.str();
}
std::string digest(const std::string &s) {
  return hex(hashBytes(14695981039346656037ULL, s.data(), s.size()));
}
json bits(const std::vector<float> &v) {
  std::vector<uint32_t> b(v.size());
  if (!v.empty())
    std::memcpy(b.data(), v.data(), v.size() * sizeof(float));
  return b;
}
std::vector<float> floats(const json &j) {
  if (!j.is_array() || j.size() > 65536)
    throw std::invalid_argument("Invalid baked calibration length");
  std::vector<float> v;
  v.reserve(j.size());
  for (const auto &x : j) {
    if (!x.is_number_unsigned() || x.get<uint64_t>() > UINT32_MAX)
      throw std::invalid_argument("Invalid baked float bits");
    const auto u = x.get<uint32_t>();
    float f;
    std::memcpy(&f, &u, sizeof(f));
    if ((u & 0x7f800000u) == 0x7f800000u)
      throw std::invalid_argument("Non-finite baked calibration");
    v.push_back(f);
  }
  return v;
}
json calibrationJson(const nntrainer::HtpPrefixCalibration &c) {
  return {{"calibrated", c.calibrated}, {"scale_k", bits(c.scale_k)},
          {"scale_v", bits(c.scale_v)}, {"q_enc", bits(c.q_enc)},
          {"out_enc", bits(c.out_enc)}, {"kmax", bits(c.kmax)},
          {"vmax", bits(c.vmax)},       {"floor_k", bits(c.floor_k)},
          {"floor_v", bits(c.floor_v)}};
}
json frameJson(const BakedPrefix::Frame &f) {
  json layers = json::array(), cal = json::array();
  for (const auto &l : f.layout->layers)
    layers.push_back({{"base", l.base},
                      {"width", l.width},
                      {"dropped_k", bits(l.dropped_k)},
                      {"dropped_v", bits(l.dropped_v)}});
  for (const auto &c : f.calibration)
    cal.push_back(calibrationJson(*c));
  return {{"position", f.layout->position},
          {"element_bytes", f.layout->element_bytes},
          {"from", f.from},
          {"plane", f.plane},
          {"chunk", f.chunk},
          {"layers", layers},
          {"calibration", cal},
          {"payload", std::filesystem::path(f.payload).filename().string()}};
}
} // namespace
BakedPrefix::json BakedPrefix::fingerprintFile(const std::string &path) {
  const auto before_size = std::filesystem::file_size(path);
  const auto before_time = std::filesystem::last_write_time(path);
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("Cannot fingerprint " + path);
  std::vector<char> buffer(1u << 20);
  uint64_t hash = 14695981039346656037ULL, total = 0;
  while (in) {
    in.read(buffer.data(), buffer.size());
    const auto n = in.gcount();
    hash = hashBytes(hash, buffer.data(), n);
    total += n;
  }
  if (!in.eof() || total != before_size ||
      std::filesystem::file_size(path) != before_size ||
      std::filesystem::last_write_time(path) != before_time)
    throw std::runtime_error("Artifact changed while fingerprinting " + path);
  return {{"bytes", total}, {"fnv1a64", hex(hash)}};
}
BakedPrefix::Writer::Writer(const std::string &directory, const json &identity,
                            std::vector<unsigned> tokens,
                            unsigned system_tokens) :
  target_(directory), identity_(identity) {
  if (directory.empty() || tokens.empty() || system_tokens < tokens.size() ||
      std::filesystem::exists(target_))
    throw std::invalid_argument(
      "Baked output must be a new directory with a nonempty prefix");
  const auto target = std::filesystem::absolute(target_);
  target_ = target.string();
  std::filesystem::create_directories(target.parent_path());
  for (unsigned i = 0; i < 100; ++i) {
    staging_ =
      (target.parent_path() /
       ("." + target.filename().string() + ".baking-" +
        std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()) +
        "-" + std::to_string(i)))
        .string();
    if (std::filesystem::create_directory(staging_))
      break;
    if (i == 99)
      throw std::runtime_error("Cannot create private bake directory");
  }
  asset_.tokens = std::move(tokens);
  asset_.system_tokens = system_tokens;
}
BakedPrefix::Writer::~Writer() {
  if (!committed_) {
    std::error_code ec;
    std::filesystem::remove_all(staging_, ec);
  }
}
std::shared_ptr<const BakedPrefix::Frame> BakedPrefix::Writer::add(
  unsigned from, unsigned plane, unsigned chunk, const Snapshot &s,
  const std::vector<std::shared_ptr<const nntrainer::HtpPrefixCalibration>>
    &cal) {
  if (!chunk || s.position > asset_.tokens.size() || s.position % chunk ||
      from + chunk != s.position || cal.size() != s.layers.size())
    throw std::invalid_argument(
      "Bake requires an original complete native call");
  const auto filename = "prefix-" + std::to_string(s.position) + ".kv";
  PrefixFlashStore::saveSnapshot(
    (std::filesystem::path(staging_) / filename).string(), s);
  auto f = std::make_shared<Frame>();
  f->from = from;
  f->plane = plane;
  f->chunk = chunk;
  f->calibration = cal;
  auto layout = std::make_shared<Snapshot>(s);
  for (auto &l : layout->layers)
    l.pages.clear();
  f->layout = std::move(layout);
  f->payload = (std::filesystem::path(target_) / filename).string();
  asset_.frames.emplace(s.position, f);
  return f;
}
BakedPrefix::json BakedPrefix::Writer::finish() {
  if (asset_.frames.empty() ||
      asset_.frames.rbegin()->first != asset_.tokens.size())
    throw std::runtime_error("Incomplete system prefix bake");
  json frames = json::array();
  for (const auto &f : asset_.frames)
    frames.push_back(frameJson(*f.second));
  json body = {{"format", "nntr-htp-native-system-prefix"},
               {"version", 1},
               {"identity", identity_},
               {"tokens", asset_.tokens},
               {"system_tokens", asset_.system_tokens},
               {"frames", frames}};
  const auto text =
    json({{"body", body}, {"checksum", digest(body.dump())}}).dump();
  if (text.size() > metadata_limit)
    throw std::length_error("Baked metadata exceeds 64 MiB");
  const auto manifest =
    (std::filesystem::path(staging_) / "manifest.json").string();
  std::ofstream out(manifest, std::ios::binary);
  out.write(text.data(), text.size());
  out.close();
  if (!out)
    throw std::runtime_error("Cannot write baked manifest");
#if defined(__linux__)
  const int fd = ::open(manifest.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    throw std::runtime_error("Cannot synchronize manifest");
  const int rc = ::fsync(fd);
  ::close(fd);
  if (rc)
    throw std::runtime_error("Cannot synchronize manifest");
#endif
  if (std::filesystem::exists(target_))
    throw std::runtime_error("Baked destination already exists");
  std::filesystem::rename(staging_, target_);
  committed_ = true;
  size_t bytes = 0;
  for (const auto &file : std::filesystem::directory_iterator(target_))
    bytes += file.file_size();
  return {
    {"directory", target_},
    {"system_tokens", asset_.system_tokens},
    {"baked_tokens", asset_.tokens.size()},
    {"remaining_system_tokens", asset_.system_tokens - asset_.tokens.size()},
    {"boundaries", asset_.frames.size()},
    {"file_bytes", bytes}};
}
BakedPrefix::Asset BakedPrefix::load(const std::string &directory,
                                     const json &identity, unsigned max_tokens,
                                     unsigned layers) {
  const auto root = std::filesystem::absolute(directory);
  const auto path = root / "manifest.json";
  if (std::filesystem::file_size(path) > metadata_limit)
    throw std::length_error("Baked manifest exceeds limit");
  std::ifstream file(path);
  json wrapped;
  file >> wrapped;
  const auto &b = wrapped.at("body");
  if (wrapped.at("checksum") != digest(b.dump()) ||
      b.at("format") != "nntr-htp-native-system-prefix" || b.at("version") != 1)
    throw std::invalid_argument(
      "Invalid baked prefix manifest/checksum/version");
  if (b.at("identity") != identity)
    throw std::invalid_argument(
      "Baked prefix model/tokenizer/runtime identity mismatch");
  Asset a;
  if (!b.at("tokens").is_array() || b.at("tokens").empty() ||
      b.at("tokens").size() > max_tokens)
    throw std::invalid_argument("Invalid baked token count");
  for (const auto &v : b.at("tokens")) {
    if (!v.is_number_unsigned() || v.get<uint64_t>() > UINT32_MAX)
      throw std::invalid_argument("Invalid baked token id");
    a.tokens.push_back(v.get<unsigned>());
  }
  a.system_tokens = b.at("system_tokens").get<unsigned>();
  if (a.system_tokens < a.tokens.size() || a.system_tokens > max_tokens)
    throw std::invalid_argument("Invalid system prefix length");
  if (!b.at("frames").is_array() || b.at("frames").empty() ||
      b.at("frames").size() > 128)
    throw std::invalid_argument("Invalid baked frame count");
  for (const auto &j : b.at("frames")) {
    auto f = std::make_shared<Frame>();
    f->from = j.at("from");
    f->plane = j.at("plane");
    f->chunk = j.at("chunk");
    auto layout = std::make_shared<Snapshot>();
    layout->position = j.at("position");
    layout->element_bytes = j.at("element_bytes");
    if (!f->chunk || f->plane != f->chunk ||
        layout->position > a.tokens.size() || !layout->position ||
        layout->position % f->chunk || f->from != layout->position - f->chunk ||
        layout->element_bytes != 2 || j.at("layers").size() != layers ||
        j.at("calibration").size() != layers)
      throw std::invalid_argument("Invalid baked frame geometry");
    for (const auto &v : j.at("layers")) {
      Snapshot::Layer l;
      l.base = v.at("base");
      l.width = v.at("width");
      l.dropped_k = floats(v.at("dropped_k"));
      l.dropped_v = floats(v.at("dropped_v"));
      if (!l.width || l.width > 65536 || l.base >= layout->position)
        throw std::invalid_argument("Invalid baked layer geometry");
      layout->layers.push_back(std::move(l));
    }
    for (const auto &v : j.at("calibration")) {
      auto c = std::make_shared<nntrainer::HtpPrefixCalibration>();
      c->calibrated = v.at("calibrated");
      c->scale_k = floats(v.at("scale_k"));
      c->scale_v = floats(v.at("scale_v"));
      c->q_enc = floats(v.at("q_enc"));
      c->out_enc = floats(v.at("out_enc"));
      c->kmax = floats(v.at("kmax"));
      c->vmax = floats(v.at("vmax"));
      c->floor_k = floats(v.at("floor_k"));
      c->floor_v = floats(v.at("floor_v"));
      f->calibration.push_back(c);
    }
    const auto name = j.at("payload").get<std::string>();
    if (name != "prefix-" + std::to_string(layout->position) + ".kv")
      throw std::invalid_argument("Invalid baked payload filename");
    f->payload = (root / name).string();
    f->layout = layout;
    size_t bytes = 48;
    for (const auto &l : layout->layers)
      bytes = nntrainer::kv::checkedAdd(
        bytes,
        nntrainer::kv::checkedMultiply(
          nntrainer::kv::checkedMultiply(layout->position - l.base, l.width),
          4));
    if (std::filesystem::is_symlink(f->payload) ||
        !std::filesystem::is_regular_file(f->payload) ||
        std::filesystem::file_size(f->payload) != bytes)
      throw std::invalid_argument("Missing/truncated baked payload");
    if (!a.frames.emplace(layout->position, f).second)
      throw std::invalid_argument("Duplicate baked position");
  }
  if (a.frames.rbegin()->first != a.tokens.size())
    throw std::invalid_argument("Incomplete baked prefix index");
  const unsigned chunk = a.frames.begin()->second->chunk;
  if (a.tokens.size() / chunk != a.frames.size())
    throw std::invalid_argument("Missing native baked boundaries");
  for (const auto &entry : a.frames)
    if (entry.second->chunk != chunk || entry.first % chunk)
      throw std::invalid_argument("Inconsistent native baked boundaries");
  return a;
}
} // namespace causallm
