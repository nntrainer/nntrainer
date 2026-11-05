// SPDX-License-Identifier: Apache-2.0
/** @file baked_prefix_check.cpp
 * @brief Persistent asset roundtrip, bit-exact calibration and rejection paths.
 */
#include <baked_prefix.h>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <htp_backend/htp_attention_state.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
using Bake = causallm::BakedPrefix;
using json = nlohmann::json;
static void require(bool ok) {
  if (!ok)
    throw std::runtime_error("Baked prefix check failed");
}
template <class F> static void rejects(F fn) {
  bool bad = false;
  try {
    fn();
  } catch (const std::exception &) {
    bad = true;
  }
  require(bad);
}
static std::string digest(const std::string &s) {
  uint64_t h = 14695981039346656037ULL;
  for (unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << h;
  return out.str();
}
int main() {
  const auto root =
    std::filesystem::temp_directory_path() /
    ("nntr-baked-check-" +
     std::to_string(
       std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(root);
  try {
    const auto weights = root / "weights.bin";
    {
      std::ofstream f(weights);
      f << "0123456789abcdef";
    }
    const auto stamp = Bake::fingerprintFile(weights.string());
    const json identity = {{"weights", stamp}, {"runtime", "original"}};
    auto page = std::make_shared<causallm::PrefixFlashStore::Snapshot::Page>();
    page->key.resize(64 * 2 * 2);
    page->value.resize(64 * 2 * 2);
    for (size_t i = 0; i < page->key.size(); ++i) {
      page->key[i] = i % 256;
      page->value[i] = (i * 3) % 256;
    }
    causallm::PrefixFlashStore::Snapshot snapshot;
    snapshot.position = 64;
    snapshot.element_bytes = 2;
    snapshot.layers.resize(1);
    snapshot.layers[0].width = 2;
    snapshot.layers[0].pages = {page};
    auto cal = std::make_shared<nntrainer::HtpPrefixCalibration>();
    cal->calibrated = true;
    cal->scale_k = {1.25f};
    cal->scale_v = {0.25f, 0.75f};
    cal->q_enc = {1e-12f, -123.0f};
    cal->out_enc = {0.5f, 32768.0f};
    cal->kmax = {3.25f};
    cal->vmax = {9.5f, 7.0f};
    const auto asset_dir = root / "asset";
    {
      Bake::Writer writer(asset_dir.string(), identity,
                          std::vector<unsigned>(64, 7), 70);
      writer.add(0, 64, 64, snapshot, {cal});
      const auto info = writer.finish();
      require(info["baked_tokens"] == 64 &&
              info["remaining_system_tokens"] == 6);
    }
    auto asset = Bake::load(asset_dir.string(), identity, 128, 1);
    require(asset.tokens.size() == 64 && asset.frames.size() == 1);
    auto restored = causallm::PrefixFlashStore::loadSnapshot(
      asset.frames.at(64)->payload, *asset.frames.at(64)->layout);
    require(restored->layers[0].pages[0]->key == page->key &&
            restored->layers[0].pages[0]->value == page->value);
    const auto &saved = *asset.frames.at(64)->calibration[0];
    require(saved.scale_k == cal->scale_k &&
            saved.q_enc.size() == cal->q_enc.size() &&
            std::memcmp(saved.q_enc.data(), cal->q_enc.data(),
                        cal->q_enc.size() * sizeof(float)) == 0);
    rejects([&] {
      Bake::Writer writer(asset_dir.string(), identity,
                          std::vector<unsigned>(64, 7), 70);
    });
    rejects([&] {
      Bake::load(asset_dir.string(), json{{"weights", "different"}}, 128, 1);
    });
    const auto mtime = std::filesystem::last_write_time(weights);
    {
      std::fstream f(weights, std::ios::in | std::ios::out);
      f.seekp(8);
      f.put('X');
    }
    std::filesystem::last_write_time(weights, mtime);
    require(Bake::fingerprintFile(weights.string()) != stamp);
    // Metadata damage is rejected before its scale values can reach inference.
    const auto manifest = asset_dir / "manifest.json";
    json doc;
    {
      std::ifstream f(manifest);
      f >> doc;
    }
    const auto original = doc;
    auto write = [&] {
      std::ofstream f(manifest);
      f << doc.dump();
    };
    doc["body"]["system_tokens"] = 75;
    write();
    rejects([&] { Bake::load(asset_dir.string(), identity, 128, 1); });
    doc = original;
    doc["body"]["frames"][0]["calibration"][0]["scale_k"][0] =
      uint32_t(0x7fc00000);
    doc["checksum"] = digest(doc["body"].dump());
    write();
    rejects([&] { Bake::load(asset_dir.string(), identity, 128, 1); });
    doc = original;
    doc["body"]["frames"][0]["payload"] = "../weights.bin";
    doc["checksum"] = digest(doc["body"].dump());
    write();
    rejects([&] { Bake::load(asset_dir.string(), identity, 128, 1); });
    doc = original;
    write();
    // Metadata can load lazily, but corrupted same-size payload must not
    // restore.
    {
      std::fstream f(asset.frames.at(64)->payload,
                     std::ios::in | std::ios::out | std::ios::binary);
      f.seekp(48);
      f.put('\x5a');
    }
    auto corrupt = Bake::load(asset_dir.string(), identity, 128, 1);
    rejects([&] {
      causallm::PrefixFlashStore::loadSnapshot(corrupt.frames.at(64)->payload,
                                               *corrupt.frames.at(64)->layout);
    });
    std::filesystem::resize_file(asset.frames.at(64)->payload, 32);
    rejects([&] { Bake::load(asset_dir.string(), identity, 128, 1); });
    const auto aborted = root / "aborted";
    {
      Bake::Writer writer(aborted.string(), identity,
                          std::vector<unsigned>(64, 7), 64);
    }
    require(!std::filesystem::exists(aborted));
    require(
      std::filesystem::exists(manifest)); // asset ownership survives readers.
    std::filesystem::remove_all(root);
    std::cout << "Persistent prefix: exact calibration, content identity, "
                 "checksum, lazy corruption, rollback PASS\n";
  } catch (...) {
    std::filesystem::remove_all(root);
    throw;
  }
}
