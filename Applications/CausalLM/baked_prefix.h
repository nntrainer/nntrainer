// SPDX-License-Identifier: Apache-2.0
/** @file baked_prefix.h
 * @brief Persistent immutable prefill assets, separate from ephemeral LRU
 * files.
 */
#ifndef NNTR_BAKED_PREFIX_H
#define NNTR_BAKED_PREFIX_H
#include <json.hpp>
#include <map>
#include <memory>
#include <prefix_flash_store.h>
#include <string>
#include <vector>
namespace nntrainer {
struct HtpPrefixCalibration;
}
namespace causallm {
class BakedPrefix {
public:
  using json = nlohmann::json;
  struct Frame {
    unsigned from = 0, plane = 0, chunk = 0;
    std::shared_ptr<const PrefixFlashStore::Snapshot> layout;
    std::vector<std::shared_ptr<const nntrainer::HtpPrefixCalibration>>
      calibration;
    std::string payload;
  };
  struct Asset {
    std::vector<unsigned> tokens;
    unsigned system_tokens = 0;
    std::map<unsigned, std::shared_ptr<const Frame>> frames;
  };
  /** Content fingerprint; no filename/mtime-only model identity. */
  static json fingerprintFile(const std::string &path);
  /** All metadata and raw float bits are checksum protected and bounded. */
  static Asset load(const std::string &directory, const json &identity,
                    unsigned max_tokens, unsigned layers);
  class Writer {
  public:
    Writer(const std::string &directory, const json &identity,
           std::vector<unsigned> tokens, unsigned system_tokens);
    ~Writer();
    Writer(const Writer &) = delete;
    Writer &operator=(const Writer &) = delete;
    std::shared_ptr<const Frame> add(
      unsigned from, unsigned plane, unsigned chunk,
      const PrefixFlashStore::Snapshot &snapshot,
      const std::vector<std::shared_ptr<const nntrainer::HtpPrefixCalibration>>
        &calibration);
    json finish();
    size_t limit() const { return asset_.tokens.size(); }

  private:
    std::string target_, staging_;
    json identity_;
    Asset asset_;
    bool committed_ = false;
  };
};
} // namespace causallm
#endif
