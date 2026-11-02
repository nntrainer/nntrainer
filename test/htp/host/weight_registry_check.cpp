// SPDX-License-Identifier: Apache-2.0
/** @file weight_registry_check.cpp
 * @brief Check non-owning WH bindings and invalid-registration recovery.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#include <cassert>
#include <cstdio>
#include <htp_weight_registry.h>
#include <limits>

int main() {
  nntrainer::HtpWeightRegistry registry;
  std::vector<uint8_t> record(32 * 64 / 2 + 12 * 64, 17);
  assert(registry.set_fc_wh_source(record.data(), 32, 64, true));
  auto *binding = registry.fcwhFind(record.data(), 32, 64);
  assert(binding && binding->bytes == record.size());
  assert(registry.set_fc_wh_source(record.data(), 32, 64, true));
  assert(registry.fcwhFind(record.data(), 32, 64) == binding);
  bool failed = false;
  try {
    registry.set_fc_wh_source(record.data(), 32, 32, true);
  } catch (const std::invalid_argument &) {
    failed = true;
  }
  assert(failed && registry.fcwhFind(record.data(), 32, 64) == binding);
  uint8_t copy[3];
  registry.fcwhRead(binding, copy, sizeof(copy), 1);
  assert(copy[0] == 17);
  record[1] = 23;
  registry.fcwhRead(binding, copy, sizeof(copy), 1);
  assert(copy[0] == 23); // The graph's weight remains authoritative.
  failed = false;
  try {
    registry.fcwhRead(binding, copy, sizeof(copy), record.size() - 1);
  } catch (const std::out_of_range &) {
    failed = true;
  }
  assert(failed);
  assert(registry.set_weight_static_quant(record.data(), .25f, 127));
  failed = false;
  try {
    registry.set_weight_static_quant(record.data(),
                                     std::numeric_limits<float>::infinity(), 0);
  } catch (const std::invalid_argument &) {
    failed = true;
  }
  assert(failed && registry.static_quant_.at(record.data()).zp == 127);
  std::puts("HTP weight registry / non-owning views / failure recovery PASS");
}
