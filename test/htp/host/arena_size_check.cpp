// SPDX-License-Identifier: Apache-2.0
/**
 * @file arena_size_check.cpp
 * @brief Arena tail limits, alignment, opt-in parsing and overflow checks.
 * @author NNTrainer contributors
 * @bug No known bugs
 */
#include "htp_arena_size.h"
#include <cassert>
#include <cstdio>
#include <limits>

int main() {
  constexpr size_t mib = size_t(1) << 20;
  assert(nntrainer::htpArenaGrainBytes(nullptr) == 64u * mib);
  assert(nntrainer::htpArenaGrainBytes("64") == 64u * mib);
  assert(nntrainer::htpArenaGrainBytes("16") == 16u * mib);
  assert(nntrainer::htpArenaGrainBytes("4") == 4u * mib);
  for (const char *invalid :
       {"", "0", "1", "2", "8", "32", "-16", "4x", "16x", "16 "}) {
    bool rejected = false;
    try {
      (void)nntrainer::htpArenaGrainBytes(invalid);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    assert(rejected);
  }
  // Real native TL2 slot geometry: a remainder of 40 slots needs 80 MiB
  // rather than 128 MiB, while both retain the complete expert records.
  constexpr size_t slot = 2019328u, cap = 256u * mib;
  assert(nntrainer::htpArenaChunkBytes(slot, 40u * slot, cap, 4u * mib) ==
         80u * mib);
  assert(nntrainer::htpArenaChunkBytes(slot, 5u * slot, cap, 4u * mib) ==
         12u * mib);
  assert(nntrainer::htpArenaChunkBytes(slot, 40u * slot, cap, 16u * mib) ==
         80u * mib);
  assert(nntrainer::htpArenaChunkBytes(slot, 40u * slot, cap, 64u * mib) ==
         128u * mib);
  for (size_t grain : {4u * mib, 16u * mib, 64u * mib}) {
    for (size_t want :
         {size_t(0), slot, grain - 1u, grain, grain + 1u, cap, cap + 1u}) {
      const size_t size = nntrainer::htpArenaChunkBytes(slot, want, cap, grain);
      assert(size >= slot && size <= cap && size % grain == 0);
      if (want <= cap)
        assert(size >= want && size - std::max(slot, want) < grain);
    }
    // A failed mapping can already have reduced cap below the preferred
    // grain. Preserve that cap rather than growing it again on retry.
    assert(nntrainer::htpArenaChunkBytes(slot, cap, 16u * mib, grain) ==
           16u * mib);
    assert(nntrainer::htpArenaChunkBytes(slot, cap, 4u * mib, grain) ==
           4u * mib);
    assert(nntrainer::htpArenaChunkBytes(cap + 1u, cap, cap, grain) == 0);
    assert(nntrainer::htpArenaChunkBytes(
             slot, std::numeric_limits<size_t>::max(), cap, grain) == 0);
  }
  assert(nntrainer::htpArenaChunkBytes(slot, slot, cap, 0) == 0);
  assert(nntrainer::htpArenaChunkBytes(slot, slot, cap, 3) == 0);
  std::puts("HTP ARENA TAIL SIZING / CAP / OVERFLOW OK");
}
