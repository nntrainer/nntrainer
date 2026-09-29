// SPDX-License-Identifier: Apache-2.0
/**
 * @file	ref_fp16_x86.h
 * @date	31 August 2026
 * @brief	__fp16 stand-in for compiling ref_ops.c as C++ on an x86 host
 *		without _Float16 (gcc < 12). ref_ops only ever converts
 *		__fp16 <-> float, so a 16-bit struct with the two conversions
 *		reproduces the Hexagon/ARM semantics exactly (RNE on store).
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#ifndef REF_FP16_X86_H
#define REF_FP16_X86_H

#include <cstdint>
#include <cstring>

#include "graph_lowering.h" /* f32_to_f16_bits */

struct ref_fp16 {
  uint16_t bits;
  ref_fp16() = default;
  ref_fp16(float f) : bits(nntrainer::hexagon::f32_to_f16_bits(f)) {}
  operator float() const {
    const uint32_t sign = (uint32_t)(bits & 0x8000u) << 16;
    uint32_t exp = (bits >> 10) & 0x1fu;
    uint32_t mant = bits & 0x3ffu;
    uint32_t out;
    if (exp == 0x1fu) {
      out = sign | 0x7f800000u | (mant << 13);
    } else if (exp == 0u) {
      if (mant == 0u) {
        out = sign;
      } else { /* subnormal: normalize */
        exp = 127u - 15u + 1u;
        while (!(mant & 0x400u)) {
          mant <<= 1;
          --exp;
        }
        out = sign | (exp << 23) | ((mant & 0x3ffu) << 13);
      }
    } else {
      out = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
  }
};
static_assert(sizeof(ref_fp16) == 2, "ref_fp16 must be 16-bit");

typedef ref_fp16 __fp16;

#endif /* REF_FP16_X86_H */
