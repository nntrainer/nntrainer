// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   unittest_half_fp16.cpp
 * @date   30 Aug 2026
 * @brief  Bit-identity and arithmetic-parity tests for the uint16-backed
 *         nntrainer::Half wrapper against native _Float16
 * @see    https://github.com/nntrainer/nntrainer
 * @see    docs/design/msvc_fp16_half_wrapper.md
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @note   The wrapper stands in for _Float16 on toolchains that have no native
 *         half (MSVC). On a toolchain that has both, the two can be compared
 *         directly, which is what this suite does: it includes half_fp16.h
 *         unconditionally, independent of which one the build selected for
 *         _FP16, so the comparison runs in both the native and the
 *         -Dfp16-impl=wrapper configurations.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <half_fp16.h>

#ifdef ENABLE_FP16

namespace {

/** @brief native half, whatever the compiler calls it */
#ifdef USE__FP16
using NativeHalf = __fp16;
#else
using NativeHalf = _Float16;
#endif

using nntrainer::Half;

inline uint16_t native_bits(NativeHalf h) {
  uint16_t b;
  std::memcpy(&b, &h, sizeof(b));
  return b;
}

inline NativeHalf native_from_bits(uint16_t b) {
  NativeHalf h;
  std::memcpy(&h, &b, sizeof(h));
  return h;
}

/** @brief NaN payloads are not required to agree; treat all NaNs as equal */
inline bool same_half(uint16_t a, uint16_t b) {
  auto is_nan = [](uint16_t x) {
    return (x & 0x7C00u) == 0x7C00u && (x & 0x03FFu) != 0u;
  };
  if (is_nan(a) && is_nan(b))
    return true;
  return a == b;
}

/** @brief a spread of floats covering subnormals, overflow and the edges */
std::vector<float> float_probe_grid() {
  std::vector<float> v;
  for (uint32_t h = 0; h < 0x10000u; ++h)
    v.push_back(static_cast<float>(native_from_bits(static_cast<uint16_t>(h))));
  const float extras[] = {0.0f,          -0.0f,         1.0f,
                          -1.0f,         65504.0f,      -65504.0f,
                          65520.0f,   // rounds to inf
                          65519.996f, // rounds to 65504
                          6.1035156e-5f, 5.9604645e-8f,
                          2.9802322e-8f, // half of the smallest subnormal
                          1e-10f,        1e10f,         3.4028235e38f,
                          -3.4028235e38f};
  for (float f : extras)
    v.push_back(f);
  for (int i = 0; i < 4096; ++i)
    v.push_back(-8.0f + static_cast<float>(i) * (16.0f / 4096.0f));
  return v;
}

} // namespace

/**
 * @brief every one of the 65536 half bit patterns widens to the same float
 */
TEST(HalfFp16, HalfToFloatIsBitIdenticalOverTheWholeDomain) {
  size_t mismatches = 0;
  for (uint32_t h = 0; h < 0x10000u; ++h) {
    const uint16_t bits = static_cast<uint16_t>(h);
    Half w;
    w.bits_ = bits;
    const float got = static_cast<float>(w);
    const float want = static_cast<float>(native_from_bits(bits));
    if (std::isnan(got) && std::isnan(want))
      continue;
    uint32_t gb, wb;
    std::memcpy(&gb, &got, sizeof(gb));
    std::memcpy(&wb, &want, sizeof(wb));
    if (gb != wb)
      ++mismatches;
  }
  EXPECT_EQ(mismatches, 0u);
}

/**
 * @brief narrowing a float rounds to the same bits as the native type does
 */
TEST(HalfFp16, FloatToHalfIsBitIdenticalOverTheProbeGrid) {
  size_t mismatches = 0;
  const std::vector<float> grid = float_probe_grid();
  for (float f : grid) {
    const uint16_t got = Half(f).bits_;
    const uint16_t want = native_bits(static_cast<NativeHalf>(f));
    if (!same_half(got, want))
      ++mismatches;
  }
  EXPECT_EQ(mismatches, 0u);
  EXPECT_GT(grid.size(), 65536u);
}

/**
 * @brief a single arithmetic operation is bit-identical to the native type
 *
 * Only one operation: with more than one, GCC and clang evaluate native
 * _Float16 with excess precision and round once at the end, while a value-type
 * half rounds every intermediate. That difference is by design and documented.
 */
TEST(HalfFp16, SingleOperationMatchesNative) {
  size_t add = 0, sub = 0, mul = 0, divi = 0, total = 0;
  for (uint32_t i = 0; i < 0x10000u; i += 7) {
    for (uint32_t k = 0; k < 0x10000u; k += 1021) {
      const uint16_t ab = static_cast<uint16_t>(i);
      const uint16_t bb = static_cast<uint16_t>(k);
      Half wa, wb;
      wa.bits_ = ab;
      wb.bits_ = bb;
      const NativeHalf na = native_from_bits(ab);
      const NativeHalf nb = native_from_bits(bb);
      ++total;
      if (!same_half((wa + wb).bits_,
                     native_bits(static_cast<NativeHalf>(na + nb))))
        ++add;
      if (!same_half((wa - wb).bits_,
                     native_bits(static_cast<NativeHalf>(na - nb))))
        ++sub;
      if (!same_half((wa * wb).bits_,
                     native_bits(static_cast<NativeHalf>(na * nb))))
        ++mul;
      if (!same_half((wa / wb).bits_,
                     native_bits(static_cast<NativeHalf>(na / nb))))
        ++divi;
    }
  }
  EXPECT_GT(total, 500000u);
  EXPECT_EQ(add, 0u);
  EXPECT_EQ(sub, 0u);
  EXPECT_EQ(mul, 0u);
  EXPECT_EQ(divi, 0u);
}

/**
 * @brief comparisons agree with the native type, NaN and signed zero included
 */
TEST(HalfFp16, ComparisonsMatchNative) {
  for (uint32_t i = 0; i < 0x10000u; i += 13) {
    for (uint32_t k = 0; k < 0x10000u; k += 2003) {
      const uint16_t ab = static_cast<uint16_t>(i);
      const uint16_t bb = static_cast<uint16_t>(k);
      Half wa, wb;
      wa.bits_ = ab;
      wb.bits_ = bb;
      const NativeHalf na = native_from_bits(ab);
      const NativeHalf nb = native_from_bits(bb);
      ASSERT_EQ(wa < wb, na < nb);
      ASSERT_EQ(wa > wb, na > nb);
      ASSERT_EQ(wa <= wb, na <= nb);
      ASSERT_EQ(wa >= wb, na >= nb);
      ASSERT_EQ(wa == wb, na == nb);
      ASSERT_EQ(wa != wb, na != nb);
    }
  }
}

/**
 * @brief `half op= float` rounds once, exactly as the native type does
 *
 * The homogeneous operator+=(Half) would narrow the float operand first and
 * round twice. These are the mixed overloads that keep the two implementations
 * in step; the SGEMM fallbacks reach them through `C[...] += beta * c_old`
 * (fallback_internal_fp16.cpp, hsgemm_loop).
 *
 * The reference side is written as an explicit widen-compute-narrow rather than
 * `native += f`: that is what the native compound assignment means once the
 * usual arithmetic conversions have run, and spelling it out avoids GCC's
 * -Wconversion-style "greater conversion rank" diagnostic, which this tree
 * builds with -Werror.
 */
TEST(HalfFp16, MixedCompoundAssignmentRoundsOnce) {
  const float rhs[] = {0.00024414062f, 0.000244140626f, 1.0f / 3.0f,
                       1e-4f,          70000.0f,        -70000.0f,
                       65504.0f,       1e-8f,           123456.75f};
  size_t mismatches = 0, checked = 0;
  for (uint32_t i = 0; i < 0x10000u; i += 3) {
    const uint16_t ab = static_cast<uint16_t>(i);
    for (float f : rhs) {
      Half w;
      w.bits_ = ab;
      w += f;
      const NativeHalf n =
        static_cast<NativeHalf>(static_cast<float>(native_from_bits(ab)) + f);
      ++checked;
      if (!same_half(w.bits_, native_bits(n)))
        ++mismatches;

      Half w2;
      w2.bits_ = ab;
      w2 *= f;
      const NativeHalf n2 =
        static_cast<NativeHalf>(static_cast<float>(native_from_bits(ab)) * f);
      ++checked;
      if (!same_half(w2.bits_, native_bits(n2)))
        ++mismatches;
    }
  }
  EXPECT_GT(checked, 100000u);
  EXPECT_EQ(mismatches, 0u);
}

/**
 * @brief the double-rounding the mixed overloads exist to prevent is real
 *
 * Narrowing the float operand first is not merely less accurate: 70000.0f is
 * above the half range, so Half(70000.0f) is +inf and the sum is +inf, while
 * rounding once gives a finite result. Pin the behaviour so a future
 * simplification that drops the mixed overloads fails here rather than in a
 * model.
 */
TEST(HalfFp16, NarrowingTheOperandFirstWouldOverflow) {
  const float big = 70000.0f;

  // narrowing first: 70000 is above the half range, so it becomes +inf
  EXPECT_TRUE(std::isinf(static_cast<float>(Half(big))));

  // ... but 65504 - 70000 = -4496 is well inside it, and the mixed overload
  // keeps it there, agreeing with the native widen-compute-narrow
  Half a(65504.0f);
  a -= big;
  const NativeHalf na = static_cast<NativeHalf>(65504.0f - big);
  EXPECT_FALSE(std::isinf(static_cast<float>(a)));
  EXPECT_FLOAT_EQ(static_cast<float>(a), -4496.0f);
  EXPECT_TRUE(same_half(a.bits_, native_bits(na)));

  // the homogeneous overload is the one that overflows: this is what `a -= big`
  // used to resolve to before the mixed overloads existed
  Half b(65504.0f);
  b -= Half(big);
  EXPECT_TRUE(std::isinf(static_cast<float>(b)));
}

/**
 * @brief the layout contract the accelerator staging code relies on
 */
TEST(HalfFp16, LayoutContract) {
  static_assert(sizeof(Half) == 2, "Half must be 2 bytes");
  static_assert(std::is_standard_layout<Half>::value,
                "Half must be std layout");
  static_assert(std::is_trivially_copyable<Half>::value,
                "Half must be trivially copyable");
  EXPECT_EQ(sizeof(Half), sizeof(NativeHalf));

  // storage interchange: a buffer written as Half reads back as native halves
  std::vector<Half> hs(256);
  std::vector<NativeHalf> ns(256);
  for (size_t i = 0; i < hs.size(); ++i) {
    const float v = static_cast<float>(i) * 0.75f - 96.0f;
    hs[i] = Half(v);
    ns[i] = static_cast<NativeHalf>(v);
  }
  EXPECT_EQ(std::memcmp(hs.data(), ns.data(), hs.size() * sizeof(Half)), 0);
}

/**
 * @brief increment / decrement, which the test utilities use on _FP16
 */
TEST(HalfFp16, IncrementDecrementMatchNative) {
  for (uint32_t i = 0; i < 0x10000u; i += 29) {
    const uint16_t ab = static_cast<uint16_t>(i);
    Half w;
    w.bits_ = ab;
    NativeHalf n = native_from_bits(ab);
    Half wpost = w++;
    NativeHalf npost = n++;
    ASSERT_TRUE(same_half(wpost.bits_, native_bits(npost)));
    ASSERT_TRUE(same_half(w.bits_, native_bits(n)));
    --w;
    --n;
    ASSERT_TRUE(same_half(w.bits_, native_bits(n)));
  }
}

/**
 * @brief std::numeric_limits<Half> answers binary16's real values.
 *
 * This is the one query that would not fail to compile if Half were left
 * unspecialized: the primary template is defined for every type and
 * value-initializes, so every value member would silently answer 0.0.
 *
 * The assertions are against binary16's own constants rather than against
 * std::numeric_limits<NativeHalf>, because the native type is NOT a usable
 * reference here. libstdc++ specializes numeric_limits for _Float16 only from
 * C++23, where __STDCPP_FLOAT16_T__ is defined; this project builds C++17
 * (C++20 on Windows), so on a native build the native limits value-initialize
 * exactly the way an unspecialized Half would. Measured on GCC 13.3:
 * is_specialized 0, max() 0, infinity() 0 under -std=c++17, and 1 / 65504 /
 * inf under -std=c++23.
 *
 * The parity check is therefore made conditional on the native type actually
 * being specialized, so it starts asserting the moment the standard level or
 * the library provides it, and does not compare against zeros until then.
 */
TEST(HalfFp16, NumericLimitsAnswerTheRealBinary16Values) {
  using WL = std::numeric_limits<Half>;

  ASSERT_TRUE(WL::is_specialized);
  EXPECT_TRUE(WL::is_signed);
  EXPECT_FALSE(WL::is_integer);
  EXPECT_FALSE(WL::is_exact);
  EXPECT_TRUE(WL::has_infinity);
  EXPECT_TRUE(WL::has_quiet_NaN);
  EXPECT_TRUE(WL::is_bounded);
  EXPECT_FALSE(WL::is_modulo);
  EXPECT_EQ(WL::radix, 2);
  EXPECT_EQ(WL::digits, 11);
  EXPECT_EQ(WL::digits10, 3);
  EXPECT_EQ(WL::max_digits10, 5);
  EXPECT_EQ(WL::min_exponent, -13);
  EXPECT_EQ(WL::min_exponent10, -4);
  EXPECT_EQ(WL::max_exponent, 16);
  EXPECT_EQ(WL::max_exponent10, 4);

  /** the value members, as bit patterns */
  EXPECT_EQ(WL::min().bits_, 0x0400);     /**< 2^-14 */
  EXPECT_EQ(WL::max().bits_, 0x7BFF);     /**< 65504 */
  EXPECT_EQ(WL::lowest().bits_, 0xFBFF);  /**< -65504 */
  EXPECT_EQ(WL::epsilon().bits_, 0x1400); /**< 2^-10 */
  EXPECT_EQ(WL::round_error().bits_, 0x3800);
  EXPECT_EQ(WL::infinity().bits_, 0x7C00);
  EXPECT_EQ(WL::denorm_min().bits_, 0x0001); /**< 2^-24 */

  /** and as values, through the same conversion every other operator uses */
  EXPECT_FLOAT_EQ(static_cast<float>(WL::max()), 65504.0f);
  EXPECT_FLOAT_EQ(static_cast<float>(WL::lowest()), -65504.0f);
  EXPECT_TRUE(std::isinf(static_cast<float>(WL::infinity())));

  /** NaN payloads are unspecified, so assert the property, not the bits */
  EXPECT_TRUE(std::isnan(static_cast<float>(WL::quiet_NaN())));
  EXPECT_TRUE(std::isnan(static_cast<float>(WL::signaling_NaN())));

  /** every member is usable in a constant expression, as the standard asks */
  static_assert(WL::max().bits_ == 0x7BFF, "max must be constexpr and 65504");
  static_assert(WL::infinity().bits_ == 0x7C00, "infinity must be constexpr");

  /**
   * Parity with the native type, once the library actually provides it. Until
   * then numeric_limits<_Float16> is unspecialized and comparing against it
   * would only assert that two things are both zero.
   */
  using NL = std::numeric_limits<NativeHalf>;
  if (NL::is_specialized) {
    EXPECT_EQ(WL::digits, NL::digits);
    EXPECT_EQ(WL::max_exponent, NL::max_exponent);
    EXPECT_EQ(WL::min_exponent, NL::min_exponent);
    EXPECT_EQ(WL::max().bits_, native_bits(NL::max()));
    EXPECT_EQ(WL::lowest().bits_, native_bits(NL::lowest()));
    EXPECT_EQ(WL::epsilon().bits_, native_bits(NL::epsilon()));
    EXPECT_EQ(WL::infinity().bits_, native_bits(NL::infinity()));
    EXPECT_EQ(WL::denorm_min().bits_, native_bits(NL::denorm_min()));
  }
}

#endif // ENABLE_FP16

int main(int argc, char **argv) {
  int result = -1;
  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Failed to init gtest\n";
    return result;
  }
  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Failed to run tests\n";
  }
  return result;
}
