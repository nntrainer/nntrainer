// SPDX-License-Identifier: Apache-2.0
/**
 * @file lfm2_router_arm_data.h
 * @brief lfm2 router arm data validation.
 * @author NNTrainer contributors
 * @bug No known bugs.
 */
// Shared fixed bit patterns; no arithmetic reference implementation.
static uint32_t bits(float x) {
  uint32_t v;
  std::memcpy(&v, &x, 4);
  return v;
}
static void put(float *p, uint32_t v) { std::memcpy(p, &v, 4); }
static void fill(float *p, size_t n, unsigned mode, bool weight) {
  const uint32_t special[] = {0,          0x80000000, 1,          0x80000001,
                              0x00800000, 0x80800000, 0x7f800000, 0xff800000,
                              0x7fc00021, 0xffc00042, 0x7f800063, 0xff800084};
  uint32_t seed = weight ? 0x783abc : 0x237abc;
  for (size_t i = 0; i < n; ++i) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    uint32_t v = (seed & 0x807fffff) | ((120u + unsigned(i % 12)) << 23);
    if (((mode == 1 && !weight) || (mode == 2 && weight) || mode == 3) &&
        i % 257 == 0)
      v = special[(i / 257) % 12];
    if (mode == 4)
      v = special[i % 6];
    if (mode == 5)
      v = (seed & 0x807fffff) | (250u << 23);
    if (mode == 6)
      v = (seed & 0x807fffff) | ((126u + unsigned(i % 3)) << 23);
    if (mode == 7)
      v = (i & 1) ? 0x80000000 : 0;
    static const unsigned powers[][2] = {
      {1, 127},  {127, 1}, {1, 254}, {254, 1}, {200, 30}, {30, 200}, {70, 70},
      {60, 60},  {30, 80}, {1, 126}, {1, 125}, {2, 124},  {1, 1},    {254, 10},
      {10, 254}, {0, 200}, {200, 0}, {0, 254}, {254, 0},  {0, 127},  {127, 0},
      {0, 0},    {60, 61}, {61, 60}, {50, 80}, {80, 50},  {2, 250},  {250, 2}};
    if (mode >= 8 && mode < 36) {
      const unsigned exponent = powers[mode - 8][weight ? 1 : 0];
      v = (seed & 0x807fffff) | (exponent << 23);
    }
    put(p + i, v);
  }
  // Every staged input differs in the sampled prefix even if a vector address
  // is reused. Retained weight vectors below never change after registration.
  if (!weight)
    put(p, 0x3f800000 + mode);
  if ((mode == 36 && !weight) || (mode == 37 && weight))
    put(p + n - 1, 1);
  if ((mode == 38 && !weight) || (mode == 39 && weight))
    put(p + n - 1, 0x7f800063);
}
