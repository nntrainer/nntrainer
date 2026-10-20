// SPDX-License-Identifier: Apache-2.0
/** @file htp_static_quant.h
 * @brief Fixed unsigned activation QDQ with round-to-nearest-even.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#ifndef NNTR_HTP_STATIC_QUANT_H
#define NNTR_HTP_STATIC_QUANT_H
#include <stdint.h>
#include <string.h>
static inline float htp_static_qdq(float x, float scale, int32_t zp) {
  uint32_t bits;
  memcpy(&bits, &x, sizeof(bits));
  if ((bits & 0x7fffffffu) > 0x7f800000u)
    return x;
  float q = x * (1.0f / scale);
  if (q <= -zp)
    return -zp * scale;
  if (q >= 255 - zp)
    return (255 - zp) * scale;
  int32_t i = (int32_t)q;
  float d = q - i;
  if (d > 0.5f || (d == 0.5f && i % 2))
    ++i;
  if (d < -0.5f || (d == -0.5f && i % 2))
    --i;
  return i * scale;
}
#endif
