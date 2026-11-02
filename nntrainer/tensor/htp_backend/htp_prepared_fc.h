// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 * @file htp_prepared_fc.h
 * @brief Little-endian field readers for prepared weight artifacts.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#ifndef __HTP_PREPARED_FC_H__
#define __HTP_PREPARED_FC_H__

#include <cstdint>

namespace nntrainer {

inline uint32_t preparedFcU32(const uint8_t *p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
         (uint32_t(p[3]) << 24);
}

inline uint64_t preparedFcU64(const uint8_t *p) {
  return preparedFcU32(p) | (uint64_t(preparedFcU32(p + 4)) << 32);
}

} // namespace nntrainer
#endif // __HTP_PREPARED_FC_H__
