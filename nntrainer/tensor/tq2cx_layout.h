// SPDX-License-Identifier: Apache-2.0
/**
 * @file	tq2cx_layout.h
 * @date	06 October 2026
 * @brief	Stored layouts of a TQ2CX record (where each mask unit lives).
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs
 *
 * A TQ2CX record holds N output rows of U = ceil(K / 32) mask units
 * ({uint32 pos; uint32 neg;}, 8 bytes, 32 input taps each), then N fp32
 * scales. The VALUES of a unit and the meaning of its bits never change; a
 * layout only says where unit (n, u) is stored:
 *
 *  "rm"      (v1, row-major) unit (n, u) at n * U + u; scales at N * U * 8;
 *            record = N * U * 8 + N * 4 bytes.
 *  "t2r16v2" (v2, tiled) rows in tiles of R = 16, padded with zero units to
 *            Npad = ceil(N / 16) * 16; inside a tile, blocks of V = 2 units,
 *            block b holding the 16 rows x w_b units row-major,
 *            w_b = min(V, U - b * V) (a short tail block is packed tight):
 *              unit (n, u) at (n / R) * R * U + (u / V) * V * R
 *                             + (n % R) * w_b + u % V
 *            scales at Npad * U * 8; record zero-padded to a multiple of 64
 *            bytes. Sixteen rows read 16 contiguous bytes each, side by side,
 *            which is the access a GPU work-group of row lanes makes.
 *
 * A pack names its layout once, in the safetensors header:
 * __metadata__["nntr_tq2cx_layout"] (mirrored as the nntr_config key
 * "tq2cx_layout"); a pack without it is "rm". The process runs one stored
 * layout (tq2cxStoredLayout()), set before the weights are sized; every
 * kernel, CPU and GPU, indexes the units through it, and the loader refuses a
 * pack whose header names another one.
 */

#ifndef __NNTRAINER_TQ2CX_LAYOUT_H__
#define __NNTRAINER_TQ2CX_LAYOUT_H__
#ifdef __cplusplus

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

namespace nntrainer {

/**
 * @brief A TQ2CX stored layout: R rows per tile (0 = row-major) and V units
 *        per block.
 */
struct Tq2cxLayout {
  unsigned int R = 0; /**< rows per tile, 0 = row-major (v1) */
  unsigned int V = 1; /**< units per block inside a tile */

  /** @brief true for a tiled (v2) layout */
  bool tiled() const { return R != 0; }

  /** @brief equality */
  bool operator==(const Tq2cxLayout &o) const {
    return R == o.R && (R == 0 || V == o.V);
  }

  /** @brief inequality */
  bool operator!=(const Tq2cxLayout &o) const { return !(*this == o); }
};

/** @brief v1: row-major */
constexpr Tq2cxLayout TQ2CX_LAYOUT_RM{0, 1};

/** @brief v2: 16-row tiles of 2-unit blocks */
constexpr Tq2cxLayout TQ2CX_LAYOUT_T2R16V2{16, 2};

/** @brief safetensors __metadata__ key naming the layout of a pack */
constexpr const char *TQ2CX_LAYOUT_META_KEY = "nntr_tq2cx_layout";

/** @brief nntr_config key mirroring it */
constexpr const char *TQ2CX_LAYOUT_CONFIG_KEY = "tq2cx_layout";

/**
 * @brief The id of a layout ("rm", "t2r16v2", ...).
 */
inline std::string tq2cxLayoutName(const Tq2cxLayout &l) {
  if (!l.tiled())
    return "rm";
  return "t2r" + std::to_string(l.R) + "v" + std::to_string(l.V);
}

/**
 * @brief Parse a layout id. Only the layouts the kernels implement are
 *        accepted: "rm" (an empty id is "rm") and "t2r16v2".
 * @retval false the id names no supported layout (@a out untouched)
 */
inline bool tq2cxLayoutFromName(const std::string &name, Tq2cxLayout &out) {
  if (name.empty() || name == "rm") {
    out = TQ2CX_LAYOUT_RM;
    return true;
  }
  if (name == "t2r16v2") {
    out = TQ2CX_LAYOUT_T2R16V2;
    return true;
  }
  return false;
}

/**
 * @brief The layout this process stores and reads TQ2CX records in. "rm"
 *        until setTq2cxStoredLayout().
 */
Tq2cxLayout tq2cxStoredLayout();

/**
 * @brief Set the stored layout. Call before any TQ2CX weight is sized (model
 *        initialize) and before the first TQ2CX kernel is built.
 */
void setTq2cxStoredLayout(const Tq2cxLayout &l);

/** @brief mask units of one row: ceil(K / 32) */
inline size_t tq2cxUnits(size_t K) { return (K + 31) / 32; }

/** @brief stored rows: N, or N padded to whole tiles */
inline size_t tq2cxPaddedRows(const Tq2cxLayout &l, size_t N) {
  return l.tiled() ? (N + l.R - 1) / l.R * l.R : N;
}

/** @brief bytes of the mask part = the offset of the scale tail */
inline size_t tq2cxMaskBytes(const Tq2cxLayout &l, size_t N, size_t K) {
  return tq2cxPaddedRows(l, N) * tq2cxUnits(K) * 8;
}

/** @brief bytes of one whole record (masks, scales, padding) */
inline size_t tq2cxRecordBytes(const Tq2cxLayout &l, size_t N, size_t K) {
  const size_t b = tq2cxMaskBytes(l, N, K) + N * sizeof(float);
  return l.tiled() ? (b + 63) / 64 * 64 : b;
}

/**
 * @brief Unit index of (row n, unit u) in a record of U units per row.
 */
inline size_t tq2cxUnitIndex(const Tq2cxLayout &l, size_t n, size_t u,
                             size_t U) {
  if (!l.tiled())
    return n * U + u;
  const size_t b = u / l.V;
  const size_t w = std::min<size_t>(l.V, U - b * l.V);
  return (n / l.R) * l.R * U + b * l.V * l.R + (n % l.R) * w + u % l.V;
}

/**
 * @brief Rewrite one record from layout @a from to layout @a to.
 * @param src record in @a from (tq2cxRecordBytes(from, N, K) bytes)
 * @param dst record in @a to (tq2cxRecordBytes(to, N, K) bytes, written in
 *        full: padding units and tail bytes are zero)
 */
void tq2cxRelayout(const Tq2cxLayout &from, const Tq2cxLayout &to,
                   const void *src, void *dst, size_t N, size_t K);

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __NNTRAINER_TQ2CX_LAYOUT_H__ */
