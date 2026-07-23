// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   hexagon_repack.h
 * @date   23 July 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @brief  Host-side (arch-agnostic) repack of standard block_q4_0 bytes into
 * the Hexagon HTP "q4x4x2" tile layout used by ggml-hexagon's cDSP matmul
 * kernel. Pure data shuffling - no NEON/AVX/Hexagon intrinsics involved, so
 * this compiles and is testable on any host regardless of target arch.
 */

#ifndef __HEXAGON_REPACK_H__
#define __HEXAGON_REPACK_H__

#include <cstddef>

namespace nntrainer {

/**
 * @brief Repack standard block_q4_0 bytes (per 32-element block: 2-byte fp16
 * delta + 16-byte packed nibbles) into the Hexagon HTP q4x4x2 tile layout:
 * every 8 consecutive q4_0 blocks (256 elements) become one tile of 128
 * packed-nibble bytes, with all tiles' nibbles stored first followed by all
 * tiles' 8 fp16 deltas. Matches ggml-hexagon's repack_row_q4x4x2()
 * (ggml/src/ggml-hexagon/ggml-hexagon.cpp), so weights packed here need no
 * further repacking once loaded into Hexagon rpcmem.
 *
 * @param dst output buffer, same total size as src (9*N/16 bytes per row)
 * @param src input block_q4_0 bytes (as produced by quantize_q4_0)
 * @param data_size total weight size in bytes
 * @param M number of rows
 * @param N number of elements per row; must be divisible by 256 (Q4_0
 * already requires divisibility by 32, this format tiles 8 blocks at a
 * time)
 * @throw std::invalid_argument if N is not divisible by 256
 */
void repack_q4_0_to_htp_q4x4x2(void *dst, const void *src, size_t data_size,
                                unsigned int M, unsigned int N);

/**
 * @brief Inverse of repack_q4_0_to_htp_q4x4x2 - unpacks Hexagon HTP q4x4x2
 * tiles back to standard block_q4_0 bytes. Used for host-side verification;
 * not needed on the hot path (the DSP kernel consumes q4x4x2 directly).
 *
 * @param dst output block_q4_0 bytes
 * @param src input q4x4x2-tiled bytes
 * @param data_size total weight size in bytes
 * @param M number of rows
 * @param N number of elements per row; must be divisible by 256
 * @throw std::invalid_argument if N is not divisible by 256
 */
void unpack_htp_q4x4x2_to_q4_0(void *dst, const void *src, size_t data_size,
                                unsigned int M, unsigned int N);

} // namespace nntrainer

#endif /* __HEXAGON_REPACK_H__ */
