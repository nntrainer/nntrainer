// SPDX-License-Identifier: Apache-2.0
/**
 * @file hvx_hadamard_f32.h
 * @brief Normalized full-width Hadamard transforms for online rotation.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#ifndef NNTR_HVX_HADAMARD_F32_H
#define NNTR_HVX_HADAMARD_F32_H

#include <stdint.h>

/** Supports powers of two and 44 times a power of two, up to 8192.
 * The latter uses the Paley H44, with first row [1, -1, ...], matching
 * SpinQuant's get_had44 / matmul_hadU convention. No zero padding and no
 * block-diagonal approximation: every output depends on the entire row.
 */
int hvx_hadamard_supported(uint32_t width);

/** In-place, normalized transform; mode 1 is the scalar timing reference,
 * mode 2 uses HVX on Hexagon (portable scalar elsewhere). Invalid arguments
 * return -1 without touching x. The caller owns width floats, aligned to
 * float only. Weights must be transformed consistently before quantization;
 * applying this to an existing checkpoint changes the model.
 */
int hvx_hadamard_f32(float *x, uint32_t width, uint32_t mode);

/** Contiguous independent rows. The 704-wide HVX path packs two rows
 * into the two halves of a register during H44; an odd final row uses
 * the single-row path. No row padding or cross-row mixing is performed.
 */
int hvx_hadamard_rows_f32(float *x, uint32_t rows, uint32_t width,
                          uint32_t mode);

#endif
