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

/** Supports powers of two, 44 times a power of two (up to 8192),
 * and LFM2 SR15's 1536-wide dephased H12 x H128 transform.
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
/** Mode 4 applies the Gemma4 Paley-I input signs, then mode 2.
 * Supported widths for mode 4 are 2816 and 704. */
int hvx_hadamard_f32(float *x, uint32_t width, uint32_t mode);

/** Contiguous independent rows. The 704-wide HVX path packs two rows
 * into the two halves of a register during H44; an odd final row uses
 * the single-row path. No row padding or cross-row mixing is performed.
 */
int hvx_hadamard_rows_f32(float *x, uint32_t rows, uint32_t width,
                          uint32_t mode);

/** Transform then apply the checkpoint's immutable width-element XOR mask.
 * A null mask is equivalent to the unsigned API. Masks need only uint32_t
 * alignment and must not overlap x. The aligned 1536/2048-wide HVX paths fuse
 * XOR into their final store, preserving transform-then-XOR bit order.
 */
int hvx_hadamard_signed_f32(float *x, uint32_t width, uint32_t mode,
                            const uint32_t *mask);
int hvx_hadamard_signed_rows_f32(float *x, uint32_t rows, uint32_t width,
                                 uint32_t mode, const uint32_t *mask);

/** Aligned mode-2 transform with two extrema per row, including zero.
 * Width must be 1536 or 2048, and x must be 128-byte aligned. Extrema owns
 * 2 * rows floats, disjoint from x and mask. NaNs propagate to both extrema;
 * quantization's usual zero clamp therefore preserves its NaN behavior.
 * Invalid arguments return -1 without modifying x or extrema. */
int hvx_hadamard_signed_rows_extrema_f32(float *x, uint32_t rows,
                                         uint32_t width, const uint32_t *mask,
                                         float *extrema);

#endif
