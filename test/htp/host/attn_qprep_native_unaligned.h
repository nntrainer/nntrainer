// SPDX-License-Identifier: Apache-2.0
/**
 * @file attn_qprep_native_unaligned.h
 * @brief attn qprep native unaligned validation.
 * @author NNTrainer contributors
 * @bug No known bugs.
 */
/** Host-only model of the SDK Hexagon HVX_UVector alignment contract.
 * libnative omits HVX_UVector. Mapping it to HEXAGON_Vect1024 would instead
 * require its host 8-byte alignment, incorrectly rejecting valid float rows.
 * Hexagon SDK hexagon_types.h defines HEXAGON_UVect1024 with aligned(4).
 * Keep all arithmetic in libnative and retain sanitizer alignment checks.
 */
#ifndef NNTR_ATTN_QPREP_NATIVE_UNALIGNED_H
#define NNTR_ATTN_QPREP_NATIVE_UNALIGNED_H
#include <hvx_hexagon_protos.h>
typedef HEXAGON_Vect1024 nntr_native_uvec __attribute__((aligned(4)));
#define HVX_UVector nntr_native_uvec
_Static_assert(sizeof(HVX_UVector) == 128, "HVX vector byte size");
_Static_assert(_Alignof(HVX_UVector) == 4, "HVX unaligned float-row alignment");
#endif
