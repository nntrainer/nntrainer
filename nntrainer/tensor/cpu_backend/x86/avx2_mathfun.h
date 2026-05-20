/**
 * @file   avx2_mathfun.h
 * @date   03 Apr 2026
 * @brief  This is collection of sin, cos function with AVX2 SIMD
 * @see    https://github.com/nntrainer/nntrainer
 * @author Julien Pommier (original cephes-based algorithm)
 * @bug    No known bugs except for NYI items
 *
 */

/** AVX2 implementation of sin, cos

   Inspired by Intel Approximate Math library, and based on the
   corresponding algorithms of the cephes math library.

   Ported from the NEON implementation (neon_mathfun.h/hxx) to AVX2
   intrinsics, processing 8 floats at a time instead of 4.
*/

/** Copyright (C) 2011  Julien Pommier

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

#ifndef __AVX2_MATHFUN_H_
#define __AVX2_MATHFUN_H_

#include <immintrin.h>

/**
 * @brief     sincos function with AVX2 SIMD
 * @param[in] x input register variable (__m256, 8 floats)
 * @param[out] ysin sin register variable (__m256, 8 floats)
 * @param[out] ycos cos register variable (__m256, 8 floats)
 */
inline void sincos256_ps(__m256 x, __m256 *ysin, __m256 *ycos);

/**
 * @brief     sin function with AVX2 SIMD
 * @param[in] x input register variable (__m256, 8 floats)
 * @return    __m256 result of sin(x)
 */
inline __m256 sin256_ps(__m256 x);

/**
 * @brief     cos function with AVX2 SIMD
 * @param[in] x input register variable (__m256, 8 floats)
 * @return    __m256 result of cos(x)
 */
inline __m256 cos256_ps(__m256 x);

#include "avx2_mathfun.hxx"

#endif /* __AVX2_MATHFUN_H_ */
