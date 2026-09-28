/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file   remote.h
 * @brief  Host stand-in for the FastRPC remote.h: the session handle type
 *         and the fixed-width names the qaic-generated header uses.
 */
#ifndef REMOTE_H
#define REMOTE_H
#include <stdint.h>
typedef uint64_t remote_handle64;
typedef uint8_t uint8;
typedef int8_t int8;
typedef uint16_t uint16;
typedef int16_t int16;
typedef uint32_t uint32;
typedef int32_t int32;
typedef uint64_t uint64;
typedef int64_t int64;
#endif
