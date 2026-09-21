// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hex_image.h
 * @date	31 August 2026
 * @brief	.hexcfg (HexModelConfig as key=value text) and raw file helpers
 *		shared by nntr_hexpack, the x86 reference runner and the device
 *		harness
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#ifndef __CAUSALLM_HEXAGON_HEX_IMAGE_H__
#define __CAUSALLM_HEXAGON_HEX_IMAGE_H__

#include <cstdint>
#include <string>
#include <vector>

#include "graph_lowering.h"

namespace nntrainer::hexagon {

/** Write/read the .hexcfg text file: the 11 HexModelConfig fields,
 * weight_layout=tiled32|w4cx_down8 and, for w4cx_down8, i8_tensors=<names>
 * (hex_i8_names). A file without the layout key is a pre-v4 image and is
 * rejected, as is a layout no kernel reads yet (w4cx).
 * @throw std::runtime_error */
void write_hexcfg(const std::string &path, const HexModelConfig &cfg);
HexModelConfig read_hexcfg(const std::string &path);

/** HexTensorBit mask <-> comma-separated names in the fixed order
 * embed,q,k,v,o,gate,up,down (the make_w8cx_bin.py --i8-tensors spelling).
 * @throw std::runtime_error on an unknown name */
std::string hex_i8_names(uint32_t mask);
uint32_t hex_i8_mask(const std::string &names);
const char *hex_layout_name(uint32_t layout);

/** Whole-file IO. @throw std::runtime_error on failure or size mismatch */
void write_file(const std::string &path, const void *data, uint64_t size);
void read_file_into(const std::string &path, void *dst, uint64_t size);
std::vector<uint8_t> read_file(const std::string &path);

} // namespace nntrainer::hexagon
#endif // __CAUSALLM_HEXAGON_HEX_IMAGE_H__
