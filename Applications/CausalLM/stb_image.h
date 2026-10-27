// SPDX-License-Identifier: MIT
/**
 * @file   stb_image.h
 * @date   23 October 2026
 * @brief  Forwarding header for the vendored stb_image v2.30 (stb_image.inc)
 * @see    https://github.com/nothings/stb
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details The YOLO runners include "stb_image.h"; the library itself lives in
 *          stb_image.inc next to this file. This used to be a symlink, which
 *          does not survive every checkout (Windows, archives), so it is a
 *          one-line header instead. Define STB_IMAGE_IMPLEMENTATION in exactly
 *          one translation unit before including it, as with upstream stb.
 */
#include "stb_image.inc"
