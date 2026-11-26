// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file   kv_img_write.cl
 * @brief  Key/value cache scatter kernels for OpenCL images.
 * @see    https://github.com/nntrainer/nntrainer
 *
 * @details Extracted from attention_kernels.cpp.
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifndef KIMG_GSH
#define KIMG_GSH 0
#endif
__constant sampler_t kimgw_smp =
  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
// K: one work item per (t, h, d-texel). A K texel is 8 halves along d and the
// scatter writes every channel of every row it touches, so every texel it
// covers is FULL -- no read-modify-write anywhere on the K side. Same source,
// same rows, same folded mapping as k_scatter_ohwi.
__kernel void k_scatter_ohwi_img(__global const half *src,
                                 __write_only image2d_t dst, const int M,
                                 const int hKV, const int d, const int max_S,
                                 const int position, const int src_off) {
  const int t = get_global_id(0);
  const int h = get_global_id(1);
  const int dt = get_global_id(2);
  const int dtex = d >> 3;
  if (t >= M || h >= hKV || dt >= dtex) return;
  const long so =
    (long)src_off + (long)t * hKV * d + (long)h * d + (long)(dt << 3);
  half tmp[8];
  for (int l = 0; l < 8; ++l) tmp[l] = src[so + l];
  const int lin_row = h * max_S + position + t;
  const int g = 1 << KIMG_GSH;
  write_imageui(dst,
                (int2)(((lin_row & (g - 1)) * dtex) + dt, lin_row >> KIMG_GSH),
                as_uint4(vload8(0, tmp)));
}
// V: one work item per (sequence texel, h, channel). The texels touched are
// [position>>3, (position+M+7)>>3); lanes inside [position, position+M) come
// from src exactly as v_scatter_ohwi_t reads it, every other lane is read back
// from the image and written unchanged -- so this writes precisely the elements
// the buffer scatter writes and nothing else.
__kernel void v_scatter_ohwi_t_img(__global const half *src,
                                   __read_write image2d_t dst, const int M,
                                   const int hKV, const int d, const int max_S,
                                   const int position, const int src_off) {
  const int j = get_global_id(0);
  const int h = get_global_id(1);
  const int x = get_global_id(2);
  const int tex0 = position >> 3;
  const int ntex = ((position + M + 7) >> 3) - tex0;
  if (j >= ntex || h >= hKV || x >= d) return;
  const int tex = tex0 + j;
  const int2 c = (int2)(tex, h * d + x);
  half tmp[8];
  vstore8(as_half8(read_imageui(dst, c)), 0, tmp);
  for (int l = 0; l < 8; ++l) {
    const int p = (tex << 3) + l;
    if (p >= position && p < position + M)
      tmp[l] = src[(long)src_off + (long)(p - position) * hKV * d +
                   (long)h * d + x];
  }
  write_imageui(dst, c, as_uint4(vload8(0, tmp)));
}
// Decode-step K and V image scatters in ONE launch (M == 1): lanes
// [0, d/8) of get_global_id(2) are k_scatter_ohwi_img's (head, d-texel) work
// items, lanes [d/8, d/8 + d) are v_scatter_ohwi_t_img's (head, channel) work
// items for the single texel a one-row step touches. Same reads, same texel
// coordinates, same stores as the two kernels above.
__kernel void kv_scatter_ohwi_img(__global const half *k_src,
                                  __write_only image2d_t k_dst,
                                  __global const half *v_src,
                                  __read_write image2d_t v_dst, const int hKV,
                                  const int d, const int k_max_S,
                                  const int v_max_S, const int position,
                                  const int k_src_off, const int v_src_off) {
  const int h = get_global_id(1);
  const int z = get_global_id(2);
  const int dtex = d >> 3;
  if (h >= hKV) return;
  if (z < dtex) {
    const int dt = z;
    const long so = (long)k_src_off + (long)h * d + (long)(dt << 3);
    half tmp[8];
    for (int l = 0; l < 8; ++l) tmp[l] = k_src[so + l];
    const int lin_row = h * k_max_S + position;
    const int g = 1 << KIMG_GSH;
    write_imageui(k_dst,
                  (int2)(((lin_row & (g - 1)) * dtex) + dt, lin_row >> KIMG_GSH),
                  as_uint4(vload8(0, tmp)));
  } else if (z < dtex + d) {
    const int x = z - dtex;
    const int tex = position >> 3;
    const int2 c = (int2)(tex, h * d + x);
    half tmp[8];
    vstore8(as_half8(read_imageui(v_dst, c)), 0, tmp);
    for (int l = 0; l < 8; ++l) {
      const int p = (tex << 3) + l;
      if (p >= position && p < position + 1)
        tmp[l] = v_src[(long)v_src_off + (long)(p - position) * hKV * d +
                       (long)h * d + x];
    }
    write_imageui(v_dst, c, as_uint4(vload8(0, tmp)));
  }
}
// V, mode 2: the same rows, with NO image read anywhere in the write path. Mode
// 1's read-modify-write is itself an image read, and an image read is what
// pollutes the texture cache on this driver (see the gather), so at long spans
// mode 1 may be re-introducing the very hazard it closes. Here the partial
// texels are rebuilt from the concat V cache instead: lane p of texel `tex` is
// absolute position base_abs + (tex<<3) + p, mapped through the ring exactly as
// cacheRow() does, and +0.0h at or past the frontier (`to_abs`) -- what the
// one-time mirror fill left there. This needs the slab to BE complete, which it
// is with the side-fills on (NNTR_KV_STAGE off, the default) AND with the
// NNTR_MHA_CLMEM boundary gather skipped: that gather writes the slab at
// ABSOLUTE rows with no ring mapping, so on a ringed layer whose mirror is live
// it fills rows past the ring cap with the wrong content (and past the
// allocation) -- harmless while nothing reads them back, fatal here.
__kernel void v_scatter_ohwi_t_img_slab(__global const half *vcache,
                                        __write_only image2d_t dst,
                                        const int hKV, const int d,
                                        const int tex0, const int ntex,
                                        const int base_abs, const int ring_cap,
                                        const int to_abs) {
  const int j = get_global_id(0);
  const int h = get_global_id(1);
  const int x = get_global_id(2);
  if (j >= ntex || h >= hKV || x >= d) return;
  const int tex = tex0 + j;
  const long hd = (long)hKV * d;
  half tmp[8];
  for (int l = 0; l < 8; ++l) {
    const int a = base_abs + (tex << 3) + l;
    half v = (half)0.0h;
    if (a >= 0 && a < to_abs) {
      const int phys = ring_cap > 0 ? (a % ring_cap) : a;
      v = vcache[(long)phys * hd + (long)h * d + x];
    }
    tmp[l] = v;
  }
  write_imageui(dst, (int2)(tex, h * d + x), as_uint4(vload8(0, tmp)));
}
// Inverse gathers on the image path: with the writes on the image, a BUFFER
// read of the mirror is the same aliasing hazard reversed (the boundary sync
// would hand the host slab a pre-write row), so gather from the image too.
__kernel void mirror_fill_img(__write_only image2d_t dst, const int w,
                              const int h, const uint v) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if (x >= w || y >= h) return;
  write_imageui(dst, (int2)(x, y), (uint4)(v, v, v, v));
}
__kernel void mirror_touch_img(__write_only image2d_t dst, const int x,
                               const int y) {
  write_imageui(dst, (int2)(x, y), (uint4)(0u, 0u, 0u, 0u));
}
__kernel void k_gather_ohwi_img(__read_only image2d_t src, __global half *dst,
                                const int M, const int hKV, const int d,
                                const int max_S, const int position) {
  const int t = get_global_id(0);
  const int h = get_global_id(1);
  const int dt = get_global_id(2);
  const int dtex = d >> 3;
  if (t >= M || h >= hKV || dt >= dtex) return;
  const int lin_row = h * max_S + position + t;
  const int g = 1 << KIMG_GSH;
  const uint4 vv = read_imageui(
    src, kimgw_smp,
    (int2)(((lin_row & (g - 1)) * dtex) + dt, lin_row >> KIMG_GSH));
  half tmp[8];
  vstore8(as_half8(vv), 0, tmp);
  const long dof = (long)t * hKV * d + (long)h * d + (long)(dt << 3);
  for (int l = 0; l < 8; ++l) dst[dof + l] = tmp[l];
}
__kernel void v_gather_ohwi_t_img(__read_only image2d_t src, __global half *dst,
                                  const int M, const int hKV, const int d,
                                  const int max_S, const int position) {
  const int t = get_global_id(0);
  const int h = get_global_id(1);
  const int x = get_global_id(2);
  if (t >= M || h >= hKV || x >= d) return;
  const int p = position + t;
  const uint4 vv = read_imageui(src, kimgw_smp, (int2)(p >> 3, h * d + x));
  half tmp[8];
  vstore8(as_half8(vv), 0, tmp);
  dst[(long)t * hKV * d + (long)h * d + x] = tmp[p & 7];
}
