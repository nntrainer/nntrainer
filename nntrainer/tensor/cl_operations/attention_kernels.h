// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file	attention_kernels.h
 * @date	28 August 2024
 * @brief	Common attention OpenCL kernels
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Yash Singh <yash.singh@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __ATTENTION_KERNELS_H__
#define __ATTENTION_KERNELS_H__

#include <cl_context.h>
#include <engine.h>
#include <opencl_buffer.h>
#include <opencl_kernel.h>

#include <string>

namespace nntrainer {

/**
 * @brief     Rotary Embedding process
 * @param[in] in _FP16 * input
 * @param[in] out _FP16 * output
 * @param[out] freqs_cos cosine of the frequencies
 * @param[out] freqs_sin sine of the frequencies
 * @param[in] cos_ vector of cos values
 * @param[in] sin_ vector of sin values
 * @param[in] batch size of batch
 * @param[in] channel channel of input
 * @param[in] height height of input
 * @param[in] width width of input
 * @param[in] dim hidden dim size
 * @param[in] from sequence order
 * @param[in] max_timestep max timestep
 * @param[in] in_size size of input
 * @param[in] out_size size of output
 */
void rotary_emb_cl(float *in, float *out,
                   const std::vector<std::vector<float>> &freqs_cos,
                   const std::vector<std::vector<float>> &freqs_sin,
                   const std::vector<float> &cos_,
                   const std::vector<float> &sin_, unsigned int batch,
                   unsigned int channel, unsigned int height,
                   unsigned int width, unsigned int dim, unsigned int from,
                   unsigned int max_timestamp, unsigned int in_size,
                   unsigned int out_size);

#ifdef ENABLE_FP16

/**
 * @brief     Rotary Embedding process
 * @param[in] in _FP16 * input
 * @param[in] out _FP16 * output
 * @param[out] freqs_cos cosine of the frequencies
 * @param[out] freqs_sin sine of the frequencies
 * @param[in] cos_ vector of cos values
 * @param[in] sin_ vector of sin values
 * @param[in] batch size of batch
 * @param[in] channel channel of input
 * @param[in] height height of input
 * @param[in] width width of input
 * @param[in] dim hidden dim size
 * @param[in] from sequence order
 * @param[in] max_timestep max timestep
 * @param[in] in_size size of input
 * @param[in] out_size size of output
 */
void rotary_emb_cl(_FP16 *in, _FP16 *out,
                   const std::vector<std::vector<float>> &freqs_cos,
                   const std::vector<std::vector<float>> &freqs_sin,
                   const std::vector<float> &cos_,
                   const std::vector<float> &sin_, unsigned int batch,
                   unsigned int channel, unsigned int height,
                   unsigned int width, unsigned int dim, unsigned int from,
                   unsigned int max_timestamp, unsigned int in_size,
                   unsigned int out_size);

#endif

/// GPU two-1x1-conv attention for prefill (ML Drift section 3.7
/// algorithm). Three-kernel pipeline:
///   K1: qk_matmul_f16    Q @ K^T  -> scores  (per head, full d reduce)
///   K2: softmax_row_f16  in-place row softmax over the N_kv axis
///   K3: sv_matmul_f16    scores @ V  -> O    (per head)
/// All three operate on FP16 storage (uint16 bits) with FP32 register
/// accumulators. GQA is handled inside the kernels via head_q / gqa.
/// `svm_inputs == true` passes Q/K/V/O directly via SVM pointers; the
/// `scores` buffer is always a grow-only cl_mem scratch.
/// Returns false if shape unsupported (caller falls back to CPU).
bool two_conv_attention_prefill_f16_cl(
  const uint16_t *Q_host, const uint16_t *K_host, const uint16_t *V_host,
  uint16_t *O_host, unsigned int M, unsigned int N_kv, unsigned int num_heads_Q,
  unsigned int num_heads_KV, unsigned int head_dim, bool causal,
  bool svm_inputs = false);

/// SVM-direct GPU RoPE (residency-friendly rotary embedding). Rotates the
/// (k, k+head_dim/2) pairs of each [M, num_heads * head_dim] FP16 (uint16-bit)
/// row by a precomputed cos/sin LUT, keeping the activation on the device (no
/// host round-trip). `in`/`out` may alias for in-place rotation (Q); pass a
/// distinct `out` to rotate-and-scatter (K into its cache slice).
///   in/out:        [M, num_heads * head_dim] FP16 bits, row-major.
///   cos_lut/sin_lut: [max_pos, head_dim/2] FP16 bits; row (start_pos + t)
///   used. start_pos:     absolute sequence position of row 0 (cache_index).
/// in/out are bound SVM-direct when `svm_inputs` (no host round-trip), else
/// uploaded/read-back via cl_mem scratch. The cos/sin LUT is a constant table
/// staged through cl_mem and cached PER SLOT across calls: each distinct
/// (cos_lut, sin_lut, head_dim/2) gets its OWN resident device buffer, uploaded
/// once and reused, so repeated RoPE calls incur no per-call LUT upload AND
/// models that alternate RoPE slots per layer (Gemma4 sliding<->full) hit the
/// cache on every transition instead of re-uploading (the caller must hand a
/// STABLE host pointer per slot for this to hold). `max_positions` is the row
/// count of the cos_lut/sin_lut tables.
/// Returns false if unsupported (caller falls back to the host RoPE).
/// in_clmem/out_clmem (static GPU_CLMEM residency): bind that side as a device
/// cl_mem (the tensor's planner sub-buffer) instead of the SVM pointer; mixed
/// cl_mem/SVM args are valid (e.g. K: cl_mem in -> SVM cache-slice out).
/// drain_svm_out=false skips the trailing clFinish on an SVM output (clFlush
/// instead, the submission point preserved): valid when every consumer of the
/// rotated output is a same-queue GPU kernel (the staged image-attention
/// chain; its non-image fallbacks drain separately before reading) -- the
/// per-call drain measured 19ms of GPU idle per 1K prefill (rope->rope).
/// write_off: when nonzero, the rotated output is written to
/// out[write_off + ..] so the K rotation can target a STABLE base handle
/// (cache_key) at a SCALAR per-token row offset (recordable) instead of an
/// offset-baked SVM slice pointer. Default 0 == byte-identical.
bool rope_inplace_f16_cl(const uint16_t *in, uint16_t *out,
                         const uint16_t *cos_lut, const uint16_t *sin_lut,
                         unsigned int M, unsigned int num_heads,
                         unsigned int head_dim, unsigned int start_pos,
                         unsigned int max_positions, bool svm_inputs = false,
                         void *in_clmem = nullptr, void *out_clmem = nullptr,
                         bool drain_svm_out = true, unsigned int write_off = 0);

/// SVM-direct flat FP16 copy (out[i] = in[i], i in [0, N)). Used to scatter a
/// V projection slice into its KV-cache window on the device without a host
/// round-trip (residency). `svm_inputs == true` binds in/out via SVM pointers;
/// otherwise host pointers are uploaded/read-back via cl_mem scratch.
/// in_clmem: bind the source as a device cl_mem (GPU_CLMEM-resident V
/// projection) instead of the SVM pointer.
/// out_clmem: bind the destination as a device cl_mem (kernel-chain staging
/// temp); when set, the trailing clFinish is skipped — the copy stays an
/// in-order enqueue consumed by downstream kernels (no host drain).
/// drain=false: skip the trailing clFinish even for an SVM destination
/// (side-fill writes consumed only after a later full queue drain, e.g. the
/// KV-cache slice read by host decode after the lm_head lower).
/// Returns false if unsupported (caller falls back to a host copy).
bool gpu_copy_f16_cl(const uint16_t *in, uint16_t *out, unsigned int N,
                     bool svm_inputs = false, void *in_clmem = nullptr,
                     void *out_clmem = nullptr, bool drain = true);

/// Row-offset KV side-fill: writes into a STABLE out_base at [write_off + i]
/// (write_off = cache_index * num_heads_KV * head_dim) instead of an
/// offset-baked destination pointer. Byte-identical, but keeps the destination
/// handle stable so a recorded decode KV-write can replay with write_off
/// overridden per token (cl_qcom_recordable_queues overrides scalars, not SVM
/// pointers). SVM-only; returns false otherwise.
bool gpu_copy_f16_row_cl(const uint16_t *in, uint16_t *out_base, unsigned int N,
                         int write_off, bool svm_inputs = false,
                         void *in_clmem = nullptr,
                         void *out_base_clmem = nullptr, bool drain = true);

/// Grow-only cl_mem staging buffer for kernel-chain temporaries (e.g. the
/// rotated-K staging temp between RoPE and k_scatter). *buf / *cap persist at
/// the caller (opaque void* so callers need not include CL headers); the
/// buffer is recreated only when `bytes` outgrows *cap. Returns false on
/// allocation failure (*buf left null; caller falls back to the SVM path).
bool ensure_cl_stage_buf(void **buf, size_t *cap, size_t bytes);

/// int8-KV variant of two_conv_attention_prefill_f16_cl. Same shapes,
/// but K/V are stored as signed int8 bytes with a per-(token, head)
/// FP16 amax scale (paper §3.7). The kernel dequantizes inline by
/// folding the scale into the QK^T and SV reductions.
///   K_i8_host, V_i8_host:  [N_kv, num_heads_KV * head_dim] int8
///   K_scale_host, V_scale_host: [N_kv, num_heads_KV] fp16-bit
bool two_conv_attention_prefill_f16_kvi8_cl(
  const uint16_t *Q_host, const int8_t *K_i8_host, const int8_t *V_i8_host,
  const uint16_t *K_scale_host, const uint16_t *V_scale_host, uint16_t *O_host,
  unsigned int M, unsigned int N_kv, unsigned int num_heads_Q,
  unsigned int num_heads_KV, unsigned int head_dim, bool causal,
  bool svm_inputs = false);

/// image2d_from_buffer variant: Q/K/V viewed as RGBA UINT32 image2d
/// (16 bytes = 8 halves per texel), 8x fewer memory transactions vs
/// scalar half loads. Requires head_dim, HD_Q, HD_KV all multiples of 8.
/// SVM inputs not supported in this variant (image2d_from_buffer needs
/// cl_mem; the wrapper copies host-host to scratch cl_mem first).
bool two_conv_attention_prefill_f16_img_cl(
  const uint16_t *Q_host, const uint16_t *K_host, const uint16_t *V_host,
  uint16_t *O_host, unsigned int M, unsigned int N_kv, unsigned int num_heads_Q,
  unsigned int num_heads_KV, unsigned int head_dim, bool causal);

/// §3.8 OHWI K-cache variant of two_conv_attention_prefill_f16_cl.
/// Same three-kernel pipeline but K is laid out as [H_kv, S_max, d]
/// (per-head contiguous, paper's "convolution weight" form) rather
/// than the default row-major [N_kv, H_kv * d]. V is still in concat
/// layout — only K1 (qk_matmul_f16_ohwi) is replaced; K2 (softmax)
/// and K3 (sv_matmul_f16) are reused unchanged.
///
///   K_host: per-batch base of the OHWI cache, i.e.
///           &cache_key[batch][0][0][0]; total size H_kv*S_max*d halves.
///   max_seq_len: the allocated S_max (head stride in the OHWI layout).
///                The kernel only reads N_kv rows.
///
/// Opt-in: the caller selects this entry point explicitly (the GPU MHA path
/// with an OHWI-laid-out KV cache). Unlike the concat _f16_cl wrapper, this
/// path has no "force-broken" gate.
bool two_conv_attention_prefill_f16_ohwi_cl(
  const uint16_t *Q_host, const uint16_t *K_host, const uint16_t *V_host,
  uint16_t *O_host, unsigned int M, unsigned int N_kv, unsigned int num_heads_Q,
  unsigned int num_heads_KV, unsigned int head_dim, unsigned int max_seq_len,
  bool causal, bool svm_inputs = false,
  unsigned int local_window = 0); // >0: sliding-window mask (n+W <= q_pos)

/// §3.8 FULL-OHWI variant: K is OHWI [H_kv, S_max, d] AND V is OHWI-
/// reversed [H_kv, d, S_max]. Same three-kernel pipeline as the
/// half-OHWI variant; only K3 (sv_matmul) changes — uses
/// sv_matmul_f16_ohwi which reads V at stride-1 per n (cache-
/// friendly across the reduction). Caller must scatter both K and V
/// caches into their respective OHWI layouts. Same SVM input
/// semantics as _ohwi_cl.
bool two_conv_attention_prefill_f16_ohwi_full_cl(
  const uint16_t *Q_host, const uint16_t *K_host, const uint16_t *V_host,
  uint16_t *O_host, unsigned int M, unsigned int N_kv, unsigned int num_heads_Q,
  unsigned int num_heads_KV, unsigned int head_dim, unsigned int max_seq_len,
  bool causal, bool svm_inputs = false,
  unsigned int local_window = 0); // >0: sliding-window mask (n+W <= q_pos)

/// §3.8 + paper-style image2d_from_buffer V variant. Same pipeline as
/// _ohwi_full_cl but the SV kernel reads V via image2d_from_buffer
/// (sv_matmul_f16_ohwi_img), exploiting the same texture-cache pattern
/// that gives v8c FC 87% of Adreno 830 peak. V must already be in
/// OHWI-reversed [H_kv, d, S_max] layout in a regular cl_mem (NOT SVM
/// — image2d_from_buffer requires a cl_mem-backed buffer). Q and K
/// are still SVM (qk_matmul_f16_ohwi unchanged); only V uses image2d.
///
///   V_buf_ohwi: cl_mem holding the per-batch OHWI-reversed V slab,
///               H_kv * d * max_seq_len halves, row_pitch = max_seq_len.
bool two_conv_attention_prefill_f16_ohwi_img_cl(
  const uint16_t *Q_svm, const uint16_t *K_svm, cl_mem V_buf_ohwi,
  uint16_t *O_svm, unsigned int M, unsigned int N_kv, unsigned int num_heads_Q,
  unsigned int num_heads_KV, unsigned int head_dim, unsigned int max_seq_len,
  bool causal);

/// Variant of _ohwi_img_cl that takes a pre-built cl_mem image2d view
/// over the OHWI-reversed V buffer. Saves one clCreateImage per call
/// when the caller can cache the view (e.g. once per layer at load).
bool two_conv_attention_prefill_f16_ohwi_img_view_cl(
  const uint16_t *Q_svm, const uint16_t *K_svm, cl_mem V_image_ohwi,
  uint16_t *O_svm, unsigned int M, unsigned int N_kv, unsigned int num_heads_Q,
  unsigned int num_heads_KV, unsigned int head_dim, unsigned int max_seq_len,
  bool causal);

/// §3.7 + §3.8 full image2d KV: both K and V wrapped as image2d_from_
/// buffer (OHWI K layout O=cache_size,I=d_h and OHWI-reversed V).
/// qk_matmul_f16_ohwi_img kernel reads K via texel pack (8 halves of d
/// per texel); sv_matmul_f16_ohwi_img reads V via texel pack (8 halves
/// of n per texel). Q stays SVM (scalar half loads in both kernels).
///
///   K_image_ohwi: image2d over OHWI K cl_mem.
///     width = d_h/8 texels, height = H_kv * S_max,
///     row_pitch = d_h * sizeof(half).
///   V_image_ohwi: image2d over OHWI-reversed V cl_mem (as in
///                 _img_view_cl).
///   q_clmem/o_clmem (static GPU_CLMEM residency): bind Q/O as device cl_mem
///   (the tensors' planner sub-buffers) instead of the SVM pointers.
bool two_conv_attention_prefill_f16_ohwi_kvimg_view_cl(
  const uint16_t *Q_svm, cl_mem K_image_ohwi, cl_mem V_image_ohwi,
  uint16_t *O_svm, unsigned int M, unsigned int N_kv, unsigned int num_heads_Q,
  unsigned int num_heads_KV, unsigned int head_dim, unsigned int max_seq_len,
  bool causal, float attn_softcap = 0.0f, // Gemma2-style QK soft-cap (0=off)
  void *q_clmem = nullptr, void *o_clmem = nullptr,
  unsigned int local_window = 0); // >0: sliding-window mask (n+W <= q_pos)

/// Fused single-kernel attention over the SAME two OHWI images as
/// _ohwi_kvimg_view_cl (K image [H_kv,S_max,d], reversed-V image
/// [H_kv,d,S_max]). One workgroup per (head_q, query-row m) does QK +
/// full-row softmax + S·V in-kernel with the score row in LDS, so the
/// [H,M,N_kv] scores tensor is NEVER written to DRAM (kills the 3-kernel
/// path's scores round-trip + L2 thrash) and 3 enqueues collapse to 1.
/// Q stays SVM; O written SVM. Adreno image path only (read_imageui).
/// Constraints: max_seq_len <= 1024, head_dim <= 128, both %8==0.
/// Selected by the caller; NNTR_FLASH_IMG=1 forces it on where available.
bool fused_row_attention_f16_ohwi_img_cl(
  const uint16_t *Q_svm, cl_mem K_image_ohwi, cl_mem V_image_ohwi,
  uint16_t *O_svm, unsigned int M, unsigned int N_kv, unsigned int num_heads_Q,
  unsigned int num_heads_KV, unsigned int head_dim, unsigned int max_seq_len,
  bool causal);

/// Single-kernel flash-attention prefill (paper §3.6 fusion +
/// Dao et al. 2022 online softmax). Replaces the three-kernel
/// two_conv_attention pipeline with one kernel that does QK · softmax
/// · V·S inline using local-memory K/V tiles + per-row online softmax
/// accumulators. Eliminates the global scores[H, M, N_kv] tensor
/// (~6.7 MB bandwidth per prefill on Qwen3-0.6B at S=282) and avoids
/// the VGPR spill that makes the 3-kernel path slower than CPU.
///
///   Q: [M, num_heads_Q * head_dim] fp16 row-major
///   K: [N_kv, num_heads_KV * head_dim] fp16
///   V: [N_kv, num_heads_KV * head_dim] fp16
///   O: [M, num_heads_Q * head_dim] fp16
/// GQA is resolved inside the kernel: head_kv = head_q / (num_heads_Q
/// / num_heads_KV).
///
/// Returns false on a shape mismatch or when the operands are not
/// device-resident, so a caller can fall back to the CPU path.
///
/// Second stage: real online-softmax body. K may be in OHWI layout
/// (K[head_kv*max_seq_len*d + n*d + x], the KV-mirror form)
/// or pure concat ([N_kv, HD_KV]); pass max_seq_len = the OHWI S_max
/// row-stride when K is OHWI, or 0 when K is pure concat. V is always
/// concat ([N_kv, HD_KV]); Q and O are always concat ([*, HD_Q]). This
/// matches the buffers fed by two_conv_attention_prefill_f16_ohwi_cl so
/// the flash path is bit-comparable to the 3-kernel baseline it replaces.
/// Returns false on a shape mismatch, on a head dimension no selected
/// variant can tile, or when the operands are not device-resident; the
/// caller then falls back to the staged three-kernel path.
bool flash_attention_prefill_f16_cl(
  const uint16_t *Q_host, const uint16_t *K_host, const uint16_t *V_host,
  uint16_t *O_host, unsigned int M, unsigned int N_kv, unsigned int num_heads_Q,
  unsigned int num_heads_KV, unsigned int head_dim, unsigned int max_seq_len,
  bool causal, bool svm_inputs = false, float attn_softcap = 0.0f,
  unsigned int local_window = 0u);

/// Flash-decoding (split-KV) for M=1 decode: KV axis split into chunks so
/// gws = num_heads_Q * n_chunks workgroups (restores parallelism the single
/// decode query starves on the blockq/coop_vec path). Two passes (partial +
/// reduce) via cl_mem partial buffers. SVM Q/K/V/O. Gemma4 only (softcap<=0).
/// Returns false on shape mismatch / softcap>0; caller falls back to flash.
bool flash_decode_f16_cl(const uint16_t *Q_host, const uint16_t *K_host,
                         const uint16_t *V_host, uint16_t *O_host,
                         unsigned int N_kv, unsigned int num_heads_Q,
                         unsigned int num_heads_KV, unsigned int head_dim,
                         unsigned int max_seq_len, bool svm_inputs = false,
                         float attn_softcap = 0.0f,
                         unsigned int local_window = 0u);

/**
 * @brief Pre-build the rope/scatter/copy kernel PROGRAM (rope_inplace source,
 *        file-local to attention_kernels.cpp) on the given context. Called from
 *        ClContext::initAttentionClKernels so the ~50ms first-use program build
 *        lands at model load, not inside the first timed prefill. One kernel
 *        suffices: the program cache makes the siblings free.
 */
void attention_prewarm_programs(ClContext &cc);

/**
 * @brief Create a cl_mem OHWI mirror buffer + image2d view for the K (is_v=0)
 *        or V (is_v=1) cache, enabling the Adreno image attention path. The
 *        mirror is filled by k_scatter_ohwi_cl / v_scatter_ohwi_t_cl.
 * @return true on success (*out_buf, *out_image set); false on failure.
 */
bool create_ohwi_kv_mirror(bool is_v, unsigned int num_heads_KV,
                           unsigned int head_dim, unsigned int max_S,
                           cl_mem *out_buf, cl_mem *out_image);

/**
 * @brief Release a cl_mem (buffer or image) created by create_ohwi_kv_mirror.
 *        Taken as void* so callers (e.g. the CausalLM layers) need not link
 *        OpenCL directly — the cast + clReleaseMemObject happen here, inside
 *        libnntrainer. No-op on nullptr.
 */
void release_cl_mem(void *mem);

/**
 * @brief Inverse gathers: OHWI mirror -> concat SVM cache slice rows
 *        [position, position+M). Boundary sync for the NNTR_MHA_CLMEM mode
 *        (the prefill window keeps the mirrors as the only KV store; host
 *        decode/save read the concat SVM slab). dst_svm = the slab slice
 *        base, already offset to `position`. drain=true clFinishes so the
 *        host may read dst immediately.
 */
bool k_gather_ohwi_cl(cl_mem src_buf, uint16_t *dst_svm, unsigned int M,
                      unsigned int num_heads_KV, unsigned int head_dim,
                      unsigned int max_S, unsigned int position,
                      bool drain = true);
bool v_gather_ohwi_t_cl(cl_mem src_buf, uint16_t *dst_svm, unsigned int M,
                        unsigned int num_heads_KV, unsigned int head_dim,
                        unsigned int max_S, unsigned int position,
                        bool drain = true);

/**
 * @brief Create a TIGHT-stride V image2d view (pitch = S*2 bytes, width = S/8
 *        texels) over an existing full-capacity V mirror buffer. The V image
 *        pitch is the texture-cache lever (S_max 2048 -> tight cuts
 *        sv_matmul ~63 -> ~41ms M=843); the sv kernels address V purely via
 *        image coordinates so only the scatter stride must match. *S_inout is
 *        rounded UP to the device image pitch alignment (8 *
 *        CL_DEVICE_IMAGE_PITCH_ALIGNMENT halves; Adreno: multiples of 256)
 *        and returns the stride actually used — the caller must scatter at
 *        that stride and check it still fits the buffer. void* handles so
 *        CausalLM layers need not link OpenCL.
 * @return true on success (*out_image set); false on failure.
 */
bool create_ohwi_v_image_view(void *v_buf, unsigned int num_heads_KV,
                              unsigned int head_dim, unsigned int *S_inout,
                              void **out_image);

/**
 * @brief Scatter this step's K (SVM concat [M, hKV, d]) into the OHWI K mirror
 *        buffer at row `position`. src_clmem: read the source from a device
 *        cl_mem staging temp instead of the SVM pointer (kernel-chain path —
 *        no host drain needed between the producer kernel and this scatter).
 */
/// src_off: when nonzero, the current token's rotated K is read
/// from src[src_off + ..] so the scatter SOURCE can be a STABLE base handle
/// (cache_key) at a SCALAR per-token row offset (recordable) instead of an
/// offset-baked SVM slice pointer. `position` still offsets only the DEST
/// mirror row. Default 0 == byte-identical (src points at the token).
bool k_scatter_ohwi_cl(const uint16_t *src_svm, cl_mem dst_buf, unsigned int M,
                       unsigned int num_heads_KV, unsigned int head_dim,
                       unsigned int max_S, unsigned int position,
                       void *src_clmem = nullptr, unsigned int src_off = 0);

/**
 * @brief Scatter this step's V (SVM concat [M, hKV, d]) into the reversed-OHWI
 *        V mirror buffer at column `position`. src_clmem: same semantics as
 *        k_scatter_ohwi_cl.
 */
bool v_scatter_ohwi_t_cl(const uint16_t *src_svm, cl_mem dst_buf,
                         unsigned int M, unsigned int num_heads_KV,
                         unsigned int head_dim, unsigned int max_S,
                         unsigned int position, void *src_clmem = nullptr,
                         unsigned int src_off = 0);

} // namespace nntrainer
#endif /* __ATTENTION_KERNELS_H__ */
