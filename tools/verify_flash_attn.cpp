// SPDX-License-Identifier: Apache-2.0
/**
 * @file   verify_flash_attn.cpp
 * @brief  Standalone, on-device correctness check for
 *         nntr_htp_bridge_flash_attn (ggml-hexagon's nntr-htp-bridge.cpp),
 *         isolated from the rest of nntrainer/CausalLM. Builds tiny Q/K/V/
 *         mask tensors with known values, runs them through the DSP bridge,
 *         and compares against a plain CPU reference computed the same way
 *         mha_core.cpp's compute_kcaches_fp32_reference / softmax /
 *         compute_vcache_fp32_transposed_reference do (naive triple loop,
 *         not the DSP's fused kernel) - same idea as tools/verify_qkv_batch.
 *
 * head_dim is 128 (not a small number) deliberately: it must stay a multiple
 * of 64 to exercise the HMX fast path (see
 * ggml_hexagon_supported_flash_attn_ext's callers, htp/flash-attn-ops.c's
 * op_flash_attn_ext), which is what the real model's head_dim=128 actually
 * takes - a head_dim=4 test would only exercise the HVX fallback and could
 * miss an HMX-specific bug.
 *
 * Build (on the device, or cross-compiled with the NDK):
 *   ${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ \
 *     --target=aarch64-linux-android30 -std=c++17 -O2 \
 *     -o verify_flash_attn tools/verify_flash_attn.cpp -ldl
 *   adb push verify_flash_attn /data/local/tmp/nntrainer/causallm/
 *   adb shell "cd /data/local/tmp/nntrainer/causallm && \
 *     LD_LIBRARY_PATH=. ./verify_flash_attn"
 * (libggml-hexagon.so and libcdsprpc.so must already be reachable there,
 * which they are on any device this branch has been installed to.)
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <dlfcn.h>

namespace {

uint16_t f32_to_f16_bits(float f) {
  // Reference software conversion (round-to-nearest-even is not needed here -
  // exactness on the exact boundary values used below is enough), used only
  // to build the test's Q/K/V/mask inputs.
  uint32_t x;
  memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;
  if (exp <= 0) {
    return (uint16_t)sign; // flush to zero - fine for this test's magnitudes
  }
  if (exp >= 0x1F) {
    return (uint16_t)(sign | 0x7C00u); // overflow -> inf
  }
  return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

float f16_bits_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      // Subnormal - not needed for this test's values, handle simply.
      float m = (float)mant / 1024.0f;
      float val = m * powf(2.0f, -14.0f);
      memcpy(&bits, &val, 4);
      bits |= sign;
    }
  } else if (exp == 0x1F) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

using RegisterPoolFn = int (*)(const void *, size_t);
using FlashAttnFn = int (*)(const void *, const void *, const void *,
                            const void *, void *, unsigned int, unsigned int,
                            unsigned int, unsigned int, unsigned int, float,
                            int, int);
using RpcmemAllocFn = void *(*)(int, uint32_t, int);
using RpcmemFreeFn = void (*)(void *);

} // namespace

int main(int argc, char **argv) {
  // --- Shapes: default small (hand-verifiable), or the real Qwen3-0.6B
  // config (16 heads / 8 kv heads / head_dim 128) at real prefill length if
  // argv[1]=="full" - the small case passing does not rule out a
  // scale-dependent bug (block tiling, VTCM sizing) that only shows up at
  // real n_tokens/n_kv. ---
  const std::string mode = argc > 1 ? argv[1] : "small";
  const bool full_scale = mode == "full";
  // "chunked" reproduces the real CausalLM call pattern: token 0 is forwarded
  // alone first (M=1, decode-shaped, never hits mha_core's flash-attn gate
  // since it requires step_size>1), THEN the remaining tokens arrive as one
  // bulk step with cache_from=1 - so n_kv (= cache_to) is 1 more than
  // n_tokens (= step_size), not equal to it. Neither the "small" nor "full"
  // case above tests n_kv != n_tokens at all.
  const bool chunked = mode == "chunked";
  const unsigned int head_dim = 128;
  const unsigned int n_head = (full_scale || chunked) ? 16 : 2;
  const unsigned int n_head_kv = (full_scale || chunked) ? 8 : 1;
  const unsigned int gqa_size = n_head / n_head_kv;
  const unsigned int cache_from = chunked ? 1 : 0;
  const unsigned int n_tokens = chunked ? 18 : (full_scale ? 308 : 3);
  const unsigned int n_kv = cache_from + n_tokens; // cache_to
  const float scale = 1.0f / std::sqrtf((float)head_dim);

  // --- Load the bridge, exactly like KVCacheManager / MHACoreLayer do ---
  void *rpc = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_GLOBAL);
  if (!rpc) {
    fprintf(stderr, "dlopen(libcdsprpc.so) failed: %s\n", dlerror());
    return 1;
  }
  auto rpcmem_alloc = (RpcmemAllocFn)dlsym(rpc, "rpcmem_alloc");
  auto rpcmem_free = (RpcmemFreeFn)dlsym(rpc, "rpcmem_free");
  if (!rpcmem_alloc || !rpcmem_free) {
    fprintf(stderr, "dlsym(rpcmem_alloc/free) failed\n");
    return 1;
  }

  void *bridge = dlopen("libggml-hexagon.so", RTLD_NOW | RTLD_GLOBAL);
  if (!bridge) {
    fprintf(stderr, "dlopen(libggml-hexagon.so) failed: %s\n", dlerror());
    return 1;
  }
  auto register_pool =
    (RegisterPoolFn)dlsym(bridge, "nntr_htp_bridge_register_activation_pool");
  auto flash_attn = (FlashAttnFn)dlsym(bridge, "nntr_htp_bridge_flash_attn");
  if (!register_pool || !flash_attn) {
    fprintf(stderr, "dlsym(nntr_htp_bridge_*) failed\n");
    return 1;
  }

  // --- Build inputs. Q/out stay F32 (q_is_fp16=0/out_is_fp16=0) so the CPU
  // reference below can be compared bit-for-bit-free of an extra fp16
  // round-trip; K/V/mask are F16 bit patterns, matching the real KV cache. ---
  std::vector<float> q(n_tokens * n_head * head_dim);
  std::vector<uint16_t> k_bits(n_kv * n_head_kv * head_dim);
  std::vector<uint16_t> v_bits(n_kv * n_head_kv * head_dim);
  std::vector<float> k_ref(n_kv * n_head_kv * head_dim);
  std::vector<float> v_ref(n_kv * n_head_kv * head_dim);

  unsigned int seed = 12345;
  auto next = [&]() {
    seed = seed * 1103515245u + 12345u;
    return (float)((seed >> 8) & 0xFFFF) / 65536.0f - 0.5f;
  };
  for (auto &x : q)
    x = next();
  for (size_t i = 0; i < k_ref.size(); ++i) {
    k_ref[i] = next();
    k_bits[i] = f32_to_f16_bits(k_ref[i]);
  }
  for (size_t i = 0; i < v_ref.size(); ++i) {
    v_ref[i] = next();
    v_bits[i] = f32_to_f16_bits(v_ref[i]);
  }

  std::vector<uint16_t> mask(n_tokens * n_kv);
  for (unsigned int i = 0; i < n_tokens; ++i) {
    const unsigned int valid_to = cache_from + i + 1; // matches mha_core.cpp
    for (unsigned int j = 0; j < n_kv; ++j) {
      mask[i * n_kv + j] = (j < valid_to) ? 0x0000 : 0xFC00;
    }
  }

  std::vector<float> out_dsp(n_tokens * n_head * head_dim, 0.0f);

  // --- K/V must be in a registered rpcmem pool - allocate one and copy in,
  // exactly mirroring KVCacheManager's pooling. ---
  const size_t kv_bytes = k_bits.size() * sizeof(uint16_t);
  void *kv_pool = rpcmem_alloc(25 /*heap*/, 1 /*flags*/, (int)(2 * kv_bytes));
  if (!kv_pool) {
    fprintf(stderr, "rpcmem_alloc failed\n");
    return 1;
  }
  uint8_t *k_dev = (uint8_t *)kv_pool;
  uint8_t *v_dev = (uint8_t *)kv_pool + kv_bytes;
  memcpy(k_dev, k_bits.data(), kv_bytes);
  memcpy(v_dev, v_bits.data(), kv_bytes);
  if (register_pool(kv_pool, 2 * kv_bytes) != 0) {
    fprintf(stderr, "register_activation_pool failed\n");
    return 1;
  }

  int rc = flash_attn(q.data(), k_dev, v_dev, mask.data(), out_dsp.data(),
                      n_tokens, n_head, n_head_kv, head_dim, n_kv, scale,
                      /*q_is_fp16=*/0, /*out_is_fp16=*/0);
  if (rc != 0) {
    fprintf(stderr, "FAIL: nntr_htp_bridge_flash_attn returned %d\n", rc);
    return 1;
  }

  // --- CPU reference: naive triple loop, matching
  // compute_kcaches_fp32_reference / softmax_triangle /
  // compute_vcache_fp32_transposed_reference's math exactly (GQA head
  // mapping h -> h/gqa_size, causal, scale before softmax). ---
  std::vector<float> out_ref(n_tokens * n_head * head_dim, 0.0f);
  std::vector<float> scores(n_kv);
  for (unsigned int t = 0; t < n_tokens; ++t) {
    const unsigned int valid_to = cache_from + t + 1; // matches mha_core.cpp
    for (unsigned int h = 0; h < n_head; ++h) {
      const unsigned int kvh = h / gqa_size;
      const float *qv = q.data() + (t * n_head + h) * head_dim;
      float maxv = -1e30f;
      for (unsigned int j = 0; j < valid_to; ++j) {
        const float *kv = k_ref.data() + (j * n_head_kv + kvh) * head_dim;
        float s = 0.0f;
        for (unsigned int d = 0; d < head_dim; ++d)
          s += qv[d] * kv[d];
        s *= scale;
        scores[j] = s;
        if (s > maxv)
          maxv = s;
      }
      float sum = 0.0f;
      for (unsigned int j = 0; j < valid_to; ++j) {
        scores[j] = expf(scores[j] - maxv);
        sum += scores[j];
      }
      float *outv = out_ref.data() + (t * n_head + h) * head_dim;
      for (unsigned int j = 0; j < valid_to; ++j) {
        const float w = scores[j] / sum;
        const float *vv = v_ref.data() + (j * n_head_kv + kvh) * head_dim;
        for (unsigned int d = 0; d < head_dim; ++d)
          outv[d] += w * vv[d];
      }
    }
  }

  double max_abs_err = 0.0, max_rel_err = 0.0;
  for (size_t i = 0; i < out_ref.size(); ++i) {
    double err = std::fabs((double)out_dsp[i] - (double)out_ref[i]);
    double rel = err / (std::fabs((double)out_ref[i]) + 1e-6);
    if (err > max_abs_err)
      max_abs_err = err;
    if (rel > max_rel_err)
      max_rel_err = rel;
  }

  printf("--- FP32 Q/out path ---\n");
  printf("out_ref[0..3] = %f %f %f %f\n", out_ref[0], out_ref[1], out_ref[2],
         out_ref[3]);
  printf("out_dsp[0..3] = %f %f %f %f\n", out_dsp[0], out_dsp[1], out_dsp[2],
         out_dsp[3]);
  printf("max_abs_err = %g   max_rel_err = %g\n", max_abs_err, max_rel_err);
  bool fp32_pass = max_abs_err <= 0.05;
  printf("%s\n\n", fp32_pass ? "PASS" : "FAIL");

  // --- FP16 Q/out path - this is the path the real model actually takes
  // (query_step/attention_output_step are F16 whenever ENABLE_FP16 is on,
  // independent of the model's own FC/GEMM activation dtype - see mha_core.
  // cpp's gate). The FP32 path above passing does NOT confirm this one. ---
  std::vector<uint16_t> q_bits(q.size());
  for (size_t i = 0; i < q.size(); ++i)
    q_bits[i] = f32_to_f16_bits(q[i]);
  std::vector<uint16_t> out_dsp_fp16(out_dsp.size(), 0);

  rc = flash_attn(q_bits.data(), k_dev, v_dev, mask.data(), out_dsp_fp16.data(),
                  n_tokens, n_head, n_head_kv, head_dim, n_kv, scale,
                  /*q_is_fp16=*/1, /*out_is_fp16=*/1);
  if (rc != 0) {
    fprintf(stderr, "FAIL: nntr_htp_bridge_flash_attn (fp16) returned %d\n",
            rc);
    rpcmem_free(kv_pool);
    return 1;
  }

  double max_abs_err16 = 0.0, max_rel_err16 = 0.0;
  std::vector<float> out_dsp_fp16_as_f32(out_dsp.size());
  for (size_t i = 0; i < out_ref.size(); ++i) {
    out_dsp_fp16_as_f32[i] = f16_bits_to_f32(out_dsp_fp16[i]);
    double err = std::fabs((double)out_dsp_fp16_as_f32[i] - (double)out_ref[i]);
    double rel = err / (std::fabs((double)out_ref[i]) + 1e-6);
    if (err > max_abs_err16)
      max_abs_err16 = err;
    if (rel > max_rel_err16)
      max_rel_err16 = rel;
  }
  printf("--- FP16 Q/out path (what the real model uses) ---\n");
  printf("out_ref[0..3]      = %f %f %f %f\n", out_ref[0], out_ref[1],
         out_ref[2], out_ref[3]);
  printf("out_dsp_fp16[0..3] = %f %f %f %f\n", out_dsp_fp16_as_f32[0],
         out_dsp_fp16_as_f32[1], out_dsp_fp16_as_f32[2],
         out_dsp_fp16_as_f32[3]);
  printf("max_abs_err = %g   max_rel_err = %g\n", max_abs_err16,
         max_rel_err16);
  bool fp16_pass = max_abs_err16 <= 0.05;
  printf("%s\n", fp16_pass ? "PASS" : "FAIL");

  rpcmem_free(kv_pool);

  if (!fp32_pass || !fp16_pass) {
    printf("\nOVERALL: FAIL\n");
    return 1;
  }
  printf("\nOVERALL: PASS\n");
  return 0;
}
