// SPDX-License-Identifier: Apache-2.0
/**
 * @file hadamard_moe_bench.cpp
 * @brief Fixed-routing packed-2-bit MoE and 4-bit dense-chunk benchmarks.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include "nntr_hvx.h"
#include <AEEStdErr.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <remote.h>
#include <vector>

static void require(int rc, const char *op) {
  if (rc) {
    std::fprintf(stderr, "%s failed: 0x%x\n", op, (unsigned)rc);
    std::exit(1);
  }
}
static uint32_t rng = 0x731ab923;
static uint32_t next_random() {
  rng ^= rng << 13;
  rng ^= rng >> 17;
  rng ^= rng << 5;
  return rng;
}
static uint32_t page(uint32_t n) { return (n + 4095u) & ~4095u; }

int main(int argc, char **argv) {
  // Arguments: tokens, repeats, weight bits (2=128 experts top8;
  // 4=three synthetic chunks, each 704, all active). PR #4416's actual
  // shared FFN uses two chunks of 1056. No model weights are used here.
  const uint32_t M =
    argc > 1 ? (uint32_t)std::strtoul(argv[1], nullptr, 10) : 1;
  const uint32_t reps =
    argc > 2 ? (uint32_t)std::strtoul(argv[2], nullptr, 10) : 12;
  const uint32_t bits =
    argc > 3 ? (uint32_t)std::strtoul(argv[3], nullptr, 10) : 2;
  const uint32_t norm =
    argc > 4 ? (uint32_t)std::strtoul(argv[4], nullptr, 10) : 0u;
  if (norm > 1u || !M || M > 2048 || !reps || reps > 200 ||
      (bits != 2 && bits != 4))
    return 2;
  const uint32_t K = 2816, I = 704, N = 2816, E = bits == 2 ? 128 : 3;
  const uint32_t top = bits == 2 ? 8 : 3;
  struct remote_rpc_control_unsigned_module u = {CDSP_DOMAIN_ID, 1};
  require(remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &u, sizeof(u)),
          "unsigned");
  remote_rpc_thread_params thread_params = {CDSP_DOMAIN_ID, -1, 64 * 1024};
  require(remote_session_control(FASTRPC_THREAD_PARAMS, &thread_params,
                                 sizeof(thread_params)),
          "thread stack");
  remote_handle64 session;
  require(nntr_hvx_open(nntr_hvx_URI "&_dom=cdsp", &session), "open");
  auto alloc =
    (void *(*)(int, uint32_t, int))dlsym(RTLD_DEFAULT, "rpcmem_alloc");
  auto rfree = (void (*)(void *))dlsym(RTLD_DEFAULT, "rpcmem_free");
  auto to_fd = (int (*)(void *))dlsym(RTLD_DEFAULT, "rpcmem_to_fd");
  auto map = (int (*)(int, int, void *, int, size_t, int))dlsym(RTLD_DEFAULT,
                                                                "fastrpc_mmap");
  auto unmap =
    (int (*)(int, int, void *, size_t))dlsym(RTLD_DEFAULT, "fastrpc_munmap");
  if (!alloc || !rfree || !to_fd || !map || !unmap)
    return 3;
  const uint32_t gbytes = K * 2u * I * bits / 8u, dbytes = I * N * bits / 8u;
  const uint32_t bytes = E * (page(gbytes) + page(dbytes));
  uint8_t *arena_ptr = (uint8_t *)alloc(25, 0u, (int)bytes);
  if (!arena_ptr)
    return 4;
  const int fd = to_fd(arena_ptr);
  if (fd < 0)
    return 5;
  require(map(CDSP_DOMAIN_ID, fd, arena_ptr, 0, bytes, FASTRPC_MAP_FD), "map");
  uint32_t arena;
  require(nntr_hvx_arena_attach(session, fd, bytes, &arena), "arena_attach");
  std::vector<uint32_t> gu(E), dn(E), count(E, 0), index;
  uint32_t offset = 0;
  for (uint32_t e = 0; e < E; ++e) {
    for (uint32_t projection = 0; projection < 2; ++projection) {
      const uint32_t k = projection ? I : K, n = projection ? N : 2u * I;
      const uint32_t weight_bytes = k * n * bits / 8u;
      const uint32_t per_byte = 8u / bits, tile_bytes = 1024u / per_byte;
      const int8_t palette[4] = {-7, 0, 7, 0};
      std::vector<int32_t> colsum(n, 0);
      std::vector<float> scale(n, 0.01f), bias(n, 0.0f);
      for (uint32_t b = 0; b < weight_bytes; ++b) {
        const uint8_t v = (uint8_t)next_random();
        arena_ptr[offset + b] = v;
        const uint32_t nt = (b / tile_bytes) % (n / 32u);
        for (uint32_t j = 0; j < per_byte; ++j) {
          const uint32_t slot = (b % tile_bytes) * per_byte + j;
          const uint32_t code = (v >> (j * bits)) & ((1u << bits) - 1u);
          colsum[nt * 32u + (slot % 256u) / 8u] +=
            bits == 2 ? palette[code] : (int)(code ^ 8u) - 8;
        }
      }
      uint32_t *handle = projection ? &dn[e] : &gu[e];
      if (bits == 2)
        require(nntr_hvx_weight_register_u2i4_arena(
                  session, k, n, arena, offset, palette, 4, scale.data(), n,
                  colsum.data(), n, bias.data(), n, handle),
                "register2");
      else
        require(nntr_hvx_weight_register_u8i4_arena(
                  session, k, n, arena, offset, scale.data(), n, colsum.data(),
                  n, bias.data(), n, handle),
                "register4");
      offset += page(weight_bytes);
    }
    for (uint32_t m = 0; m < M; ++m)
      for (uint32_t j = 0; j < top; ++j)
        if ((bits == 4 ? j : (m * 17u + j * 13u) % E) == e) {
          index.push_back(m);
          ++count[e];
        }
  }
  std::vector<float> weights(index.size(), bits == 2 ? 1.0f / top : 1.0f);
  std::vector<float> input((size_t)M * K), out((size_t)M * N), scalar, vector,
    hmx, up, down, hmx_up, hmx_down;
  for (auto &v : input)
    v = ((int)(next_random() % 20001u) - 10000) / 10000.0f;
  const std::vector<float> input_copy = input;
  std::vector<float> gamma(K);
  for (uint32_t k = 0; k < K; ++k)
    gamma[k] = 1.0f + 0.1f * std::sin((float)k);
  std::printf("CONFIG pre_norm=%u\n", norm);
  // Same input/routing/weights for all variants. Valid text is not expected.
  const uint32_t common =
    0x000f03e1u | (3u << 22); // GEMV, GeGLU, VTCM, four DMA queues
  const uint32_t variant[] = {0u,          0x04000000u, 0x08000000u,
                              0x0c000000u, 0x1c000000u, 0x24000000u,
                              0x28000000u, 0x2c000000u};
  auto select = [&](uint32_t v) {
    uint32_t applied = 0;
    require(nntr_hvx_moe_set_opts(session, common | variant[v], &applied),
            "opts");
    if (applied != (common | variant[v]))
      require(1, "opts echo");
  };
  auto run = [&](bool timed, uint32_t *stage) {
    if (norm) {
      if (timed)
        require(nntr_hvx_mm_u8i4_moe_layer_norm_timed(
                  session, M, K, I, N, 1u, 1e-6f, gamma.data(), K, nullptr, 0,
                  gu.data(), E, dn.data(), E, index.data(), index.size(),
                  count.data(), E, weights.data(), weights.size(), input.data(),
                  input.size(), out.data(), out.size(), stage, 31),
                "moe_norm_timed");
      else
        require(nntr_hvx_mm_u8i4_moe_layer_norm(
                  session, M, K, I, N, 1u, 1e-6f, gamma.data(), K, nullptr, 0,
                  gu.data(), E, dn.data(), E, index.data(), index.size(),
                  count.data(), E, weights.data(), weights.size(), input.data(),
                  input.size(), out.data(), out.size()),
                "moe_norm");
      return;
    }
    if (timed)
      require(nntr_hvx_mm_u8i4_moe_layer_timed(
                session, M, K, I, N, 1u, gu.data(), E, dn.data(), E,
                index.data(), index.size(), count.data(), E, weights.data(),
                weights.size(), input.data(), input.size(), out.data(),
                out.size(), stage, 31),
              "moe_timed");
    else
      require(nntr_hvx_mm_u8i4_moe_layer(
                session, M, K, I, N, 1u, gu.data(), E, dn.data(), E,
                index.data(), index.size(), count.data(), E, weights.data(),
                weights.size(), input.data(), input.size(), out.data(),
                out.size()),
              "moe");
  };
  uint32_t stage[31] = {};
  for (uint32_t v = 0; v < 8; ++v) {
    std::fprintf(stderr, "warmup M=%u bits=%u variant=%u\n", M, bits, v);
    select(v);
    run(false, stage);
    if (input != input_copy)
      require(1, "input modified");
    for (float y : out)
      if (!std::isfinite(y))
        require(1, "nonfinite warmup");
    if (v == 1)
      up = out;
    if (v == 2)
      down = out;
    if (v == 5)
      hmx_up = out;
    if (v == 6)
      hmx_down = out;
    if (v == 3)
      vector = out;
    if (v == 4)
      scalar = out;
    if (v == 7)
      hmx = out;
  }
  double err2 = 0, energy = 0, max_abs = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    if (!std::isfinite(scalar[i]) || !std::isfinite(vector[i]))
      require(1, "nonfinite");
    const double d = scalar[i] - vector[i];
    err2 += d * d;
    energy += (double)scalar[i] * scalar[i];
    max_abs = std::max(max_abs, std::fabs(d));
  }
  const double nrmse = std::sqrt(err2 / std::max(energy, 1e-30));
  std::printf("CHECK M=%u bits=%u nrmse=%.9g max_abs=%.9g\n", M, bits, nrmse,
              max_abs);
  // Reordered floating-point sums can cross an 8-bit quantization boundary.
  // This pipeline sanity check allows 0.1% normalized error; the separate
  // rotation check compares every element to a double matrix reference
  // with 3e-5 * (1 + abs(reference)) and checks energy preservation.
  if (nrmse > 1e-3)
    require(1, "scalar/HVX parity");
  auto check_hmx = [&](const char *name, const std::vector<float> &ref,
                       const std::vector<float> &candidate) {
    double error = 0, ref_energy = 0, max_error = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
      const double d = ref[i] - candidate[i];
      error += d * d;
      ref_energy += (double)ref[i] * ref[i];
      max_error = std::max(max_error, std::fabs(d));
    }
    const double rms = std::sqrt(error / std::max(ref_energy, 1e-30));
    std::printf("%s M=%u bits=%u nrmse=%.9g max_abs=%.9g\n", name, M, bits, rms,
                max_error);
    // FP16 rotation is checked separately against FP32 at <=0.2% NRMSE.
    // Dynamic INT8 bin changes amplify that error in this random-weight
    // pipeline, especially the near-cancelling 4-bit dense output. This 2%
    // guard detects broken integration; it is not a model-quality bound.
    if (!std::isfinite(rms) || rms > 0.02)
      require(1, "HMX parity");
  };
  check_hmx("CHECK_HMX_UP", up, hmx_up);
  check_hmx("CHECK_HMX_DOWN", down, hmx_down);
  check_hmx("CHECK_HMX", scalar, hmx);
  std::puts("M,bits,rep,variant,dsp_us,rpc_us,path,quant_us,requant_us,mm_us");
  for (uint32_t rep = 0; rep < reps; ++rep) {
    // Rotate/reverse the order to avoid always measuring the hot variant last.
    for (uint32_t j = 0; j < 8; ++j) {
      const uint32_t v = (rep + (rep % 2 ? 7u - j : j)) % 8u;
      select(v); // The option-setting RPC is outside the measured layer call.
      const auto begin = std::chrono::steady_clock::now();
      run(true, stage);
      const double us = std::chrono::duration<double, std::micro>(
                          std::chrono::steady_clock::now() - begin)
                          .count();
      std::printf("%u,%u,%u,%u,%u,%.3f,%u,%u,%u,%u\n", M, bits, rep, v,
                  stage[0], us, stage[29], stage[1], stage[8], stage[10]);
      std::fflush(stdout);
    }
  }
  for (uint32_t e = 0; e < E; ++e) {
    require(nntr_hvx_weight_release_u8i4(session, gu[e]), "release gu");
    require(nntr_hvx_weight_release_u8i4(session, dn[e]), "release dn");
  }
  require(nntr_hvx_arena_detach(session, arena), "detach");
  require(unmap(CDSP_DOMAIN_ID, fd, arena_ptr, bytes), "unmap");
  rfree(arena_ptr);
  require(nntr_hvx_close(session), "close");
  return 0;
}
