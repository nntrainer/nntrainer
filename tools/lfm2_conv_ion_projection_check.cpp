// SPDX-License-Identifier: Apache-2.0
/**
 * @file lfm2_conv_ion_projection_check.cpp
 * @brief Compare copied and retained affine FC output after real DSP writes.
 * @author NNTrainer contributors
 * @bug Synthetic FC weights do not measure concurrent expert I/O.
 */
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>
#include <vector>

#include <htp_backend.h>
#include <htp_wh_layout.h>

// Use the actual candidate convolution and arithmetic pragmas.
#include "../Applications/CausalLM/layers/conv_block_layer.cpp"

namespace {
constexpr unsigned C = 2048, N = 3u * C;
void check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}

struct Weight {
  std::vector<uint8_t> bytes;
  Weight() : bytes(nntrainer::whBytes(C, N) + 12u * N) {
    std::vector<int8_t> codes(size_t(C) * N);
    std::vector<float> sums(N, 0.0f), scales(N);
    std::vector<int32_t> offsets(N);
    for (unsigned n = 0; n < N; ++n) {
      scales[n] = float(1u + n % 7u) * 0.001f;
      offsets[n] = int(n % 5u) - 2;
      for (unsigned k = 0; k < C; ++k) {
        const int value = int((k * 13u + n * 7u) % 11u) - 5;
        codes[size_t(k) * N + n] = int8_t(value);
        sums[n] += float(value);
      }
    }
    nntrainer::whPack(codes.data(), C, N, bytes.data());
    const size_t tail = nntrainer::whBytes(C, N);
    std::memcpy(bytes.data() + tail, scales.data(), 4u * N);
    std::memcpy(bytes.data() + tail + 4u * N, sums.data(), 4u * N);
    std::memcpy(bytes.data() + tail + 8u * N, offsets.data(), 4u * N);
  }
  float *scale() {
    return reinterpret_cast<float *>(bytes.data() + nntrainer::whBytes(C, N));
  }
};

void gateConvState(const float *p, float *g, const float *w, float *out,
                   float *state, unsigned rows) {
  nntrainer::ThreadManager::Global().parallel_for(
    0, size_t(rows), [&](size_t r) {
      const float *a = p + r * N;
      const float *c = a + 2u * C;
      float *dst = g + r * C;
      for (unsigned j = 0; j < C; ++j)
        dst[j] = a[j] * c[j];
    });
  check(causallm::lfmConvPostGatePrefill(g, w, p, out, rows, C),
        "fused conv guard");
  std::memcpy(state, g + size_t(rows - 2u) * C, 2u * C * sizeof(float));
}

void exact(nntrainer::ComputeOps *ops, Weight &weight, unsigned rows) {
  const size_t mid = size_t(rows) * C, count = 3u * mid;
  std::vector<float> input(mid), projection(count), retained(count);
  std::vector<float> gated(mid), output(mid), state(2u * C);
  std::vector<float> ref_gated(mid), ref_output(mid), ref_state(2u * C);
  std::vector<float> conv_weights(3u * C);
  for (size_t i = 0; i < conv_weights.size(); ++i)
    conv_weights[i] = float(int(i % 17u) - 8) / 128.0f;
  auto to_fd =
    reinterpret_cast<int (*)(void *)>(dlsym(RTLD_DEFAULT, "rpcmem_to_fd"));
  check(to_fd != nullptr, "rpcmem_to_fd unavailable");
  const bool force_host =
    std::getenv("NNTR_HTP_QKV_TO_HOST") &&
    std::strcmp(std::getenv("NNTR_HTP_QKV_TO_HOST"), "1") == 0;
  for (unsigned generation = 0; generation < 3; ++generation) {
    for (size_t i = 0; i < input.size(); ++i)
      input[i] = float(int((i * 17u + generation * 41u) % 257u) - 128) /
                 float(31u + generation);
    ops->gemm_qs4cx_accel_fp32(weight.bytes.data(), weight.scale(),
                               input.data(), projection.data(), rows, N, C);
    gateConvState(projection.data(), ref_gated.data(), conv_weights.data(),
                  ref_output.data(), ref_state.data(), rows);
    // Poison between the sampled windows on every call. The check fails if
    // a CPU reader accidentally consumes a partially materialized tensor.
    std::fill(retained.begin(), retained.end(), -9876.0f);
    ops->gemm_q4_0_batch_norm_fp32(
      {weight.bytes.data()}, {weight.scale()}, input.data(), {retained.data()},
      rows, {N}, C, nullptr, {0u}, nullptr, 0.0f, nullptr, 0u, 0u, false);
    const auto *view = static_cast<const float *>(
      ops->host_view(retained.data(), count * sizeof(float)));
    check(view && view != retained.data() &&
            to_fd(const_cast<float *>(view)) >= 0,
          "retained result is not the live ION view");
    check(std::memcmp(view, projection.data(), count * sizeof(float)) == 0,
          "DSP affine copied/retained projection differs");
    if (!force_host)
      check(std::memcmp(retained.data(), projection.data(),
                        count * sizeof(float)) != 0,
            "host unexpectedly materialized; copy elimination untested");
    gateConvState(view, gated.data(), conv_weights.data(), output.data(),
                  state.data(), rows);
    check(
      std::memcmp(gated.data(), ref_gated.data(), mid * sizeof(float)) == 0 &&
        std::memcmp(output.data(), ref_output.data(), mid * sizeof(float)) ==
          0 &&
        std::memcmp(state.data(), ref_state.data(),
                    state.size() * sizeof(float)) == 0,
      "retained view convolution or state differs");
    // Exercise the supported materialization fallback before a later call
    // is allowed to reuse the staging buffer.
    ops->sync_to_host(retained.data(), count * sizeof(float));
    check(std::memcmp(retained.data(), projection.data(),
                      count * sizeof(float)) == 0,
          "sync_to_host failed to materialize all output bytes");
    std::printf("CONV_ION_DSP M=%u generation=%u exact=1 state_exact=1 "
                "sync_exact=1 force_host=%u PASS\n",
                rows, generation, unsigned(force_host));
  }
}
} // namespace

int main() {
  try {
    check(nntrainer::HtpBackend::global().enabled(), "HTP unavailable");
    auto *ops = nntrainer::get_htp_ops();
    check(ops && ops->supports_gemm_q4_0_batch_norm_fp32() &&
            ops->supports_gemm_qs4cx_accel_fp32(),
          "affine FC API unavailable");
    Weight weight;
    check(ops->set_fc_wh_source(weight.bytes.data(), C, N, true),
          "affine WH source rejected");
    for (unsigned rows : {16u, 17u, 64u, 1024u, 1920u})
      exact(ops, weight, rows);
    std::puts("CONV_ION_DSP 15 cases PASS");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "CONV_ION_DSP FAIL: %s\n", e.what());
    return 1;
  }
}
