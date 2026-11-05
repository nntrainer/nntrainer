// SPDX-License-Identifier: Apache-2.0
/** @file wh_affine_device_check.cpp
 * @brief Verify affine WH correction on a real DSP against equivalent int4
 * weights, through the generated FastRPC interface.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug See docs/gemma4-tl2-affine-handoff.md for validation limits.
 */
#include <AEEStdErr.h>
#include <remote.h>

#include "nntr_hvx.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

static void check(int rc, const char *operation) {
  if (rc) {
    std::fprintf(stderr, "%s failed: 0x%x\n", operation, unsigned(rc));
    std::exit(1);
  }
}

static void check_moe(remote_handle64 session, unsigned K, unsigned I,
                      unsigned N) {
  uint32_t handles[4]; // affine gate/up and down, then reference
  for (unsigned h = 0; h < 2; ++h) {
    const unsigned wk = h ? I : K, wn = h ? N : 2 * I;
    std::vector<int8_t> codes(wk * wn), reference(codes.size());
    std::vector<float> scale(wn, .001f), bias(wn, 0), affine_bias(2 * wn, 0);
    std::vector<int32_t> sums(wn, 0), reference_sums(wn, 0);
    for (unsigned n = 0; n < wn; ++n) {
      const int offset = int(n % 11) - 5;
      affine_bias[wn + n] = float(offset);
      for (unsigned k = 0; k < wk; ++k) {
        const int code = int((k * 13 + n * 7) % 5) - 2;
        codes[k * wn + n] = int8_t(code);
        reference[k * wn + n] = int8_t(code + offset);
        sums[n] += code;
        reference_sums[n] += code + offset;
      }
    }
    check(nntr_hvx_weight_register_u8i4(session, wk, wn, codes.data(),
                                        codes.size(), scale.data(), wn,
                                        sums.data(), wn, affine_bias.data(),
                                        affine_bias.size(), handles + h),
          "affine MoE registration");
    check(nntr_hvx_weight_register_u8i4(
            session, wk, wn, reference.data(), reference.size(), scale.data(),
            wn, reference_sums.data(), wn, bias.data(), wn, handles + 2 + h),
          "reference MoE registration");
  }
  for (unsigned M : {1u, 2u, 3u, 4u, 64u, 65u, 129u}) {
    std::vector<float> input(M * K), actual(M * N), expected(M * N);
    std::vector<uint32_t> rows(M);
    std::vector<float> routing(M);
    for (unsigned r = 0; r < M; ++r) {
      rows[r] = M - 1 - r;
      routing[r] = .25f * (1 + r % 4);
    }
    for (size_t i = 0; i < input.size(); ++i)
      input[i] = float(int((i * 17) % 127) - 63) / 31.f;
    for (unsigned pass = 0; pass < 3; ++pass) {
      for (unsigned mode = 0; mode < 2; ++mode) {
        uint32_t *h = handles + 2 * mode;
        check(nntr_hvx_mm_u8i4_moe_layer(
                session, M, K, I, N, 1, h, 1, h + 1, 1, rows.data(), M, &M, 1,
                routing.data(), M, input.data(), input.size(),
                mode ? expected.data() : actual.data(), actual.size()),
              "affine MoE matmul");
      }
      for (float f : actual)
        check(!std::isfinite(f), "nonfinite MoE output");
      check(std::memcmp(actual.data(), expected.data(),
                        actual.size() * sizeof(float)),
            "affine MoE/reference mismatch");
    }
    std::printf("AFFINE_MOE_DSP M=%u K=%u I=%u N=%u bit_identical=1\n", M, K, I,
                N);
  }
  for (auto h : handles)
    check(nntr_hvx_weight_release_u8i4(session, h), "MoE release");
}

int main() {
  remote_rpc_thread_params thread_params = {CDSP_DOMAIN_ID, -1, 64 * 1024};
  check(remote_session_control(FASTRPC_THREAD_PARAMS, &thread_params,
                               sizeof(thread_params)),
        "thread params");
  remote_rpc_control_unsigned_module unsigned_pd = {CDSP_DOMAIN_ID, 1};
  check(remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &unsigned_pd,
                               sizeof(unsigned_pd)),
        "unsigned PD");
  remote_handle64 session = 0;
  const std::string uri = std::string(nntr_hvx_URI) + "&_dom=cdsp";
  check(nntr_hvx_open(uri.c_str(), &session), "open");
  for (auto shape : {std::pair<unsigned, unsigned>{96, 128}, {2816, 2816}}) {
    const unsigned K = shape.first, N = shape.second;
    std::vector<int8_t> q(K * N), reference(K * N);
    std::vector<float> scale(N), bias(N, 0), affine_bias(2 * N, 0);
    std::vector<int32_t> sums(N, 0), reference_sums(N, 0);
    for (unsigned n = 0; n < N; ++n) {
      const int offset = int(n % 11) - 5;
      scale[n] = .01f * (1 + n % 7);
      affine_bias[N + n] = float(offset);
      for (unsigned k = 0; k < K; ++k) {
        const int code = int((k * 13 + n * 7) % 5) - 2;
        q[k * N + n] = int8_t(code);
        reference[k * N + n] = int8_t(code + offset);
        sums[n] += code;
        reference_sums[n] += code + offset;
      }
    }
    uint32_t handles[2];
    check(nntr_hvx_weight_register_u8i4(session, K, N, q.data(), int(q.size()),
                                        scale.data(), N, sums.data(), N,
                                        affine_bias.data(),
                                        int(affine_bias.size()), &handles[0]),
          "affine registration");
    check(nntr_hvx_weight_register_u8i4(session, K, N, reference.data(),
                                        int(reference.size()), scale.data(), N,
                                        reference_sums.data(), N, bias.data(),
                                        N, &handles[1]),
          "reference registration");
    for (unsigned M : {1u, 2u, 64u, 65u, 128u}) {
      std::vector<float> input(M * K), actual(M * N), expected(M * N);
      for (size_t i = 0; i < input.size(); ++i)
        input[i] = float(int((i * 17) % 127) - 63) / 31.f;
      check(nntr_hvx_mm_u8i4_layer(session, M, K, handles, 1, input.data(),
                                   int(input.size()), actual.data(),
                                   int(actual.size())),
            "affine matmul");
      check(nntr_hvx_mm_u8i4_layer(session, M, K, handles + 1, 1, input.data(),
                                   int(input.size()), expected.data(),
                                   int(expected.size())),
            "reference matmul");
      if (M == 1) {
        std::vector<float> head(N);
        check(nntr_hvx_lm_head_wh_f32(session, handles, 1, input.data(), K,
                                      head.data(), N),
              "WH LM head row");
        check(std::memcmp(head.data(), actual.data(), N * sizeof(float)) != 0,
              "WH LM head/FC row mismatch");
        std::puts("AFFINE_LM_HEAD_DSP bit_identical=1");
      }
      float maximum = 0;
      for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]))
          check(1, "nonfinite output");
        maximum = std::fmax(maximum, std::fabs(actual[i] - expected[i]));
      }
      std::printf("AFFINE_DSP M=%u K=%u N=%u max_error=%.9g bit_identical=%d\n",
                  M, K, N, maximum,
                  std::memcmp(actual.data(), expected.data(),
                              actual.size() * sizeof(float)) == 0);
      check(maximum != 0, "affine/reference mismatch");
    }
    for (auto h : handles)
      check(nntr_hvx_weight_release_u8i4(session, h), "release");
  }
  for (uint32_t flags : {0u, 0x21u, 0xcf03e1u}) {
    uint32_t applied = 0;
    check(nntr_hvx_moe_set_opts(session, flags, &applied), "MoE options");
    check(applied != flags, "MoE options mismatch");
    std::printf("MOE_FLAGS 0x%x\n", applied);
    check_moe(session, 96, 192, 128);
    check_moe(session, 2816, 704, 2816);
  }
  check(nntr_hvx_close(session), "close");
  std::puts("AFFINE_DSP_PASS");
}
