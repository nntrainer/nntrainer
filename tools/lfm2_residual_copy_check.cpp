// SPDX-License-Identifier: Apache-2.0
/**
 * @file lfm2_residual_copy_check.cpp
 * @brief Actual residual layer versus its previous copy/add sequence.
 * @author NNTrainer contributors
 * @bug See the project issue tracker.
 */
#include <cpu_ops_table.h>
#include <layer_context.h>
#include <var_grad.h>
#include <weight.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

// Include the implementation so guard checks exercise the same internal
// helper as the app, without adding a testing API to the layer.
#include "../Applications/CausalLM/layers/residual_add.cpp"

namespace {
void check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}

struct ProbeOps : nntrainer::CpuComputeOps {
  unsigned copies = 0, adds = 0;
  bool reject = false;
  void scopy_fp32(unsigned n, const float *x, unsigned sx, float *y,
                  unsigned sy) override {
    check(!reject, "input context used for output copy");
    ++copies;
    CpuComputeOps::scopy_fp32(n, x, sx, y, sy);
  }
  void ele_add_fp32(unsigned n, const float *x, const float *y, float *z,
                    float a, float b, unsigned sy, unsigned sz) override {
    check(!reject, "input context used for output add");
    ++adds;
    CpuComputeOps::ele_add_fp32(n, x, y, z, a, b, sy, sz);
  }
};

std::shared_ptr<nntrainer::ContextData> context(ProbeOps &ops) {
  auto c = std::make_shared<nntrainer::ContextData>();
  c->setComputeOps(&ops);
  return c;
}

void fill(nntrainer::Tensor &x, nntrainer::Tensor &y, bool special) {
  const uint32_t edge[] = {0u,          0x80000000u, 0x00000001u, 0x80000001u,
                           0x3f800000u, 0xbf800000u, 0x00800000u, 0x80800000u,
                           0x7f7fffffu, 0xff7fffffu, 0x7f800000u, 0xff800000u,
                           0x7fc00001u, 0xffc00123u, 0x3f800001u, 0xbf800001u};
  uint32_t rng = 0x837251u;
  for (unsigned i = 0; i < x.size(); ++i) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    uint32_t a = (rng & 0x807fffffu) | ((119u + i % 16u) << 23u);
    uint32_t b = (rng & 0x807fffffu) | ((119u + (i / 16u) % 16u) << 23u);
    if (special) {
      a = edge[i % 16u];
      b = edge[(i / 16u) % 16u];
    }
    std::memcpy(x.getData<float>() + i, &a, 4);
    std::memcpy(y.getData<float>() + i, &b, 4);
  }
}

void run(unsigned rows, unsigned width, bool special) {
  const nntrainer::TensorDim dim(1, 1, rows, width, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP32);
  ProbeOps output_ops, input_ops;
  input_ops.reject = true;
  auto output_context = context(output_ops), input_context = context(input_ops);
  nntrainer::Tensor x(dim, true), y(dim, true), got(dim, true), ref(dim, true);
  x.setContextData(input_context);
  y.setContextData(input_context);
  got.setContextData(output_context);
  ref.setContextData(output_context);
  fill(x, y, special);
  const std::vector<float> saved_x(x.getData<float>(),
                                   x.getData<float>() + x.size());
  const std::vector<float> saved_y(y.getData<float>(),
                                   y.getData<float>() + y.size());
  ref.copy(x);
  check(ref.add_i(y) == 0, "reference add failed");
  output_ops.copies = output_ops.adds = 0;

  causallm::ResidualAddLayer layer;
  nntrainer::InitLayerContext init({dim, dim}, {true}, false, "residual_check",
                                   "", 0.0f, {"NCHW", "FP32", "FP32"});
  layer.finalize(init);
  nntrainer::Var_Grad vx(&x, nullptr), vy(&y, nullptr), vz(&got, nullptr);
  nntrainer::RunLayerContext run_context("residual_check", false, 0.0f, false,
                                         1.0f, output_context, false, {},
                                         {&vx, &vy}, {&vz}, {});
  layer.incremental_forwarding(run_context, 0, rows, false);
  check(std::memcmp(ref.getData<float>(), got.getData<float>(),
                    got.size() * sizeof(float)) == 0,
        "residual output is not bit-identical");
  check(std::memcmp(saved_x.data(), x.getData<float>(), x.size() * 4u) == 0 &&
          std::memcmp(saved_y.data(), y.getData<float>(), y.size() * 4u) == 0,
        "residual modified an input");
  check(got.getContextData() == output_context, "output context changed");
  const bool selected = rows > 1 && width == 2048;
  check(output_ops.adds == 1 && output_ops.copies == (selected ? 0u : 1u),
        "unexpected copy/add dispatch counts");
  std::printf("RESIDUAL_COPY M=%u W=%u special=%u copies=%u adds=%u exact=1\n",
              rows, width, special, output_ops.copies, output_ops.adds);
}

void guards() {
  using nntrainer::Tensor;
  using nntrainer::TensorDim;
  const TensorDim d(1, 1, 2, 2048, nntrainer::Tformat::NCHW,
                    nntrainer::Tdatatype::FP32);
  Tensor storage(TensorDim(1, 1, 4, 2048), true), x(d, true), y(d, true);
  Tensor z = storage.getSharedDataTensor(d, 0, true);
  Tensor overlap = storage.getSharedDataTensor(d, 1, true);
  ProbeOps ops;
  z.setContextData(context(ops));
  check(!causallm::lfmPrefillAdd(z, y, z, 2), "lhs alias selected");
  check(!causallm::lfmPrefillAdd(x, z, z, 2), "rhs alias selected");
  check(!causallm::lfmPrefillAdd(overlap, y, z, 2),
        "partial lhs overlap selected");
  check(!causallm::lfmPrefillAdd(x, overlap, z, 2),
        "partial rhs overlap selected");
  Tensor broadcast(TensorDim(1, 1, 1, 2048), true);
  check(!causallm::lfmPrefillAdd(x, broadcast, z, 2), "broadcast selected");
  Tensor wide(TensorDim(1, 1, 2, 4096), true);
  Tensor strided = wide.getSharedDataTensor(d, 0, false);
  check(!strided.getContiguous(), "strided fixture is contiguous");
  check(!causallm::lfmPrefillAdd(strided, y, z, 2), "strided input selected");
  check(!causallm::lfmPrefillAdd(x, y, strided, 2), "strided output selected");
#ifdef ENABLE_FP16
  Tensor fp16(TensorDim(1, 1, 2, 2048, nntrainer::Tformat::NCHW,
                        nntrainer::Tdatatype::FP16),
              true);
  check(!causallm::lfmPrefillAdd(fp16, fp16, fp16, 2), "FP16 selected");
#endif
  check(ops.adds == 0 && ops.copies == 0, "rejected shape performed work");
  std::puts("RESIDUAL_COPY_GUARDS alias/partial/stride/broadcast/FP16 PASS");
}
} // namespace

int main() {
  try {
    for (unsigned m : {2u, 17u, 1024u, 2048u}) {
      run(m, 2048, false);
      run(m, 2048, true);
    }
    run(1, 2048, false);
    run(2, 2816, false);
    guards();
    std::puts("RESIDUAL_COPY_PASS");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "RESIDUAL_COPY_FAIL: %s\n", e.what());
    return 1;
  }
}
