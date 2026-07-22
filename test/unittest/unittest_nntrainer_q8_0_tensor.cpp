// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_nntrainer_q8_0_tensor.cpp
 * @date   14 May 2026
 * @brief  Unit tests for Q8_0_Tensor + the surrounding factory/dispatch
 *         plumbing (TensorDim::DataType::Q8_0, QScheme::Q8_0,
 *         ModelTensorDataTypeInfo::WQ80A32). Mirrors what Q4_0_Tensor
 *         already provides for the 4-bit path.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cstring>
#include <gtest/gtest.h>

#include <compute_ops.h>
#include <q8_0_tensor.h>
#include <quantizer.h>
#include <tensor.h>
#include <tensor_dim.h>

namespace {

// 32 elements per Q8_0 block × 34 bytes/block. Pick a tensor wide enough to
// span multiple blocks so we exercise multi-block byte-size accounting.
constexpr unsigned int H = 32;
constexpr unsigned int W = 64;
constexpr unsigned int NUM_BLOCKS = (H * W) / QK8_0;
constexpr size_t EXPECTED_BYTES = NUM_BLOCKS * sizeof(nntrainer::block_q8_0);

nntrainer::TensorDim q80_dim() {
  return nntrainer::TensorDim(
    1, 1, H, W,
    {nntrainer::Tformat::NCHW, nntrainer::TensorDim::DataType::Q8_0});
}

// Build a deterministic Q8_0 buffer where each block has scale=1.0 (fp16
// 0x3C00) and qs[i] = (block_idx + i) % 256 cast to int8.
std::vector<uint8_t> makeSyntheticQ80Buffer() {
  std::vector<uint8_t> buf(EXPECTED_BYTES, 0);
  for (unsigned int b = 0; b < NUM_BLOCKS; ++b) {
    uint8_t *block = buf.data() + b * sizeof(nntrainer::block_q8_0);
    // fp16 1.0 = 0x3C00 (little-endian)
    block[0] = 0x00;
    block[1] = 0x3C;
    for (int l = 0; l < 32; ++l) {
      block[2 + l] = static_cast<uint8_t>((b + l) & 0xFF);
    }
  }
  return buf;
}

} // namespace

TEST(Q8_0_Tensor, datatype_enum_and_block_layout) {
  // The added enum + struct layout must be in sync with GGML's block_q8_0:
  //  - 2-byte fp16 scale + 32 int8 quants = 34 bytes.
  EXPECT_EQ(static_cast<int>(nntrainer::TensorDim::DataType::Q8_0),
            // Q8_0 was inserted right after Q4_0 in the enum.
            static_cast<int>(nntrainer::TensorDim::DataType::Q4_0) + 1);
  EXPECT_EQ(QK8_0, 32u);
  EXPECT_EQ(sizeof(nntrainer::block_q8_0), 34u);
  EXPECT_EQ(sizeof(nntrainer::block_q8_0), 34u);
}

TEST(Q8_0_Tensor, qscheme_value) {
  // Default-constructed Q8_0_Tensor should report QScheme::Q8_0.
  nntrainer::Q8_0_Tensor t("test", nntrainer::Tformat::NCHW);
  EXPECT_EQ(t.q_scheme(), nntrainer::QScheme::Q8_0);
}

TEST(Q8_0_Tensor, rejects_invalid_dim) {
  // batch != 1
  EXPECT_THROW(nntrainer::Q8_0_Tensor(
                 nntrainer::TensorDim(2, 1, H, W,
                                      {nntrainer::Tformat::NCHW,
                                       nntrainer::TensorDim::DataType::Q8_0}),
                 false),
               std::invalid_argument);

  // channel != 1
  EXPECT_THROW(nntrainer::Q8_0_Tensor(
                 nntrainer::TensorDim(1, 2, H, W,
                                      {nntrainer::Tformat::NCHW,
                                       nntrainer::TensorDim::DataType::Q8_0}),
                 false),
               std::invalid_argument);

  // width not divisible by 32
  EXPECT_THROW(nntrainer::Q8_0_Tensor(
                 nntrainer::TensorDim(1, 1, H, 17,
                                      {nntrainer::Tformat::NCHW,
                                       nntrainer::TensorDim::DataType::Q8_0}),
                 false),
               std::invalid_argument);
}

TEST(Q8_0_Tensor, allocate_and_size_accounting) {
  nntrainer::Q8_0_Tensor t(q80_dim(), true);
  EXPECT_EQ(t.size(), EXPECTED_BYTES);
  EXPECT_EQ(t.getMemoryBytes(), EXPECTED_BYTES);

  // getData must return a valid (non-null), readable buffer after allocate().
  void *data = t.getData();
  ASSERT_NE(data, nullptr);
}

TEST(Q8_0_Tensor, copy_from_buffer_is_byte_exact) {
  auto src = makeSyntheticQ80Buffer();
  nntrainer::Q8_0_Tensor t(q80_dim(), src.data());

  // After construction with a buffer, the tensor contents must match byte-wise.
  const uint8_t *dst = reinterpret_cast<const uint8_t *>(t.getData());
  ASSERT_NE(dst, nullptr);
  for (size_t i = 0; i < src.size(); ++i) {
    ASSERT_EQ(dst[i], src[i]) << "byte mismatch at offset " << i;
  }
}

TEST(Q8_0_Tensor, factory_dispatch_creates_q8_0_tensor) {
  // Going through nntrainer::Tensor (not Q8_0_Tensor directly) should still
  // produce a Q8_0-backed tensor that reports size()/getMemoryBytes() in
  // packed-Q8_0 units.
  nntrainer::Tensor t(q80_dim(), true);
  EXPECT_EQ(t.getDataType(), nntrainer::TensorDim::DataType::Q8_0);
  EXPECT_EQ(t.size(), EXPECTED_BYTES);
  EXPECT_EQ(t.getMemoryBytes(), EXPECTED_BYTES);
}

TEST(Q8_0_Tensor, set_zero_zeroes_every_byte) {
  auto src = makeSyntheticQ80Buffer();
  nntrainer::Q8_0_Tensor t(q80_dim(), src.data());
  t.setZero();

  const uint8_t *dst = reinterpret_cast<const uint8_t *>(t.getData());
  for (size_t i = 0; i < EXPECTED_BYTES; ++i) {
    ASSERT_EQ(dst[i], 0u) << "non-zero byte at offset " << i;
  }
}

TEST(Q8_0_Tensor, quantize_dequantize_round_trip) {
  // Create an FP32 tensor and fill it with floating point values.
  nntrainer::TensorDim fp32_dim(
    1, 1, H, W,
    {nntrainer::Tformat::NCHW, nntrainer::TensorDim::DataType::FP32});
  nntrainer::Tensor src(fp32_dim, true);

  float *src_data = src.getData<float>();
  for (unsigned int i = 0; i < H * W; ++i) {
    src_data[i] =
      static_cast<float>(i % 32) * 1.5f - 24.0f; // Range: -24 to +22.5
  }

  // Quantize FP32 -> Q8_0
  nntrainer::GgmlQuantizer q(nntrainer::QScheme::Q8_0);
  nntrainer::Tensor q_tensor =
    q.quantize(src, nntrainer::TensorDim::DataType::Q8_0);
  EXPECT_EQ(q_tensor.getDataType(), nntrainer::TensorDim::DataType::Q8_0);

  // Dequantize Q8_0 -> FP32
  nntrainer::Tensor deq_tensor =
    q.dequantize(q_tensor, nntrainer::TensorDim::DataType::FP32);
  EXPECT_EQ(deq_tensor.getDataType(), nntrainer::TensorDim::DataType::FP32);

  nntrainer::Tensor W_t = src.transpose("0:2:1");
  const float *src_t_data = W_t.getData<float>();
  const float *deq_data = deq_tensor.getData<float>();

  // Check the reconstruction error per block of 32
  for (unsigned int b = 0; b < NUM_BLOCKS; ++b) {
    float max_abs_val = 0.0f;
    for (int l = 0; l < 32; ++l) {
      float val = std::abs(src_t_data[b * 32 + l]);
      if (val > max_abs_val) {
        max_abs_val = val;
      }
    }

    float error_bound = max_abs_val / 127.0f;
    // Rounding error of 8-bit can be up to 0.5 * step_size (which is
    // max_abs_val / 127.0). Allowing a very small tolerance for float
    // precision.
    for (int l = 0; l < 32; ++l) {
      float diff = std::abs(src_t_data[b * 32 + l] - deq_data[b * 32 + l]);
      EXPECT_LE(diff, error_bound + 1e-5f)
        << "Excessive error at block " << b << ", element " << l
        << ": original=" << src_t_data[b * 32 + l]
        << ", reconstructed=" << deq_data[b * 32 + l];
    }
  }
}

TEST(Q8_0_Tensor, indirect_conv_q80_vs_fp16) {
#ifdef ENABLE_FP16
  auto *ops = nntrainer::getComputeOps();
  if (!ops->supports_gemm_q4_0_indirect_conv_q8_0() ||
      !ops->supports_gemm_q4_0_indirect_conv_fp16()) {
    SUCCEED() << "Indirect quantized conv GEMM is not supported on this "
                 "platform, skipping.";
    return;
  }

  // Create a synthetic NCHW FP16 input tensor
  unsigned int B = 1, C = 32, IH = 8, IW = 8;
  nntrainer::TensorDim in_dim(
    B, C, IH, IW,
    {nntrainer::Tformat::NCHW, nntrainer::TensorDim::DataType::FP16});
  nntrainer::Tensor in_fp16(in_dim, true);
  _FP16 *in_data = in_fp16.getData<_FP16>();
  for (unsigned int i = 0; i < in_dim.getDataLen(); ++i) {
    in_data[i] = static_cast<_FP16>(static_cast<float>(i % 16) * 0.1f - 0.8f);
  }

  // Create a Q8_0 pre-quantized version of the input
  // Since the input has contiguous channel blocks of size 32 at each spatial
  // pixel, we first flatten it to [IH*IW, C] = [64, 32] layout and quantize it
  // to Q8_0!
  nntrainer::TensorDim flat_dim(
    1, 1, IH * IW, C,
    {nntrainer::Tformat::NCHW, nntrainer::TensorDim::DataType::FP32});
  nntrainer::Tensor flat_fp32(flat_dim, true);
  // Reorder NCHW to [IH*IW, C] spatial-major layout
  float *flat_data = flat_fp32.getData<float>();
  for (unsigned int h = 0; h < IH; ++h) {
    for (unsigned int w = 0; w < IW; ++w) {
      for (unsigned int c = 0; c < C; ++c) {
        flat_data[(h * IW + w) * C + c] =
          static_cast<float>(in_data[(c * IH + h) * IW + w]);
      }
    }
  }

  nntrainer::GgmlQuantizer q80(nntrainer::QScheme::Q8_0);
  nntrainer::Tensor in_q80 =
    q80.quantize(flat_fp32, nntrainer::TensorDim::DataType::Q8_0);

  // Create a Q4_0 filter [CRS, out_ch] = [288, 32] for a 3x3 conv with 32 in_ch
  unsigned int out_ch = 32, k = 3;
  unsigned int CRS = C * k * k; // 32 * 9 = 288
  nntrainer::TensorDim filter_dim(
    1, 1, CRS, out_ch,
    {nntrainer::Tformat::NCHW, nntrainer::TensorDim::DataType::FP32});
  nntrainer::Tensor filter_fp32(filter_dim, true);
  float *f_data = filter_fp32.getData<float>();
  for (unsigned int i = 0; i < filter_dim.getDataLen(); ++i) {
    f_data[i] = static_cast<float>(i % 8) * 0.05f - 0.2f;
  }

  nntrainer::GgmlQuantizer q40(nntrainer::QScheme::Q4_0);
  nntrainer::Tensor filter_q40 =
    q40.quantize(filter_fp32, nntrainer::TensorDim::DataType::Q4_0);

  // Configure convolution geometry
  nntrainer::ConvGatherParams geom;
  geom.in_ch = C;
  geom.in_h = IH;
  geom.in_w = IW;
  geom.k_h = k;
  geom.k_w = k;
  geom.pad_t = 1;
  geom.pad_l = 1;
  geom.stride_h = 1;
  geom.stride_w = 1;
  geom.dil_h = 1;
  geom.dil_w = 1;
  geom.out_w = IW; // same padding => out_w = IW

  // Allocate outputs
  unsigned int OH = IH, OW = IW;
  nntrainer::TensorDim out_dim(
    1, 1, OH * OW, out_ch,
    {nntrainer::Tformat::NCHW, nntrainer::TensorDim::DataType::FP16});
  nntrainer::Tensor out_fp16(out_dim, true);
  nntrainer::Tensor out_q80(out_dim, true);

  // 1. Run indirect conv on FP16 input (internally quantizes to Q8_0 on the
  // fly)
  in_fp16.convQ4_0Indirect(filter_q40, out_fp16, geom);

  // 2. Run indirect conv directly on pre-quantized Q8_0 input (skips on-the-fly
  // quantization)
  in_q80.convQ4_0Indirect(filter_q40, out_q80, geom);

  // Verify that the two outputs are mathematically identical (or extremely
  // close within float tolerance)
  const _FP16 *fp16_out = out_fp16.getData<_FP16>();
  const _FP16 *q80_out = out_q80.getData<_FP16>();

  for (unsigned int i = 0; i < out_dim.getDataLen(); ++i) {
    float diff = std::abs(static_cast<float>(fp16_out[i]) -
                          static_cast<float>(q80_out[i]));
    EXPECT_LT(diff, 1e-4f) << "Mismatch at index " << i << ": FP16-indirect="
                           << static_cast<float>(fp16_out[i])
                           << ", Q8_0-direct="
                           << static_cast<float>(q80_out[i]);
  }
#else
  SUCCEED() << "ENABLE_FP16 is not defined, skipping.";
#endif
}

TEST(Q8_0_Tensor, indirect_conv_nhwc_vs_nchw) {
#ifdef ENABLE_FP16
  auto *ops = nntrainer::getComputeOps();
  if (!ops->supports_gemm_q4_0_indirect_conv_fp16()) {
    SUCCEED() << "Indirect quantized conv GEMM is not supported on this "
                 "platform, skipping.";
    return;
  }

  // Create a synthetic NCHW FP16 input tensor
  unsigned int B = 1, C = 32, IH = 8, IW = 8;
  nntrainer::TensorDim in_nchw_dim(
    B, C, IH, IW,
    {nntrainer::Tformat::NCHW, nntrainer::TensorDim::DataType::FP16});
  nntrainer::Tensor in_nchw(in_nchw_dim, true);
  _FP16 *in_nchw_data = in_nchw.getData<_FP16>();
  for (unsigned int i = 0; i < in_nchw_dim.getDataLen(); ++i) {
    in_nchw_data[i] =
      static_cast<_FP16>(static_cast<float>(i % 16) * 0.1f - 0.8f);
  }

  // Create a synthetic NHWC FP16 input tensor with identical values rearranged
  // physically
  nntrainer::TensorDim in_nhwc_dim(
    B, C, IH, IW,
    {nntrainer::Tformat::NHWC, nntrainer::TensorDim::DataType::FP16});
  nntrainer::Tensor in_nhwc(in_nhwc_dim, true);
  _FP16 *in_nhwc_data = in_nhwc.getData<_FP16>();
  for (unsigned int h = 0; h < IH; ++h) {
    for (unsigned int w = 0; w < IW; ++w) {
      for (unsigned int c = 0; c < C; ++c) {
        // NCHW physical layout index: (c * IH + h) * IW + w
        // NHWC physical layout index: (h * IW + w) * C + c
        in_nhwc_data[(h * IW + w) * C + c] =
          in_nchw_data[(c * IH + h) * IW + w];
      }
    }
  }

  // Create a Q4_0 filter [CRS, out_ch] = [288, 32] for a 3x3 conv with 32 in_ch
  unsigned int out_ch = 32, k = 3;
  unsigned int CRS = C * k * k; // 32 * 9 = 288
  nntrainer::TensorDim filter_dim(
    1, 1, CRS, out_ch,
    {nntrainer::Tformat::NCHW, nntrainer::TensorDim::DataType::FP32});
  nntrainer::Tensor filter_fp32(filter_dim, true);
  float *f_data = filter_fp32.getData<float>();
  for (unsigned int i = 0; i < filter_dim.getDataLen(); ++i) {
    f_data[i] = static_cast<float>(i % 8) * 0.05f - 0.2f;
  }

  nntrainer::GgmlQuantizer q40(nntrainer::QScheme::Q4_0);
  nntrainer::Tensor filter_q40 =
    q40.quantize(filter_fp32, nntrainer::TensorDim::DataType::Q4_0);

  // Configure NCHW geometry
  nntrainer::ConvGatherParams geom_nchw;
  geom_nchw.in_ch = C;
  geom_nchw.in_h = IH;
  geom_nchw.in_w = IW;
  geom_nchw.k_h = k;
  geom_nchw.k_w = k;
  geom_nchw.pad_t = 1;
  geom_nchw.pad_l = 1;
  geom_nchw.stride_h = 1;
  geom_nchw.stride_w = 1;
  geom_nchw.dil_h = 1;
  geom_nchw.dil_w = 1;
  geom_nchw.out_w = IW;
  geom_nchw.is_nhwc = false;

  // Configure NHWC geometry
  nntrainer::ConvGatherParams geom_nhwc = geom_nchw;
  geom_nhwc.is_nhwc = true;

  // Allocate outputs
  unsigned int OH = IH, OW = IW;
  nntrainer::TensorDim out_dim(
    1, 1, OH * OW, out_ch,
    {nntrainer::Tformat::NCHW, nntrainer::TensorDim::DataType::FP16});
  nntrainer::Tensor out_nchw(out_dim, true);
  nntrainer::Tensor out_nhwc(out_dim, true);

  // Run indirect conv
  in_nchw.convQ4_0Indirect(filter_q40, out_nchw, geom_nchw);
  in_nhwc.convQ4_0Indirect(filter_q40, out_nhwc, geom_nhwc);

  // Verify that the two outputs are 100% mathematically identical
  const _FP16 *nchw_out_ptr = out_nchw.getData<_FP16>();
  const _FP16 *nhwc_out_ptr = out_nhwc.getData<_FP16>();

  for (unsigned int i = 0; i < out_dim.getDataLen(); ++i) {
    float diff = std::abs(static_cast<float>(nchw_out_ptr[i]) -
                          static_cast<float>(nhwc_out_ptr[i]));
    EXPECT_LT(diff, 1e-5f) << "Mismatch at index " << i
                           << ": NCHW=" << static_cast<float>(nchw_out_ptr[i])
                           << ", NHWC=" << static_cast<float>(nhwc_out_ptr[i]);
  }
#else
  SUCCEED() << "ENABLE_FP16 is not defined, skipping.";
#endif
}

int main(int argc, char **argv) {
  int result = -1;
  try {
    ::testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }
  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS" << std::endl;
  }
  return result;
}
