// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_nntrainer_q8_0_tensor.cpp
 * @date   14 May 2026
 * @brief  Unit tests for Q8_0_Tensor and the Q8_0 GgmlQuantizer path
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <random>

#include <cpu_backend.h>
#include <q8_0_tensor.h>
#include <quantizer.h>
#include <tensor.h>
#include <tensor_dim.h>

namespace {

// Two blocks per row, so size accounting covers several blocks.
constexpr unsigned int H = 32;
constexpr unsigned int W = 64;
constexpr unsigned int NUM_BLOCKS = (H * W) / QK8_0;
constexpr size_t EXPECTED_BYTES = NUM_BLOCKS * sizeof(nntrainer::block_q8_0);

nntrainer::TensorDim q80_dim(unsigned int h = H) {
  return nntrainer::TensorDim(
    1, 1, h, W,
    {nntrainer::Tformat::NCHW, nntrainer::TensorDim::DataType::Q8_0});
}

/// Every block has scale 1.0 (fp16 0x3C00) and qs[l] = block + l.
std::vector<uint8_t> makeQ80Buffer() {
  std::vector<uint8_t> buf(EXPECTED_BYTES, 0);
  for (unsigned int b = 0; b < NUM_BLOCKS; ++b) {
    uint8_t *block = buf.data() + b * sizeof(nntrainer::block_q8_0);
    block[0] = 0x00;
    block[1] = 0x3C;
    for (int l = 0; l < QK8_0; ++l)
      block[2 + l] = static_cast<uint8_t>(b + l);
  }
  return buf;
}

} // namespace

TEST(Q8_0_Tensor, block_layout_matches_ggml) {
  EXPECT_EQ(QK8_0, 32);
  EXPECT_EQ(sizeof(nntrainer::block_q8_0), 34u);
}

TEST(Q8_0_Tensor, qscheme_id_is_stable) {
  // The id is written into quantized files, so it must not move.
  EXPECT_EQ(static_cast<uint16_t>(nntrainer::QScheme::Q8_0), 0x7);

  nntrainer::Q8_0_Tensor t("test", nntrainer::Tformat::NCHW);
  EXPECT_EQ(t.q_scheme(), nntrainer::QScheme::Q8_0);
}

TEST(Q8_0_Tensor, rejects_invalid_dim_n) {
  auto dim = [](unsigned b, unsigned c, unsigned w) {
    return nntrainer::TensorDim(
      b, c, H, w,
      {nntrainer::Tformat::NCHW, nntrainer::TensorDim::DataType::Q8_0});
  };
  EXPECT_THROW(nntrainer::Q8_0_Tensor(dim(2, 1, W), false),
               std::invalid_argument);
  EXPECT_THROW(nntrainer::Q8_0_Tensor(dim(1, 2, W), false),
               std::invalid_argument);
  EXPECT_THROW(nntrainer::Q8_0_Tensor(dim(1, 1, 17), false),
               std::invalid_argument);
}

TEST(Q8_0_Tensor, size_is_in_packed_bytes) {
  nntrainer::Q8_0_Tensor t(q80_dim(), true);
  EXPECT_EQ(t.size(), EXPECTED_BYTES);
  EXPECT_EQ(t.getMemoryBytes(), EXPECTED_BYTES);
  EXPECT_NE(t.getData(), nullptr);
}

TEST(Q8_0_Tensor, copy_from_buffer_is_byte_exact) {
  auto src = makeQ80Buffer();
  nntrainer::Q8_0_Tensor t(q80_dim(), src.data());

  const uint8_t *dst = static_cast<const uint8_t *>(t.getData());
  ASSERT_NE(dst, nullptr);
  EXPECT_EQ(std::vector<uint8_t>(dst, dst + EXPECTED_BYTES), src);
}

TEST(Q8_0_Tensor, set_zero_clears_every_byte) {
  auto src = makeQ80Buffer();
  nntrainer::Q8_0_Tensor t(q80_dim(), src.data());
  t.setZero();

  const uint8_t *dst = static_cast<const uint8_t *>(t.getData());
  EXPECT_EQ(std::vector<uint8_t>(dst, dst + EXPECTED_BYTES),
            std::vector<uint8_t>(EXPECTED_BYTES, 0));
}

TEST(Q8_0_Tensor, tensor_creates_q8_0_backend) {
  nntrainer::Tensor t(q80_dim(), true);
  EXPECT_EQ(t.getDataType(), nntrainer::TensorDim::DataType::Q8_0);
  EXPECT_EQ(t.size(), EXPECTED_BYTES);
  EXPECT_EQ(t.getMemoryBytes(), EXPECTED_BYTES);
}

TEST(Q8_0_Tensor, shared_tensor_offset_is_in_blocks) {
  auto src = makeQ80Buffer();
  nntrainer::Tensor t(q80_dim(), src.data());

  // Row 1 starts W elements, i.e. W / QK8_0 blocks, into the buffer.
  nntrainer::Tensor row = t.getSharedDataTensor(q80_dim(1), W);
  EXPECT_EQ(row.getData<uint8_t>(),
            t.getData<uint8_t>() + (W / QK8_0) * sizeof(nntrainer::block_q8_0));
}

TEST(Q8_0_Tensor, shared_tensor_rejects_offset_inside_a_block_n) {
  auto src = makeQ80Buffer();
  nntrainer::Tensor t(q80_dim(), src.data());

  nntrainer::Tensor bad = t.getSharedDataTensor(q80_dim(1), QK8_0 / 2);
  EXPECT_THROW(bad.getData<uint8_t>(), std::invalid_argument);
}

TEST(Q8_0_Tensor, quantizer_rejects_partial_block_rows_n) {
  // The quantizer works on the transposed weight, so the height is the row
  // length that has to hold whole blocks.
  nntrainer::Tensor src(1, 1, 48, W);
  nntrainer::GgmlQuantizer q(nntrainer::QScheme::Q8_0);
  EXPECT_THROW(q.quantize(src, nntrainer::TensorDim::DataType::Q8_0),
               std::invalid_argument);
}

TEST(Q8_0_Tensor, repack_interleaves_four_rows) {
  constexpr unsigned int M = 8, N = 64, NB = N / QK8_0;
  std::vector<uint8_t> rows(M * NB * Q8_0_SIZE);
  std::mt19937 rng(7);
  for (auto &b : rows)
    b = static_cast<uint8_t>(rng());

  std::vector<uint8_t> x4(rows.size());
  nntrainer::repack_q8_0(x4.data(), rows.data(), rows.size(), M, N);

  // Each q8_0x4 block: 4 fp16 scales, then bytes 8k..8k+7 of row r at
  // qs[32k + 8r].
  auto row_block = [&](unsigned int r, unsigned int b) {
    return rows.data() + (r * NB + b) * Q8_0_SIZE;
  };
  for (unsigned int g = 0; g < M / 4; ++g) {
    for (unsigned int b = 0; b < NB; ++b) {
      const uint8_t *blk = x4.data() + (g * NB + b) * 4 * Q8_0_SIZE;
      for (unsigned int r = 0; r < 4; ++r) {
        const uint8_t *src = row_block(4 * g + r, b);
        EXPECT_EQ(std::memcmp(blk + 2 * r, src, 2), 0);
        for (unsigned int k = 0; k < 4; ++k)
          EXPECT_EQ(std::memcmp(blk + 8 + 32 * k + 8 * r, src + 2 + 8 * k, 8),
                    0)
            << "group " << g << ", block " << b << ", row " << r;
      }
    }
  }

  std::vector<uint8_t> back(rows.size());
  nntrainer::unpack_q8_0(x4.data(), back.data(), x4.size(), M, N);
  EXPECT_EQ(back, rows);
}

TEST(Q8_0_Tensor, repack_rejects_partial_row_groups_n) {
  // Rows are interleaved in fours, so 6 rows do not fit.
  std::vector<uint8_t> rows(6 * Q8_0_SIZE), x4(rows.size());
  EXPECT_THROW(
    nntrainer::repack_q8_0(x4.data(), rows.data(), rows.size(), 6, QK8_0),
    std::invalid_argument);
  EXPECT_THROW(
    nntrainer::unpack_q8_0(x4.data(), rows.data(), rows.size(), 6, QK8_0),
    std::invalid_argument);
}

TEST(Q8_0_Tensor, quantize_dequantize_round_trip) {
  nntrainer::Tensor src(1, 1, H, W);
  float *src_data = src.getData<float>();
  for (unsigned int i = 0; i < H * W; ++i)
    src_data[i] = static_cast<float>(i % 32) * 1.5f - 24.0f;

  nntrainer::GgmlQuantizer q(nntrainer::QScheme::Q8_0);
  nntrainer::Tensor q_tensor =
    q.quantize(src, nntrainer::TensorDim::DataType::Q8_0);
  ASSERT_EQ(q_tensor.getDataType(), nntrainer::TensorDim::DataType::Q8_0);

  nntrainer::Tensor deq =
    q.dequantize(q_tensor, nntrainer::TensorDim::DataType::FP32);
  ASSERT_EQ(deq.getDataType(), nntrainer::TensorDim::DataType::FP32);

  // The quantizer works on the transposed weight, one block per 32 values.
  nntrainer::Tensor src_t = src.transpose("0:2:1");
  const float *ref = src_t.getData<float>();
  const float *out = deq.getData<float>();
  for (unsigned int b = 0; b < NUM_BLOCKS; ++b) {
    float amax = 0.0f;
    for (int l = 0; l < QK8_0; ++l)
      amax = std::max(amax, std::abs(ref[b * QK8_0 + l]));

    // Half a quantization step, plus slack for the fp16 scale.
    const float tol = amax / 127.0f;
    for (int l = 0; l < QK8_0; ++l) {
      const unsigned int i = b * QK8_0 + l;
      EXPECT_NEAR(out[i], ref[i], tol) << "block " << b << ", element " << l;
    }
  }
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
