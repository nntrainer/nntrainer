// SPDX-License-Identifier: Apache-2.0
/**
 * @file lfm2_embedding_stream_check.cpp
 * @brief Exercise actual FP32 embedding row reads without a resident table.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#include <embedding_layer.h>
#include <layer_context.h>
#include <thread_manager.h>
#include <var_grad.h>
#include <weight.h>
#include <weight_file_reader.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>

namespace {
constexpr unsigned rows = 300, width = 2048, tokens = 4;
constexpr size_t q4_stride = 18 * (width / 32);
constexpr size_t offset = 13;

void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}

template <typename T> std::vector<T *> pointers(std::vector<T> &values) {
  std::vector<T *> result;
  for (auto &value : values)
    result.push_back(&value);
  return result;
}

struct Fixture {
  causallm::EmbeddingLayer layer;
  std::vector<nntrainer::Weight> weights;
  std::vector<nntrainer::Var_Grad> inputs, outputs;
  std::unique_ptr<nntrainer::RunLayerContext> context;

  Fixture(bool stream, int fd, float scale, bool q4 = false) {
    layer.setProperty(
      {"in_dim=" + std::to_string(rows), "out_dim=" + std::to_string(width),
       "scale=" + std::to_string(scale),
       std::string("stream_fp32=") + (stream && !q4 ? "true" : "false")});
    const nntrainer::TensorDim::TensorType f32(
      ml::train::TensorDim::Format::NCHW, ml::train::TensorDim::DataType::FP32);
    nntrainer::InitLayerContext init(
      {nntrainer::TensorDim({1, 1, 1, tokens}, f32)}, {true}, false,
      "embedding", "", 0.0f, {"NCHW", q4 ? "Q4_0" : "FP32", "FP32"}, 1.0f,
      ml::train::ExecutionMode::INFERENCE, ml::train::CPU);
    layer.finalize(init);
    for (const auto &spec : init.getWeightsSpec()) {
      const bool requested_virtual = std::get<12>(spec);
      require(requested_virtual == (stream || q4),
              "layer requested incorrect storage");
      // Construct the identical resident reference explicitly even when
      // the Q4 inference policy requested virtual storage.
      const bool is_virtual = stream;
      nntrainer::Tensor variable(
        std::get<0>(spec), !is_virtual, nntrainer::Initializer::NONE, "table",
        nntrainer::QScheme::PER_TENSOR_AFFINE, is_virtual);
      weights.emplace_back(variable, nntrainer::Tensor(), nntrainer::Tensor(),
                           "table");
    }
    require(weights.size() == 1, "expected one embedding table");
    auto &weight = weights[0].getVariableRef();
    require(weight.isVirtual() == stream, "virtual storage opt-in mismatch");
    weight.setFileOffset(offset);
    weight.read(nntrainer::FileReadSource{fd}, offset, true, fd);
    require(!stream || !weight.isAllocated(), "streamed table was allocated");
    for (const auto &dim : init.getInputDimensions())
      inputs.emplace_back(dim, nntrainer::Initializer::NONE, false, true, "in");
    for (const auto &spec : init.getOutSpecs())
      outputs.emplace_back(spec.variable_spec.dim, nntrainer::Initializer::NONE,
                           false, true, "out");
    context = std::make_unique<nntrainer::RunLayerContext>(
      "embedding", false, 0.0f, false, 1.0f, nullptr, false, pointers(weights),
      pointers(inputs), pointers(outputs),
      std::vector<nntrainer::Var_Grad *>{});
  }

  std::vector<char> save(const std::string &path,
                         nntrainer::TensorDim::DataType dtype =
                           nntrainer::TensorDim::DataType::NONE) {
    weights[0].setAsGradientFirstAccess();
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    require(out.is_open(), "save output open failed");
    layer.save(out, *context, false, ml::train::ExecutionMode::INFERENCE, false,
               dtype);
    out.close();
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in),
            std::istreambuf_iterator<char>()};
  }

  std::vector<float> run(const std::vector<unsigned> &ids, unsigned from) {
    auto *input = inputs[0].getVariableRef().getData<float>();
    for (size_t i = 0; i < ids.size(); ++i)
      input[i] = static_cast<float>(ids[i]);
    layer.incremental_forwarding(*context, from, from + ids.size(), false);
    const auto *out = outputs[0].getVariableRef().getData<float>();
    require(!weights[0].getVariableRef().isVirtual() ||
              !weights[0].getVariableRef().isAllocated(),
            "lookup materialized the table");
    return {out, out + ids.size() * width};
  }
};
} // namespace
#endif

int main() {
#if defined(__linux__)
  char path[] = "embedding-stream-XXXXXX";
  int write_fd = -1, read_fd = -1;
  std::string save_path;
  try {
    setenv("NNTR_HTP_PAGE_CACHE", "0", 1);
    nntrainer::ThreadManagerConfig threads;
    threads.compute_threads = 4;
    threads.enable_affinity = false;
    nntrainer::ThreadManager::setConfig(threads);
    write_fd = mkstemp(path);
    require(write_fd >= 0, "mkstemp failed");
    save_path = std::string(path) + ".saved";
    std::vector<unsigned char> bytes(offset + rows * width * sizeof(float), 0);
    for (unsigned i = 0; i < rows * width; ++i) {
      const float value = (static_cast<int>(i % 137) - 68) * 0.03125f;
      std::memcpy(bytes.data() + offset + i * sizeof(float), &value,
                  sizeof(value));
    }
    require(write(write_fd, bytes.data(), bytes.size()) ==
              ssize_t(bytes.size()),
            "fixture write failed");
    require(fsync(write_fd) == 0, "fixture sync failed");
    read_fd = nntrainer::openWeightFile(path);
    require((fcntl(read_fd, F_GETFL) & O_DIRECT) != 0,
            "reader is not direct IO");
    for (float scale : {1.0f, -0.75f}) {
      Fixture resident(false, read_fd, scale), stream(true, read_fd, scale);
      for (const auto &ids :
           {std::vector<unsigned>{7, 0, 3, 7}, std::vector<unsigned>{2},
            std::vector<unsigned>{1, 6, 2, 0}}) {
        auto expected = resident.run(ids, 23);
        auto actual = stream.run(ids, 23);
        require(expected.size() == actual.size() &&
                  std::memcmp(expected.data(), actual.data(),
                              expected.size() * sizeof(float)) == 0,
                "streamed rows differ from resident rows");
      }
      // Repeated rows from simultaneous prefill jobs and scaled cache hits.
      for (unsigned repeat = 0; repeat < 8; ++repeat) {
        const auto expected = resident.run({7, 7, 7, 7}, 0);
        const auto actual = stream.run({7, 7, 7, 7}, 0);
        require(actual == expected, "concurrent repeated rows differ");
      }
      auto &source = stream.weights[0].getVariableRef();
      source.setFileOffset(offset + width * sizeof(float));
      require(stream.run({0}, 0) == resident.run({1}, 0),
              "changed source offset used a stale row");
      source.setFileOffset(offset);
      require(stream.run({0}, 0) == resident.run({0}, 0),
              "restored source offset used a stale row");
      source.read(nntrainer::FileReadSource{read_fd}, offset, true, -2);
      bool bad_fd_rejected = false;
      try {
        stream.run({0}, 0);
      } catch (const std::runtime_error &) {
        bad_fd_rejected = true;
      }
      require(bad_fd_rejected, "changed fd used a stale cached row");
      source.read(nntrainer::FileReadSource{read_fd}, offset, true, read_fd);
      const auto saved = stream.save(save_path);
      require(
        saved.size() == bytes.size() - offset &&
          std::memcmp(saved.data(), bytes.data() + offset, saved.size()) == 0,
        "streamed save did not preserve exact unscaled table bytes");
      bool quantization_rejected = false;
      try {
        stream.save(save_path, nntrainer::TensorDim::DataType::Q4_0);
      } catch (const std::invalid_argument &) {
        quantization_rejected = true;
      }
      require(quantization_rejected, "streamed quantized export accepted");
      bool rejected = false;
      try {
        stream.run({0, rows, 0, 0}, 0);
      } catch (const std::invalid_argument &) {
        rejected = true;
      }
      require(rejected, "out-of-range token accepted");
    }
    Fixture resident(false, read_fd, 1.0f), cached(true, read_fd, 1.0f);
    for (unsigned i = 0; i < 128; ++i)
      cached.run({i}, 0);
    cached.run({0},
               0); // Make row 0 recent; inserting row 128 must evict row 1.
    cached.run({128}, 0);
    Fixture truncated(true, read_fd, 1.0f);
    require(ftruncate(write_fd, offset + width * sizeof(float)) == 0,
            "truncate failed");
    require(cached.run({0, 127, 128, 0}, 0) ==
              resident.run({0, 127, 128, 0}, 0),
            "cached hit attempted I/O or changed exact row data");
    bool evicted = false;
    try {
      cached.run({1}, 0);
    } catch (const std::runtime_error &) {
      evicted = true;
    }
    require(evicted, "cache exceeded its 128-row bound or did not use LRU");
    bool rejected = false;
    try {
      truncated.run({0, rows - 1, rows - 2, 0}, 0);
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    require(rejected, "truncated table accepted");
    rejected = false;
    try {
      cached.save(save_path);
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    require(rejected, "truncated export used cached rows or accepted EOF");
    // Q4_0 rows include 64 distinct half-precision scale blocks. A raw
    // packed cache must retain all of them, not just the token's nibbles.
    require(ftruncate(write_fd, offset + rows * q4_stride) == 0,
            "Q4 resize failed");
    std::vector<uint8_t> packed(rows * q4_stride);
    for (size_t r = 0; r < rows; ++r)
      for (size_t b = 0; b < width / 32; ++b) {
        uint8_t *block = packed.data() + r * q4_stride + b * 18;
        const uint16_t scale = 0x2000u + ((r * 37 + b * 19) % 0x1400u);
        std::memcpy(block, &scale, sizeof(scale));
        for (size_t j = 0; j < 16; ++j)
          block[2 + j] = uint8_t(r * 71 + b * 23 + j * 13);
      }
    require(pwrite(write_fd, packed.data(), packed.size(), offset) ==
                ssize_t(packed.size()) &&
              fsync(write_fd) == 0,
            "Q4 fixture write failed");
    for (float scale : {1.0f, -0.75f}) {
      Fixture expected(false, read_fd, scale, true);
      Fixture actual(true, read_fd, scale, true);
      for (const auto &ids : {std::vector<unsigned>{7, 7, 7, 7},
                              std::vector<unsigned>{299, 0, 9, 299},
                              std::vector<unsigned>{128, 65, 257, 65},
                              std::vector<unsigned>{299}})
        for (unsigned repeat = 0; repeat < 3; ++repeat)
          require(actual.run(ids, 23) == expected.run(ids, 23),
                  "Q4 cached/scaled/repeated rows changed");
      auto &source = actual.weights[0].getVariableRef();
      source.setFileOffset(offset + q4_stride);
      require(actual.run({0, 0, 1, 1}, 0) == expected.run({1, 1, 2, 2}, 0),
              "Q4 source offset used stale rows");
      source.setFileOffset(offset);
      require(actual.run({0, 1, 0, 1}, 0) == expected.run({0, 1, 0, 1}, 0),
              "Q4 restored offset used stale rows");
      source.read(nntrainer::FileReadSource{read_fd}, offset, true, -2);
      bool rejected = false;
      try {
        actual.run({0, 0, 1, 1}, 0);
      } catch (const std::runtime_error &) {
        rejected = true;
      }
      require(rejected, "Q4 changed fd reused stale rows");
      source.read(nntrainer::FileReadSource{read_fd}, offset, true, read_fd);
      require(actual.run({0}, 0) == expected.run({0}, 0),
              "Q4 restored source differs");
    }
    Fixture q4_expected(false, read_fd, 1.0f, true);
    Fixture q4_cache(true, read_fd, 1.0f, true);
    for (unsigned r = 0; r < 128; ++r)
      q4_cache.run({r}, 0);
    q4_cache.run({0}, 0);
    q4_cache.run({128}, 0);
    require(ftruncate(write_fd, offset) == 0, "Q4 truncate failed");
    require(q4_cache.run({0, 127, 128, 0}, 0) ==
              q4_expected.run({0, 127, 128, 0}, 0),
            "Q4 cached hits read the file or changed values");
    bool q4_evicted = false;
    try {
      q4_cache.run({1, 1, 1, 1}, 0);
    } catch (const std::runtime_error &) {
      q4_evicted = true;
    }
    require(q4_evicted, "Q4 cache exceeded 128 rows or accepted EOF");
    unlink(save_path.c_str());
    close(read_fd);
    close(write_fd);
    unlink(path);
    std::puts(
      "FP32/Q4 direct embedding: exact rows, concurrent cache hits, LRU, "
      "source changes, bounded save, bounds and EOF passed");
    return 0;
  } catch (const std::exception &error) {
    if (read_fd >= 0)
      close(read_fd);
    if (write_fd >= 0)
      close(write_fd);
    unlink(path);
    if (!save_path.empty())
      unlink(save_path.c_str());
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
#else
  std::puts("FP32 direct embedding check requires Linux/Android O_DIRECT");
  return 77;
#endif
}
