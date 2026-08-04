// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   nntrainer_neuron_smoke.cpp
 * @date   30 Jul 2026
 * @brief  Smoke test for nntrainer's MediaTek NeuroPilot (Neuron Runtime)
 *         backend. Loads a user-provided .dla file, executes it via a
 *         neuron_graph layer, and validates output against a golden file.
 *         Standalone binary (no gtest, no CausalLM/app dependency).
 * @see    https://github.com/nnstreamer/nntrainer
 *
 * @usage
 *   nntrainer_neuron_smoke <model.dla> [golden.bin] [options]
 *
 * Options:
 *   --in-shape=B:C:H:W   input shape of the .dla network  (default 1:1:1:1)
 *   --out-shape=B:C:H:W  output shape of the .dla network (default 1:1:1:1)
 *   --in-dtype=TYPE      input tensor dtype, e.g. FP32, UINT8 (default FP32)
 *   --out-dtype=TYPE     output tensor dtype (default FP32)
 *   --input=<file>       raw input data matching --in-dtype (default: zeros)
 *   --dump=<file>        write the produced output to <file> (to seed a golden)
 *
 * A quantized .dla (e.g. a UINT8 MobileNet) needs --in-dtype=UINT8
 * --out-dtype=UINT8; get the values from your model's own reported I/O specs
 * (dumped by TFLite/ncc-tflite tooling), not by guessing. Getting this wrong
 * does not fail loudly: the nntrainer-side buffer would just be sized 4x too
 * large for a 1-byte type, the layer's ">= required bytes" check still
 * passes, and the model silently receives the wrong bytes.
 *
 * @details
 * This test harness:
 * 1. Reaches the neuron backend through Engine::Global() (registered from
 *    engine.cpp when built with -Denable-neuron=true), or explicitly via
 *    NNTRAINER_NEURON_CONTEXT_SO.
 * 2. Builds a 2-layer model programmatically: input -> neuron_graph.
 * 3. Compiles + initializes, which loads the .dla and caches its I/O geometry.
 * 4. Feeds an input buffer (all-zeros unless --input is given).
 * 5. Runs inference.
 * 6. Compares the output buffer against a golden file (byte-for-byte).
 * 7. Reports pass/fail.
 *
 * The model is built programmatically rather than from an ini file so that no
 * temporary file has to be written on the device.
 *
 * Note the .dla carries its own weights, so unlike the QNN backend no
 * IN_TENSOR entries are declared on the layer -- only the OUT_TENSOR shape,
 * which nntrainer needs in order to size the output tensor.
 *
 * Golden file format: raw binary dump of the output tensor in native byte
 * order (no headers, no metadata). Seed one with --dump=golden.bin after
 * confirming the values are correct.
 *
 * Environment variables (for testing):
 * - QUICK_DOT_AI_NEURON_LIB: override libneuron_runtime.so path
 *   (e.g., point at the SDK's dummy/lib copy for host testing).
 * - QUICK_DOT_AI_NEURON_NULL_DEVICE: set to "1" to use the no-hardware
 *   stub runtime (requires QUICK_DOT_AI_NEURON_LIB to be set).
 * - NNTRAINER_NEURON_CONTEXT_SO: explicit path to libneuron_context.so, used
 *   when the automatic registration in engine.cpp did not find it.
 * - NNTRAINER_NEURON_SMOKE_VERBOSE: set to "1" for detailed logging.
 */

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <engine.h>
#include <layer.h>
#include <model.h>
#include <nntrainer_error.h>

namespace {

bool g_verbose = false;

void log_verbose(const char *fmt, ...) {
  if (!g_verbose)
    return;
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  fprintf(stderr, "\n");
}

std::vector<uint8_t> read_binary_file(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  const std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> buf(static_cast<size_t>(size));
  if (size > 0 && !file.read(reinterpret_cast<char *>(buf.data()), size)) {
    throw std::runtime_error("Failed to read file: " + path);
  }
  return buf;
}

void write_binary_file(const std::string &path, const void *data,
                       size_t bytes) {
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + path);
  }
  file.write(static_cast<const char *>(data),
             static_cast<std::streamsize>(bytes));
  if (!file) {
    throw std::runtime_error("Failed to write file: " + path);
  }
}

/**
 * @brief pull the value out of a "--key=value" argument
 * @return true when @a arg carried @a key, with @a out set to the value
 */
bool match_opt(const std::string &arg, const std::string &key,
               std::string &out) {
  const std::string prefix = "--" + key + "=";
  if (arg.rfind(prefix, 0) != 0)
    return false;
  out = arg.substr(prefix.size());
  return true;
}

void usage(const char *argv0) {
  std::cerr
    << "Usage: " << argv0 << " <model.dla> [golden.bin] [options]\n"
    << "\nOptions:\n"
    << "  --in-shape=B:C:H:W   input shape of the .dla  (default 1:1:1:1)\n"
    << "  --out-shape=B:C:H:W  output shape of the .dla (default 1:1:1:1)\n"
    << "  --in-dtype=TYPE      input dtype: FP32, FP16, UINT8, UINT16,\n"
    << "                       QINT8, QINT16 (default FP32)\n"
    << "  --out-dtype=TYPE     output dtype, same set as --in-dtype\n"
    << "  --input=<file>       raw input bytes matching --in-dtype/--in-shape\n"
    << "                       (default: all zeros)\n"
    << "  --dump=<file>        write produced output to <file>\n"
    << "\nEnvironment variables:\n"
    << "  QUICK_DOT_AI_NEURON_LIB          override libneuron_runtime.so path\n"
    << "  QUICK_DOT_AI_NEURON_NULL_DEVICE  set to 1 for the no-hardware stub\n"
    << "  NNTRAINER_NEURON_CONTEXT_SO      explicit libneuron_context.so path\n"
    << "  NNTRAINER_NEURON_SMOKE_VERBOSE   set to 1 for detailed logging\n";
}

/**
 * @brief bytes-per-element for the dtype strings this test supports, and the
 * "model_tensor_type=<W>-<A>" activation half nntrainer needs to make the
 * input layer actually produce that dtype (see InputLayer::finalize(), which
 * only overrides its default FP32 output dim with
 * InitLayerContext::getActivationDataType() -- i.e. without this property the
 * input tensor stays FP32 no matter what dtype is requested here).
 *
 * Deliberately narrower than nntrainer's full TensorDim::DataType list: BCQ,
 * UINT4, Q4_K, Q6_K, Q4_0, QS4CX are sub-byte or block-packed formats with no
 * simple elems*bytes layout, so a smoke test built around flat byte buffers
 * cannot represent them correctly -- better to reject them than silently
 * mishandle the packing.
 */
size_t dtype_bytes(const std::string &dtype) {
  if (dtype == "FP32")
    return 4;
  if (dtype == "FP16" || dtype == "UINT16" || dtype == "QINT16")
    return 2;
  if (dtype == "UINT8" || dtype == "QINT8")
    return 1;
  throw std::invalid_argument(
    "unsupported --in-dtype/--out-dtype: " + dtype +
    " (supported: FP32, FP16, UINT8, UINT16, QINT8, QINT16)");
}

/**
 * @brief Widen `raw` (packed in @a dtype's natural on-disk width) into
 * `out`, one float per element.
 *
 * @details ml::train::Model::inference() takes a "float *", and that is a
 * real type, not just a pointer convention: NetworkGraph::getInputDimension()
 * (network_graph.cpp, identify_as_model_input) reports the input layer's
 * *declared* input dimension, which stays FP32 regardless of
 * model_tensor_type -- that property only overrides the input layer's
 * *output* dimension (InputLayer::finalize()), which is what the downstream
 * neuron_graph layer actually reads. So the buffer hard-required by the
 * ccapi boundary is genuinely float32; nntrainer converts it to the
 * managed tensor's real dtype via a plain truncating cast while copying
 * (see __fallback_copy_fp32_u8: `Y[i] = static_cast<uint8_t>(X[i])`).
 * Concretely: to place the quantized byte value 127 into the network, this
 * function must produce the float 127.0f, not the byte 0x7F verbatim.
 *
 * A raw --input file is still packed in @a dtype's natural width (1 byte
 * per element for UINT8, matching how a real quantized image would be
 * dumped) -- this function is what bridges that natural, compact format to
 * the float32 layout the ccapi boundary actually requires.
 */
void widen_to_float(const std::string &dtype, const std::vector<uint8_t> &raw,
                    std::vector<float> &out) {
  const size_t n = out.size();
  if (dtype == "FP32") {
    std::memcpy(out.data(), raw.data(), n * sizeof(float));
  } else if (dtype == "UINT8") {
    for (size_t i = 0; i < n; ++i)
      out[i] = static_cast<float>(raw[i]);
  } else if (dtype == "UINT16") {
    const auto *p = reinterpret_cast<const uint16_t *>(raw.data());
    for (size_t i = 0; i < n; ++i)
      out[i] = static_cast<float>(p[i]);
  } else if (dtype == "QINT16") {
    const auto *p = reinterpret_cast<const int16_t *>(raw.data());
    for (size_t i = 0; i < n; ++i)
      out[i] = static_cast<float>(p[i]);
  } else {
    throw std::invalid_argument(
      "--input widening not implemented for --in-dtype=" + dtype +
      " (supported: FP32, UINT8, UINT16, QINT16)");
  }
}

/**
 * @brief batch + total element count parsed out of a "B:C:H:W" shape string.
 *
 * Deliberately does not go through ml::train::Model::getInputDimension() /
 * getOutputDimension(): those read NetworkGraph::input_dims_ / label_dims_,
 * and label_dims_ is only populated for layers where requireLabel() is true
 * (loss layers, for *training* labels). This model has no loss layer, so
 * getOutputDimension() throws "the graph has no node identified as output!"
 * even though inference() itself works fine (it only calls
 * getOutputDimension() when a label buffer is passed, which this smoke test
 * never does). Parsing the shape we were given sidesteps that entirely.
 */
struct ParsedShape {
  unsigned int batch;
  size_t elems;
};

ParsedShape parse_shape(const std::string &shape) {
  std::vector<unsigned long> fields;
  size_t start = 0;
  while (start <= shape.size()) {
    size_t pos = shape.find(':', start);
    std::string tok = shape.substr(
      start, pos == std::string::npos ? std::string::npos : pos - start);
    if (tok.empty())
      throw std::invalid_argument("malformed shape (empty field): " + shape);
    fields.push_back(std::stoul(tok));
    if (pos == std::string::npos)
      break;
    start = pos + 1;
  }
  if (fields.size() != 4)
    throw std::invalid_argument(
      "shape must have exactly 4 colon-separated fields (B:C:H:W): " + shape);

  size_t elems = 1;
  for (auto f : fields)
    elems *= static_cast<size_t>(f);
  return ParsedShape{static_cast<unsigned int>(fields[0]), elems};
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    usage(argv[0]);
    return EXIT_FAILURE;
  }

  std::string dla_path;
  std::string golden_path;
  std::string in_shape = "1:1:1:1";
  std::string out_shape = "1:1:1:1";
  std::string in_dtype = "FP32";
  std::string out_dtype = "FP32";
  std::string input_file;
  std::string dump_file;

  // positional: <model.dla> then optional <golden.bin>; the rest are --opts
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    std::string val;
    if (match_opt(arg, "in-shape", val)) {
      in_shape = val;
    } else if (match_opt(arg, "out-shape", val)) {
      out_shape = val;
    } else if (match_opt(arg, "in-dtype", val)) {
      in_dtype = val;
    } else if (match_opt(arg, "out-dtype", val)) {
      out_dtype = val;
    } else if (match_opt(arg, "input", val)) {
      input_file = val;
    } else if (match_opt(arg, "dump", val)) {
      dump_file = val;
    } else if (arg == "-h" || arg == "--help") {
      usage(argv[0]);
      return EXIT_SUCCESS;
    } else if (arg.rfind("--", 0) == 0) {
      std::cerr << "ERROR: unknown option: " << arg << std::endl;
      usage(argv[0]);
      return EXIT_FAILURE;
    } else if (dla_path.empty()) {
      dla_path = arg;
    } else if (golden_path.empty()) {
      golden_path = arg;
    } else {
      std::cerr << "ERROR: unexpected extra argument: " << arg << std::endl;
      return EXIT_FAILURE;
    }
  }

  if (dla_path.empty()) {
    std::cerr << "ERROR: no .dla path given" << std::endl;
    usage(argv[0]);
    return EXIT_FAILURE;
  }

  // --in-dtype has to round-trip through model_tensor_type="<W>-<A>"
  // (model_common_properties.h ModelTensorDataTypeInfo::EnumStr), which is a
  // fixed enum of specific pairs, not "any dtype paired with itself". QINT8
  // has no QINT8-QINT8 entry there (only QINT8-FP16/FP32/UINT16), so it can't
  // be used as --in-dtype here even though it's valid as --out-dtype (the
  // neuron_graph layer's own tensor_dtype property has no such restriction).
  static const std::vector<std::string> valid_in_dtypes = {
    "FP32", "FP16", "UINT16", "UINT8", "QINT16"};
  if (std::find(valid_in_dtypes.begin(), valid_in_dtypes.end(), in_dtype) ==
      valid_in_dtypes.end()) {
    std::cerr << "ERROR: --in-dtype=" << in_dtype
              << " has no <dtype>-<dtype> entry in nntrainer's "
                 "model_tensor_type enum. Supported here: FP32, FP16, "
                 "UINT16, UINT8, QINT16."
              << std::endl;
    return EXIT_FAILURE;
  }

  const char *verbose_env = std::getenv("NNTRAINER_NEURON_SMOKE_VERBOSE");
  g_verbose = (verbose_env != nullptr && std::string(verbose_env) == "1");

  log_verbose("[smoke] verbose logging enabled");
  log_verbose("[smoke] DLA path: %s", dla_path.c_str());
  log_verbose("[smoke] in-shape: %s (%s)  out-shape: %s (%s)", in_shape.c_str(),
              in_dtype.c_str(), out_shape.c_str(), out_dtype.c_str());
  if (!golden_path.empty()) {
    log_verbose("[smoke] golden file: %s", golden_path.c_str());
  }

  try {
    // ---------------------------------------------------------------------
    // Step 1: reach the neuron backend.
    // ---------------------------------------------------------------------
    log_verbose("[smoke] looking up neuron context...");
    auto &engine = nntrainer::Engine::Global();

    // engine.cpp registers libneuron_context.so automatically when nntrainer
    // was built with -Denable-neuron=true. Allow an explicit override so the
    // plugin can be pointed at directly.
    if (const char *so = std::getenv("NNTRAINER_NEURON_CONTEXT_SO")) {
      log_verbose("[smoke] explicitly registering context from %s", so);
      engine.registerContext(so, "");
    }

    // getRegisteredContext throws std::invalid_argument when absent.
    try {
      (void)engine.getRegisteredContext("neuron");
    } catch (const std::exception &e) {
      std::cerr << "ERROR: neuron context not registered: " << e.what() << "\n"
                << "  Was nntrainer built with -Denable-neuron=true, and is\n"
                << "  libneuron_context.so on LD_LIBRARY_PATH? You can also\n"
                << "  set NNTRAINER_NEURON_CONTEXT_SO to its full path."
                << std::endl;
      return EXIT_FAILURE;
    }
    log_verbose("[smoke] neuron context registered");

    // ---------------------------------------------------------------------
    // Step 2: build input -> neuron_graph programmatically.
    // ---------------------------------------------------------------------
    log_verbose("[smoke] creating model...");
    auto model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
    if (model == nullptr) {
      std::cerr << "ERROR: failed to create model" << std::endl;
      return EXIT_FAILURE;
    }

    model->addLayer(
      ml::train::createLayer("input", {"name=in", "input_shape=" + in_shape}));

    // The .dla holds its own weights, so only the OUT_TENSOR geometry is
    // declared here; that is what nntrainer needs to size the output tensor.
    // Vector-valued properties are comma separated (see base_properties.h).
    model->addLayer(ml::train::createLayer(
      "neuron_graph",
      {"name=neuron_layer", "input_layers=in", "path=" + dla_path,
       "dim=" + out_shape, "tensor_dtype=" + out_dtype,
       "tensor_type=OUT_TENSOR", "engine=neuron"}));

    // model_tensor_type="<weight>-<activation>" is what actually makes the
    // input layer emit in_dtype: InputLayer::finalize() only overrides its
    // default-FP32 output dim with InitLayerContext::getActivationDataType(),
    // which reads this property. Without it, --in-dtype=UINT8 would be
    // silently ignored and the input tensor would stay FP32.
    model->setProperty({"batch_size=1", "epochs=1",
                        "model_tensor_type=" + in_dtype + "-" + in_dtype});

    log_verbose("[smoke] compiling model...");
    int ret = model->compile(ml::train::ExecutionMode::INFERENCE);
    if (ret != ML_ERROR_NONE) {
      std::cerr << "ERROR: failed to compile model, error code: " << ret
                << std::endl;
      return EXIT_FAILURE;
    }
    log_verbose("[smoke] model compiled");

    // initialize() drives neuron_graph::finalize().
    ret = model->initialize(ml::train::ExecutionMode::INFERENCE);
    if (ret != ML_ERROR_NONE) {
      std::cerr << "ERROR: failed to initialize model, error code: " << ret
                << std::endl;
      return EXIT_FAILURE;
    }
    log_verbose("[smoke] model initialized");

    // ---------------------------------------------------------------------
    // Step 3: prepare the input buffer.
    // ---------------------------------------------------------------------
    const ParsedShape parsed_in = parse_shape(in_shape);
    const ParsedShape parsed_out = parse_shape(out_shape);
    const size_t in_elem_bytes = dtype_bytes(in_dtype);
    const size_t out_elem_bytes = dtype_bytes(out_dtype);
    // Natural on-disk size for --input, e.g. 1 byte/elem for UINT8 -- see
    // widen_to_float() for why this is NOT the same as the in-memory buffer
    // size handed to inference().
    const size_t in_file_bytes = parsed_in.elems * in_elem_bytes;
    const size_t out_bytes = parsed_out.elems * out_elem_bytes;
    log_verbose("[smoke] input[0] dim %s dtype %s (%zu elems, %zu bytes on "
                "disk / %zu bytes as float32)",
                in_shape.c_str(), in_dtype.c_str(), parsed_in.elems,
                in_file_bytes, parsed_in.elems * sizeof(float));
    log_verbose("[smoke] output[0] dim %s dtype %s (%zu elems, %zu bytes)",
                out_shape.c_str(), out_dtype.c_str(), parsed_out.elems,
                out_bytes);

    // Single input assumed (matches the single "input" layer built above).
    // Always float32 in memory -- see widen_to_float()'s doc comment for why
    // this is true regardless of --in-dtype. This buffer must outlive the
    // inference() call below.
    std::vector<float> in_storage(parsed_in.elems, 0.0f);

    if (!input_file.empty()) {
      log_verbose("[smoke] loading input from %s", input_file.c_str());
      std::vector<uint8_t> raw = read_binary_file(input_file);
      if (raw.size() != in_file_bytes) {
        std::cerr << "ERROR: --input size mismatch. File has " << raw.size()
                  << " bytes, expected " << in_file_bytes << " bytes ("
                  << parsed_in.elems << " " << in_dtype << " elements)"
                  << std::endl;
        return EXIT_FAILURE;
      }
      widen_to_float(in_dtype, raw, in_storage);
    }

    std::vector<float *> in_ptrs{in_storage.data()};

    // ---------------------------------------------------------------------
    // Step 4: run inference.
    // ---------------------------------------------------------------------
    log_verbose("[smoke] running inference...");
    std::vector<float *> outputs =
      model->inference(parsed_in.batch, in_ptrs, {});
    if (outputs.empty() || outputs[0] == nullptr) {
      std::cerr << "ERROR: inference returned no output" << std::endl;
      return EXIT_FAILURE;
    }
    log_verbose("[smoke] inference completed");
    log_verbose("[smoke] output size: %zu bytes (%zu %s elems)", out_bytes,
                parsed_out.elems, out_dtype.c_str());

    if (g_verbose) {
      // Printed as raw bytes rather than floats: outputs[0] is only a
      // float*-typed pointer by ccapi convention (see the in_ptrs comment
      // above) -- for a non-FP32 out_dtype, dereferencing it as float would
      // read garbage across the wrong element boundaries.
      const auto *out_raw = reinterpret_cast<const uint8_t *>(outputs[0]);
      const size_t n = out_bytes < 16 ? out_bytes : 16;
      fprintf(stderr, "[smoke] first %zu output byte(s):", n);
      for (size_t i = 0; i < n; ++i) {
        fprintf(stderr, " %02x", out_raw[i]);
      }
      fprintf(stderr, "\n");
    }

    if (!dump_file.empty()) {
      write_binary_file(dump_file, outputs[0], out_bytes);
      std::cout << "wrote " << out_bytes << " bytes to " << dump_file
                << std::endl;
    }

    // ---------------------------------------------------------------------
    // Step 5: compare against the golden file.
    // ---------------------------------------------------------------------
    if (golden_path.empty() || golden_path == "/dev/null") {
      std::cout << "PASS (inference ran; no golden file to compare)"
                << std::endl;
      std::cout << "Expected golden file size: " << out_bytes << " bytes"
                << std::endl;
      return EXIT_SUCCESS;
    }

    log_verbose("[smoke] loading golden file...");
    std::vector<uint8_t> golden = read_binary_file(golden_path);
    if (golden.size() != out_bytes) {
      std::cerr << "ERROR: size mismatch. Golden: " << golden.size()
                << " bytes, output: " << out_bytes << " bytes" << std::endl;
      return EXIT_FAILURE;
    }

    const uint8_t *out_bytes_p = reinterpret_cast<const uint8_t *>(outputs[0]);
    if (std::memcmp(golden.data(), out_bytes_p, out_bytes) != 0) {
      std::cerr << "ERROR: output mismatch vs golden file" << std::endl;
      int reported = 0;
      const int max_reports = 10;
      for (size_t i = 0; i < out_bytes && reported < max_reports; ++i) {
        if (golden[i] != out_bytes_p[i]) {
          fprintf(stderr, "  byte %zu: expected 0x%02x got 0x%02x\n", i,
                  golden[i], out_bytes_p[i]);
          reported++;
        }
      }
      if (reported == max_reports) {
        std::cerr << "  (and possibly more...)" << std::endl;
      }
      return EXIT_FAILURE;
    }

    std::cout << "PASS (output matches golden file)" << std::endl;
    return EXIT_SUCCESS;

  } catch (const std::exception &e) {
    std::cerr << "EXCEPTION: " << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << "EXCEPTION: unknown error" << std::endl;
    return EXIT_FAILURE;
  }
}
