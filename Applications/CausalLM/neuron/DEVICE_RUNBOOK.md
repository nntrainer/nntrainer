# MediaTek Neuron Backend — Device Smoke Test Runbook

**Date:** 2026-07-30  
**Target Device:** MediaTek mt6991 (Dimensity 9400)  
**Milestone:** 1 (Pluggable nntrainer Context + neuron_graph layer + standalone smoke test)

## Overview

This package contains a standalone smoke-test binary that validates the nntrainer MediaTek Neuron backend on device. It loads a precompiled `.dla` (Deep Learning Archive) file, executes it via the Neuron Runtime, and compares output against a golden reference file.

## Contents

```
neuron-smoke-device-bundle/
├── nntrainer_neuron_smoke        # Standalone test binary (arm64-v8a)
├── libneuron_context.so          # Neuron backend plugin library
├── libnntrainer.so               # Required (DT_NEEDED)
├── libccapi-nntrainer.so         # Required (DT_NEEDED)
├── libc++_shared.so              # Required (DT_NEEDED)
├── libneuron_runtime.so          # MediaTek Neuron Runtime (mt6991-specific)
├── model.dla                     # Precompiled model
├── golden_output.bin             # Expected output (optional; see Step 5)
├── DEVICE_RUNBOOK.md             # This file
└── setup.sh                      # Device setup script
```

The first five files all come out of `builddir/jni/arm64-v8a/` after a build
with `-Denable-neuron=true`. The three `DT_NEEDED` libraries are not optional —
without them the dynamic linker refuses to start the binary.

## Prerequisites

- **Device:** MediaTek mt6991 (Dimensity 9400) or compatible SoC with Neuron NPU support
- **Android version:** 12+ (NDK 23+ compatible)
- **Root access:** Required for pushing files to `/system/lib64` or `/vendor/lib64`
- **adb:** Android Debug Bridge, configured on your development host

## Setup Steps

### 1. Connect Device and Verify adb

```bash
adb devices
# You should see your device listed with "device" status.
```

### 2. Extract and Push Files to Device

```bash
# On development host:
cd neuron-smoke-device-bundle

# Push everything to /data/local/tmp (world-writable, no root needed).
adb push . /data/local/tmp/
adb shell chmod +x /data/local/tmp/nntrainer_neuron_smoke

# Verify files arrived:
adb shell ls -lh /data/local/tmp/
```

Or just run `./setup.sh`, which pushes the same set and fails loudly if a
required library is absent.

### 3. Set Runtime Library Path and Run the Test

The test cannot infer the `.dla`'s I/O geometry, so pass the real shapes.
The `1:1:1:1` defaults only fit a trivial network; a mismatch surfaces as an
explicit buffer-size error from the layer, not as silent bad output.

```bash
# On device shell:
adb shell

# Inside the device shell:
cd /data/local/tmp
export LD_LIBRARY_PATH=/data/local/tmp:$LD_LIBRARY_PATH
export NNTRAINER_NEURON_SMOKE_VERBOSE=1

# First run: execute the graph and record the output.
./nntrainer_neuron_smoke model.dla \
    --in-shape=1:3:224:224 --out-shape=1:1000:1:1 --dump=golden_output.bin
```

Full usage:

```
nntrainer_neuron_smoke <model.dla> [golden.bin] [options]

  --in-shape=B:C:H:W   input shape of the .dla  (default 1:1:1:1)
  --out-shape=B:C:H:W  output shape of the .dla (default 1:1:1:1)
  --input=<file>       raw float32 input (default: all zeros)
  --dump=<file>        write produced output to <file>
```

**Expected output:**
```
[smoke] verbose logging enabled
[smoke] DLA path: model.dla
[smoke] in-shape: 1:3:224:224  out-shape: 1:1000:1:1
[smoke] looking up neuron context...
[smoke] neuron context registered
[smoke] creating model...
[smoke] compiling model...
[smoke] model compiled
[smoke] model initialized
[smoke] input[0] dim 1:3:224:224 (150528 elems)
[smoke] output[0] dim 1:1000:1:1 (1000 elems)
[smoke] running inference...
[smoke] inference completed
[smoke] output size: 4000 bytes (1000 floats)
[smoke] first 8 output value(s): ...
wrote 4000 bytes to golden_output.bin
PASS (inference ran; no golden file to compare)
Expected golden file size: 4000 bytes
```

### 4. Interpret Results

| Exit Code | Outcome |
|-----------|---------|
| **0** | PASS — inference ran, output matches golden file |
| **1** | FAIL — runtime error, size mismatch, or output mismatch |

**Common failure modes:**

- **"neuron context not registered"** → `libneuron_context.so` not found or not linked properly. Check `LD_LIBRARY_PATH`.
- **"Failed to open file"** → `.dla`, golden file, or binary not found. Verify all files are in `/data/local/tmp`.
- **"size mismatch"** → Golden file is wrong size; regenerate it or compare output sizes.
- **"output mismatch"** → NPU computation differs from reference. Check device state, verify `.dla` was compiled for mt6991.

## Detailed Test Flow

1. **Backend Registration:** The smoke test calls `Engine::Global().getRegisteredContext("neuron")`, which:
   - Loads `libneuron_context.so` via dlopen.
   - Calls `NeuronContext::init()` → `NeuronApi::load("libneuron_runtime.so")`.
   - Registers the `neuron_graph` layer factory.

2. **Model Construction:** Builds a 2-layer model programmatically via
   `ml::train::createLayer` + `addLayer` (no ini file, so nothing has to be
   written to the device filesystem):
   - **Input layer:** `input_shape=<--in-shape>`.
   - **neuron_graph layer:** `path=<model.dla>`, `dim=<--out-shape>`,
     `tensor_dtype=FP32`, `tensor_type=OUT_TENSOR`, `engine=neuron`.

   Only an `OUT_TENSOR` entry is declared. Unlike the QNN backend, no
   `IN_TENSOR` weight tensors are requested: a `.dla` carries its own weights,
   so nntrainer only needs the output geometry in order to size the output
   tensor.

3. **Initialization:** `compile()` then `initialize()` drives
   `NeuronGraph::finalize()`, which records the `.dla` path and requests the
   output tensor. The runtime itself is created lazily on first
   `forwarding()` (`NeuronVar::makeRuntime`, idempotent and cached per path).

4. **Inference:** `model->inference(batch, inputs)` runs the input layer
   (zeros, or `--input`) and forwards through the neuron_graph layer, which:
   - Validates each nntrainer tensor is at least the size the network needs.
   - Sets input buffers via `NeuronRuntime_setInput` (index-based: nntrainer
     input *i* maps to Neuron handle *i*).
   - Sets output buffers via `NeuronRuntime_setOutput`.
   - Calls `NeuronRuntime_inference`.

5. **Validation:** Compares the output buffer byte-for-byte against the golden
   file, reporting the first mismatching bytes on failure.

## Troubleshooting

### Symbol Resolution Errors

If you see errors like `undefined reference to NeuronRuntime_*`, the dynamic linker couldn't find or dlopen `libneuron_runtime.so`. Verify:
- File is in `/data/local/tmp` with execute permissions.
- `LD_LIBRARY_PATH` includes `/data/local/tmp`.
- Device has Neuron NPU support (check `getprop ro.hardware | grep -i neuron` or similar).

### Buffer Size Mismatch

An error like:

```
neuron_graph output 0: nntrainer tensor buffer is 4 bytes but the Neuron
network requires at least 4000 bytes; adjust this layer's dim/tensor_dtype
properties to match
```

means the shape you passed doesn't match the `.dla`. This is the layer's own
check, which runs before inference, so it fails cleanly instead of producing
garbage. Pass the correct `--in-shape` / `--out-shape`.

To determine the `.dla`'s shape:
```bash
# On a host with the NeuroPilot SDK:
neuron_sdk/host/bin/ncc-tflite --info model.dla
# (Prints I/O metadata, if available)
```

Failing that, run with `NNTRAINER_NEURON_SMOKE_VERBOSE=1` — the required byte
count in the error message tells you the size, and dividing by 4 gives the
float32 element count to distribute across your shape.

### Segfaults or Crashes

- Enable verbose logging: `export NNTRAINER_NEURON_SMOKE_VERBOSE=1`.
- Check `dmesg` for kernel messages or MMU faults: `adb shell dmesg | tail -50`.
- If the NPU itself crashes, the Neuron Runtime may return an error code; the smoke test will report it.

## Generating Golden Files

`--dump=<file>` writes whatever the NPU produced. That makes it easy to seed a
golden, but note what it does and does not prove: a dumped file will always
match on the next run, so on its own it is a **self-consistency check, not a
correctness check**.

To get a golden that actually validates numerics:

1. Run a known-good reference (e.g. TFLite executing the source model on CPU)
   with the same input the smoke test uses — all zeros by default, or whatever
   you pass to `--input`.
2. Save its output as raw float32, native byte order, no header:
   ```cpp
   std::ofstream out("golden_output.bin", std::ios::binary);
   out.write((const char *)output_tensor.data(), output_tensor.size() * sizeof(float));
   ```
3. Push it and re-run the smoke test with the golden as the second argument.

Because the comparison is an exact `memcmp`, expect it to be strict: a
quantized `.dla` will generally not reproduce a float reference bit-for-bit.
For those models, either compare against a golden captured from a trusted
Neuron run on the same SoC, or treat `--dump` output as a regression baseline
after inspecting the values by hand.

## Environment Variables

- **`QUICK_DOT_AI_NEURON_LIB`:** Override the path to `libneuron_runtime.so`. Useful for testing against the SDK's dummy stub on a host without an NPU.
- **`QUICK_DOT_AI_NEURON_NULL_DEVICE`:** Set to `1` to use the no-hardware stub runtime (requires `QUICK_DOT_AI_NEURON_LIB` to point at `neuron_sdk/dummy/lib/libneuron_runtime.so`).
- **`NNTRAINER_NEURON_CONTEXT_SO`:** Explicit path to `libneuron_context.so`. Use this when the automatic registration in `engine.cpp` did not find the plugin on `LD_LIBRARY_PATH`.
- **`NNTRAINER_NEURON_SMOKE_VERBOSE`:** Set to `1` for detailed logging of each step.

## Next Steps (Phase 2+)

- **Layer properties:** Integrate quant params (input_quant_param / output_quant_param) into the layer config so the app layer can access scale/offset for de/requantization.
- **Multi-layer models:** Test models with multiple neuron_graph layers or mixed CPU/NPU execution.
- **Dynamic shapes:** Support `NeuronRuntime_setInputShape` for models with variable input dimensions.
- **Zero-copy I/O:** Wire up true ION/DMA-BUF buffers (currently all buffers are non-ION for simplicity).
- **Integration with CausalLM:** Plug the neuron_graph layer into the full app for LLM inference benchmarks.

## Support

For issues or questions:
1. Check the verbose log output (`NNTRAINER_NEURON_SMOKE_VERBOSE=1`).
2. Review the nntrainer Neuron backend docs: `nntrainer/docs/backend_guide/NEURON_BUILD.md`.
3. Consult the MediaTek NeuroPilot SDK documentation (especially the Neuron Runtime API in `neuron_sdk/*/include/neuron/api/RuntimeAPI.h`).
