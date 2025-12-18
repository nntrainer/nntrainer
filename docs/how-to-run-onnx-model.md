# How to run ONNX Model with NNTrainer

NNTrainer introduces early-stage support for running ONNX models. Currently, only a limited set of ONNX operations are supported, and the feature is in an initial development phase. Users can experiment with loading ONNX models and running simple graphs through the provided API.

## Prerequisites

You must have the following packages installed:

- `protobuf` (both library and `protoc` compiler)
- Standard NNTrainer dependencies

Install protobuf and required development tools:

```bash
sudo apt-get update
sudo apt-get install protobuf-compiler libprotobuf-dev
```

Make sure `protoc` is available:

```bash
protoc --version
```

Additionally, install NNTrainer dependencies if not already done:

```bash
sudo add-apt-repository ppa:nnstreamer/ppa
sudo apt-get update
sudo apt install meson ninja-build
sudo apt install gcc g++ pkg-config libopenblas-dev libiniparser-dev libjsoncpp-dev libcurl3-dev tensorflow2-lite-dev nnstreamer-dev libglib2.0-dev libgstreamer1.0-dev libgtest-dev ml-api-common-dev flatbuffers-compiler ml-inference-api-dev
```

Initialize NNTrainer submodules:

```bash
git submodule sync && git submodule update --init --depth 1
```

## Building NNTrainer with ONNX Support

During the Meson configuration step, you must explicitly enable ONNX interpreter support.  
By default, it is **disabled**.

To enable it, pass the following flag to the Meson command:

```bash
meson build -Denable-onnx-interpreter=true
ninja -C build install
```

This will trigger additional settings:

- Add `protobuf` dependency.
- Automatically generate `onnx.pb.cc` and `onnx.pb.h` from `onnx.proto` using `protoc`.
- Include the generated sources into the NNTrainer build.

## Running an Example ONNX Application

A minimal example application is provided at:

```
Applications/ONNX/jni/main.cpp
```

The example shows how to load and run an ONNX model using the NNTrainer API:

```cpp
#include <iostream>
#include <layer.h>
#include <model.h>
#include <nntrainer-api-common.h>
#include <optimizer.h>
#include <util_func.h>

int main() {
  auto model = ml::train::createModel();

  try {
    std::string path = "../../../../Applications/ONNX/jni/llama_example.onnx";
    model->load(path, ml::train::ModelFormat::MODEL_FORMAT_ONNX);
  } catch (const std::exception &e) {
    std::cerr << "Error during load: " << e.what() << "\n";
    return 1;
  }

  try {
    model->compile();
  } catch (const std::exception &e) {
    std::cerr << "Error during compile: " << e.what() << "\n";
    return 1;
  }

  try {
    model->initialize();
  } catch (const std::exception &e) {
    std::cerr << "Error during initialize: " << e.what() << "\n";
    return 1;
  }

  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

  return 0;
}
```

After building NNTrainer, you can run the example application:

```bash
cd build/Applications/ONNX/jni
./nntrainer_onnx_example
```

It will:

- Load the provided ONNX model (`llama_example.onnx`)
- Compile and initialize the model
- Print a summary of the model structure

Then the output should look like this:

```bash
================================================================================
          Layer name          Layer type    Output dimension         Input layer
================================================================================
         onnx__add_6               input             1:1:1:1
--------------------------------------------------------------------------------
 onnx__add_6/generat            multiout             1:1:1:1         onnx__add_6
--------------------------------------------------------------------------------
                 sin               input            1:1:1:64
--------------------------------------------------------------------------------
 sin/generated_out_0            multiout            1:1:1:64                 sin
--------------------------------------------------------------------------------

...

--------------------------------------------------------------------------------
   model_norm_cast_1                cast          1:1:1:2048      model_norm_mul
--------------------------------------------------------------------------------
    model_norm_mul_1            multiply          1:1:1:2048   model_norm_weight
                                                              model_norm_cast_1
--------------------------------------------------------------------------------
      lm_head_matmul              matmul         1:1:1:50304    model_norm_mul_1
                                                               onnx__matmul_3531
================================================================================
```

## Notes and Limitations

- **Supported Operations**: Only a subset of ONNX operations is currently supported.
- **Model Compatibility**: Complex ONNX models may not load properly.
- **Development Stage**: ONNX interpreter support is still experimental and evolving.
---
