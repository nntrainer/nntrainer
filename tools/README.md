# Tools

This section explains how to utilize the NNTrainer tools.

## Running Unit Tests on Android Devices

Here we will guide you through running unit tests on an Android device.

#### Preparing for Android Testing

Prerequisite: Install and configure the NDK

```
$ ./tools/android_test.sh
```

#### Generating Layer Golden Data

```
$ meson build [flags...]
$ cd build
$ adb push res/ /data/local/tmp/nntr_android_test
```

Please note that golden data is necessary to execute layer-related tests.

#### Executing Unit Tests on the Android Device

```
$ adb shell
$ cd /data/local/tmp/nntr_android_test/
$ export LD_LIBRARY_PATH=.
$ ./unittest_nntrainer_tensor
```


## LFM2 SR15 MoE for HTP

Convert the rotated SR15 checkpoint to the single supported LFM2 HTP pack:

```bash
python tools/convert_lfm2_sr15.py /path/to/sr15-checkpoint \
  --encodings /path/to/encodings \
  --output /path/to/lfm2-24ba2b-htp-w4a32 \
  --tokenizer-source /path/to/tokenizer \
  --rotation both
python test/htp/host/lfm2_conversion_check.py
```

Requires NumPy and a host C compiler. The output contains
`LFM2-24B-A2B-Ternary-INT4.bin`, `nntr_config.json`, tokenizer files,
`activation_scales.bin` and `conversion_report.json` with tensor and
rotation offsets and verification metrics. Existing destinations are
refused, and every source tensor must be consumed exactly once.

Experts use native `TL2CX_WH`; projections, the router and LM head use
`QS4CX_WH_AFFINE`. Embeddings use `Q4_0`, short convolutions use `DW3_INT4`,
and norms and expert bias use FP32. The converter verifies serialized
records and reports embedding requantization error. Activation quantization
is dynamic; the activation sidecar is retained but not consumed at runtime.
Both expert rotations are required for correct SR15 inference.

The model factory and QuickDotAI API require
`lfm2_execution_profile=htp_fused` and `lfm2_weight_format=sr15_v2`.
The catalog id is `lfm2-24ba2b-htp`, backend NPU, quantization W4A32;
its directory is `<model_base_path>/lfm2-24ba2b-htp-w4a32`.
See the [LFM2 HTP guide](../docs/lfm2-htp-user-guide.md) for the pack
contract, build, runtime settings and validation limits.
