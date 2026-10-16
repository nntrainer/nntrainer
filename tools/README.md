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


## LFM2 ternary MoE for HTP

Convert a local BF16-dequantized LFM2 checkpoint directly to the Gemma4 HTP
record layouts, without an intermediate FP32 model:

```bash
python tools/convert_lfm2_htp.py lfm2-24ba2b-expert-ternary-2bit \
  --output lfm2-24ba2b-ternary-tl2-npu \
  --tokenizer-source lfm2-24ba2b
python test/htp/host/lfm2_conversion_check.py
```

Requires NumPy and a host C compiler. The output contains
`nntr_lfm2_tl2_affine.bin`, `nntr_config.json`, tokenizer files and
`conversion_report.json` with every tensor's name, shape, byte offset and
verification metrics. Existing destinations are refused. Source files are
read only. Every source tensor must be consumed exactly once.

Experts preserve their per-output-channel ternary values exactly in native
`TL2CX_WH` records, with gate/up fused per expert. FCs, attention and conv
projections, and the LM head use `QS4CX_WH_AFFINE`, recovering the original
INT4 grids without min/max requantization. BF16 round trips are exact;
execution uses the recovered FP32 grid, whose small difference from BF16 is
reported separately. Short-conv taps are preserved as FP32 in newest-first
order. Embedding, norms and router values are also preserved as FP32.
Embedding scale is 1; routing is sigmoid plus expert bias; FFNs use SwiGLU.

The HTP build registers `Lfm2MoeForCausalLM` with
`lfm2_execution_profile=htp_fused` and `lfm2_weight_format=htp_fused_v1`.
The QuickDotAI catalog id is `lfm2-24ba2b-htp`, backend NPU, quantization
W4A32; its API directory is `<model_base_path>/lfm2-24ba2b-htp-w4a32`.
Rebuild both ARM libraries and the v81 DSP library: the LFM2 profile adds
64-expert sigmoid routing, WH dense FFNs wider than 8192 channels, and
64-wide INT8 decode attention. Prefill uses `attention_kv_dtype=q8_row`
for LFM2's unnormalized values; the fixed-scale Gemma path is unchanged.
Provide that matching DSP library just as for the Gemma4 HTP descriptor.
The current LFM2 profile supports one prompt per handle, a single prefill
block (default 512 tokens), and a 2048-token context. Raise `init_seq_len`
within `max_seq_len` before loading for longer prompts; chunked prefill,
weight reload and KV snapshots are refused. Conversion and host checks do
not establish device inference accuracy or throughput.
