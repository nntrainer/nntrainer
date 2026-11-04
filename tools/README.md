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


## Model pack converters for the Hexagon NPU

`tools/convert_lfm2_sr15.py` builds the LFM2 24B-A2B HTP pack and
`tools/convert_tl2_model.cpp` is the native helper behind the Gemma4 TL2
pack conversion. Both are documented in
[Model packs](../docs/quickai/model-packs.md).
