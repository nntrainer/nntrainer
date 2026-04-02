---
title: How to run examples - Android
...

# How to run examples - Android

If at any point of this guide, the user encounters an error or some unexpected behavior, the first step should be checking the *Troubleshooting* section at the end of this document.

## Prerequisites

First, the user should ensure they have all necessary dependencies installed.

```bash
$ sudo apt-get update
$ sudo apt-get install tar wget gzip libglib2.0-dev libjson-glib-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libunwind-dev googletest liborc-0.4-dev flex bison libopencv-dev pkg-config python3-dev python3-numpy python3 meson ninja-build libflatbuffers-dev flatbuffers-compiler protobuf-compiler
```

After that, it is necessary to configure the NDK tool chain to build the `nntrainer` library and its applications for Android. In this guide, the version for Linux x86/64 is used. It's also recommended to use release *r26d*.

It can be downloaded from the [Google webpage](https://developer.android.com/ndk/downloads), or by executing the command below.

```bash
# note: it's possible to replace 'r26d' in URL with another correct release number, but versions before 'r23c' must also have platform 'linux-x86_64' instead of 'linux'
$ wget https://dl.google.com/android/repository/android-ndk-r26d-linux.zip
```

After the NDK is downloaded, it needs to be unzipped & added to environment variables for later steps.

```bash
$ ls
android-ndk-r26d-linux.zip
$ unzip android-ndk-r26d-linux.zip
$ ls
android-ndk-r26d
# note: exports below can also be added to ~/.bash_profile or ~/.bashrc to set them permanently
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PWD}/android-ndk-r26d/
$ export PATH=$PATH:${PWD}/android-ndk-r26d/
$ export ANDROID_NDK=${PWD}/android-ndk-r26d/
```

## Build `nntrainer`

Once the NDK tool chain is configured, the user can build the `nntrainer.so` for Android.

Currently, the architecture is set to `arm64-v8a` in the `APP_ABI` variable in `jni/Application.mk` file. If the user wishes to use `armeabi-v7a`, they need to make the modification in the file themself.

There are two ways to build the `nntrainer` for Android.

### Build using shell script

For the user's convenience, the build process has been automated in a shell script in `tools/package_android.sh`. It is the recommended way of building the library.

```bash
$ ./tools/package_android.sh
$ ls builddir/android_build_result
Android.mk  conf  examples  include  lib
$ ls builddir/android_build_result/lib/arm64-v8a
libcapi-nntrainer.so  libccapi-nntrainer.so  libc++_shared.so  libnnstreamer-native.so  libnntrainer.so
```

### Build using `meson`

Build can also be conducted manually using `meson`.

```bash
$ meson setup build -Dplatform=android -Dopenblas-num-threads=1 -Denable-tflite-interpreter=false -Denable-tflite-backbone=false -Denable-fp16=true -Denable-opencl=true -Dhgemm-experimental-kernel=false
$
$ meson compile -C build
$
$ pushd build/jni
$ ndk-build NDK_PROJECT_PATH=./ APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk -j $(nproc)
$ popd
```

Compiled libraries are located in the `build/jni/libs/<ARCH>` directory.

```bash
$ ls build/jni/libs/arm64-v8a
libcapi-nntrainer.so  libccapi-nntrainer.so  libc++_shared.so  libnnstreamer-native.so  libnntrainer.so
```

## Build applications

The user can also build applications located in the `Applications` directory. 

In this document, `LogisticRegression` application is used as an example, but all applications containing files `jni/Application.mk` and `jni/Android.mk` can be build accordingly.

```bash
$ cd Applications/LogisticRegression/jni
$ ls
Android.mk  Application.mk  CMakeLists.txt  main.cpp  meson.build
$
$ ndk-build NDK_PROJECT_PATH=./ APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk -j $(nproc)
$
$ ls
Android.mk  Application.mk  CMakeLists.txt  libs  main.cpp  meson.build  obj
$ ls libs/arm64-v8a
libc++_shared.so  libnntrainer.so  nntrainer_logistic
```

## Deploying on device & running

After successful compilation, the binary can be pushed to an Android device using `adb push`.

```bash
$ adb shell mkdir -p /data/local/tmp/nntrainer
$ adb push <BINARY_NAME> /data/local/tmp/nntrainer/
```

Accordingly, all shared libraries inside compilation directory, and additional resources can also be pushed to the device.

Then, it can be ran using ADB shell.

```bash
$ adb shell chmod +x /data/local/tmp/nntrainer/<BINARY_NAME>
$ adb shell ./data/local/tmp/nntrainer/<BINARY_NAME>
```

## Troubleshooting

If build process fails, the user should check if the `ANDROID_NDK` environment variable is set, and if the NDK toolkit location is added to `PATH` and `LD_LIBRARY_PATH` variables.

___

Some applications require the built libraries to be present in root directory of the repository, in directory `libs` (eg. `libs/arm64-v8a`). If the user gets an error 

```
Android NDK: ERROR:Android.mk:nntrainer: LOCAL_SRC_FILES points to a missing file
```

copying the compiled libraries from their build directory to `libs` in root directory may solve the problem. 

___

If deployment on mobile device fails, the user may try the command `adb devices` to make sure their mobile device is visible to ADB. 

If it's not, make sure the Developer Options on the device are turned on, and *ADB Debugging* is enabled.

___

If after launching the application on device, shared libraries cannot be found during runtime, the user can add the deployment directory to `LD_LIBRARY_PATH` env variable, and then try launching the app.

```bash
$ adb shell
> export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/data/local/tmp/nntrainer
> ./data/local/tmp/nntrainer/<BINARY_NAME>
```

___


