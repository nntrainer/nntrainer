# Building nntrainer with the QNN (Qualcomm AI Engine Direct) backend

The QNN backend lets nntrainer offload supported subgraphs to the
Qualcomm Hexagon NPU on Snapdragon devices via Qualcomm's *AI Engine
Direct* (QNN) runtime.

The Qualcomm QNN SDK itself is **proprietary** ("Confidential and
Proprietary - Qualcomm Technologies, Inc."). It cannot be redistributed
inside this repository. nntrainer therefore follows the standard
**bring-your-own-SDK** pattern used by CUDA, ROCm, and oneAPI: the user
installs the SDK locally and points the build at it via a meson option.

This document describes how to do that.

## 1. Obtain the QNN SDK

The SDK is distributed by Qualcomm via the *Qualcomm Software Center* /
*Qualcomm Package Manager*:

  https://qpm.qualcomm.com/

Search for **Qualcomm AI Engine Direct SDK** (a.k.a. QNN SDK). Download
requires a (free) Qualcomm developer account and acceptance of
Qualcomm's license terms.

### Tested versions

nntrainer's QNN backend has been verified against:

| QNN SDK    | Status |
| ---------- | ------ |
| 2.20.x     | OK     |
| 2.22.x     | OK     |
| 2.24.x     | OK     |

Older 1.x SDKs and 2.10-2.18 use a different op-package layout and are
not supported. Newer 2.x releases generally work but are
unverified — please report regressions.

## 2. Extract the SDK

After unpacking, the SDK root contains roughly this layout:

```
qnn-2.x.x.x/
├── bin/                    # qnn-net-run, qnn-context-binary-generator, ...
├── include/
│   ├── QNN/                # QnnInterface.h, QnnTensor.h, QnnTypes.h, ...
│   ├── HTP/                # HTP-specific headers
│   └── System/             # QnnSystemInterface.h ...
├── lib/
│   ├── x86_64-linux-clang/ # Host libraries (libQnnCpu.so, libQnnHtp.so ...)
│   └── aarch64-android/    # Device libraries
└── share/
```

nntrainer's build only consumes headers under `include/`; the runtime
shared libraries (`libQnnHtp.so`, `libQnnCpu.so`, ...) are dlopen'd at
runtime — see §5.

## 3. Configure the build

Pass the SDK root via `-Dqnn-sdk-root` together with `-Denable-npu=true`:

```bash
meson setup build \
    -Denable-npu=true \
    -Dqnn-sdk-root=/opt/qcom/aistack/qnn-2.24.0.240626
ninja -C build
```

The build will validate the path early and produce a clear error if it
is missing or not a directory:

```
nntrainer/qnn/meson.build:9:0: ERROR: enable-npu=true requires
-Dqnn-sdk-root=<path-to-qcom-qnn-sdk>. Obtain the QNN SDK from
https://qpm.qualcomm.com/ and pass its root directory. See
docs/backend_guide/QNN_BUILD.md.
```

When `enable-npu=false` (the default) the `qnn-sdk-root` option is
ignored — default builds need nothing from Qualcomm and are unaffected
by all of this.

## 4. What gets built

With `enable-npu=true`, the QNN integration is compiled into a
**separate plugin shared library** (`libqnn_context.so`). The main
`libnntrainer.so` is unchanged. The plugin is loaded dynamically by
`Engine::registerContext` only when the user actually requests the
`"qnn"` backend; this keeps the QNN runtime dependency from leaking
into binaries that do not need it.

The plugin contains:

- `nntrainer/qnn/jni/qnn_rpc_manager.{h,cpp}` — RPC memory wiring for
  shared-buffer DMA between the host and the HTP.
- `nntrainer/qnn/jni/qnn_context_var.h` — `QNNBackendVar` (the typed
  payload reachable through `ContextData::as<QNNBackendVar>()`).
- `nntrainer/qnn/jni/iotensor_wrapper.hpp` — adapter around the QNN
  sample-app `IOTensor` / `DataUtil` utilities (these utilities live
  inside the SDK, *not* in this repo).
- `nntrainer/qnn/jni/qnn/op/QNN{Linear,Graph}.{h,cpp}` — Layer-level
  ops that capture into a QNN graph (see ARCHITECTURE.md §5 for why
  QNN integrates at Layer granularity rather than at ComputeOps op
  granularity).
- `nntrainer/qnn/jni/qnn/qnn_properties.{h,cpp}` — string ⇄ struct
  converters for QNN-specific layer properties (quantization params,
  tensor shapes, ...).

Everything else under `nntrainer/qnn/jni/qnn/{Log,PAL,Utils,WrapperUtils}`
that previously sat in-tree was Qualcomm sample code. It is now
expected to come from the SDK (or, where the SDK does not ship a
particular utility, to be replaced by an equivalent from elsewhere in
nntrainer). Issues uncovered while doing this should be filed against
this repo, not Qualcomm.

## 5. Runtime: locating libQnnHtp.so

At run time the QNN runtime libraries (`libQnnHtp.so`,
`libQnnHtpV73Stub.so`, `libQnnSystem.so`, ...) need to be reachable by
`dlopen`. The simplest option is `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=/opt/qcom/aistack/qnn-2.24.0.240626/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
./build/test/unittest/unittest_nntrainer_qnn   # or any binary using the qnn ctx
```

On Android / device builds, place the matching `aarch64-android/`
libraries in `/data/local/tmp/qnn/` (or whichever path your APK / shell
binary expects) and adjust `LD_LIBRARY_PATH` accordingly.

We deliberately do *not* burn an `rpath` into the nntrainer binaries —
the SDK install path is environment-specific and an rpath would either
be wrong on every other developer's machine or force everyone to use
the same layout.

## 6. Verifying the build

Quick sanity check that the QNN context registers without exercising
the NPU itself:

```bash
ninja -C build test
build/test/unittest/unittest_nntrainer_engine --gtest_filter='*QNN*'
```

Tests that exercise the actual HTP runtime are gated behind the
presence of the SDK shared libraries and a Snapdragon device with HTP
support. If you only have the headers (no `libQnnHtp.so` runtime,
no device), expect those tests to be skipped.

## 7. Troubleshooting

- **`fatal error: 'QnnInterface.h' file not found`** — `qnn-sdk-root`
  points at the wrong directory, or your SDK layout has the headers
  somewhere unexpected. Confirm `<sdk-root>/include/QNN/QnnInterface.h`
  exists.
- **`undefined reference to QnnInterface_getProviders`** — the QNN
  runtime is `dlopen`'d, not link-time linked, so this should not
  occur during build. If it does, the build is mis-configured: file
  an issue.
- **`libQnnHtp.so: cannot open shared object file`** at run time —
  see §5 (`LD_LIBRARY_PATH`).
- **Wrong HTP architecture stub** (`HtpV68Stub` vs `HtpV73Stub` etc.) —
  pick the stub matching your SoC. The QNN SDK README has a table.

## 8. Why this is not a git submodule

The QNN SDK has no public git mirror. Qualcomm distributes it
exclusively through the Software Center as versioned tarballs that
require accepting a license. A submodule would either point at
nothing (no public URL) or at a private mirror that most contributors
cannot access. The BYO-SDK pattern matches the precedent set by
CUDA / ROCm / oneAPI for the same reason.
