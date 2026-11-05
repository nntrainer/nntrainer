# Runtime environment-variable reference

Environment variables read by nntrainer at run time, and by its test harnesses.

**This file lists only variables that are read by code in this tree, but it is not a complete
list.** Every entry below names the symbol that reads it, and every one was checked with a
`getenv` grep over `nntrainer/`, `Applications/`, `api/` and `test/`. Nothing here is
aspirational. The many GPU-backend tuning and diagnostic variables are not listed here; the
GPU ones that matter for running CausalLM are documented in
[`Applications/CausalLM/README_GPU.md`](../Applications/CausalLM/README_GPU.md).

## Maintenance rule

**A variable gets a row here in the same change that adds the `getenv` call that reads it, and
loses its row in the change that removes the call.** A reference that documents flags the tree
does not read is worse than no reference: a reader cannot tell which half is real, and there is no
mechanical way to find out short of re-grepping the source. The check is one line:

```sh
git grep -hoE '(std::)?getenv\("[A-Za-z_][A-Za-z_0-9]*"\)' -- nntrainer Applications api test \
  | grep -oE '"[^"]+"' | tr -d '"' | sort -u
```

Every name listed below must appear in that command's output. The reverse does not hold yet: the
command also prints the unlisted backend variables mentioned above.

**Environment variables are a stopgap, not an interface.** A per-hardware or per-model behaviour
selected by an environment variable is a decision the code has not yet learned to derive. The
direction of travel is from a variable to a device-capability query or a resolved execution plan
— see [`backend_guide/ARCHITECTURE_REFACTOR.md`](backend_guide/ARCHITECTURE_REFACTOR.md) §10 —
never the reverse. Prefer adding a capability over adding a flag; where a flag is unavoidable,
give it a working default so that a bare run needs no environment at all.

---

## Runtime

| Variable | Meaning | Default | Read by |
|---|---|---|---|
| `NNTR_NUM_THREADS` | Number of CPU compute worker threads. Takes priority over the `NNTR_NUM_THREADS` compile flag. | `hardware_concurrency() / 2` (minimum 1) | `nntrainer/utils/thread_manager.h: ThreadManagerConfig::defaultComputeThreads` |
| `NNTR_XE3_SYNC` | `1`: `clFinish` after every OpenCL dispatch that binds shared virtual memory. Off by default: the in-order queue orders kernel-to-kernel hand-offs, and the one ordering the drain was providing (a shared-memory unmap behind a running kernel) is kept by `enqueueSVMUnmap` on coarse-grain devices. A diagnostic only. | unset = off | `nntrainer/opencl/opencl_command_queue_manager.cpp: needsCoarseSVMDrain` |
| `NNTR_MHA_GPU` | `0`: keep attention on the host on the OpenCL gpu engine (the per-row loop, or the host flash arm). Otherwise a layer built with `use_gemm_attention` runs the device attention (flash prefill/decode, or the image arm below). | on for the gpu engine (`ClContext::initialize` sets it unless already set), unset on the cpu and cuda engines | `Applications/CausalLM/layers/mha_core.cpp: mha_device_attention_on`, `MHACoreLayer::one_batch_incremental_forwarding`; `Applications/CausalLM/kv_ring.h: kvRingArmAvailable` |
| `NNTR_KV_IMG_ATTN` | `0`: on Adreno, serve the device attention from the buffer (flash) kernels instead of the image2d K/V mirrors. | on for the gpu engine on an Adreno (Qualcomm vendor id) device, set by `ClContext::initialize`; unset elsewhere | `Applications/CausalLM/kv_ring.h: imageAttnRequested` |
| `NNTR_XE3_FC_SYNC` | `1`: `clFinish` after every quantised FC GEMM/GEMV on the OpenCL v8c path. A diagnostic only. | unset = off | `nntrainer/tensor/cl_operations/blas_kernel_interface.cpp: dotCl_v8c` |
| `NNTR_CLMEM_ALIAS_SVM` | `0`: give the device activation plane its own allocation instead of backing it with the shared SVM plane (`CL_MEM_USE_HOST_PTR`). Needed on a driver that shadow-copies a host-pointer buffer instead of aliasing it. Any other value: aliased, even for a model that asks for a private plane. | unset = aliased (one physical plane), except on Adreno for a model whose `prefersPrivateActivationPlane()` is true (private plane) | `nntrainer/tensor/cl_operations/cl_buffer_pool.cpp: ClBufferPool::devicePlaneBaseLocked` |
| `NNTR_V8C_DROP_PLAIN` | `1`: after the v8c device repack of a QS4CX weight, release the plain payload's pages. `0`: keep it resident. | unset = on for Android when the heap bypass is on, off elsewhere | `nntrainer/tensor/qs4cx_tensor.cpp: qs4cxDropPlainOn` |
| `NNTR_LOAD_ZEROCOPY` | `1`: when the loader reads through a file mapping (`-Dmmap-read=true`), build the v8c device weight straight from the mapping and never copy the nibbles into the tensor. `0`: copy. | unset = follows `qs4cxDropPlainOn()` | `nntrainer/layers/fc_layer_cl.cpp: FullyConnectedLayerCl::read` |
| `NNTR_LOAD_PREFETCH` | `0`: do not read the weight file ahead of the load workers. Otherwise, when the loader reads `.bin` weights through a file mapping (`-Dmmap-read=true`, not FSU), one helper thread issues `posix_fadvise(POSIX_FADV_WILLNEED)` over the file front to back, up to half of `MemAvailable`, until the workers finish. POSIX only. | unset = on | `nntrainer/models/neuralnet.cpp: WeightFileReadAhead` |
| `NNTRAINER_PATH` | A **single** directory searched for dynamically loadable layer and optimizer plugins, in addition to the build-time configured plugin path. Ignored with a warning if the path does not exist. | unset | `nntrainer/app_context.cpp: getPluginPaths` |

## Native HTP and MoE configuration

Native HTP/MoE/LFM2 runtime tuning now uses the `htp` and `moe` objects in
`nntr_config.json`. The former runtime environment variables are no longer
read; see the [configuration reference](gemma4-htp-code-guide.md#7-runtime-configuration-and-options)
for defaults, migration and strict validation. `ADSP_LIBRARY_PATH` remains
the DSP library search path. `NNTR_HTP_DIAGNOSTICS` is a build-script option,
not a runtime setting.

`htp.load_prefetch` overrides `NNTR_LOAD_PREFETCH` when explicitly set;
otherwise the shared loader retains its environment fallback. LFM2's CLI
sets the config override to `false` by default. Its `num_threads` JSON preset
also controls the compute thread count; it does not write process environment
variables. The public API does not install the CLI preset.

## QNN backend

Both are path overrides for the QNN backend's on-device assets; both are optional and both fall
back to a working default.

| Variable | Meaning | Default | Read by |
|---|---|---|---|
| `QUICK_DOT_AI_BASE_DIR` | Base directory the QNN backend resolves its relative asset paths against. | the current working directory, else a built-in fallback path | `nntrainer/qnn/qnn_context.cpp: resolve_quick_dot_ai_base_dir` |
| `QUICK_DOT_AI_QNN_BACKEND_EXT_CONFIG_PATH` | Path to the HTP backend-extensions JSON configuration. A relative value is resolved against the base directory above. | `<base>/htp_backend_ext_config.json` | `nntrainer/qnn/qnn_context.cpp: resolve_backend_extensions_config_path` |

## Test harnesses

Not read by the library. These configure the unit tests and the on-device test scripts; when they
are unset the affected tests skip rather than fail.

| Variable | Meaning | Read by |
|---|---|---|
| `NNTRAINER_RESOURCE_PATH` | Root for test resource files. When set, it replaces the built-in fallback search bases. | `test/nntrainer_test_util.cpp: getResPath` |
| `NNTRAINER_CAUSALLM_FIXTURE_DIR` | Override for the CausalLM differential-test fixture root. Needed on device, where the source-relative default does not resolve. | `test/unittest/models/causallm_test_utils.cpp: findFixtureDir` |
| `NNTR_QUANTIZE_BIN` | Path to the `nntr_quantize` binary used by the quantized-model tests. Set automatically by the on-device test script. | `test/unittest/models/causallm_test_utils.cpp` (`QUANTIZE_BIN_ENV`) |

---

## Notes

- The backend of a layer is chosen with the `engine=` model/layer property, validated in
  `nntrainer/engine.cpp: Engine::parseComputeEngine` against the registered contexts. Which
  contexts are brought up at all is selected by the `NNTR_ENGINE` environment variable (read
  through `nntr_engine_env()` in `nntrainer/utils/env_compat.h`), and the CausalLM application
  derives its default `engine=` from it (`causallm_engine()` in
  `Applications/CausalLM/llm_util.hpp`).
- Build-time options (`-Denable-opencl`, `-Denable-htp`, and the rest) gate *availability*, not
  selection; see `meson_options.txt`.
