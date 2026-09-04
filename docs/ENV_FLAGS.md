# Runtime environment-variable reference

Environment variables read by nntrainer at run time, and by its test harnesses.

**This file lists only variables that are read by code in this tree.** Every entry below names
the symbol that reads it, and every one was checked with a `getenv` grep over `nntrainer/`,
`Applications/`, `api/` and `test/`. Nothing here is aspirational.

## Maintenance rule

**A variable gets a row here in the same change that adds the `getenv` call that reads it, and
loses its row in the change that removes the call.** A reference that documents flags the tree
does not read is worse than no reference: a reader cannot tell which half is real, and there is no
mechanical way to find out short of re-grepping the source. The check is one line:

```sh
git grep -hoE '(std::)?getenv\("[A-Za-z_][A-Za-z_0-9]*"\)' -- nntrainer Applications api test \
  | grep -oE '"[^"]+"' | tr -d '"' | sort -u
```

Every name that command prints should appear below, and vice versa.

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
| `NNTRAINER_PATH` | A **single** directory searched for dynamically loadable layer and optimizer plugins, in addition to the build-time configured plugin path. Ignored with a warning if the path does not exist. | unset | `nntrainer/app_context.cpp: getPluginPaths` |

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

- There is no environment variable that selects the compute backend. The backend is chosen with
  the `engine=` model/layer property, validated in `nntrainer/engine.cpp:
  Engine::parseComputeEngine` against the values in `nntrainer/utils/base_properties.h:
  ComputeEngineTypeInfo::EnumStr`.
- Build-time options (`-Denable-opencl`, `-Denable-htp`, and the rest) gate *availability*, not
  selection; see `meson_options.txt`.
