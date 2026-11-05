# QuickDotAI C API

The public, handle-based C API of nntrainer's CausalLM engine. The canonical
header is `quick_dot_ai_api.h`; it is self-contained for C and C++
consumers. The Android library ([QuickDotAI](../Android/QuickDotAI/README.md))
and the command-line test tool `quick_dot_ai_test` are built on it.

The API covers model discovery through a catalog of string ids, load and
destroy, streaming text generation from a prompt, from OpenAI-style
messages or from a raw OpenAI JSON request, cancellation, grammar-constrained
tool output through xgrammar, multimodal input for vision models,
performance metrics, and a load diagnostic string.

## Build and link

`Applications/CausalLM/build_android.sh --htp` (or without `--htp` for the
CPU variant) produces, under `builddir_app/<variant>/`:

- `libquick_dot_ai_api.so`: the public C ABI in this directory.
- `libcausallm.so`: models, layers, tokenizer, xgrammar.
- `libnntrainer.so`, `libccapi-nntrainer.so` (under
  `builddir/android_build_result/lib/arm64-v8a/`): the engine.

A native consumer includes `quick_dot_ai_api.h` and links
`libquick_dot_ai_api.so`. At run time the four libraries above plus
`libc++_shared.so` must be on the library path; the HTP build also needs
`libsdkl.so` (HexKL) on the library path and `libnntr_hvx_skel.so` in the
directory passed as `native_lib_dir`. See
[Build from source](../../../docs/quickai/build.md).

## Lifecycle

1. Inspect `getModelCatalogJson()` or choose a known catalog id.
2. `loadModelHandleByName(backend, model_id, quantization, native_lib_dir, model_base_path, &handle)`.
3. Run a streaming entry point as often as needed; each call is an
   independent prompt.
4. `cancelModelHandle(handle)` from any thread to stop at the next token.
5. `unloadModelHandle(handle)` to drop the weights and keep the handle, or
   `destroyModelHandle(handle)` to release everything.

Every function returns an `ErrorCode`: `CAUSAL_LM_ERROR_NONE` (0),
`INVALID_PARAMETER` (1), `MODEL_LOAD_FAILED` (2), `INFERENCE_FAILED` (3),
`NOT_INITIALIZED` (4), `INFERENCE_NOT_RUN` (5), `UNSUPPORTED` (6),
`UNKNOWN` (99). After `MODEL_LOAD_FAILED`, `getLastModelLoadError()` returns
the engine's reason (thread-local, valid until the next load on that
thread).

| Group | Functions |
|---|---|
| Catalog | `getModelCatalogJson()`: JSON array of `{id, family, display_name, runtime, backend_mask, capabilities}` |
| Load, unload | `loadModelHandleByName`, `unloadModelHandle`, `destroyModelHandle`, `configureSpeculativeDecoding`, `setOptions` (process-wide `use_chat_template`, `debug_mode`, `verbose`, `chat_template_name`), `loadModelHandle` / `loadModel` (deprecated enum-based forms) |
| Generate | `runModelHandleStreaming(h, prompt, cb, user)`, `runModelHandleWithMessages(h, msgs, n, add_generation_prompt, &out)`, `runModelHandleWithMessagesStreaming(...)`, `runModelHandleWithJsonStreaming(h, json, cb, user)`, `applyChatTemplate(msgs, n, add_generation_prompt, &formatted)` |
| Tools | `runModelHandleWithTool(h, prompt, &out, tool_name, tool_schema)`: output constrained to the tool's JSON schema |
| Multimodal | `runMultimodalHandleStreaming`, `runMultimodalHandleWithMessages[Streaming]`, `runMultimodalMultiImageHandle[WithMessages]Streaming`, `encodeModelHandle` / `freeEmbedding`, `encodeImageModelHandle` / `freeImageEmbedding` |
| Control | `cancelModelHandle(h)` |
| Metrics | `getPerformanceMetricsHandle(h, &m)`: `prefill_tokens`, `prefill_duration_ms`, `generation_tokens`, `generation_duration_ms`, `total_duration_ms`, `initialization_duration_ms`, `peak_memory_kb`, `ttft_ms`, `first_token_ms`, `decode_steady_ms`, `decode_steady_tokens`, `prefill_drain_ms` |
| Errors | `getLastModelLoadError()` |
| QNN KV cache | `save/load/resetQnnKvCacheHandle`: QNN models only; `UNSUPPORTED` otherwise |

### Behaviour every run call shares

- **Backend.** `BackendType` takes effect. `CPU` and `GPU` select the
  nntrainer compute engine for the process on the first load, through
  `nntrainer::requestEngine()` (`nntrainer/utils/env_compat.h`); the API sets
  no environment variable. A `CPU` request does not bring up the OpenCL
  context. A later load asking for the other engine is logged and runs on the
  first. An `NNTR_ENGINE` exported by the caller wins over the request. `NPU`
  models leave the engine alone.
- **Sessions.** Each run starts a fresh session: the KV position of a CausalLM
  model is rewound first, so N requests on one handle each behave like the
  first. A JSON request opts out with `"continue_session": true` and sends only
  the new turn.
- **Stream filter.** The callback never sees the chat turn markers
  (`<turn|>`, `<|turn>`, `<|im_end|>`, `<|im_start|>`), and eight consecutive
  `<pad>` deltas cancel the generation.
- **Saved KV prefix.** `runModelHandleWithJsonStreaming()` accepts
  `{"kv_prefix": {"static", "dynamic", "mode"}}` (see the header). The dump is
  `<model_dir>/kv_prefix/<hash>.kv`; `kv_prefix_path.h` defines the name and is
  tested against the reference value (`unittest_kv_prefix_path`).
- **Peak memory.** `PerformanceMetrics::peak_memory_kb` is the request's peak of
  anonymous RSS plus accelerator memory (`models/footprint_sampler.h`), which
  includes the load when the first request loads the model; `ru_maxrss` only
  when `/proc` gives no number.

Streaming callbacks (`CausalLmTokenCallback`) receive each delta on the
calling thread; returning nonzero from the callback cancels at the next
token boundary. Chat sessions are a Kotlin-level feature of the Android
library; the C API has no session object.

## The HTP models

The HTP build (`ENABLE_HEXKL`) registers four `NATIVE` descriptors with the
`NPU` backend and the `STREAMING` and `MESSAGES_API` capabilities:

| Catalog id | Directory under `model_base_path` |
|---|---|
| `gemma4-26b-a4b-htp` | `gemma4-26b-a4b-htp-w4a32/` |
| `gemma4-26b-a4b-htp-long` | `gemma4-26b-a4b-htp-long-w4a32/` (same weights, 30k-context pack for the KV-prefix demo) |
| `gemma4-26b-a4b-htp-2bit-4bit` | `gemma4-26b-a4b-2bit-4bit/` (rotated checkpoint, 2-bit TL2 experts, 4-bit layers 28-29) |
| `lfm2-24ba2b-htp` | `lfm2-24ba2b-htp-w4a32/` |

`model_base_path` NULL or empty means `/sdcard/Download/aistudio-mobile/models/`.
Load with `CAUSAL_LM_BACKEND_NPU` and `CAUSAL_LM_QUANTIZATION_W4A32`; other
combinations return `MODEL_LOAD_FAILED`; `getLastModelLoadError()` gives the
reason.

Rules specific to these models:

- **`native_lib_dir` is required** and must be an absolute directory that
  contains the matching v81 `libnntr_hvx_skel.so`. The API sets
  `ADSP_LIBRARY_PATH` to it before opening the DSP. Errors: `HTP requires
  native_lib_dir with the v81 DSP skel`, `HTP native_lib_dir is missing
  libnntr_hvx_skel.so`, `Conflicting HTP runtime option: ADSP_LIBRARY_PATH`
  (the variable was already set to a different directory).
- **One active HTP model per process.** A second load while one is active
  fails with `Only one active QuickDotAI HTP model is supported`. Unloading
  or destroying the active handle stops its workers, releases the DSP
  session and the weight registries, and resets the runtime configuration,
  so another handle can load.
- **The pack is checked before loading.** The pack's `nntr_config.json`
  must declare the model's profile and format (`gemma4_execution_profile`
  `htp_fused` with `gemma4_weight_format` `htp_fused_v1`, or
  `lfm2_execution_profile` `htp_fused` with `lfm2_weight_format` `sr15_v2`),
  for Gemma4 all four engines (`moe_engine`, `attention_engine`,
  `attn_proj_engine`, `dense_ffn_engine`) on `htp`, one of the accepted
  dtype sets, and no KV snapshot keys (`use_kvcache`, `save_kvcache`). `config.json` must be the flat text
  config (`Gemma4ForCausalLM`, hidden 2816, 30 layers, 128 experts;
  `Lfm2MoeForCausalLM`). A mismatch fails with `Gemma4 HTP requires the
  validated NPU/W4A32 preset`, `Unsupported Gemma4 HTP pack or KV snapshot
  option` or `LFM2 HTP requires the NPU/W4A32 sr15_v2 fused pack`. The
  contract is in [Model packs](../../../docs/quickai/model-packs.md).
- **Runtime settings come from the pack.** The `htp` and `moe` objects of
  `nntr_config.json` are parsed and validated before the DSP opens; an
  unknown key or wrong type fails the load with the key named (see
  [Configuration](../../../docs/quickai/configuration.md)). The configuration
  is fixed while the model is loaded. The API never applies the LFM2
  command-line preset.
- **Each run is an independent prompt.** The KV caches are rebuilt per run;
  the weights stay resident. KV snapshot and resume are not exposed for
  these models. Multi-model packs (`architectures[]` with `model_dirs[]`)
  are rejected for the HTP ids (`HTP supports a single model per handle`).
- **Context.** The Gemma4 chat preset holds 2,304 positions, 2,048 of them for
  the prompt; a longer rendered prompt is truncated at the end without an error (nntrainer issue
  #4474). Size the pack's `init_seq_len` / `max_seq_len` to your prompts,
  or send only what fits.
- **Chat input is plain text or messages**; the API applies the pack's
  template (`chat_template.jinja`). The output ends with the end-of-turn
  marker `<turn|>`, which a UI usually hides.

Example (Gemma4, messages API):

```c
CausalLmHandle h = NULL;
ErrorCode rc = loadModelHandleByName(CAUSAL_LM_BACKEND_NPU, "gemma4-26b-a4b-htp",
                                     CAUSAL_LM_QUANTIZATION_W4A32,
                                     "/data/app/.../lib/arm64",            /* holds libnntr_hvx_skel.so */
                                     "/sdcard/Download/aistudio-mobile/models", &h);
if (rc != CAUSAL_LM_ERROR_NONE) { fputs(getLastModelLoadError(), stderr); return rc; }

CausalLMChatMessage msg = { .role = "user", .content = "Summarize the text above in three sentences." };
rc = runModelHandleWithMessagesStreaming(h, &msg, 1, true, on_token, user_data);
/* on_token(const char *delta, void *user) returns 0 to continue, nonzero to cancel */

PerformanceMetrics m; getPerformanceMetricsHandle(h, &m);
destroyModelHandle(h);
```

## Testing the API on a device

`quick_dot_ai_test` (built with `-Denable-api-test=true`, which the Android
script enables) runs one generation through the API, or with
`QDA_HTP_LIFECYCLE=1` the HTP lifecycle check: load, repeat parity,
cancellation from the callback and from another thread, unload during
generation, reload, and a bad pack. See
[Command line](../../../docs/quickai/cli.md#the-c-api-lifecycle-test).

## Model extensions

A downstream project can add models this library does not implement by shipping
`libqai_ext_model.so` next to the other native libraries. The host holds the
single Factory, descriptor registry and callback registry; the plugin imports
them and registers into them. Nothing about a plugin's models is known here.

| Extension point | Where | Purpose |
|---|---|---|
| `Factory::registerModel` | `factory.h` | architecture string to class |
| `register_model_descriptor` | `model_descriptor.h` | catalog entry (id, backends, `QDA_CAP_*`, `sd_variant_id`) |
| `register_model_path` | `model_descriptor.h` | pack directory when it is not `config_name` lower-cased |
| `ModelCallbackRegistry::register_for` | `model_callbacks.h` | per-architecture prompt format, KV length, speculative decoding, multimodal |

**Load protocol** (`ext_model_abi.h`, `ext_model_loader.h`). On the first read of
the model registry the host `dlopen`s `libqai_ext_model.so` (or the path in
`QAI_EXT_MODEL_PATH`), calls `qai_ext_model_abi_info()` and compares the plugin's
`QDA_EXT_ABI_VERSION`, `sizeof(ModelDescriptor)` and `sizeof(ModelCallbacks)`
with its own. Only on a match does it call `qai_ext_model_register()`. A plugin
must not register from a static constructor: loading it has to be harmless
until the ABI is accepted. `ModelDescriptor` and `ModelCallbacks` are copied by
value across the boundary; change either only together with
`QDA_EXT_ABI_VERSION` and a rebuild of the plugin.

**Diagnostics.** `getExtensionModelStatus()` returns one line:
`absent: ...` (no plugin on the library path: a normal public build),
`loaded: ...` or `failed: ...` (a plugin is present but refused or broken: ABI
mismatch, missing symbol, unresolved dependency). The Kotlin SDK exposes the
same through `NativeCausalLm.extensionStatus()` and logs a failure at error
level; only a plugin file that is not there counts as absent.

**Hooks.**

- *Speculative decoding.* `configureSpeculativeDecoding` uses the first
  sub-model of the handle whose architecture registered
  `configure_speculative_decoding`, so a `[vision encoder, LLM]` handle reaches
  the LLM's callback.
- *Multimodal.* The `[producer, consumer]` composer (QNN builds with the
  experimental multimodal API) calls `ModelCallbacks::multimodal` when the
  architectures registered it: `encode_image` and `image_row_token_ids` on the
  producer; `image_begin_token`/`image_end_token` and `bind_image_span` on the
  consumer. With none registered it is the plain text/image/text splice.
- *Planar input.* `QDA_CAP_PLANAR_CHW` marks a vision encoder that takes planar
  CHW pixels; the Kotlin SDK reads it as `Capability.PLANAR_CHW`.

## Compatibility

Consumers must use `quick_dot_ai_api.h` and `libquick_dot_ai_api.so`; the
former `causal_lm_api.h` ABI is not provided and the enum-based `ModelType`
is a deprecated compatibility shim (string catalog ids are canonical).
Android builds use the same `libcausallm.so` core for the native tools and
the app packaging.
