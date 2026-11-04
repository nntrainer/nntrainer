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

Streaming callbacks (`CausalLmTokenCallback`) receive each delta on the
calling thread; returning nonzero from the callback cancels at the next
token boundary. Chat sessions are a Kotlin-level feature of the Android
library; the C API has no session object. Each run is one prompt.

## The HTP models

The HTP build (`ENABLE_HEXKL`) registers two `NATIVE` descriptors with the
`NPU` backend and the `STREAMING` and `MESSAGES_API` capabilities:

| Catalog id | Directory under `model_base_path` |
|---|---|
| `gemma4-26b-a4b-htp` | `gemma4-26b-a4b-htp-w4a32/` |
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
- **Context.** The Gemma4 chat preset holds 1,280 positions; a longer
  rendered prompt is truncated at the end without an error (nntrainer issue
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

## Compatibility

Consumers must use `quick_dot_ai_api.h` and `libquick_dot_ai_api.so`; the
former `causal_lm_api.h` ABI is not provided and the enum-based `ModelType`
is a deprecated compatibility shim (string catalog ids are canonical).
Android builds use the same `libcausallm.so` core for the native tools and
the app packaging.
