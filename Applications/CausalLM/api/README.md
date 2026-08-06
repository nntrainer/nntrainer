# QuickDotAI C API

This directory contains the public, handle-based C API used by the CausalLM
command-line test and the QuickDotAI Android AAR. The canonical header is
`quick_dot_ai_api.h`; it is self-contained for both C and C++ consumers.

The API supports model-catalog discovery, model load and destruction,
streaming text and multimodal inference, cancellation, chat sessions, tool
constraints through xgrammar, and performance metrics. Model selection uses
stable string catalog identifiers rather than the removed global `ModelType`
API.

## Build and link

The standalone Android build produces:

- `libcausallm.so`: models, layers, tokenizer integration, and xgrammar.
- `libquick_dot_ai_api.so`: the public C ABI in this directory.
- `libnntrainer.so` and `libccapi-nntrainer.so`: engine dependencies.

From `Applications/CausalLM`, run `./build_android.sh --app`. The AAR
mode is CPU-only unless QNN is explicitly enabled and never installs to a
device without `--install`. See the parent
[README](../README.md#4-android-build--test) for cache, QNN, and
device-install options. The no-option command builds the same canonical
CPU-native libraries and tools without Gradle.

Native consumers should include `quick_dot_ai_api.h` and link
`libquick_dot_ai_api.so`. Its transitive shared libraries must be available to
the platform loader. Android packages built by `build_android.sh` stage those
dependencies automatically.

## Lifecycle

The normal handle-based lifecycle is:

1. Inspect `getModelCatalogJson()` or choose a known catalog id.
2. Load with `loadModelHandleByName()`.
3. Run a handle-based streaming or chat-session entry point.
4. Use `cancelModelHandle()` when cancellation is required.
5. Release the model with `destroyModelHandle()`.

Refer to declarations and ownership comments in
[`quick_dot_ai_api.h`](quick_dot_ai_api.h) for the complete API contract.

## Compatibility

Consumers must use `quick_dot_ai_api.h` and `libquick_dot_ai_api.so`; the
former `causal_lm_api.h` ABI is not provided. Android builds use the same
canonical `libcausallm.so` core for native tools and app packaging.
