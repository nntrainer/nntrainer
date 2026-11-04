# Troubleshooting and known limits

Messages are quoted as the app, the API (`getLastModelLoadError()`) or the
command-line tool prints them. The app shows load errors as
`Load failed: [<code>] <message>`; the command line prints
`[!] FATAL ERROR: <message>`.

## Installing and starting the app

| Symptom | Cause and fix |
|---|---|
| `adb install` fails with `INSTALL_FAILED_UPDATE_INCOMPATIBLE` | A Quick.AI build signed with another key is installed. Uninstall it first. |
| A warning about 16 KB memory pages at install or first start | Expected. HexKL's `libsdkl.so` and the DSP image are 4 KB-aligned; the supported phone uses 4 KB pages and runs the app. Not validated on a 16 KB-page device. |
| The model sheet says `not found` although the pack is on the phone | **BASE PATH** must be the directory that *contains* `gemma4-26b-a4b-htp-w4a32/`, not the pack directory itself, and the pack directory name must be exact. Also check that all-files access was granted (Settings > Apps > Quick.AI). |
| The FAMILY list has no `gemma4-26b-a4b` or `lfm2` | The installed build is not an HTP build. Rebuild with `build_android.sh --htp --app`. |

## Loading a model

| Message | Cause and fix |
|---|---|
| `Load failed: not found: …/nntr_config.json` | The pack directory is missing under the base path. The previously loaded model is kept. |
| `Load failed: [MODEL_LOAD_FAILED] Failed to open file …` | The directory exists but a file in it is missing or unreadable (weights, tokenizer, `config.json`). The previous model has already been released. |
| `HTP requires native_lib_dir with the v81 DSP skel` | The API was called without `native_lib_dir`. Pass the app's `applicationInfo.nativeLibraryDir`. |
| `HTP native_lib_dir is missing libnntr_hvx_skel.so` | The DSP image is not in that directory: the APK was built without `jniLibs.useLegacyPackaging = true`, or the skel was stripped (`keepDebugSymbols`), or the build was not `--htp`. |
| `Conflicting HTP runtime option: ADSP_LIBRARY_PATH` | The variable was already set to a different directory in this process. |
| `Only one active QuickDotAI HTP model is supported` | Unload or destroy the active HTP handle before loading another. |
| `Gemma4 HTP requires NPU / W4A32` (library) or `INVALID_PARAMETER` (API) | The request asked for another backend or quantization. The HTP models are NPU + W4A32 only. |
| `Gemma4 HTP requires the validated NPU/W4A32 preset`, `Unsupported Gemma4 HTP pack or KV snapshot option`, `LFM2 HTP requires the NPU/W4A32 sr15_v2 fused pack` | The pack's `nntr_config.json` does not match the contract: missing profile or format key, an engine not set to `htp`, a dtype set that is not one of the accepted ones, or `use_kvcache` / `save_kvcache` present. See [Model packs](model-packs.md). |
| `Gemma4 HTP requires an HTP build`, `LFM2 HTP requires an HTP build` | The library was built without `--htp`. |
| `Unknown runtime configuration: <section>.<key>` | A key in the `htp` or `moe` object is misspelled or does not exist in this build. See [Configuration](configuration.md). |
| `Invalid runtime configuration: <section>.<key>` | Wrong JSON type (a boolean written as `0`/`1` or a string, a float for an integer key, a negative number for an unsigned key). |
| `htp.page_cache must be -1, 0 or 1`, `moe.prediction must be auto, off, …`, `Invalid moe.hadamard mode`, … | An out-of-range value. The message names the allowed values. |
| `Conflicting moe.cache_experts settings` (also `moe.hadamard`, `moe.hadamard_scope`) | A legacy flat key (`moe_cache_experts`, `moe_hadamard`, `moe_hadamard_scope`) and its nested form disagree. Remove the flat key. |
| `htp must be an object`, `moe must be an object` | The `htp` / `moe` entry is not a JSON object. |
| `htp/moe runtime configuration requires an HTP model` | An `htp` or `moe` object in a pack that is not an HTP model. |
| `HTP configuration cannot change while a session is open` | A load tried to apply different `htp` / `moe` values while an HTP session was open. Unload first. |
| `HTP supports a single model per handle`, `HTP profile requires its dedicated descriptor` | A multi-model pack (`architectures[]` + `model_dirs[]`) named an HTP model. Load the HTP pack by its own catalog id. |
| The LFM2 pack fails with a message about `conversion_report.json` or rotations | The report is missing or does not belong to this weight file, or the pack is the legacy unrotated layout. Reconvert with `tools/convert_lfm2_sr15.py --rotation both` and keep the report with the weights. |
| Loading takes very long on the first start of a pack with an untied `QS4CX` head | The head is being converted to WH slices and cached as `<weights>.lmhead_wh` (about 0.8 s on the S26 Ultra); later loads read the cache. A directory the app cannot write skips the cache and converts every time. |

## Generating

| Symptom | Cause and fix |
|---|---|
| The answer starts with `<|channel>thought` or rambles past the summary | The pack's `chat_template.jinja` is not the one from `Applications/CausalLM/res/gemma4/htp_chat/`. The built-in template engine rejected a variant and fell back to a bare turn format; logcat shows `Chat template apply failed`. Copy the preset's template into the pack. |
| The answer never stops, or always runs to the same length | The pack is a benchmark configuration with EOS disabled (`bad_word_ids` containing the EOS ids, or a `generation_config.json` without `eos_token_id`). Use the chat preset's `generation_config.json` (EOS `[1, 106, 50]`). |
| A long prompt gives an answer that ignores its end | The rendered prompt exceeded the pack's positions (1,280 in the Gemma4 chat preset) and was truncated at the end without an error (nntrainer issue #4474). Shorten the prompt or raise `init_seq_len` / `max_seq_len` in the pack. In the app, the HTP models receive only the newest message for this reason. |
| The model answers as if earlier turns never happened | By design in the app for the HTP models (see above). Pass the transcript yourself through the messages API if your prompt fits. |
| Two runs of the same prompt give different token IDs | Check that nothing changed between them: `htp.profile` 2 or 3 alters the execution path and can change IDs; a KV snapshot run differs from a single prefill because the int8 KV scales are calibrated on the rows present. Ordinary greedy runs reproduce the same IDs. |
| Much slower than the numbers in [Performance](performance.md) | On the command line: the DSP image was not found and the model silently fell back to the CPU; look for `nntr_hvx_open failed … falling back to CPU` in the log and set `ADSP_LIBRARY_PATH`. In the app: the phone is hot (the Power tab shows thermal status), or `htp.page_cache` is `0` with a small expert cache. |
| Out of memory, the app is killed, or the phone reboots | A 4,096-token context with large resident caches can exceed the phone's memory. Reduce `max_seq_len`, keep `moe.cache_experts` at 16, set `moe.cache_grow` to 0 or `htp.page_cache` to 0, and avoid loading a CPU model in the same process. |

## Command line

| Symptom | Cause and fix |
|---|---|
| `CANNOT LINK EXECUTABLE … library "libcausallm.so" not found` (or another `.so`) | `LD_LIBRARY_PATH` does not include the staging directory. Use the `run_causallm.sh` wrapper that `--install` writes, or export it yourself. |
| Runs, but at CPU speed; log shows `HTP backend disabled, falling back to CPU` | `ADSP_LIBRARY_PATH` does not point at the directory holding `libnntr_hvx_skel.so`. The tool does not set it. |
| `Usage: nntr_causallm <model_path> [input_prompt]` | No pack path was given. |
| `Unknown architecture: …` | `config.json` is not the flat text configuration the loader expects (for Gemma4, `architectures` must be `["Gemma4ForCausalLM"]`). |
| `Cannot open htp.token_trace`, `Cannot write generated IDs: …` | The trace path is not writable from the shell user; use `/data/local/tmp`. |

## Building

| Symptom | Cause and fix |
|---|---|
| `--htp requires HEXAGON_SDK_ROOT` (or `DEFAULT_HEXAGON_TOOLS_ROOT`, `HEXKL_ROOT`, `an explicit DSP target`) | Source the Hexagon SDK's `setup_sdk_env.source` and export `HEXKL_ROOT` and `HEX_ARCH=v81`. |
| `Error: the Gemma4 HTP runtime requires HEX_ARCH=v81.` | Only v81 is supported. |
| `Error: --htp and --qnn select separate native variants.` | Build the two variants separately. |
| `Error: missing …/libsdkl.so` | `HEXKL_SDK_VER` (default `6.4.0.2`) does not match the `lib/<ver>` directory of your HexKL. |
| `enable-htp=true requires the generated FastRPC stub …` or `The generated FastRPC stub is older than …` | Run `nntrainer/tensor/htp_backend/generate_stub.sh` after changing the IDL; the Android script does this itself. |
| `Error: skel has undefined symbols the DSP image will not provide` | A DSP source uses a symbol outside the allowed runtime set; the skel build refuses rather than failing on the phone. |
| `verify_native_package.py` reports `HTP runtime is incomplete`, `Missing libraries`, `DSP image changed`, `Expected v81` | The AAR or APK does not carry every staged library, the skel in the package is not byte-identical to the built one (stripped or replaced), or it is not a v81 image. Check `jniLibs.useLegacyPackaging` and `keepDebugSymbols` in the consuming app. |
| Host model tests fail with `Key is not found … fully_connected` | An OpenCL-enabled host build defaults to the GPU engine. Run the tests with `NNTR_ENGINE=cpu`. |
| `CausalLM tokenizer library not found` | The arm64 tokenizer archive was not built; the script builds it with `build_tokenizer_android.sh`, which needs `cargo`. |
| `HTP_GRAPH_E_VERSION` at the first decode | The DSP image and the ARM library come from different builds. Rebuild and redeploy the stub, the skel and the libraries together. |

## Known limits

- **Devices.** Only Hexagon v81 (Galaxy S26 Ultra, Snapdragon SM8850,
  Android 16) is supported by the build; the build script refuses other
  DSP targets. 16 KB-page devices are not validated.
- **One HTP model per process**, loaded through one handle. Unload before
  loading another.
- **Context.** The Gemma4 chat preset holds 1,280 positions; prompts that do
  not fit are truncated at the end without an error (nntrainer issue
  #4474). The app sends only the newest message to the HTP models.
- **No conversation memory** across calls in the native engine; every run
  is an independent prompt. KV snapshot and resume are available only to
  the command-line tool through the pack configuration, not through the
  API.
- **Greedy decoding** is the validated mode. Sampling works by asking the
  DSP for the logits but has not been measured.
- **Accuracy** has not been scored. Validation is token-level parity between
  runs of this runtime and its CPU reference, not a perplexity or benchmark
  result.
- **Thermal behaviour** was not measured; the published numbers are short
  runs on a cooled phone.
- **Open observations.** One intermittent expert-load rebind failure was
  seen once during snapshot experiments on the native Gemma4 pack and did
  not recur in eleven repeats; its cause is not established. The
  static-encoding Gemma4 checkpoint generates coherent text but has not
  been shown to match the original simulation on an agent test case (it
  chose a different tool call).
- **Packs** start from checkpoints prepared outside this repository; see
  [Model packs](model-packs.md#where-packs-come-from).
