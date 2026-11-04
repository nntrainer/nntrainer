# Command line

`nntr_causallm` runs a model pack without the app. It is the tool behind
every number in [Performance](performance.md) and the quickest way to check
a pack, a configuration or a new build on the phone.

## Stage the binaries

The easiest way is `build_android.sh --htp --install` (see
[Build from source](build.md)), which pushes everything to
`/data/local/tmp/Quick.AI` and writes two wrappers there:

```bash
adb shell sh /data/local/tmp/Quick.AI/run_causallm.sh /sdcard/Download/aistudio-mobile/models/gemma4-26b-a4b-htp-w4a32
```

The wrappers `cd` into that directory and export `LD_LIBRARY_PATH`,
`ADSP_LIBRARY_PATH` and `NNTR_NUM_THREADS` (8 for the HTP build). The second
wrapper, `run_test.sh` for `quick_dot_ai_test`, does not load the HTP models
yet: it exports `ADSP_LIBRARY_PATH` with a trailing `:`, which the API refuses
with `Conflicting HTP runtime option: ADSP_LIBRARY_PATH`. Run
`quick_dot_ai_test` with the explicit command in
[The C API lifecycle test](#the-c-api-lifecycle-test) instead.

To stage by hand, push these files into one directory and set the same three
variables yourself:

```bash
D=/data/local/tmp/Quick.AI
adb shell mkdir -p $D
# from the repository root
for f in builddir_app/htp/nntr_causallm builddir_app/htp/quick_dot_ai_test \
         builddir_app/htp/libcausallm.so builddir_app/htp/libquick_dot_ai_api.so \
         builddir/android_build_result/lib/arm64-v8a/libnntrainer.so \
         builddir/android_build_result/lib/arm64-v8a/libccapi-nntrainer.so \
         $ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so \
         nntrainer/tensor/htp_backend/dsp/build/libnntr_hvx_skel.so \
         $HEXKL_ROOT/lib/$HEXKL_SDK_VER/armv8_android26/libsdkl.so; do adb push $f $D/; done
adb shell chmod 755 $D/nntr_causallm $D/quick_dot_ai_test

adb shell "cd $D && LD_LIBRARY_PATH=$D ADSP_LIBRARY_PATH=$D NNTR_NUM_THREADS=8 \
  ./nntr_causallm /sdcard/Download/aistudio-mobile/models/gemma4-26b-a4b-htp-w4a32"
```

`ADSP_LIBRARY_PATH` must name the directory that holds `libnntr_hvx_skel.so`.
The command-line tool does not set it. **If the skel is not found, the run
does not fail**: the backend logs `nntr_hvx_open failed ...; HTP backend
disabled, falling back to CPU. Is libnntr_hvx_skel.so on ADSP_LIBRARY_PATH?`
and the model runs on the CPU, many times slower. Check the log when a run
is unexpectedly slow. `libcdsprpc.so` comes from the phone's vendor
partition and is not pushed.

## Run a pack

```text
Usage: nntr_causallm <model_path> [input_prompt]
  <model_path>   : Path to model directory
  [input_prompt] : Optional input text (uses sample_input or chat_input if omitted)
```

There are no other flags. Everything else is read from the pack's
`nntr_config.json`:

| What | Where it comes from |
|---|---|
| The prompt | The second argument, used as given: it is **not** rendered through the chat template, so include the model's turn markup yourself. Without it, `chat_input` from `nntr_config.json` rendered through the pack's chat template, else the raw `sample_input`. |
| Generation length | `num_to_generate`; the context is `init_seq_len` (prompt capacity) and `max_seq_len` (prompt plus generation). The environment variables `NNTR_MAX_SEQ_LEN` and `NNTR_INIT_SEQ_LEN` override them for one run. |
| Sampling | `generation_config.json` (`do_sample`, EOS ids). `bad_word_ids` in `nntr_config.json` suppresses ids, which is how a fixed-length benchmark disables the end-of-turn stop. |
| Runtime policy | The `htp` and `moe` objects; see [Configuration](configuration.md). Values are fixed until the process ends. |
| CPU threads | `NNTR_NUM_THREADS` if set, else the pack's `num_threads`. |

Prompts longer than `init_seq_len` are fed in chunks of that size. A prompt
longer than `max_seq_len` minus the generation budget is cut, so size the
pack's context to the longest prompt you intend to send.

The run prints the prefill and generation timings, then
`[e2e time]: <ms> ms` and `Max Resident Set Size: <KB> KB`. An error is
printed as `[!] FATAL ERROR: <reason>` with exit status 1; a configuration
problem names the offending key (for example `Unknown runtime
configuration: htp.pagecache`).

Two ways to capture the generated token IDs for a comparison:

- `"htp": {"token_trace": "/data/local/tmp/htp/trace.json"}` in the pack
  writes `{input_ids, output_ids}` at the end of each generation. Copy the
  file between runs to keep several results.
- `NNTR_GENERATED_IDS=/data/local/tmp/ids.txt` writes the generated IDs one
  per line (any model, any engine).

Set `NNTR_INIT_TRACE=1` to print one `[init-trace]` line per load phase with
its time and resident memory.

## The LFM2 preset

For an LFM2 SR15 pack the command-line tool merges a measured performance
preset into the configuration and prints `[LFM2] SR15 HTP CLI defaults
active; explicit configuration values take precedence`. Keys the pack sets
explicitly win; `"htp": {"lfm2_default_preset": false}` disables the preset.
The QuickDotAI API does not apply it; see
[Configuration](configuration.md#the-lfm2-cli-preset).

## Reuse a prefilled system prompt (KV snapshot)

A long static prefix (a system prompt, a document) can be prefilled once and
reused. This is a feature of the command-line tool and the C++ model class;
the public API does not expose it. In `nntr_config.json`:

```json
{
  "system_prompt": {
    "head_prompt": "<the static text>",
    "tail_prompt": "",
    "kvcache": {"pre_computed_cache_path": "/data/local/tmp/htp/static.kv"}
  },
  "sample_input": "<the dynamic part of the prompt>"
}
```

- If the file at `pre_computed_cache_path` does **not** exist, the run
  prefills `head_prompt` only, writes `<path>` and `<path>.q2` (the int8
  attention scales) and exits without decoding.
- If it exists, the run loads it, prefills `sample_input` plus `tail_prompt`
  behind it and decodes.
- `chat_input` clears `head_prompt` / `tail_prompt`, so use `sample_input`
  with this feature.

Generation after a snapshot can differ from a single prefill of the same
text, because the int8 KV scale calibration depends on the rows present
when it ran. The one recorded long-context run of this workflow is kept in
[`docs/validation/gemma4-htp-kv-cache-s25u-20261015.md`](../validation/gemma4-htp-kv-cache-s25u-20261015.md)
(in Korean; a Galaxy S25 Ultra with a v79 build of that time).

## The C API lifecycle test

`quick_dot_ai_test` exercises the public C API the way the app does: load,
generate twice and compare, cancel from the callback and from another
thread, unload during generation, reload, and recover from a bad pack. It
prints one `[HTP-API-TEST] PASS <step>` line per check and exits 0 when all
pass.

```bash
D=/data/local/tmp/Quick.AI
adb shell "cd $D && LD_LIBRARY_PATH=$D ADSP_LIBRARY_PATH=$D QUICKAI_NATIVE_LIB_DIR=$D QDA_HTP_LIFECYCLE=1 \
  ./quick_dot_ai_test gemma4-26b-a4b-htp 'I bought a red car yesterday' true W4A32 1 /sdcard/Download/aistudio-mobile/models"
```

Arguments: `<catalog id> [prompt] [chat_template true|false] [quant W4A32|W16A16|W8A16|W32A32] [verbose] [model_base_path]`.
`QUICKAI_NATIVE_LIB_DIR` is the directory with `libnntr_hvx_skel.so` (the
API's `native_lib_dir`); `QUICKAI_MODEL_BASE_PATH` can replace the last
argument. Without `QDA_HTP_LIFECYCLE` the tool runs one ordinary generation
and prints the init, prefill, generation and peak-memory figures. To keep
the per-step token traces, configure `htp.token_trace` in the pack and set
`QDA_HTP_TRACE_PREFIX=/data/local/tmp/htp/run`; the harness copies the trace
to `<prefix>-<step>.json` after each generation. The same binary accepts
`lfm2-24ba2b-htp`.

## Other tools in the same directory

| Tool | Purpose |
|---|---|
| `nntr_quantize` | The CPU-model quantizer (Q4_0, Q6_K, QS4CX, …). It does not produce HTP packs; see [Model packs](model-packs.md). |
| `nntr_safetensors_info` | Lists the tensors of a `.safetensors` file. |
| `htp_continuation_check` | Built with `-Denable-htp-device-check=true`; compares resident continuation with single-row feeding on a pack. |
