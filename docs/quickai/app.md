# Using the Quick.AI app

Quick.AI is the Android app that runs language models on the phone through
the QuickDotAI library. On a Galaxy S26 Ultra it runs Gemma4 26B-A4B and
LFM2 24B-A2B on the Hexagon NPU; it also drives the CPU models of nntrainer
and Gemma4 E2B through LiteRT-LM when their files are present. This page is
for people who use the app. Developers who want to embed the engine should
read the [QuickDotAI library](../../Applications/CausalLM/Android/QuickDotAI/README.md)
page instead.

## What you need

- The Quick.AI APK installed (`QuickDotAIAPP-debug.apk`, see
  [Build from source](build.md#install-the-app)). The app's package name is
  `com.example.quickdotaiapp`; it needs Android 13 or newer.
- A phone with a Hexagon v81 DSP for the NPU models (Galaxy S26 Ultra,
  Android 16).
- A model pack on the phone, under
  `/sdcard/Download/aistudio-mobile/models/` by default (see
  [Model packs](model-packs.md)). The APK does not contain model weights.

The first launch asks for "all files" access, which the app needs to read a
pack from `/sdcard`. Grant it in the system screen that opens, or later under
Settings > Apps > Quick.AI. Android also shows a one-time notice that the app
is not built for 16 KB memory pages; the supported phone uses 4 KB pages and
runs it (see [Troubleshooting](troubleshooting.md#known-limits)).

## Load a model

Tap the model chip under the top bar. The model sheet has three rows of
choices:

| Row | Choices | Meaning |
|---|---|---|
| **FAMILY** | `gemma4-26b-a4b`, `lfm2`, `qwen3`, … | The model. The list contains only the models the installed build can run. |
| **RUNTIME** | `Quick.AI`, `LiteRT` | `Quick.AI` is the native nntrainer engine. `LiteRT` appears for the Gemma4 E2B `.litertlm` file. |
| **BACKEND** | `NPU`, `CPU`, `GPU` | The processor. The HTP models offer `NPU` only. |
| **MODEL VARIANT** | e.g. `Gemma4 26B A4B MoE (HTP)`, `… (HTP, 30k demo)` | Shown only when one family/runtime/backend choice has several packs. It picks the exact pack; the folder check and the load follow it. |

The chip under the rows reads `HTP · W4A32 · <model name>` for the NPU
models. **BASE PATH** is the directory that contains the pack directory,
not the pack itself; the line under it says `<pack>/nntr_config.json found`
or `not found`. Tap **Load model**. The status line reads
`Loaded gemma4-26b-a4b-htp …` after about ten seconds for the 7.5 GB Gemma4
pack. A build that contains the Gemma4 NPU preset selects it by default.

- **Reload model** replaces the loaded engine with a fresh one; **Unload**
  releases the model and the DSP session.
- A base path without the pack shows `Load failed: not found: …` and keeps
  the model that was loaded. A pack directory that exists but fails to load
  shows `Load failed: [MODEL_LOAD_FAILED] …` with the engine's reason, and by
  then the previous model has been released.
- Only one NPU model can be loaded at a time. Loading a different one
  unloads the first.
- **Session settings** (system prompt, temperature, thinking) exist for the
  LiteRT runtime only. The native engine uses the pack's own chat template
  and greedy decoding.

## Chat

Type in the composer and tap **Send**. The answer streams into the
transcript; the status strip shows the elapsed time, the token count and
the current speed, and a line under each answer records the time to first
token, the speed and the token count.

- **Stop** replaces the send button while an answer streams and cancels at
  the next token; the turn is marked `stopped`.
- **Clear conversation** (top bar) empties the transcript.
- The **`{ }` toggle** switches the composer to JSON mode: paste or **Load
  .json** an OpenAI-style request (`{"messages": […], "tools": […]}`). The
  header line tells you whether it parsed and how it will be sent. A request
  with `tools` goes through the raw JSON path, the only one that forwards
  tools to the model. The bundled summarization request lives in the app as
  `gemma4_moe_messages.json` and is also the Power tab's `HTP 1K` prompt.
- The **attach** button inserts a text file (`.txt`, up to 1 MB) into the
  message, loads a `.json` request, or attaches an image for a vision model
  (the NPU models are text only).
- The end-of-turn marker `<turn|>` is hidden in the bubble; the Metrics
  screen's raw output and the token trace show it.

**The NPU models have no conversation memory in the app.** Their packs hold
1,280 positions for the Gemma4 chat preset, and a prompt that does not fit
is cut at the end without a warning (nntrainer issue #4474). The app
therefore sends only your newest message to these models, with the pack's
template, and nothing from earlier turns. Each question stands alone. The
other engines receive the whole transcript.

## Metrics

The Metrics screen shows the last run's time to first token, steady decode
time per token (median over tokens 2 to N), generation and prefill speeds
with their token counts and durations, and peak memory with the
initialisation time. A table lists the recent runs, and a collapsible panel
shows the raw output of the last run, end-of-turn marker included.

## Power

The Power tab measures the whole phone's energy per generated token from
the battery gauge while a model answers a fixed prompt, repeating the run to
get a median.

1. Load a model, unplug the phone (the tab waits for the cable; a debug
   switch allows plugged measurements, which the gauge reports wrongly).
2. Choose **PROMPT** (the composer's text, 256 / 1K / 2K / 4K tokens of
   bundled text, or `HTP 1K`, the validated Gemma4 summarization request),
   **WINDOW TOKENS** (64 / 512 / 1024 tokens of decoding to measure),
   **REPEATS** (1, 2, 3 or 5) and **BASELINE** (5 / 10 / 20 seconds of idle power
   before each run).
3. Tap **Measure**. Each repeat waits until the phone is cool (thermal
   status nominal, headroom below 0.80, up to 90 s), records the baseline,
   sends the prompt, measures through the decode window, then cools down.

The result shows energy per token (median mJ/token, with the number of
throttled runs), decode, prefill and baseline power in W, the speed, the
run energy and the battery and NPU temperatures, and a per-run list.
`mJ/token` is the energy above the baseline over the decode window, divided
by the tokens generated in it. Runs with no tokens or no gauge movement are
skipped and say why.

Files land in `Download/QuickAI/power/`: `runs.csv` gets one row per run
(columns `time, model_id, runtime, backend, prompt, prompt_tokens,
window_tokens, gen_tokens, tok_s, prefill_w, decode_w, baseline_w, peak_w,
mj_per_token, run_energy_j, bat_c_start, bat_c_end, npu_c_max, cpu_c_max,
gpu_c_max, thermal_status_max, throttled, skipped`), and each session writes
a `<date>-<model>-<backend>.json` with every sample. **Copy CSV** puts the
rows on the clipboard.

## Device tests

Three instrumentation tests drive the app's own controls on a phone. They
need the model pack on the phone, the messages file staged somewhere the
app can read, and both APKs installed (the test APK is built with
`:QuickDotAIAPP:assembleDebugAndroidTest`, see [Build from source](build.md)):

```bash
adb push Applications/CausalLM/Android/QuickDotAIAPP/src/main/res/raw/gemma4_moe_messages.json \
  /sdcard/Download/aistudio-mobile/messages.json
adb install -r -t Applications/CausalLM/Android/QuickDotAIAPP/build/outputs/apk/androidTest/debug/QuickDotAIAPP-debug-androidTest.apk
R=com.example.quickdotaiapp.test/androidx.test.runner.AndroidJUnitRunner
B=/sdcard/Download/aistudio-mobile/models
M=/sdcard/Download/aistudio-mobile/messages.json

# Gemma4: prompt format (ends at <turn|> once, no thought channel), chat == JSON
adb shell am instrument -w -r -e class 'com.example.quickdotaiapp.HtpAppLifecycleTest#chatUsesGemma4PromptFormat' \
  -e htpModelBase $B -e htpMessages $M $R
# Lifecycle: load, two identical runs, Stop, unload during generation, reload, bad path keeps the model
adb shell am instrument -w -r -e class 'com.example.quickdotaiapp.HtpAppLifecycleTest#loadGenerateStopRepeatReloadAndReportInvalidPath' \
  -e htpModelBase $B -e htpMessages $M $R
# LFM2 from the default base path, across an Activity restart
adb shell am instrument -w -r -e class 'com.example.quickdotaiapp.HtpAppLifecycleTest#lfm2ChatOpensFromDefaultPathAfterActivityRestart' \
  -e htpDefaultPathCheck true $R
```

`-e htpModelId lfm2-24ba2b-htp` runs the lifecycle test on the LFM2 pack and
`-e htpChatPrompt '<text>'` replaces the short chat prompt. A test whose
arguments are missing is skipped rather than failed. Each run leaves
screenshots, inputs, outputs and token traces under the app's private files
directory (`htp-chat-validation/`, `htp-app-validation/`,
`htp-default-path-validation/`); read them with
`adb exec-out run-as com.example.quickdotaiapp tar c files | tar x`.

## When something goes wrong

See [Troubleshooting](troubleshooting.md). The most common causes are a base
path that points at the pack instead of its parent, a pack without the
chat-preset files, and an APK signed with a different key than the one
already installed.
