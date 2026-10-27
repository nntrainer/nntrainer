# Audio Detection — nntrainer End-to-End Port

Port of a 12-class sound event detector: a fine-tuned **ced_tiny** backbone with
**xi-vector** pooling. Unlike the plain CED port next door, this one runs the
whole pipeline in C++ — wav in, detections out — so it can be diffed directly
against the reference `run_demo.py`.

## Pipeline

```
dog_tizen.wav
  -> int16 PCM, channel 0, / 32768                        (readWav16)
  -> 1 s windows, 1 s stride                              (runAudioFile)
  -> STFT 512/160, Hann, center+reflect -> power          (logMelSpectrogram)
  -> 64-bin htk mel filterbank -> 10 log10 -> floor at max-80 dB
  -> init_bn (per-frequency BatchNorm)                    | nntrainer graph
  -> patch conv 16x16 -> +pos_embed -> 12x encoder blocks |
  -> LayerNorm -> xi pooling -> LayerNorm -> Linear(12)   |
  -> sigmoid, top-3, per-class thresholds                 (runAudioFile)
```

Everything from the model's `init_bn` to `Linear(12)` is the nntrainer graph;
the front-end and the threshold logic are the port's pre/post-processing, which
is exactly how the reference splits them too (`run_demo.py` applies the sigmoid
itself because xi pooling returns raw logits).

## What differs from the plain CED port

| | CED (`res/ced`) | this |
|---|---|---|
| input | mel `.bin`, 64 x 1012 | **wav**, 1 s windows -> 64 x 101 |
| pooling | mean over tokens | **xi**: learned per-frame precision, Gaussian posterior mean |
| classes | 527 AudioSet tags | 12 SED classes with per-class thresholds |
| sigmoid | inside the graph | in post-processing (matches `forward_head`) |
| `top_db` | 120 | **80** |

Three details are easy to get wrong:

- **`top_db` is 80, not the torchaudio default.** The checkpoint was trained
  that way; 120 shifts every quiet bin and moves the logits.
- **`time_pos_embed` is wider than the deployed window.** The table is sized for
  the 1012-frame pretraining window, and the model slices it to the actual grid
  width. The converter slices it the same way, so a 1 s window uses columns
  0..5 of 63.
- **The attention KV cache must be rewound per window.** Each window is an
  independent clip; without resetting `cache_index` the write position keeps
  advancing across windows and runs past `max_timestep`.

## Conversion

```bash
python weight_converter.py \
  --ckpt   /path/to/pytorch_audio_detection/pytorch/ckpt \
  --out-dir /path/to/audio-detection
cp /path/to/pytorch_audio_detection/pytorch/dog_tizen.wav /path/to/audio-detection/
```

This writes `config.json`, `nntr_config.json` and the FP32 weight blob. The
front-end geometry, the labels and the thresholds all live in `config.json`, so
the runner needs no side files — the Hann window and the mel filterbank are
derived at run time (they reproduce the ones stored in the checkpoint to 1.8e-7
and 5.2e-6 respectively).

## Reference extraction and verification

```bash
python extract_reference.py --out-dir /path/to/audio-detection/ref

AD_REF_DIR=/path/to/audio-detection/ref \
  ./builddir/Applications/CausalLM/nntr_causallm /path/to/audio-detection
```

`extract_reference.py` runs the upstream PyTorch model and dumps the input
waveform, per-window mel, `init_bn` output, encoder output, pooled vector,
logits and final probabilities, plus `expected.txt` — the reference table. With
`AD_REF_DIR` set the runner compares its own probabilities against
`window*_probs.bin` for every window and prints one `[AD_REF]` verdict
(`AD_REF_TOL` sets the budget, default 1e-3).

## Reproduce / cross-verify in one command

```bash
# from the repo root, after: meson setup builddir -Denable-transformer=true \
#                                 -Denable-app=true && ninja -C builddir
Applications/CausalLM/res/audio_detection/verify_e2e.sh
```

It runs the upstream PyTorch reference, converts the weights, and compares the
two at all three points below, printing one PASS/FAIL per check and exiting
non-zero if any budget is missed — so it works as a gate. `--skip-reference`
reuses an existing `ref/` directory; `--pytorch-dir`, `--build-dir`,
`--work-dir` and `--wav` override the defaults. `TOL_FRONTEND_DB`,
`TOL_MODEL_LOGITS` and `TOL_E2E_PROBS` override the budgets.

### Provenance of the verified numbers

| | |
|---|---|
| reference repo | `/home/seungbaek/projects/0824/pytorch_audio_detection/pytorch` |
| reference entry point | `run_demo.py` (its table is also checked into `inference_results.md`) |
| `ckpt/weights.pt` | `ff0acbba4ea3696c739f717a68ce8db2ed62d055a3edb33b812f1122f2cbebad` |
| `ckpt/config.yaml` | `78c1e2b1558a0ab2c9249843ef30d69cfb4a719472acc09b981aaae59a67aa50` |
| `dog_tizen.wav` | `063e3967252e2bfcea89b55e44ab38900784d0297a2e2b222442b43b9f21797b` |
| torch / torchaudio | 2.11.0+cu130 / 2.11.0+cu130 |
| numpy / scipy / soundfile | 2.2.6 / 1.15.3 / 0.13.1 |
| host | x86-64, gcc 11.4, FP32, `NNTR_NUM_THREADS=4` |

`verify_e2e.sh` re-checks the two checksums on every run and says so when they
drift, because the numbers below only mean something for that exact checkpoint
and clip.

## Verified results (x86, FP32, dog_tizen.wav, 16 windows)

| stage | compared against | result |
|---|---|---|
| front-end | `window*_mel.bin`, all 16 windows | max_abs_diff **1.3e-03 dB** |
| model, mel input | `window0_logits.bin` | max_abs_diff **1.76e-04** |
| full e2e | `window*_probs.bin`, all 16 windows | max_abs_diff **3.29e-04**, NaN 0, PASS |

Diffing the printed table against `expected.txt` leaves 15 of 17 lines
byte-identical. The two that differ are third-decimal rounding only:

```
-  [   8-   9s] knock=0.516, ...        reference 0.516379833
+  [   8-   9s] knock=0.517, ...
-  [  12-  13s] dog_bark=0.867, ...     reference 0.866539299
+  [  12-  13s] dog_bark=0.866, ...
```

Both reference values sit within 3.3e-04 of a `%.3f` rounding boundary
(0.5164, 0.8665), so an FP32-level difference flips the displayed digit. Every
top-3 ordering, every label and every threshold decision matches.

## W8A8 on Android

`fc_layer_dtype=Q8_0` quantizes the encoder's projections to int8 with **one
scale per output channel**, and the runtime quantizes each activation with **one
scale for the whole tensor**. Both symmetric, so the inner loop is an int8 dot
into int32 (`sdot` on ARM) with a single multiply per output element.

```bash
nntr_quantize <fp32 model dir> --fc_dtype Q8_0 --isa ARM \
  -o <out dir> --output_bin nntr_ad_pcw8a8.bin
```

`--isa` is a no-op for Q8_0: unlike Q4_0 the blocks are not ISA-interleaved, so
one file runs on x86 and ARM.

Measured on SM-S938N (Snapdragon 8 Elite), arm64-v8a, dog_tizen.wav, 16 windows,
front end plus graph with model load excluded, against the PyTorch reference:

| scheme | per window | realtime | worst prob diff | weights |
|---|---|---|---|---|
| channel-wise w / tensor-wise a | **13.9 ms** | **0.0138x** | 3.07e-02 | 6.0 MB |
| per-32-block, both sides | 23.1 ms | 0.0230x | 1.79e-02 | 6.0 MB |
| FP32 | 17.1 ms | 0.0170x | 1.79e-03 | 20.9 MB |

Block-scaled W8A8 is slower than FP32 here because the kernel multiplies by a
fresh scale every 32 values on both operands, and these GEMMs are small (M=24,
K=192) so that dominates. Lifting the scales out of the inner loop is what makes
8-bit actually pay.

Detections match the reference in every case; the coarser activation scale only
swaps the top-1 of window 2, where the reference's own top two are 0.011 apart.

A ready-to-run device package with the prebuilt arm64 binaries, the quantized
weights, the reference dumps and a one-command device verifier lives outside the
repo at `0824/quickai_audio_detection/`.
