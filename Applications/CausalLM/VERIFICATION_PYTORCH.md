# Verification: nntrainer W8A8 vs PyTorch FP32

Verified the `feat/ced-w8a8-int8-resident` branch's CED audio detector (W8A8,
int8-resident activations) against the original PyTorch FP32 checkpoint using
identical front-end settings.

## Setup

- **Branch**: `feat/ced-w8a8-int8-resident` (commit `58ccc9ebb`)
- **Build**: x86 Meson (`builddir_x86`), `-Denable-transformer=true -Denable-app=true`
- **Model**: `nntr_ad3s_q8.bin` (channel-wise Q8_0 FC weights, W8A8 int8 activations)
- **PyTorch checkpoint**: `pytorch_audio_detection/pytorch/ckpt/weights.pt`
  (ced_tiny, embed_dim=192, depth=12, heads=3, xi pooling, 12-class head)
- **Front-end** (matched exactly):
  - n_fft=512, hop=160, n_mels=64, f_min=0, f_max=8000
  - log_mode="natural" (ln(max(mel, 1e-5)))
  - normalize_divisor=32767.0
  - window=48000 (3s), stride=16000 (1s)
- **Test clips**: 4 wav files, 67 windows total, 12 classes per window

## Method

1. Ran the nntrainer x86 binary on all 4 clips with `AD_JSON=1`, producing
   per-window JSON with all 12 sigmoid scores.
2. Ran PyTorch inference with a custom script (`pt_inference_matched.py`) that
   uses torchaudio MelSpectrogram with the exact same front-end parameters
   (natural log, n_fft=512, norm/32767) instead of the model's built-in
   FrontEnd (which uses AmplitudeToDB / dB scale).
3. Compared all 804 scores (67 windows × 12 classes) window-by-window.

**Key finding**: The pre-existing `pt3s_out/` PyTorch results were generated
with mismatched front-end settings (dB scale, n_fft=1024, normalize/32768),
producing a false 0.079 mean difference. With matched front-end, the
difference drops to 0.0025.

## Results

### nntrainer W8A8 vs PyTorch FP32 (matched front-end)

| clip | windows | det | top-1 | max diff | mean diff | >0.01 | >0.05 |
|------|---------|-----|-------|----------|------------|-------|-------|
| 16k_0128_barking_only | 13 | 13/13 | 13/13 | 0.0078 | 0.0022 | 0 | 0 |
| 16k_0128_stranger_barking | 13 | 13/13 | 13/13 | 0.0372 | 0.0025 | 4 | 0 |
| 16k_good_night | 25 | 25/25 | 24/25 | 0.0325 | 0.0028 | 5 | 0 |
| 16k_wake_up | 16 | 16/16 | 16/16 | 0.0170 | 0.0023 | 2 | 0 |
| **TOTAL** | **67** | **67/67** | **66/67** | **0.0372** | **0.0025** | **11** | **0** |

- **Detections identical**: 67/67 (100.0%)
- **Top-1 identical**: 66/67 (98.5%)
- **Score diff, mean**: 0.002493
- **Score diff, max**: 0.037200
- **Scores > 0.05**: 0

The single top-1 mismatch is `16k_good_night` window 0: PyTorch picks
knock (0.2318) vs nntrainer clap (0.2332) — a 0.0014 gap, both well below
any detection threshold.

### nntrainer W8A8 vs TFLite reference (for context)

| | |
|---|---|
| detections identical | 67/67 (100%) |
| top-1 identical | 66/67 (98.5%) |
| score diff, mean | 0.0026 |
| score diff, max | 0.0300 |
| scores > 0.05 | 0 |

## Per-class max/mean diff (nntrainer W8A8 vs PyTorch FP32)

| class | max | mean |
|-------|-----|------|
| dog_bark | 0.015515 | 0.002495 |
| cat_meow | 0.008006 | 0.002170 |
| doorbell | 0.009454 | 0.002189 |
| glass_break | 0.010514 | 0.002696 |
| clap | 0.037200 | 0.003780 |
| knock | 0.012669 | 0.002974 |
| alert | 0.006814 | 0.002277 |
| speech | 0.011751 | 0.002311 |
| baby_cry | 0.016968 | 0.002105 |
| cough | 0.010249 | 0.002205 |
| scream | 0.009493 | 0.002182 |
| water | 0.007773 | 0.002537 |

## Conclusion

The W8A8 quantization error (nntrainer vs PyTorch FP32) is **0.0025 mean /
0.0372 max**, with **zero detection mismatches** and **zero scores over 0.05**.
This is consistent with the documented W8A8 residual of ~0.0035 mean after
quantization, and confirms the porting is correct: the remaining difference
is purely from int8 weight/activation quantization, not from a porting bug.

Signed-off-by: Seungbaek Hong <sb92.hong@samsung.com>
