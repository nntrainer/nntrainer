# Quick.AI app screenshots

Captured directly with `adb exec-out screencap -p` on November 4 and 5, 2026.
The PNG files are unedited device captures. Their captions and alternative
text live in the English and Korean guide Markdown; the renderer links each
thumbnail to its original image.

| File | Device | State |
| --- | --- | --- |
| `02-model-selection.png` | Galaxy S26 Ultra SM-S948U | Default HTP variant selected; includes the newer MODEL VARIANT controls |
| `03-model-loaded.png` | Galaxy S26 Ultra SM-S948N | Existing rotated TL2 pack loaded |
| `04-chat-response.png` | SM-S948N | One completed short question |
| `05-metrics.png` | SM-S948U | Updated Metrics benchmark setup, November 5; no model loaded |

Both devices run Android 16 with 4096-byte pages and Quick.AI version 2.0
(versionCode 2). The installed APKs' source commits were not verified. The
captures illustrate the interface and are not evidence of an end-to-end build
or benchmark of the guide's pinned PR head. The newer model picker was checked
against `ModelSheet.kt` at `fa078c94d1d73609c5584e948931384d69f9364c`.

The model selection screenshot uses `/sdcard/Download/aistudio-mobile/models`.
The inference screenshots use the existing compatible pack under
`/sdcard/Download/aistudio-mobile/models_1022/gemma4-26b-a4b-htp-w4a32`:
`nntr_gemma4_26b_a4b_tl2_rotated.bin`, Q4_0 embeddings, TL2CX_WH experts,
QS4CX_WH_AFFINE projections and HTP head, `init_seq_len=2048`,
`max_seq_len=2304`, `num_to_generate=384`.

Prompt: `Explain neural networks in two sentences.`

App-reported results: 20 input / 45 output tokens, TTFT 1.39 s,
steady decode 41.5 ms/tok, generation 19.6 tok/s (2295 ms),
prefill 15 tok/s (1331 ms), peak memory 3.9 GB, initialization 2.3 s.
This is one UI capture run, without repeated timing or thermal control.

After the November 4 captures, the model was unloaded and the SM-S948N app's original base
path was restored. No model pack, installed APK, SDK or build artifact was
modified for these captures.

The November 5 Metrics capture replaces the previous last-chat metrics image.
It shows the default 1K prefill / 128 decode / 3 repeats choices and the thermal
wait switched off. Scroll below the switch for Run; a loaded, idle model is
required. The controls were checked against `MetricsScreen.kt`, `AppViewModel.kt`
and `PowerPrompts.kt` at `67d2903a279a3f5604074f001bbf8f814dfb9f82`.
No model was loaded and no measurement was run for this capture.

The inspected SM-S948U installation reports version 2.0 (versionCode 2), last
updated November 4 at 19:53:28. Its base APK SHA-256 is
`5e944ad293abebd12cf894796af1c812db14f322fec4b32c431698c639c894d9`.
This identifies the installed artifact; it does not establish its source commit.
