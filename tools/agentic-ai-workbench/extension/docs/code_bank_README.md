# Code bank and feature checklist

Step 1 of generating nntrainer CausalLM C++ for any Hugging Face model: learn
from the hand-written models in `Applications/CausalLM`, then, for a new model,
tick the feature boxes, find what is new, and see what can be reused.

| File (`engine/knowledge/`) | What it does |
|---|---|
| `code_bank.py` | Local SQLite database of reusable code, with its life cycle and search |
| `capture_causallm.py` | Fills the bank from `Applications/CausalLM` (once; re-run to refresh) |
| `builtin_features.py` | The checklist boxes, written from the code that implements each feature |
| `checklist.py` | Ticks the boxes for a checkpoint, finds unexplained parts, plans against the bank |
| `cpp_extract.py` | Small C++ reader used by the capture step |

Standard library only. `huggingface_hub`, `torch`/`transformers`,
`sentence-transformers` and Ollama are optional extras.

## Commands

Run from `tools/agentic-ai-workbench/extension/engine`:

```bash
# Fill (or refresh) the bank from this checkout
python -m knowledge.capture_causallm

# Checklist + plan for a model
python -m knowledge.checklist path/to/hf_snapshot          # reads safetensors headers only
python -m knowledge.checklist Qwen/Qwen3-0.6B --hub        # needs huggingface_hub
python -m knowledge.checklist path/to/config.json --via-transformers   # needs torch + transformers

# Look inside the bank
python -m knowledge.code_bank stats
python -m knowledge.code_bank list --kind hook --status verified
python -m knowledge.code_bank show hook:Qwen3Transformer::createAttention
python -m knowledge.code_bank search "q_norm attention" --kind hook
```

The bank lives at `tools/agentic-ai-workbench/code_bank/code_bank.db`. The
folder is in the repo, but its `.gitignore` keeps the database out of git, so
the bank stays local. When the engine runs from an installed copy of the
extension instead of the checkout, set `NNTRAINER_ROOT` to the checkout so the
same file is used. `WORKBENCH_CODE_BANK` overrides the path completely.

## What is stored

| Kind | Content |
|---|---|
| `hook` | One method body (`createAttention`, `createMlp`, ...), tagged with the features it implements |
| `layer` | A custom layer (`.h`/`.cpp`), from `models/<arch>/` or the shared `layers/` |
| `converter` | A HF → nntrainer weight converter (`res/**`, reference fixture generators) |
| `framework` | Base classes (`Transformer`, `CausalLM`, `SentenceTransformer`) |
| `model` | Each `models/<arch>/` folder as written, for reference |

`framework`, `model` and model-local `layer` entries hold full files, so
generation does not depend on `Applications/CausalLM/models` being present.

## Life cycle and the reuse rule

```
generated ──approve (user)──▶ approved ──verify (machine)──▶ verified ──▶ reused automatically
                                 ▲                              │
                                 └── approve after a fix ── failed_verification
```

* Only `verified` entries are reused (`CodeBank.find_reusable`).
* `approved` entries are only shown as examples; `generated`, `rejected`,
  `failed_verification` and `withdrawn` entries are never returned.
* Verification needs evidence with a `method`. The target for generated code
  is the same check the upstream reference tests use: fp32 logits within
  `logits_atol_fp32` (0.01) of HF transformers, plus matching tokens.

**Captured entries:** an entry is `verified` only if an upstream reference
test (`test/unittest/models/unittest_causallm_*_reference.cpp`) can reach it.
Reachability is worked out from the class hierarchy: every constructor in the
hierarchy, plus the methods the test calls, followed through the code with
C++ dispatch rules. Everything else hand-written is `approved`: merged upstream,
but no reference test covers it. That includes GPT-OSS, the slim/cached MoE
variants, TimmViT, and the base `Transformer::createAttention` used by Llama.
The workbench records this evidence; it does not re-run those tests.

## The checklist

1. **Boxes.** Each box is checked against `config.json` and the checkpoint's
   tensor names and shapes (only the safetensors headers are read). Rules
   include the implementing code's limits; for example, GPT-OSS's YaRN
   constants are fixed, Gemma3's sliding layers use RoPE theta 10000, and
   `qwen_moe` always re-normalises the top-k weights.
2. **Coverage.** Every tensor and every config key must be explained by a
   ticked box, unless it is metadata or has a null/false value. Whatever is
   left is a new part, with the hook it most likely belongs to.
3. **Plan.** Each hook is matched against the bank, with three outcomes:
   * **reuse:** a verified entry implements exactly the ticked features;
   * **match, NOT verified:** only an approved entry fits;
   * **NEW code needed:** the closest entries are listed as starting points.

Boxes learned from approved code are stored in the bank
(`CodeBank.put_feature`) and are checked alongside the built-in ones.

## Step 2: the Claude CLI writes the C++

The pipeline step **C++ Generator (Claude CLI)** (`engine/agents/causallm_codegen.py`, library in
`engine/knowledge/codegen.py`) runs after lowering:

1. **Read the model.** It reads `config.json` and the tensor names and shapes without downloading
   weights. It tries a local snapshot first, then the Hub's safetensors headers, then the
   transformers model on the meta device.
2. **Plan.** It runs the checklist and plans against the bank. Each hook gets one of three actions:
   * **INHERIT:** verified base code fits, so the hook is not overridden;
   * **COPY:** verified model code fits, and Claude must copy it unchanged;
   * **WRITE:** nothing fits, so Claude writes it, starting from the closest bank code and the
     Hugging Face modelling source.
3. **Generate and check.** The Claude CLI writes `<slug>_workbench_causallm.{h,cpp}`, plus
   `weight_converter.py` when no converter fits. `knowledge/codegen_validator.py` checks the reply:
   * class layout and constructor order;
   * COPY bodies are identical to the bank's; INHERIT hooks are not overridden;
   * no `setupParameters` override;
   * layer types, properties and names, and layer registration;
   * forbidden symbols;
   * the converter handles every tensor.

   Errors go back to Claude, up to `codegenMaxAttempts` times.
4. **Write the files.** Everything goes to `tools/agentic-ai-workbench/generated/<slug>/`: the
   sources, `meson.build`, `register_<slug>_workbench.cpp`, `res/` configs, `review.json` and the
   prompts and replies (`.codegen/`).
5. **Review.** New parts go into the bank as `generated` and appear in the webview's **Review** tab.
   Each part opens next to the Hugging Face code, with a diff against the code it started from.
   * **Approve:** the part becomes `approved` in the bank, and its learned checklist box is saved.
   * **Edit in editor → Re-check edits:** re-validates the files after you change them.
   * **Regenerate:** your comment goes to Claude.
   * **Reject.**

A whole-model match (Gemma4, LFM2) is copied from the bank with renamed classes. No LLM call is
made for it.

Settings: `useClaudeCodegen`, `claudePath` (empty = PATH, `~/.local/bin`, or the copy bundled with the
Claude Code VS Code extension), `claudeModel`, `claudeCliTimeout`, `codegenMaxAttempts`,
`outputDirectory`. The prompt goes to the CLI through stdin, with tools, sessions and MCP turned off,
from an empty working folder.

## Step 3: verify against Hugging Face

`knowledge/verify.py` (Review tab **Verify**, `python -m agents.codegen_review <review.json> verify`) is
the machine step that turns `approved` entries into `verified` ones:

* **Fixture:** `knowledge/verify_fixture.py` shrinks the model's own `config.json` to a tiny model of the
  same architecture, with random weights (init 0.2 plus noise on 1-D params). It converts the weights with
  the generated `weight_converter.py` through its `save_<model>_for_nntrainer(...)` entry point, which
  the validator requires, and records Hugging Face's prefill logits and greedy tokens. The layout is
  upstream's fixture layout.
* **Harness:** binds the generated class to `CausalLMTestAdapter` and upstream's
  `runFp32DifferentialChecks`. The pass rule is upstream's: fp32 atol 0.01 plus the tokens.
* **Build:** uses the rules, flags and libraries of the workbench build (`nb/build.ninja`). Nothing
  upstream is edited.
* **Bank:** a pass calls `mark_verified` for every approved entry the model uses, with evidence (method
  `hf-logits`, max diff, atol, tokens, files hash). A logit or token mismatch calls `mark_failed` for
  the workbench-written ones only. A result stays valid until the generated `.h`/`.cpp`/converter
  change.

**Build host:**
* Verify builds in Docker or natively. Docker (`knowledge/docker_host.py`) is Ubuntu 24.04 with gcc-13,
  building in a volume that holds a synced copy of the checkout; the checkout itself is never mounted.
  The choice is `aiCompilerWorkbench.buildHost`; `auto` uses Docker when it is running.
* The evidence records which host was used.

## Step 4: run and profile on a device

`knowledge/run_model.py` (Review tab **Build, Run & Profile**, or
`codegen_review <review.json> run`) runs verified code with the model's real weights:
* The generated converter makes the `.bin`.
* A runner app modelled on upstream `main.cpp` is built in Docker.
* It is loaded on this PC as a runtime image (`nntr-workbench-run:<model>-<hash>`, with the weights
  mounted).
* It is run R times, with optional CPU/memory limits.
* Prefill/generation tokens/s, peak memory and the output go to `review["runs"]` and the Profiler panel.

Not done yet: Android phones, which need the NDK build and `adb push` of the same bundle.

## Found while writing the rules

These are notes on the upstream code. Nothing there was changed.

* Windows CausalLM build (CI builds Windows without `enable-transformer`, so none of this is tested
  there). Verify works around each of these inside the workbench:
  * `main.cpp` includes `xlm_roberta.h` unguarded, but `models/meson.build` adds `xlm_roberta` only on
    non-Windows, so `nntr_causallm.exe` does not build.
  * `causal_conv1d_layer.dll` exports nothing, so MSVC writes no `causal_conv1d_layer.lib`, yet the link
    lines list it.
  * The tokenizer link line lacks `synchronization.lib`, which Rust std 1.78+ needs.
  * `jni/prepare_encoder.ps1` copies `json.hpp` to `<builddir>\..\Applications\CausalLM`, which only
    works for a build dir directly under the checkout.
* nntrainer's weight loading strips spaces from file paths (`C:\Users\A B\...` becomes
  `C:\Users\AB\...`); verify passes the fixture by its 8.3 short name.
* `meson setup -Denable-test=true` extracts `packaging/causallm_reference_*.tar.gz` into the source tree
  (`test/unittest/models/`). With `core.autocrlf=true` that shows tracked fixture files as modified
  (line endings only).

* `Gemma3Transformer::setupParameters` never runs. The constructors dispatch
  to the base `setupParameters`, so the Gemma3 constructor sets the embedding
  scale itself. `attn_logit_softcapping` is only read in that method, so a
  non-null value is ignored. The checklist reports such a model as needing
  new code.
* `qwen_moe`'s `forwarding()` does not re-normalise the top-k weights;
  `incremental_forwarding()` does.
* The base `createAttention` applies `sliding_window` to every layer except
  each N-th one (`sliding_window_pattern`, default 5) whenever
  `sliding_window` is set. A uniform window (Mistral-style) therefore needs
  the Qwen-style attention or new code.
