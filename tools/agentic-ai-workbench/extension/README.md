# AI Compiler Workbench

A VS Code extension that turns a Hugging Face causal language model into nntrainer C++
(`Applications/CausalLM` style), checks it against Hugging Face, and runs and profiles it.

* **Generates** the model's C++ with the Claude Code CLI. Code that was already verified for another
  model is reused from a local **code bank** instead of being written again.
* **Shows** the model as two graphs side by side: the Hugging Face model and the nntrainer model.
* **Checks** the code: a compile check, then **Verify**, which builds it and compares its output with
  Hugging Face on a tiny copy of the model.
* **Runs and profiles** the model with its real weights in Docker: tokens/s, memory, and the time of
  every layer.

## What you need

* The nntrainer checkout, opened as the folder in VS Code.
* Python 3.10 or newer. The workbench sets up its own environment on first use.
* The Claude Code CLI, signed in once (run `claude` in a terminal).
* Docker (Docker Desktop on Windows) on an x86-64 computer, for Verify and Run.

## Quick start

1. Build and install the extension:

   ```
   cd tools/agentic-ai-workbench/extension
   npx @vscode/vsce package
   code --install-extension ai-compiler-workbench-<version>.vsix
   ```
2. Open the panel: command palette (`Ctrl+Shift+P`) → **AI Compiler Workbench: Open Agentic Graph
   Visualiser**.
3. Click **Check Env** once. It sets up Python.
4. Run **AI Compiler Workbench: Set Up Docker Build** once from the command palette.
5. Type a model, for example `Qwen/Qwen2.5-0.5B-Instruct`, or pick a local model folder with **📁**.
   Then click **▶ Run Pipeline**, or **⚡ Check only** for a quick look without downloading the weights.
6. Open the **Review** tab:
   * **✔ Approve** or **✖ Reject** each newly written part, or ask for **⟳ Regenerate**.
   * **Verify** compares the model with Hugging Face.
   * **▶ Build, run & profile** runs it with its real weights.

   By default the pipeline compiles, verifies and profiles by itself once nothing waits for review.

## The panel

| Part | What it shows |
|---|---|
| Agent Pipeline | Each step of the run and its result |
| Model Graph / nntrainer Graph | The two graphs. Click a node to see it in both, with its weights |
| C++ Code | The generated `.h` and `.cpp` |
| Review | The generated parts, compile and Verify results, Docker status, runs |
| Chat | Ask Claude about the current run (it explains; it never edits files) |
| Logs, Profiler, Artifacts | The log, speed and per-layer timing, and the files written |

## Where the output goes

All under `tools/agentic-ai-workbench/` in the checkout:

* `generated/<model>/`: the generated C++, `weight_converter.py`, and `review.json`;
* `code_bank/`: the code bank;
* `workbench_output/`: the graphs and the state of the last pipeline run.

The rest of the checkout is not changed, unless you turn on `installGeneratedFiles`.

## Commands

All are in the command palette under **AI Compiler Workbench**:

* Open Agentic Graph Visualiser, Run Pipeline, Check Model Only, Stop Pipeline
* Build, Run & Profile Generated Model
* Set Up / Check Python Environment
* Set Up Docker Build, Check Docker Setup, Run Docker Self-Test, Clean Up Docker
* Set Up nntrainer Build (native build, instead of Docker)

## Settings

The most useful ones (search for `aiCompilerWorkbench` in Settings):

* `claudeModel`, `claudePath`: which Claude model and CLI generate the code;
* `buildHost`: `auto`, `docker` or `native`;
* `autoVerify`, `autoProfile`, `autoProfilePerLayer`: what the pipeline does by itself;
* `runMaxNewTokens`, `runRepeats`, `runCpus`, `runMemory`: how a model is run.

Settings that pick a program or a folder (`pythonPath`, `claudePath`, `cxxCompiler`, `outputDirectory`,
...) are read from your User settings only, never from a repository's `.vscode/settings.json`, and the
workbench runs only in a trusted workspace.

An Anthropic API key is optional. It is only for a few extras (chat when the Claude CLI is missing,
Auto-Fix). Set it with **AI Compiler Workbench: Set Anthropic API Key**; it is kept in VS Code's secret
storage, never in a settings file.

## More

* [docs/SETUP.md](docs/SETUP.md): setup step by step, how Verify and Run work, and fixes for common
  problems.
* [docs/code_bank_README.md](docs/code_bank_README.md): the code bank and the feature checklist.
