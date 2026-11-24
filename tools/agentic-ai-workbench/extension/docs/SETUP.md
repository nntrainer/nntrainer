# Setting up the AI Compiler Workbench

## 1. Install the extension

In VS Code, open **Extensions** (`Ctrl+Shift+X`), click **`…` → Install from VSIX…**, and pick
`tools/agentic-ai-workbench/ai-compiler-workbench-<version>.vsix`. Reload the window when VS Code asks.

From a terminal you can run `code --install-extension <path-to-vsix>` instead.

## 2. Open the nntrainer folder

Open the nntrainer checkout with **File → Open Folder…**. The workbench writes its output, the code bank
and generated models under that checkout's `tools/agentic-ai-workbench/`. If you work in a different
folder, set `aiCompilerWorkbench.nntrainerRepoPath` to the checkout instead.

## 3. Python (automatic)

You only need some **Python 3.10 or newer** installed. Python 3.12 from python.org is a safe choice; on
Windows, tick *Add python.exe to PATH* in its installer.

When you first click **▶ Run Pipeline** or **Check Env**, the workbench offers to set up its own
environment:

* **Finding a Python.** It tries, in order:
  * the interpreter selected in VS Code's Python extension;
  * the Windows `py` launcher;
  * the standard install folders;
  * `python` / `python3` on PATH.

  It skips the Microsoft Store `python3` placeholder, and prefers 3.10–3.13, which torch publishes
  builds for on every platform.
* **Creating the environment.** It creates a virtual environment at
  `%LOCALAPPDATA%\ai-compiler-workbench\venv` on Windows, or `~/.local/share/ai-compiler-workbench/venv`
  elsewhere. Your other Python installs are not touched.
* **Installing packages.** It installs a CPU build of torch, then transformers, huggingface_hub and
  safetensors. Optional packages (LangChain, LangGraph) are attempted last; if they fail, the pipeline
  still runs.
* **Fallback.** If torch has no build for the first Python it found, it tries the next one.

It is a one-time download of about 1 GB. To use an environment you already have (for example a conda
env), set `aiCompilerWorkbench.pythonPath` to its interpreter.

## 4. Claude CLI (for C++ generation)

Sign in once by running `claude` in a terminal. If `claude` is not on PATH, the workbench uses the copy
bundled with the Claude Code VS Code extension. To use a specific one, set `aiCompilerWorkbench.claudePath`.
`claudeModel` picks the model (default `sonnet`), and `claudeCliTimeout` the time limit for one call
(default 900 seconds).

The **Chat** panel uses the same Claude CLI. Every question is sent with the latest run as context: the
model, the plan, review status, check and compile results, and the generated `.h`/`.cpp`. The chat also
remembers the last few turns, kept in `chat_history.json` in the output folder. It explains and advises
but never edits files; changes go through the Review tab.

**Optional Anthropic API key.** Without the Claude CLI, the chat can use an Anthropic API key instead, and
the key also enables suggestions for unsupported ops, Auto-Fix and the graph Focus fallback. Run
**AI Compiler Workbench: Set Anthropic API Key** (or **Clear Anthropic API Key**). The key is kept in VS
Code's secret storage and reaches the engine only through an environment variable. A key left in the old
`aiCompilerWorkbench.anthropicApiKey` user setting is moved there once. These extras need the optional
LangChain packages; the workbench never installs them by itself and shows the command that does.

## 5. Compiling generated code (optional)

The **Compile** button (Review tab, and next to the file tabs in the C++ Code tab) runs a compile check.
It compiles the generated `.cpp` files in syntax-only mode against the nntrainer checkout's headers. If an
nntrainer build exists (the workbench's own `tools/agentic-ai-workbench/nb`, or `build*/` in the
checkout), it uses that build's exact flags from `compile_commands.json`.

* **When the pipeline compiles:** only when nothing waits for review. Otherwise it skips compiling, and
  when you approve or accept the last waiting part you are asked whether to compile now.
* **On compile errors:** they are listed in the Review tab. **Fix with Claude** sends them back to Claude
  to correct the code.
* **In Docker:** when Docker is the build host (`buildHost` `docker`, or `auto` with Docker running and
  its build set up), the check runs in the Docker build container with its gcc-13 and the build's own
  flags. Then this computer needs no C++ compiler at all, and the pipeline goes on to Verify in Docker.
* **Otherwise it needs** a C++ compiler on this computer: `clang++` or `g++` on PATH, or on Windows Visual
  Studio Build Tools with the C++ workload. It is found even outside a Developer prompt (vswhere +
  `vcvars64.bat`). Or set `aiCompilerWorkbench.cxxCompiler`, which always compiles on this computer.

This is a compile check, not a build. **Verify** (next section) builds and runs the model.

## 6. Verify against Hugging Face

**Verify** builds the generated model and runs it next to Hugging Face. A pass is what makes approved
code **verified** in the code bank, the only kind the workbench reuses for other models.

1. **Set up the build once.** `aiCompilerWorkbench.buildHost` picks where it happens. The default,
   `auto`, uses Docker when it's running.
   * **Docker (recommended):** command palette → **AI Compiler Workbench: Set Up Docker Build**.
     * It builds an Ubuntu 24.04 image with gcc-13, like upstream's Ubuntu CI (`engine/docker/Dockerfile.build`,
       about 1 GB, built once).
     * It starts one build container with a Docker volume (`nntr-wb-<id>`) and the workbench folder
       mounted at `/wb`.
     * The checkout is **copied** into the volume, file by file. Later runs copy only what changed.
       Build scripts are converted to LF line endings on the way, because your checkout uses
       `core.autocrlf=true`.
     * Your checkout is never mounted, so the container can't change it. Results come back to the
       workbench folder only.
     * nntrainer is built in the volume, at `/work/src/build-wb`, like upstream's usual `builddir`.
       Building there is much faster than through a Windows mount. Copies of `build.ninja` and
       `compile_commands.json` are kept in `tools/agentic-ai-workbench/docker/build`.
     * See **Docker on any computer** below for the checks and tools that make this work everywhere.
   * **Native:** **AI Compiler Workbench: Set Up nntrainer Build**, or **⚙ Set up build** in the Review tab.
     * On Windows it uses MSVC with meson and `configurations/windows-native.ini`, into
       `tools/agentic-ai-workbench/nb`.
     * meson and ninja go into `tools/agentic-ai-workbench/.buildenv`.
   * Both builds use `-Denable-test=false`, because with tests on, meson re-extracts upstream's fixtures
     into `test/`.
   * Setup won't install things that live outside the workbench folder. If one is missing, it says so with
     the command to run:
     * git submodules: `git submodule sync && git submodule update --init --depth 1`
     * for Docker: Docker Desktop
     * for native Windows: Visual Studio Build Tools (C++), plus Rust for the tokenizer library
       (`winget install Rustlang.Rustup`, then `rustup default stable`)
2. **Verify** (Review tab) runs once nothing is waiting for review. With `aiCompilerWorkbench.autoVerify`
   on (the default), the pipeline also runs it after a passing compile check. It does four things:
   * It makes a tiny random-weight copy of the architecture (2 layers, hidden size 64) with
     transformers, and converts its weights with the generated `weight_converter.py`.
   * It compiles the generated `.cpp` and a test harness with the build's own flags, then links them
     against the build's libraries.
   * It runs upstream's `runFp32DifferentialChecks`
     (`test/unittest/models/causallm_test_utils.cpp`). That is the same rule the reference tests use:
     fp32 logits within 0.01 of Hugging Face, and all greedy tokens equal.
   * It shows the result in the Review tab: the largest logit difference and both token lists.
3. **The result:**
   * **Pass:** the approved parts become `verified` in the bank, and the next model with the same features
     reuses them without asking Claude.
   * **Mismatch:** the Claude-written parts become `failed verification`. **Fix with Claude** sends the
     numbers back. The new code needs review again before it is verified.
   * **Build errors, a crash, or a converter that fails on the tiny model** don't change the bank.
     Fix with Claude is offered for these too.

The tiny model's weights use `initializer_range` 0.2, and its norm weights and biases get noise. With the
usual 0.02 init and zero biases, attention is almost uniform and biases vanish. A wrong RoPE, a missing
bias or an ignored norm weight could then still match within 0.01.

Output: `generated/<model>/verify/` (fixture, harness, `gtest.json`).

### Docker on any computer

The Docker build is meant to work on any **x86-64** computer (Windows with Docker Desktop, or Linux with
Docker Engine). ARM computers such as Apple Silicon Macs are not supported yet.

* **System check.** Command palette → **AI Compiler Workbench: Check Docker Setup**. It also runs by itself
  before Set Up Docker Build, Verify and Run in Docker, and stops a step only for what that step needs.
  It checks: Docker installed and running, that it builds for x86-64, Docker's memory and CPUs, free disk
  (here and inside Docker), the git submodules, the Python packages, the Claude CLI, the proxy, whether
  Docker Hub, the Ubuntu servers and Hugging Face can be reached, how much the workbench keeps in Docker,
  and the last self-test. Each problem comes with a fix. The Review tab shows the last result under
  **Docker setup**.
* **Memory.** Builds run as many parallel compiles as Docker's memory allows (one per 1.5 GB, at most one
  per CPU). If the compiler is still killed for lack of memory, the build retries once with half the
  jobs, then says to give Docker more memory. Below 6 GB the system check warns. On Windows with the WSL 2
  backend, Docker's memory is set in `%UserProfile%\.wslconfig` (`[wsl2]` → `memory=12GB`, then
  `wsl --shutdown` and restart Docker Desktop).
* **Proxy.** `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` from the environment, or VS Code's own `http.proxy`
  / `http.noProxy`, are passed to the Docker image build and the container. A proxy on `localhost` is
  reached as `host.docker.internal`. Passwords in a proxy URL never appear on a command line or in a log.
* **Same image for everyone.** `engine/docker/Dockerfile.build` pins the Ubuntu base image by digest; the
  image tag is the Dockerfile's hash, so a change rebuilds it once. Your build in the volume is kept.
* **Self-test.** **AI Compiler Workbench: Run Docker Self-Test** proves the whole Docker path once: the
  system check, the build setup, a full `ninja -C /work/src/build-wb` (every target), then upstream's own
  tiny Qwen3 reference test (`unittest_causallm_qwen3_reference.cpp`) against Hugging Face, fp32 within
  0.01. No Claude call and no model download. The fixture is unpacked only inside Docker, never into your
  checkout. Result: `tools/agentic-ai-workbench/docker/selftest/` (`result.json`, `log.txt`). The first
  run takes a while because of the full build.
* **Clean Up Docker.** **AI Compiler Workbench: Clean Up Docker** lists what the workbench keeps in Docker
  with sizes, then removes either **Old images only** (build images of an earlier Dockerfile, run images
  older than the newest one of each model; the build stays) or **Everything** (also the build container
  and its volume; the next Docker step sets the build up again). It asks before removing anything.

## 7. Build, run and profile on a device

Once a model has passed Verify, it is run with its **real** weights and profiled:

* **Automatically in the pipeline** (`aiCompilerWorkbench.autoProfile`, on by default): after Verify passes,
  the pipeline waits for its background weight download and then does everything below with the **Run**
  settings. When it can't, one log line says why (Verify did not pass, Docker is not running, the weights
  are not on this computer, the model does not fit Docker's memory, or the setting is off).
* **By hand:** **Build, Run & Profile on a device** (Review tab), or the command
  **AI Compiler Workbench: Build, Run & Profile Generated Model**.

1. **Weights.** It uses the model's Hugging Face safetensors already on this computer: the local folder,
   the pipeline's download for this model (`.ai_compiler_cache/<model>/weights`, or `.w/<short id>` on
   Windows), or the weights of the last run. If there are none, it asks before downloading them into
   `tools/agentic-ai-workbench/.w`. They are converted by the generated `weight_converter.py`, and the
   result, `generated/<model>/run/model/nntr_<model>_fp32.bin`, is reused while nothing changes.
   Before anything slow, the memory the run needs is estimated from the checkpoint's tensor shapes
   (FP32 weights × 1.15, the KV cache and 0.4 GB; Qwen2.5-0.5B: 2.7 GB estimated, 2.6 GB measured). If it
   can't fit in Docker's memory, or in the **Memory** limit you set, the run stops there and says so.
2. **Build in Docker.** A small runner app is compiled with the Docker build's flags. It is modelled on
   upstream's `Applications/CausalLM/main.cpp`, with the chat template and the generated class.
3. **Load it on the device.** For **This PC**, the runner and its libraries
   (`generated/<model>/deploy/linux-x86_64`) become a runtime image,
   `nntr-workbench-run:<model>-<hash>`, on this computer. The weights are copied once per version into
   Docker's storage and read from there (reading them through the Windows mount took about a minute).
   A run the out-of-memory killer stops (exit 137) is reported as "ran out of memory", with the numbers.
4. **Run and profile.** The image runs the prompt *Runs* times. The Review tab and the **Profiler**
   panel show:
   * the generated text;
   * prefill and generation tokens/s, total time, peak memory and load time for each run;
   * earlier runs, for comparison.

   **CPUs** / **Memory** (for example `2` and `4g`) limit the container, to imitate a smaller device.
5. **Per-layer timing.** This is on by default: the **Per-layer timing** box in the Run section, and
   `aiCompilerWorkbench.autoProfilePerLayer` for the pipeline. After the measured runs, the model runs
   once more against a second nntrainer build with its profiler on, and every layer is timed.
   * **Phases.** A run reads the whole prompt in one call, then makes one call per generated token. Each
     layer's time is kept apart for the two, and every view has a **Whole run / Prompt / Generating**
     switch. Reading the prompt is limited by computation; generating, by reading the weights from memory.
   * **The Profiler panel** shows the time per layer type, then the slowest single layers (for example
     `output_of_causallm (tie_word_embeddings)`, the output head that turns the last hidden state into a
     score for every word).
   * **The per-layer view** opens from the **⤢ All N layers + block map** button, in the panel or in the
     Run section:
     * **Table.** Every layer, in the order it ran: its type, its decoder block, the prompt call's ms, ms per
       generated token, the total and its share. Click a column to sort, filter by type or name, and use
       **Show all** for more than the top 40. The 5 slowest layers are marked ●.
     * **GB/s.** For the layers with at least 1 MB of weights: the weights (read once per generated token,
       sizes from the checkpoint) divided by the time per token. When every weight layer reaches about the
       same GB/s, the time is set by memory speed, not by a slow layer.
     * **Block map.** The decoder blocks (rows) by the layers in a block (columns: `wq`, `ffn_up`, ...),
       coloured by time per call. A square outlined in amber is more than 1.5× the same layer's median over
       the blocks, by at least 1% of the call: one block is slower than the others. (The prompt is a single
       call per layer, so smaller gaps are mostly noise.) Click a square to find its row in the table. Below
       the map, **Outside the blocks** lists the layers that aren't in a block, such as the output head.
   * **Overlap.** nntrainer times each layer type and each layer, and these overlap. Shares are computed
     within the types and within the layers, so nothing is counted twice.
   * **Accuracy.** The profiled run is slower than the normal build. Compare layers by their shares, not
     by their ms. The tokens/s shown come from the normal runs.
   * **First time.** nntrainer is built with its profiler on (`-Denable-profile=true`). Docker sets up
     that second build in the same volume (`/work/src/build-wb-prof`, mirror
     `tools/agentic-ai-workbench/docker/build-prof`); this takes about as long as the first normal build.
     Its runner and image are separate (`deploy/linux-x86_64-prof`, `nntr-workbench-run:<model>-prof-<hash>`).
   * **Disk.** If the profiled run's prompt takes far longer than in the measured runs (the weights were
     read from disk), it is timed once more.

An **Android phone** is the next target: the same build plus the Android NDK, with the files loaded onto
the phone with `adb`. A Docker image can't run on Android, so loading means pushing the files. The x86_64
emulator isn't a target, because upstream builds Android for arm64 only.

Settings: `autoProfile` (on), `autoProfilePerLayer` (on), `runMaxNewTokens` (64), `runRepeats` (3), `runCpus`, `runMemory`, `buildHost`.

## 8. Models already on your computer

A plain folder from `save_pretrained` or a manual download works without downloading anything. It needs:
`config.json`, the `*.safetensors` weights, and the tokenizer files. You can:

* click **📁** and pick the folder; the bar shows what is in it (architecture, weight size, parameter count,
  tokenizer), or what is missing;
* paste the folder path into the model box. Quotes around it are fine.

A run from a folder is fully offline. Nothing is sent to or fetched from the Hugging Face Hub. GGUF-only
folders are not supported yet.

**⚡ Check only** (also *Check Model Only* in the command palette) is the quick look. It reads only
`config.json` and the tensor headers (names and shapes), then runs the checklist, writes the plan and
generates the C++. The model is never loaded into memory, so it suits big models. For a Hub id it downloads
only the config and the safetensors headers. It builds no graphs; a full **Run** does that.

## If something fails

The **AI Compiler Workbench** output panel shows every command and its output. Common fixes:

| Message | Fix |
|---|---|
| No Python 3.10 or newer was found | Install Python 3.12, then click **Check Env** again |
| no working venv module | Debian/Ubuntu: `sudo apt install python3-venv` |
| torch has no build for Python 3.x | Handled automatically when another Python is installed; otherwise install Python 3.12 |
| Windows path-length limit | Enable long paths (Settings → System → For developers), or set `aiCompilerWorkbench.venvPath` to a short folder such as `C:\wbenv` |
| network problem | Check the connection / proxy. `HTTPS_PROXY` (or VS Code's `http.proxy`) is honoured by pip, the downloads and Docker |
| Claude Code CLI not found | Install Claude Code and run `claude` once, or set `aiCompilerWorkbench.claudePath` |
| No C++ compiler found (Compile) | Install Visual Studio Build Tools with the C++ workload or `winget install LLVM.LLVM` (Windows) / install `g++`, or set `aiCompilerWorkbench.cxxCompiler` |
| Verify: "the workbench's nntrainer build is not set up yet" | Run **Set Up nntrainer Build** once |
| Set Up nntrainer Build: "git submodules" missing | `git submodule sync && git submodule update --init --depth 1` in the checkout |
| Set Up nntrainer Build: "path_too_long" | Enable long paths (Settings → System → For developers), or move the checkout to a shorter folder |
| Verify: fixture_failed with "auto_map" | Models with custom Hugging Face code are not verified yet (built-in transformers architectures only) |
| Docker is installed but not running | Start Docker Desktop, then try again |
| Docker setup check: "not x86-64" | The Docker build supports x86-64 computers only for now |
| "the compiler ran out of memory in Docker" | Give Docker more memory (see **Docker on any computer**), then try again |
| Run: "needs about X GB of memory to run" | Give Docker more memory, raise or clear `runMemory`, or use a smaller model |
| Docker setup check: old images can be freed | **Clean Up Docker** → Old images only |
| Docker self-test failed | Read `tools/agentic-ai-workbench/docker/selftest/log.txt`; the failing stage is named in the message |
| Run: "verify the model against Hugging Face first" | Click **Verify** in the Review tab; only verified code runs with real weights |
| Run: the weights are not on this computer yet | Accept the download prompt, or run the pipeline on the model folder first |
| Pipeline: "Profiling skipped: …" | The rest of the line says why; fix that, or use **Build, Run & Profile** in the Review tab |
| "Per-layer timing did not run: the runner printed no per-layer report" | The profiling build is not in use; run **Set Up Docker Build**, then tick **Per-layer timing** again |
| Weight download: `[Errno 2] ... .incomplete` | Windows path-length limit. Fixed in 0.4.3 (weights go to a short `.w/` folder); enabling long paths also works |

To start over, delete the venv folder and click **Check Env**.
