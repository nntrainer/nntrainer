// No build step: plain CommonJS, requiring only Node's built-ins and the
// "vscode" module VS Code injects at runtime.
const vscode = require("vscode");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");
const readline = require("readline");
const crypto = require("crypto");

let output;
let panel;
let currentProc = null;
let outDir = null;

function activate(context) {
  output = vscode.window.createOutputChannel("AI Compiler Workbench");

  context.subscriptions.push(
    vscode.commands.registerCommand("aiCompilerWorkbench.openWorkbench", () => openPanel(context)),
    vscode.commands.registerCommand("aiCompilerWorkbench.runPipeline", () => runPipeline(context)),
    vscode.commands.registerCommand("aiCompilerWorkbench.stopPipeline", () => stopPipeline()),
    vscode.commands.registerCommand("aiCompilerWorkbench.checkEnvironment", () => ensureEnvironment(true)),
    vscode.commands.registerCommand("aiCompilerWorkbench.runDeviceProfiler", () => runDeviceProfiler(context)),
    vscode.commands.registerCommand("aiCompilerWorkbench.buildNntrainer", () => buildNntrainerFromSource(context))
  );
}

function deactivate() {
  stopPipeline();
}

// ---------------------------------------------------------------------
// Webview panel (Presentation Layer)
// ---------------------------------------------------------------------

function openPanel(context) {
  if (panel) {
    panel.reveal(vscode.ViewColumn.One);
    return;
  }

  panel = vscode.window.createWebviewPanel(
    "aiCompilerWorkbench",
    "Agentic Graph Visualiser",
    vscode.ViewColumn.One,
    { enableScripts: true, retainContextWhenHidden: true }
  );

  const htmlPath = path.join(context.extensionPath, "webview", "main.html");
  // Per-load nonce so the CSP can allow our single inline <script> without
  // 'unsafe-inline' (which would let injected markup execute via onerror etc.).
  const nonce = crypto.randomBytes(16).toString("base64");
  panel.webview.html = fs.readFileSync(htmlPath, "utf-8").replace(/__NONCE__/g, nonce);

  const config = vscode.workspace.getConfiguration("aiCompilerWorkbench");
  const savedNntrainerPath = config.get("nntrainerPath") || config.get("nntrainerRepoPath") || "";
  post({ type: "settingsInfo", nntrainerPath: savedNntrainerPath });

  panel.onDidDispose(() => {
    panel = null;
    stopPipeline();
  });

  panel.webview.onDidReceiveMessage(async (msg) => {
    switch (msg.type) {
      case "runPipeline":
        await runPipeline(context, msg.model);
        break;
      case "stopPipeline":
        stopPipeline();
        break;
      case "sendChat":
        await sendChat(context, msg.message);
        break;
      case "openGeneratedFile":
        await openGenerated(msg.filename);
        break;
      case "checkEnvironment":
        await ensureEnvironment(true);
        break;
      case "exportArtifacts":
        await exportArtifacts();
        break;
      case "exportFile":
        await exportFile(msg.filename, msg.content);
        break;
      case "runDeviceProfiler":
        await runDeviceProfiler(context);
        break;
      case "runNntrainerPipeline":
        await runNntrainerPipeline(context, msg.nntrainerPath);
        break;
      case "graphFocusQuery":
        await runGraphFocusQuery(context, msg.graph, msg.query);
        break;
    }
  });

  return panel;
}

// ---------------------------------------------------------------------
// Run Pipeline: spawns engine/orchestrator_main.py, streams JSON events
// ---------------------------------------------------------------------

async function runPipeline(context, modelNameArg) {
  const p = openPanel(context);

  let modelName = modelNameArg;
  if (!modelName) {
    modelName = await vscode.window.showInputBox({
      prompt: "HuggingFace model name or local path",
      placeHolder: "e.g. Qwen/Qwen2.5-1.5B",
    });
  }
  if (!modelName) {
    return;
  }

  const workspaceFolder = vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders[0];
  if (!workspaceFolder) {
    vscode.window.showErrorMessage("Open a folder first -- generated files are written into your workspace.");
    return;
  }
  outDir = path.join(workspaceFolder.uri.fsPath, "nntrainer_out");
  fs.mkdirSync(outDir, { recursive: true });

  const ready = await ensureEnvironment(false);
  if (!ready) {
    return;
  }

  stopPipeline(); // in case a previous run is still going

  const pythonPath = vscode.workspace.getConfiguration("aiCompilerWorkbench").get("pythonPath") || "python3";
  const apiKey = vscode.workspace.getConfiguration("aiCompilerWorkbench").get("anthropicApiKey");
  const causallmRoot = vscode.workspace.getConfiguration("aiCompilerWorkbench").get("causallmProjectRoot") || "";
  const installGenerated = !!vscode.workspace.getConfiguration("aiCompilerWorkbench").get("installGeneratedFiles");
  const generatedHeaderDir = vscode.workspace.getConfiguration("aiCompilerWorkbench").get("generatedHeaderDirectory") || "include/generated";
  const generatedSourceDir = vscode.workspace.getConfiguration("aiCompilerWorkbench").get("generatedSourceDirectory") || "src/generated";
  const engineDir = path.join(context.extensionPath, "engine");
  const scriptPath = path.join(engineDir, "orchestrator_main.py");

  const args = [scriptPath, modelName, "--out", outDir];
  if (causallmRoot) args.push("--causallm-root", causallmRoot);
  if (installGenerated) args.push("--install-generated");
  args.push("--generated-header-dir", generatedHeaderDir, "--generated-source-dir", generatedSourceDir);
  // Pass the key via the environment, never argv -- argv is world-readable
  // via `ps` / /proc on multi-user machines.
  const spawnEnv = { ...process.env };
  if (apiKey) spawnEnv.ANTHROPIC_API_KEY = apiKey;

  output.show(true);
  output.appendLine(`\n--- Running pipeline for "${modelName}" ---`);
  post({ type: "pipelineStarted", model: modelName });

  currentProc = spawn(pythonPath, args, { cwd: engineDir, env: spawnEnv });

  const rl = readline.createInterface({ input: currentProc.stdout });
  rl.on("line", (line) => {
    if (!line.trim()) return;
    let event;
    try {
      event = JSON.parse(line);
    } catch {
      output.appendLine(line); // non-JSON stdout noise, keep it in the output channel only
      return;
    }
    output.appendLine(`[${event.event}] ${event.message || event.detail || ""}`);
    post({ type: "engineEvent", event });
  });

  currentProc.stderr.on("data", (d) => output.append(d.toString()));

  currentProc.on("close", (code) => {
    currentProc = null;
    if (code !== 0 && code !== null) {
      post({ type: "engineEvent", event: { event: "error", stage: "process", message: `Engine exited with code ${code}` } });
    }
  });

  currentProc.on("error", (err) => {
    currentProc = null;
    vscode.window.showErrorMessage(`Could not start Python (${pythonPath}). Check aiCompilerWorkbench.pythonPath.`);
    post({ type: "engineEvent", event: { event: "error", stage: "process", message: err.message } });
  });
}

function stopPipeline() {
  if (currentProc) {
    currentProc.kill();
    currentProc = null;
    post({ type: "pipelineStopped" });
  }
}

// ---------------------------------------------------------------------
// Chat: spawns agents/chat_agent.py against the last saved state.json
// ---------------------------------------------------------------------

async function sendChat(context, message) {
  if (!outDir) {
    post({ type: "engineEvent", event: { event: "chat", role: "assistant", content: "Run the pipeline at least once in this workspace first." } });
    return;
  }

  const pythonPath = vscode.workspace.getConfiguration("aiCompilerWorkbench").get("pythonPath") || "python3";
  const apiKey = vscode.workspace.getConfiguration("aiCompilerWorkbench").get("anthropicApiKey") || "";
  const engineDir = path.join(context.extensionPath, "engine");

  post({ type: "engineEvent", event: { event: "chat", role: "user", content: message } });

  const chatEnv = { ...process.env };
  if (apiKey) chatEnv.ANTHROPIC_API_KEY = apiKey;
  const proc = spawn(pythonPath, ["-m", "agents.chat_agent", outDir, message], { cwd: engineDir, env: chatEnv });
  let stdout = "";
  proc.stdout.on("data", (d) => (stdout += d.toString()));
  proc.stderr.on("data", (d) => output.append(d.toString()));
  proc.on("close", () => {
    for (const line of stdout.split("\n")) {
      if (!line.trim()) continue;
      try {
        const event = JSON.parse(line);
        post({ type: "engineEvent", event });
      } catch {
        // ignore stray non-JSON output
      }
    }
  });
}

// ---------------------------------------------------------------------
// Environment check / one-click install (torch/transformers/langchain)
// ---------------------------------------------------------------------

async function ensureEnvironment(explicit) {
  const pythonPath = vscode.workspace.getConfiguration("aiCompilerWorkbench").get("pythonPath") || "python3";
  const check = await runPython(pythonPath, ["-c", "import torch, transformers, langchain"], null);

  if (check.ok) {
    if (explicit) {
      vscode.window.showInformationMessage("Environment OK -- torch, transformers, and langchain are importable.");
    }
    return true;
  }

  if (check.notFound) {
    const choice = await vscode.window.showErrorMessage(
      `Python interpreter "${pythonPath}" was not found. Install Python 3.10+, then set aiCompilerWorkbench.pythonPath if it's not "python3" on your PATH.`,
      "Open Settings"
    );
    if (choice === "Open Settings") {
      vscode.commands.executeCommand("workbench.action.openSettings", "aiCompilerWorkbench.pythonPath");
    }
    return false;
  }

  const choice = await vscode.window.showWarningMessage(
    "torch/transformers/langchain aren't installed in this Python environment yet. Install them now?",
    "Install now",
    "Cancel"
  );
  if (choice !== "Install now") {
    return false;
  }
  return installDependencies(pythonPath);
}

function installDependencies(pythonPath) {
  const requirementsPath = path.join(__dirname, "engine", "requirements.txt");
  return vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: "Installing AI Compiler Workbench dependencies", cancellable: false },
    async (progress) => {
      output.show(true);
      output.appendLine(`\n--- Installing dependencies via ${pythonPath} -m pip ---`);
      progress.report({ message: "This can take a few minutes the first time..." });

      const result = await runPython(pythonPath, ["-m", "pip", "install", "-r", requirementsPath], (chunk) => {
        output.append(chunk);
      });

      if (!result.ok) {
        output.appendLine(`\nInstall failed (exit code ${result.code}).`);
        vscode.window.showErrorMessage("Dependency install failed -- see the output panel for details.");
        return false;
      }
      output.appendLine("\nInstall complete.");
      vscode.window.showInformationMessage("Dependencies installed.");
      return true;
    }
  );
}

function runPython(pythonPath, args, onData) {
  return new Promise((resolve) => {
    let proc;
    try {
      proc = spawn(pythonPath, args);
    } catch {
      resolve({ ok: false, notFound: true, code: null });
      return;
    }
    proc.on("error", (err) => resolve({ ok: false, notFound: err.code === "ENOENT", code: null }));
    if (onData) {
      proc.stdout.on("data", (d) => onData(d.toString()));
      proc.stderr.on("data", (d) => onData(d.toString()));
    } else {
      proc.stdout.resume();
      proc.stderr.resume();
    }
    proc.on("close", (code) => resolve({ ok: code === 0, notFound: false, code }));
  });
}

// ---------------------------------------------------------------------

async function openGenerated(filename) {
  if (!outDir) return;
  const filePath = path.join(outDir, "generated", filename);
  try {
    const doc = await vscode.workspace.openTextDocument(filePath);
    await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
  } catch {
    output.appendLine(`Could not open ${filePath}`);
  }
}

async function exportArtifacts() {
  if (!outDir) {
    vscode.window.showInformationMessage("Nothing to export yet -- run the pipeline first.");
    return;
  }
  const uri = await vscode.window.showSaveDialog({
    defaultUri: vscode.Uri.file(path.join(outDir, "report.json")),
    filters: { JSON: ["json"] },
  });
  if (!uri) return;

  const statePath = path.join(outDir, "state.json");
  if (fs.existsSync(statePath)) {
    fs.copyFileSync(statePath, uri.fsPath);
    vscode.window.showInformationMessage(`Exported to ${uri.fsPath}`);
  } else {
    vscode.window.showWarningMessage("No completed run found to export yet.");
  }
}

async function exportFile(filename, content) {
  const defaultDir = outDir || (vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders[0].uri.fsPath) || ".";
  const uri = await vscode.window.showSaveDialog({
    defaultUri: vscode.Uri.file(path.join(defaultDir, filename)),
  });
  if (!uri) return;
  fs.writeFileSync(uri.fsPath, content, "utf-8");
  vscode.window.showInformationMessage(`Exported ${filename} to ${uri.fsPath}`);
}

// ---------------------------------------------------------------------
// Unified "Run On-Device Profiling" pipeline.
//
// One nntrainer path in, from the webview's own input field. This is the
// path the user should reach for -- it classifies what they gave it and
// does whatever is needed next, all streamed into the SAME panel:
//   - already a built prefix (has include/nntrainer + lib/)    -> profile directly
//   - a git clone of the nntrainer source (has meson.build+.git) -> build it first, then profile
//   - neither                                                   -> tell them what's wrong
//
// The only native (non-webview) UI left is the one thing that genuinely
// can't move into the panel: the interactive terminal + confirmation for
// the apt/sudo step, since a background child process cannot answer a
// password prompt. Everything else -- prerequisites, submodule sync,
// meson/ninja build, compiling the generated model, running it on-device,
// and the resulting bottleneck breakdown -- streams into this same
// Agentic Graph Visualiser window via the existing engineEvent channel,
// and the profiling results land in the Profiler Dashboard automatically
// (the "profile" event is already wired to renderProfile() there).
// ---------------------------------------------------------------------

function classifyNntrainerPath(p) {
  if (!p) return "empty";
  if (fs.existsSync(path.join(p, "include", "nntrainer"))) return "prefix";
  if (fs.existsSync(path.join(p, "meson.build")) && fs.existsSync(path.join(p, ".git"))) return "repo";
  return "unknown";
}

// Post a log line into the same panel the child-process agents stream
// into, for the handful of orchestration messages that originate in
// extension.js itself rather than in a spawned Python agent.
function postLog(message, level) {
  output.appendLine(message);
  post({ type: "engineEvent", event: { event: "log", level: level || "info", message } });
}

function postAgentStatus(agent, status, detail) {
  post({ type: "engineEvent", event: { event: "agent_status", agent, status, detail: detail || "" } });
}

async function runNntrainerPipeline(context, nntrainerPathArg) {
  openPanel(context);

  if (!outDir || !fs.existsSync(path.join(outDir, "state.json"))) {
    vscode.window.showWarningMessage(
      "Run the pipeline at least once in this workspace first (AI Compiler Workbench: Run Pipeline)."
    );
    return;
  }

  const config = vscode.workspace.getConfiguration("aiCompilerWorkbench");
  const nntrainerPath = (nntrainerPathArg || "").trim() || config.get("nntrainerPath") || config.get("nntrainerRepoPath");

  if (!nntrainerPath) {
    vscode.window.showWarningMessage(
      "Enter a path in the nntrainer field first -- either an existing install prefix, or a git clone of the nntrainer source."
    );
    return;
  }

  output.show(true);
  post({ type: "deviceProfilerStarted" });
  postLog(`\n--- Resolving nntrainer path: "${nntrainerPath}" ---`);

  const kind = classifyNntrainerPath(nntrainerPath);
  let prefix;

  if (kind === "prefix") {
    postLog("Found an existing nntrainer install (include/nntrainer + lib/ present) -- skipping build.");
    prefix = nntrainerPath;
  } else if (kind === "repo") {
    postLog("Detected an nntrainer source repo -- building it first, then profiling in the same run.");
    prefix = await buildNntrainerFromRepo(context, nntrainerPath);
    if (!prefix) {
      return; // buildNntrainerFromRepo already logged/reported the specific failure
    }
  } else {
    vscode.window.showWarningMessage(
      `"${nntrainerPath}" is neither a valid nntrainer install (needs include/nntrainer) nor a git clone of the ` +
        `nntrainer source (needs meson.build + .git). Check the path and try again.`
    );
    return;
  }

  await config.update("nntrainerPath", prefix, vscode.ConfigurationTarget.Workspace);

  // Chain straight into compiling + running on-device -- the resulting
  // "profile" event is already wired to the Profiler Dashboard panel.
  await runDeviceProfilerCore(context, prefix);
}

// ---------------------------------------------------------------------
// Profile On-Device (standalone command / backward compatible entry
// point): prompts for a prefix via the old dialog chain if none is
// configured yet, then delegates to the same core used by the unified
// pipeline above.
// ---------------------------------------------------------------------

async function runDeviceProfiler(context) {
  openPanel(context);

  if (!outDir || !fs.existsSync(path.join(outDir, "state.json"))) {
    vscode.window.showWarningMessage(
      "Run the pipeline at least once in this workspace first (AI Compiler Workbench: Run Pipeline)."
    );
    return;
  }

  const nntrainerPath = await ensureNntrainerPath(context);
  if (!nntrainerPath) {
    return; // user cancelled the prompt
  }

  await runDeviceProfilerCore(context, nntrainerPath);
}

// Core: spawns engine/device_profiler_main.py against an already-resolved
// nntrainer prefix, streaming into the panel. Shared by both the unified
// pipeline and the standalone "Profile On-Device" command.
function runDeviceProfilerCore(context, nntrainerPath) {
  return new Promise((resolve) => {
    const pythonPath = vscode.workspace.getConfiguration("aiCompilerWorkbench").get("pythonPath") || "python3";
    const engineDir = path.join(context.extensionPath, "engine");
    const scriptPath = path.join(engineDir, "device_profiler_main.py");

    output.show(true);
    postLog(`\n--- Profiling on-device against nntrainer at "${nntrainerPath}" ---`);
    postAgentStatus("profiler", "running", "Compiling + running on-device...");

    const proc = spawn(pythonPath, [scriptPath, outDir, nntrainerPath], { cwd: engineDir });

    const rl = readline.createInterface({ input: proc.stdout });
    rl.on("line", (line) => {
      if (!line.trim()) return;
      let event;
      try {
        event = JSON.parse(line);
      } catch {
        output.appendLine(line);
        return;
      }
      output.appendLine(`[${event.event}] ${event.message || event.detail || ""}`);
      post({ type: "engineEvent", event });
    });

    proc.stderr.on("data", (d) => output.append(d.toString()));

    proc.on("close", (code) => {
      if (code !== 0 && code !== null) {
        post({ type: "engineEvent", event: { event: "error", stage: "device_profiler", message: `Profiler exited with code ${code}` } });
        postAgentStatus("profiler", "error", `exit ${code}`);
      }
      resolve();
    });

    proc.on("error", (err) => {
      vscode.window.showErrorMessage(`Could not start Python (${pythonPath}). Check aiCompilerWorkbench.pythonPath.`);
      post({ type: "engineEvent", event: { event: "error", stage: "device_profiler", message: err.message } });
      postAgentStatus("profiler", "error", err.message);
      resolve();
    });
  });
}

// Ensures aiCompilerWorkbench.nntrainerPath is set and looks like a real
// install (has include/nntrainer). Prompts the user if not, offering to
// locate an existing build, build one from source now (Ubuntu), or open
// the nntrainer repo on GitHub to build manually. Used only by the
// standalone "Profile On-Device" command -- the unified pipeline reads
// its path straight from the webview's own input field instead.
async function ensureNntrainerPath(context) {
  const config = vscode.workspace.getConfiguration("aiCompilerWorkbench");
  let nntrainerPath = config.get("nntrainerPath");

  if (nntrainerPath && fs.existsSync(path.join(nntrainerPath, "include", "nntrainer"))) {
    return nntrainerPath;
  }

  const choice = await vscode.window.showInformationMessage(
    nntrainerPath
      ? `The configured nntrainer path ("${nntrainerPath}") doesn't look like a valid install (no include/nntrainer found).`
      : "Profile On-Device needs a local nntrainer install (built from github.com/nntrainer/nntrainer).",
    "Build nntrainer from source...",
    "Locate existing install...",
    "Open nntrainer on GitHub",
    "Cancel"
  );

  if (choice === "Build nntrainer from source...") {
    const uris = await vscode.window.showOpenDialog({
      canSelectFiles: false,
      canSelectFolders: true,
      openLabel: "Select nntrainer repo root (git clone)",
    });
    if (!uris || uris.length === 0) {
      return null;
    }
    return buildNntrainerFromRepo(context, uris[0].fsPath);
  }

  if (choice === "Locate existing install...") {
    const uris = await vscode.window.showOpenDialog({
      canSelectFiles: false,
      canSelectFolders: true,
      openLabel: "Select nntrainer install prefix (contains include/ and lib/)",
    });
    if (!uris || uris.length === 0) {
      return null;
    }
    const selected = uris[0].fsPath;
    if (!fs.existsSync(path.join(selected, "include", "nntrainer"))) {
      vscode.window.showWarningMessage(
        `"${selected}" doesn't contain include/nntrainer -- select the --prefix directory you passed to meson, not the source checkout.`
      );
      return null;
    }
    await config.update("nntrainerPath", selected, vscode.ConfigurationTarget.Workspace);
    return selected;
  }

  if (choice === "Open nntrainer on GitHub") {
    vscode.env.openExternal(vscode.Uri.parse("https://github.com/nntrainer/nntrainer"));
    vscode.window.showInformationMessage(
      "Build it with: meson --prefix=<install-dir> build && ninja -C build install -- then run " +
        "'Profile On-Device' again and point it at <install-dir>."
    );
  }

  return null;
}

// ---------------------------------------------------------------------
// Build nntrainer From Source (Ubuntu only). Two callers:
//   - buildNntrainerFromSource(context): standalone command, prompts for
//     the repo itself
//   - runNntrainerPipeline() / ensureNntrainerPath() above: already have a
//     repo path, call buildNntrainerFromRepo(context, repoPath) directly
//
// Uses the nntrainer_builder.py agent for everything that doesn't need
// sudo (validation, submodule sync, meson/ninja), and opens exactly one
// interactive terminal + one confirmation dialog for the apt/sudo step,
// since that genuinely cannot be scripted from a background process.
// Every status update streams into the same panel via postLog/postAgentStatus
// (or via the agent's own bus events, forwarded the same way runPipeline
// forwards orchestrator_main.py's).
// ---------------------------------------------------------------------

async function buildNntrainerFromSource(context) {
  if (process.platform !== "linux") {
    vscode.window.showWarningMessage(
      "Build nntrainer From Source is Ubuntu-only for now. On other platforms, build nntrainer manually " +
        "(see github.com/nntrainer/nntrainer) and use 'Locate existing install...' instead."
    );
    return null;
  }

  const config = vscode.workspace.getConfiguration("aiCompilerWorkbench");
  let repoPath = config.get("nntrainerRepoPath");

  if (!repoPath || !fs.existsSync(path.join(repoPath, "meson.build")) || !fs.existsSync(path.join(repoPath, ".git"))) {
    vscode.window.showInformationMessage(
      "Select your local clone of the nntrainer repo (must be a real 'git clone', not a downloaded zip -- " +
        "submodule setup needs actual git metadata)."
    );
    const uris = await vscode.window.showOpenDialog({
      canSelectFiles: false,
      canSelectFolders: true,
      openLabel: "Select nntrainer repo root",
    });
    if (!uris || uris.length === 0) {
      return null;
    }
    repoPath = uris[0].fsPath;
  }

  return buildNntrainerFromRepo(context, repoPath);
}

// Shared build core: validates the repo, prompts for an install prefix,
// runs the prep phase, opens a terminal for apt install, runs the build
// phase, validates the result. Returns the resolved prefix or null.
async function buildNntrainerFromRepo(context, repoPath) {
  if (process.platform !== "linux") {
    vscode.window.showWarningMessage(
      "Build nntrainer From Source is Ubuntu-only for now. On other platforms, build nntrainer manually " +
        "(see github.com/nntrainer/nntrainer) and point 'Profile On-Device' at an existing install instead."
    );
    return null;
  }

  const hasMeson = fs.existsSync(path.join(repoPath, "meson.build"));
  const hasGit = fs.existsSync(path.join(repoPath, ".git"));
  if (!hasMeson || !hasGit) {
    vscode.window.showWarningMessage(
      `"${repoPath}" doesn't look like a git clone of nntrainer ` +
        `(missing ${!hasMeson ? "meson.build" : ""}${!hasMeson && !hasGit ? " and " : ""}${!hasGit ? ".git" : ""}). ` +
        `Clone it fresh with: git clone https://github.com/nntrainer/nntrainer`
    );
    return null;
  }

  const config = vscode.workspace.getConfiguration("aiCompilerWorkbench");
  await config.update("nntrainerRepoPath", repoPath, vscode.ConfigurationTarget.Workspace);

  const defaultPrefix = path.join(repoPath, "nntrainer-install");
  const prefix = await vscode.window.showInputBox({
    prompt: "Install prefix for the nntrainer build (where include/ and lib/ will end up)",
    value: defaultPrefix,
  });
  if (!prefix) {
    return null;
  }

  output.show(true);
  postLog(`\n--- Building nntrainer from "${repoPath}" into prefix "${prefix}" ---`);

  const pythonPath = vscode.workspace.getConfiguration("aiCompilerWorkbench").get("pythonPath") || "python3";
  const engineDir = path.join(context.extensionPath, "engine");

  // Phase 1: validate + sync submodules + get apt commands (no sudo).
  let aptCmds = [];
  try {
    const result = await runNntrainerBuilderAgent(pythonPath, engineDir, repoPath, prefix, "prep");
    if (!result.ok) {
      const err = result.error || "Unknown error";
      postAgentStatus("nntrainer_builder", "error", err);
      vscode.window.showErrorMessage(`nntrainer build failed: ${err}`);
      return null;
    }
    aptCmds = result.apt_commands || [];
  } catch (e) {
    postAgentStatus("nntrainer_builder", "error", e.message);
    vscode.window.showErrorMessage(`nntrainer build agent failed: ${e.message}`);
    return null;
  }

  // Phase 2: apt install (only if needed). If all packages are already
  // installed, skip this step entirely.
  if (aptCmds.length > 0) {
    postLog("System packages are missing -- installing via apt...");
    const terminal = vscode.window.createTerminal({ name: "nntrainer apt-install", cwd: repoPath });
    terminal.show(true);
    for (const cmd of aptCmds) {
      terminal.sendText(cmd);
    }

    const confirmed = await vscode.window.showInformationMessage(
      "Dependency install commands were sent to the terminal. Enter your sudo password if prompted, " +
        "then click 'Done' once they've finished.",
      "Done",
      "Cancel"
    );
    if (confirmed !== "Done") {
      postAgentStatus("nntrainer_builder", "error", "cancelled during dependency install");
      return null;
    }
    postLog("System dependencies installed.");
  } else {
    postLog("All system packages already installed -- skipping apt step.");
  }

  // Phase 3: meson + ninja build (no sudo).
  try {
    const result = await runNntrainerBuilderAgent(pythonPath, engineDir, repoPath, prefix, "build");
    if (!result.ok) {
      const err = result.error || "Build failed";
      postAgentStatus("nntrainer_builder", "error", err);
      vscode.window.showErrorMessage(`nntrainer build failed: ${err}`);
      return null;
    }
  } catch (e) {
    postAgentStatus("nntrainer_builder", "error", e.message);
    vscode.window.showErrorMessage(`nntrainer build agent failed: ${e.message}`);
    return null;
  }

  await config.update("nntrainerPath", prefix, vscode.ConfigurationTarget.Workspace);
  postLog(`nntrainer built and installed to: ${prefix}`);
  return prefix;
}

// Runs the nntrainer_builder.py agent for one phase ("prep" or "build"),
// forwarding its bus events into the panel exactly like runPipeline()
// forwards orchestrator_main.py's, then resolving with its final JSON result.
function runNntrainerBuilderAgent(pythonPath, engineDir, repoPath, prefix, phase) {
  return new Promise((resolve, reject) => {
    const proc = spawn(pythonPath, ["-m", "agents.nntrainer_builder", repoPath, prefix, phase], { cwd: engineDir });
    let lastJson = null;

    const rl = readline.createInterface({ input: proc.stdout });
    rl.on("line", (line) => {
      if (!line.trim()) return;
      let event;
      try {
        event = JSON.parse(line);
      } catch {
        output.appendLine(line);
        return;
      }
      // Bus events have an "event" key (log/agent_status/etc); the final
      // result line instead has an "ok" key -- keep the latter aside.
      if (Object.prototype.hasOwnProperty.call(event, "ok")) {
        lastJson = event;
        return;
      }
      output.appendLine(`[${event.event}] ${event.message || event.detail || ""}`);
      post({ type: "engineEvent", event });
    });

    let stderr = "";
    proc.stderr.on("data", (d) => {
      stderr += d.toString();
      output.append(d.toString());
    });

    proc.on("close", (code) => {
      if (!lastJson) {
        reject(new Error(code !== 0 ? `Agent exited with code ${code}: ${stderr}` : "Agent produced no JSON result"));
        return;
      }
      resolve(lastJson);
    });

    proc.on("error", (err) => reject(err));
  });
}

// ---------------------------------------------------------------------
// Graph Focus: user types a plain-language request next to a graph pane
// ("show me the self-attention layers", "isolate the bottleneck"). Spawns
// graph_focus_agent.py against the last completed pipeline's state.json,
// which resolves the request to a set of node ids (rule-based first,
// LLM fallback if configured) and streams a "graph_focus" bus event that
// the webview uses to dim everything else in that pane.
// ---------------------------------------------------------------------

function runGraphFocusQuery(context, graphKey, query) {
  return new Promise((resolve) => {
    if (!outDir || !fs.existsSync(path.join(outDir, "state.json"))) {
      vscode.window.showWarningMessage("Run the pipeline at least once before focusing the graph.");
      resolve();
      return;
    }

    const pythonPath = vscode.workspace.getConfiguration("aiCompilerWorkbench").get("pythonPath") || "python3";
    const apiKey = vscode.workspace.getConfiguration("aiCompilerWorkbench").get("anthropicApiKey") || "";
    const engineDir = path.join(context.extensionPath, "engine");

    // Key via env so the LLM fallback works now that it's no longer written to
    // state.json (and so it never appears in argv).
    const focusEnv = { ...process.env };
    if (apiKey) focusEnv.ANTHROPIC_API_KEY = apiKey;
    const proc = spawn(pythonPath, ["-m", "agents.graph_focus_agent", outDir, graphKey, query], { cwd: engineDir, env: focusEnv });

    const rl = readline.createInterface({ input: proc.stdout });
    rl.on("line", (line) => {
      if (!line.trim()) return;
      let event;
      try {
        event = JSON.parse(line);
      } catch {
        output.appendLine(line);
        return;
      }
      if (Object.prototype.hasOwnProperty.call(event, "ok") && !event.event) {
        return; // final result line, nothing to forward
      }
      output.appendLine(`[${event.event}] ${event.message || event.explanation || ""}`);
      post({ type: "engineEvent", event });
    });

    proc.stderr.on("data", (d) => output.append(d.toString()));

    proc.on("close", () => resolve());
    proc.on("error", (err) => {
      vscode.window.showErrorMessage(`Graph focus agent failed to start: ${err.message}`);
      resolve();
    });
  });
}

function post(message) {
  if (panel) {
    panel.webview.postMessage(message);
  }
}

module.exports = { activate, deactivate };
