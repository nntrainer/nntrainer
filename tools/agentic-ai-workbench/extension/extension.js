// No build step: plain CommonJS, requiring only Node's built-ins and the
// "vscode" module VS Code injects at runtime.
const vscode = require("vscode");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");
const readline = require("readline");
const crypto = require("crypto");
const os = require("os");

let output;
let panel;
let currentProc = null;
let runningModel = "";
let outDir = null;

function activate(context) {
  output = vscode.window.createOutputChannel("AI Compiler Workbench");

  context.subscriptions.push(
    vscode.commands.registerCommand("aiCompilerWorkbench.openWorkbench", () => openPanel(context)),
    vscode.commands.registerCommand("aiCompilerWorkbench.runPipeline", () => runPipeline(context)),
    vscode.commands.registerCommand("aiCompilerWorkbench.checkModel", () => runPipeline(context, undefined, { checkOnly: true })),
    vscode.commands.registerCommand("aiCompilerWorkbench.stopPipeline", () => stopPipeline()),
    vscode.commands.registerCommand("aiCompilerWorkbench.checkEnvironment", () => ensureEnvironment(true)),
    vscode.commands.registerCommand("aiCompilerWorkbench.runDeviceProfiler", () => runDeviceProfiler(context)),
    vscode.commands.registerCommand("aiCompilerWorkbench.buildNntrainer", () => buildNntrainerFromSource(context)),
    vscode.commands.registerCommand("aiCompilerWorkbench.setupBuild", () => runBuildSetup(context, "setup")),
    vscode.commands.registerCommand("aiCompilerWorkbench.setupDockerBuild", () => runBuildSetup(context, "setup", "docker")),
    vscode.commands.registerCommand("aiCompilerWorkbench.checkDockerSetup", () => runBuildSetup(context, "check")),
    vscode.commands.registerCommand("aiCompilerWorkbench.runDockerSelfTest", () => runBuildSetup(context, "selftest")),
    vscode.commands.registerCommand("aiCompilerWorkbench.cleanUpDocker", () => cleanUpDocker(context)),
    vscode.commands.registerCommand("aiCompilerWorkbench.generateBuildAndProfile", () => runGeneratedModel(context)),
    vscode.commands.registerCommand("aiCompilerWorkbench.setAnthropicApiKey", () => setAnthropicKey(context)),
    vscode.commands.registerCommand("aiCompilerWorkbench.clearAnthropicApiKey", () => clearAnthropicKey(context)),
    context.secrets.onDidChange(async (e) => {
      if (e.key === ANTHROPIC_KEY_SECRET) anthropicKey = (await context.secrets.get(ANTHROPIC_KEY_SECRET)) || "";
    })
  );
  // Activation finishes once the key is loaded, so the first command already has it.
  return loadAnthropicKey(context).catch((err) => output.appendLine(`Could not read the Anthropic API key: ${err.message}`));
}

// ---------------------------------------------------------------------
// The optional Anthropic API key lives in VS Code's secret storage (the
// operating system's credential store), never in settings.json. It is read
// once at activation and kept current, and only engineEnv() hands it to the
// engine, through the environment (never on a command line).
// ---------------------------------------------------------------------

const ANTHROPIC_KEY_SECRET = "aiCompilerWorkbench.anthropicApiKey";
let anthropicKey = "";

async function loadAnthropicKey(context) {
  anthropicKey = (await context.secrets.get(ANTHROPIC_KEY_SECRET)) || "";
  // Earlier versions kept the key as a plain setting: move a user-settings
  // value into secret storage once. A workspace's .vscode/settings.json is
  // the user's file, so a key found there is only reported, not edited.
  const config = vscode.workspace.getConfiguration("aiCompilerWorkbench");
  const old = config.inspect("anthropicApiKey") || {};
  if (old.globalValue) {
    if (!anthropicKey) {
      await context.secrets.store(ANTHROPIC_KEY_SECRET, old.globalValue);
      anthropicKey = old.globalValue;
    }
    try {
      await config.update("anthropicApiKey", undefined, vscode.ConfigurationTarget.Global);
      vscode.window.showInformationMessage(
        "AI Compiler Workbench: your Anthropic API key was moved from your user settings into VS Code's secret storage.");
    } catch (err) {
      vscode.window.showWarningMessage(
        "AI Compiler Workbench: your Anthropic API key is now in VS Code's secret storage, but it could not be " +
        `removed from your user settings (${err.message}). Please delete aiCompilerWorkbench.anthropicApiKey there.`);
    }
  }
  if (old.workspaceValue || old.workspaceFolderValue) {
    vscode.window.showWarningMessage(
      "AI Compiler Workbench: this workspace's .vscode/settings.json contains aiCompilerWorkbench.anthropicApiKey. " +
      "It is no longer read from there: remove it, and use \"AI Compiler Workbench: Set Anthropic API Key\" instead.");
  }
}

async function setAnthropicKey(context) {
  const value = await vscode.window.showInputBox({
    title: "Anthropic API key (optional)",
    prompt: "Only for the extras: chat when the Claude CLI is missing, suggestions for unsupported ops, " +
      "Auto-Fix and the graph Focus fallback. It is kept in VS Code's secret storage.",
    placeHolder: "sk-ant-...",
    password: true,
    ignoreFocusOut: true,
  });
  if (value === undefined) return; // cancelled
  const key = value.trim();
  if (!key) {
    vscode.window.showWarningMessage("No key entered: nothing was changed.");
    return;
  }
  await context.secrets.store(ANTHROPIC_KEY_SECRET, key);
  anthropicKey = key;
  vscode.window.showInformationMessage("Anthropic API key saved in VS Code's secret storage.");
}

async function clearAnthropicKey(context) {
  await context.secrets.delete(ANTHROPIC_KEY_SECRET);
  anthropicKey = "";
  vscode.window.showInformationMessage("Anthropic API key removed from VS Code's secret storage.");
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
        await runPipeline(context, msg.model, { checkOnly: !!msg.checkOnly });
        break;
      case "browseModelFolder":
        await browseModelFolder();
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
      case "exportFiles":
        await exportFiles(msg.files);
        break;
      case "exportFile":
        await exportFile(msg.filename, msg.content);
        break;
      case "graphImageRequest":
        await requestGraphImages(msg.keys);
        break;
      case "graphImages":
        await saveGraphImages(msg);
        break;
      case "runNntrainerPipeline":
        await runNntrainerPipeline(context, msg.nntrainerPath);
        break;
      case "graphFocusQuery":
        await runGraphFocusQuery(context, msg.graph, msg.query);
        break;
      case "setupBuild":
        await runBuildSetup(context, "setup", msg.host);
        break;
      case "buildStatus":
        await runBuildSetup(context, "status");
        break;
      case "preflightCheck":
        await runBuildSetup(context, "check");
        break;
      case "codegenAction":
        await runCodegenReview(context, msg.action, msg.partId, msg.comment, msg.run);
        break;
      case "openPath":
        await openGeneratedPath(msg.path);
        break;
    }
  });

  restoreLastReview(context);
  return panel;
}

// ---------------------------------------------------------------------
// The nntrainer checkout and the environment every engine process gets.
// Everything the workbench writes (code bank, generated code, pipeline
// output) goes under that checkout's tools/agentic-ai-workbench folder.
// ---------------------------------------------------------------------

function isNntrainerRoot(p) {
  try {
    return !!p && fs.existsSync(path.join(p, "Applications", "CausalLM")) && fs.existsSync(path.join(p, "tools"));
  } catch {
    return false;
  }
}

function resolveNntrainerRoot(context) {
  const config = vscode.workspace.getConfiguration("aiCompilerWorkbench");
  const candidates = [];
  for (const key of ["nntrainerRepoPath", "nntrainerPath"]) {
    const v = (config.get(key) || "").trim();
    if (v) candidates.push(v);
  }
  for (const key of ["causallmProjectRoot", "causallmPath"]) {
    const v = (config.get(key) || "").trim();
    if (v) candidates.push(path.resolve(v, "..", ".."));
  }
  for (const f of vscode.workspace.workspaceFolders || []) candidates.push(f.uri.fsPath);
  // Running straight from the checkout (tools/agentic-ai-workbench/extension).
  candidates.push(path.resolve(context.extensionPath, "..", "..", ".."));
  return candidates.find(isNntrainerRoot) || null;
}

function workbenchOutputDir(context) {
  const config = vscode.workspace.getConfiguration("aiCompilerWorkbench");
  const custom = (config.get("outputDirectory") || "").trim();
  if (custom) return custom;
  const root = resolveNntrainerRoot(context);
  if (root) return path.join(root, "tools", "agentic-ai-workbench", "workbench_output");
  const ws = vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders[0];
  return ws ? path.join(ws.uri.fsPath, "nntrainer_out") : null;
}

function engineEnv(context) {
  const config = vscode.workspace.getConfiguration("aiCompilerWorkbench");
  const env = { ...process.env };
  if (anthropicKey) env.ANTHROPIC_API_KEY = anthropicKey;
  const root = resolveNntrainerRoot(context);
  if (root) env.NNTRAINER_ROOT = root;
  const claudePath = (config.get("claudePath") || "").trim();
  if (claudePath) env.CLAUDE_PATH = claudePath;
  env.CLAUDE_MODEL = config.get("claudeModel") || "sonnet";
  env.CLAUDE_CLI_TIMEOUT = String(config.get("claudeCliTimeout") || 900);
  env.WORKBENCH_CODEGEN = config.get("useClaudeCodegen") === false ? "0" : "1";
  env.WORKBENCH_CODEGEN_ATTEMPTS = String(config.get("codegenMaxAttempts") || 3);
  const cxx = (config.get("cxxCompiler") || "").trim();
  if (cxx) env.WORKBENCH_CXX = cxx;
  env.WORKBENCH_AUTO_VERIFY = config.get("autoVerify") === false ? "0" : "1";
  env.WORKBENCH_AUTO_PROFILE = config.get("autoProfile") === false ? "0" : "1";
  env.WORKBENCH_AUTO_PROFILE_PER_LAYER = config.get("autoProfilePerLayer") === false ? "0" : "1";
  env.WORKBENCH_BUILD_HOST = config.get("buildHost") || "auto";
  env.WORKBENCH_RUN_TOKENS = String(config.get("runMaxNewTokens") || 64);
  env.WORKBENCH_RUN_REPEATS = String(config.get("runRepeats") || 3);
  env.WORKBENCH_RUN_CPUS = String(config.get("runCpus") || "");
  env.WORKBENCH_RUN_MEMORY = String(config.get("runMemory") || "");
  env.PYTHONIOENCODING = "utf-8";
  // Proxy for Docker (image build, container) and downloads: the
  // environment's when set, else VS Code's own http.proxy / http.noProxy.
  // The engine passes these to Docker by name only (knowledge/docker_host.py).
  const httpConfig = vscode.workspace.getConfiguration("http");
  const proxy = String(httpConfig.get("proxy") || "").trim();
  const hasProxy = ["HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"].some((k) => env[k]);
  if (proxy && !hasProxy) {
    env.HTTPS_PROXY = proxy;
    env.HTTP_PROXY = proxy;
  }
  const noProxy = httpConfig.get("noProxy");
  if (Array.isArray(noProxy) && noProxy.length && !env.NO_PROXY && !env.no_proxy) env.NO_PROXY = noProxy.join(",");
  return env;
}

// ---------------------------------------------------------------------
// Review of generated code: each Review-tab button runs
// engine/agents/codegen_review.py against the last review.json.
// ---------------------------------------------------------------------

let lastReviewPath = null;
let reviewProc = null;
let lastReview = null;
let compilePromptPending = false;

// When the last part waiting for review gets approved/accepted, offer to
// compile (the pipeline does not compile code that still needs review).
function noteReview(review) {
  const before = lastReview && lastReview.review_path === review.review_path ? lastReview : null;
  lastReview = review;
  lastReviewPath = review.review_path;
  const pendingBefore = before ? ((before.summary || {}).to_review || 0) : 0;
  const pendingNow = (review.summary || {}).to_review || 0;
  const ready = !!(review.compile_ready && review.compile_ready.ok);
  const compiled = !!(review.compile && ["ok", "failed"].includes(review.compile.status));
  if (ready && !compiled && pendingBefore > 0 && pendingNow === 0) compilePromptPending = true;
}

async function promptCompile(context) {
  compilePromptPending = false;
  const pick = await vscode.window.showInformationMessage(
    "All generated parts are reviewed. Compile the code now?", "Compile", "Later");
  if (pick === "Compile") runCodegenReview(context, "compile", "", "");
}

// After a passing compile check from the Review tab: offer to build and run
// the model against Hugging Face (a pass marks the bank entries verified).
async function promptVerify(context) {
  const v = (lastReview && lastReview.verify) || {};
  if (v.status === "passed") return;
  const pick = await vscode.window.showInformationMessage(
    "Compile check passed. Build the model and verify it against Hugging Face now?", "Verify", "Later");
  if (pick === "Verify") runCodegenReview(context, "verify", "", "");
}

// ---------------------------------------------------------------------
// The workbench's own nntrainer build (tools/agentic-ai-workbench/nb),
// used by Verify: engine/agents/build_setup.py. Its "check" action is the
// Docker system check (engine/knowledge/preflight.py); setup, Verify and
// Run in Docker also run that check by themselves first.
// ---------------------------------------------------------------------
let buildProc = null;

// After "Check Docker Setup": one notification with the outcome.
async function notifyPreflight(result) {
  if (!result) {
    vscode.window.showWarningMessage("The Docker setup check did not finish -- see the output.");
    return;
  }
  const warns = (result.counts && result.counts.warn) || 0;
  if (result.ok && !warns) {
    vscode.window.showInformationMessage(`Docker setup check: ${result.summary}`);
    return;
  }
  const pick = await vscode.window.showWarningMessage(`Docker setup check: ${result.summary}`, "Show details");
  if (pick === "Show details") output.show(true);
}

// After "Run Docker Self-Test".
async function notifySelftest(result) {
  if (!result) {
    vscode.window.showWarningMessage("The Docker self-test did not finish -- see the output.");
    return;
  }
  const mins = Math.round(((result.seconds || 0) / 60) * 10) / 10;
  if (result.ok) {
    vscode.window.showInformationMessage(`Docker self-test passed in ${mins} min: ${result.summary}`);
    return;
  }
  const pick = await vscode.window.showWarningMessage(`Docker self-test failed: ${result.summary}`, "Show details");
  if (pick === "Show details") output.show(true);
}

function fmtBytes(n) {
  return n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : `${Math.round(n / 1e6)} MB`;
}

// "Clean Up Docker": list what the workbench keeps in Docker, pick what goes,
// confirm, remove (engine/agents/build_setup.py cleanup).
async function cleanUpDocker(context) {
  const usage = await runBuildSetup(context, "cleanup", undefined, ["--list"]);
  if (!usage) return;                                   // busy, or the engine did not start
  if (!usage.ok) {
    vscode.window.showWarningMessage(`Clean Up Docker: ${usage.error || "could not read Docker's disk use"}`);
    return;
  }
  const items = usage.items || [];
  if (!items.length) {
    vscode.window.showInformationMessage("The workbench keeps nothing in Docker.");
    return;
  }
  const old = items.filter((i) => i.old);
  const pick = await vscode.window.showQuickPick([
    { label: "Old images only", value: "old",
      description: old.length ? `${fmtBytes(usage.old_total || 0)} in ${old.length} image(s) · keeps the build` : "nothing old to remove" },
    { label: "Everything", value: "all",
      description: `${fmtBytes(usage.total || 0)} · the next Docker step sets the build up again from scratch` },
  ], { placeHolder: `The workbench keeps ${fmtBytes(usage.total || 0)} in Docker. What should be removed?` });
  if (!pick) return;
  const chosen = pick.value === "old" ? old : items;
  if (!chosen.length) {
    vscode.window.showInformationMessage("Nothing old to remove: the build and the newest run image of each model stay.");
    return;
  }
  const size = chosen.reduce((s, i) => s + (i.size || 0), 0);
  const detail = chosen.map((i) => `${i.name} (${fmtBytes(i.size || 0)})`).join("\n") + (pick.value === "all"
    ? "\n\nThe build container and its volume go too: the next Verify or Run sets the Docker build up again, which takes a while."
    : "");
  const confirm = await vscode.window.showWarningMessage(
    `Remove ${chosen.length} item(s), about ${fmtBytes(size)}, from Docker?`, { modal: true, detail }, "Remove");
  if (confirm !== "Remove") return;
  const r = await runBuildSetup(context, "cleanup", undefined, ["--scope", pick.value]);
  if (!r) return;
  if (r.ok) {
    vscode.window.showInformationMessage(`Clean Up Docker: removed ${(r.removed || []).length} item(s), freed about ${fmtBytes(r.freed || 0)}.`);
  } else {
    const more = await vscode.window.showWarningMessage(
      `Clean Up Docker: ${(r.failed || []).length} item(s) could not be removed${r.error ? ": " + r.error : ""}.`, "Show details");
    if (more) output.show(true);
  }
  if (pick.value === "all") await runBuildSetup(context, "status");    // the Review tab's build state
}

const BUILD_ACTION_TITLE = {
  check: "Docker setup check",
  cleanup: "Clean Up Docker",
  selftest: "Docker self-test",
};
// The event each action's promise resolves with (the last one of that kind).
const BUILD_ACTION_RESULT = {
  check: "preflight",
  selftest: "selftest",
  cleanup: "docker_cleanup",
};

function runBuildSetup(context, action, host, extraArgs) {
  return new Promise((resolve) => {
    if (buildProc) {
      // A status refresh (opening the Review tab) just waits for next time.
      if (action !== "status") {
        vscode.window.showInformationMessage("A Docker / nntrainer build task (setup, check, clean up or self-test) is already running.");
      }
      resolve();
      return;
    }
    const engineDir = path.join(context.extensionPath, "engine");
    const extra = extraArgs || [];
    output.appendLine(`\n--- ${BUILD_ACTION_TITLE[action] || "nntrainer build: " + action}${host ? " (" + host + ")" : ""} ---`);
    if (["setup", "check", "selftest"].includes(action)) output.show(true);
    const args = ["-m", "agents.build_setup", action, ...extra];
    if (host) args.push("--host", host);
    if (action === "check") args.push("--step", "all");
    buildProc = spawn(pythonFor(), args, { cwd: engineDir, env: engineEnv(context) });
    let last = null;
    const lastOf = {};
    const rl = readline.createInterface({ input: buildProc.stdout });
    rl.on("line", (line) => {
      let event;
      try {
        event = JSON.parse(line);
      } catch {
        output.appendLine(line);
        return;
      }
      if (event.event === "build_status") last = event;
      lastOf[event.event] = event;
      if (event.event === "log") output.appendLine(event.message || "");
      post({ type: "engineEvent", event });
    });
    buildProc.stderr.on("data", (d) => output.append(d.toString()));
    let finished = false;
    const done = () => {
      if (finished) return;
      finished = true;
      buildProc = null;
      const want = action === "cleanup" && extra.includes("--list") ? "docker_usage" : BUILD_ACTION_RESULT[action];
      resolve(want ? lastOf[want] : last);
      if (action === "check") {
        notifyPreflight(lastOf.preflight);
        return;
      }
      if (action === "selftest") {
        notifySelftest(lastOf.selftest);
        return;
      }
      if (action === "cleanup") return;               // cleanUpDocker reports the outcome
      if (action !== "setup") return;
      if (last && last.ok && !(last.missing || []).length) {
        vscode.window.showInformationMessage("nntrainer build is ready. Verify is available in the Review tab.");
      } else {
        const why = last ? ((last.missing || []).join("; ") || last.why || last.status) : "see the output";
        vscode.window.showWarningMessage(`nntrainer build not ready: ${why}`);
      }
    };
    buildProc.on("close", done);
    buildProc.on("error", (err) => {
      vscode.window.showErrorMessage(`Build setup failed to start: ${err.message}`);
      done();
    });
  });
}

// `accept` decides whether this process's events still matter (a replaced
// pipeline run keeps printing until it dies; its events must not repaint the UI).
function forwardEngineLines(proc, accept) {
  const rl = readline.createInterface({ input: proc.stdout });
  rl.on("line", (line) => {
    if (!line.trim()) return;
    if (accept && !accept()) return;
    let event;
    try {
      event = JSON.parse(line);
    } catch {
      output.appendLine(line);
      return;
    }
    if (event.event === "codegen" && event.data && event.data.review_path) noteReview(event.data);
    if (event.event === "run_status" && event.status === "no_weights") pendingNoWeights = event;
    output.appendLine(`[${event.event}] ${event.message || event.detail || ""}`);
    post({ type: "engineEvent", event });
  });
  proc.stderr.on("data", (d) => {
    if (!accept || accept()) output.append(d.toString());
  });
}

// Stop a process and everything it started (the pipeline starts the Claude
// CLI, pip, ...). Windows: taskkill /T; elsewhere the process group.
function killTree(proc) {
  if (!proc || proc.exitCode !== null || proc.signalCode) return;
  try {
    if (process.platform === "win32" && proc.pid) {
      spawn("taskkill", ["/PID", String(proc.pid), "/T", "/F"], { windowsHide: true }).on("error", () => proc.kill());
    } else if (proc.pid) {
      try {
        process.kill(-proc.pid, "SIGTERM");
      } catch {
        proc.kill("SIGTERM");
      }
    } else {
      proc.kill();
    }
  } catch {
    // already gone
  }
}

// The model's weights are not on this computer: offer to download them, then run again.
let pendingNoWeights = null;

async function promptDownloadWeights(context, info, runArgs) {
  const pick = await vscode.window.showInformationMessage(
    `${info.message || "The model's weights are not on this computer."} Download them now ` +
    "(Hugging Face safetensors into tools/agentic-ai-workbench/.w)?", "Download and run", "Cancel");
  if (pick !== "Download and run") return;
  await runCodegenReview(context, "fetchweights", "", "");
  await runCodegenReview(context, "run", "", "", runArgs);
}

// Build in Docker, load on this PC, run with the real weights, profile.
async function runGeneratedModel(context) {
  openPanel(context);
  if (!lastReviewPath) await restoreLastReview(context);
  const prompt = await vscode.window.showInputBox({
    prompt: "Prompt for the generated model (it runs with its real weights in Docker on this PC)",
    value: "What is the capital of France?",
  });
  if (prompt === undefined) return;
  await runCodegenReview(context, "run", "", "", { prompt });
}

function runCodegenReview(context, action, partId, comment, runArgs) {
  return new Promise((resolve) => {
    const allowed = ["show", "approve", "reject", "regenerate", "recheck", "compile", "fixcompile",
      "verify", "fixverify", "run", "fetchweights"];
    if (!allowed.includes(action)) {
      resolve();
      return;
    }
    if (!lastReviewPath || !fs.existsSync(lastReviewPath)) {
      vscode.window.showWarningMessage("Run the pipeline first -- there is no generated code to review yet.");
      resolve();
      return;
    }
    if (reviewProc) {
      vscode.window.showInformationMessage("A review action is still running.");
      resolve();
      return;
    }
    const pythonPath = pythonFor();
    const engineDir = path.join(context.extensionPath, "engine");
    const args = ["-m", "agents.codegen_review", lastReviewPath, action];
    if (partId) args.push(String(partId));
    if (comment) args.push("--comment", String(comment));
    const r = runArgs || {};
    if (action === "run") {
      if (r.prompt) args.push("--prompt", String(r.prompt));
      for (const k of ["tokens", "repeat", "cpus", "memory", "target"]) {
        if (r[k] !== undefined && r[k] !== null && String(r[k]).trim() !== "") args.push(`--${k}`, String(r[k]).trim());
      }
      if (r.perLayer === true) args.push("--per-layer");
      output.show(true);
    }
    post({ type: "codegenBusy", action, partId });
    output.appendLine(`\n--- Review: ${action} ${partId || ""} ---`);
    reviewProc = spawn(pythonPath, args, { cwd: engineDir, env: engineEnv(context) });
    pendingNoWeights = null;
    forwardEngineLines(reviewProc);
    const done = () => {
      reviewProc = null;
      post({ type: "codegenIdle" });
      resolve();
      if (action === "run" && pendingNoWeights) {
        const info = pendingNoWeights;
        pendingNoWeights = null;
        setTimeout(() => promptDownloadWeights(context, info, r), 0);
        return;
      }
      if (compilePromptPending) setTimeout(() => promptCompile(context), 0);
      else if (action === "compile" && lastReview && lastReview.compile && lastReview.compile.status === "ok") {
        setTimeout(() => promptVerify(context), 0);
      }
    };
    reviewProc.on("close", done);
    reviewProc.on("error", (err) => {
      vscode.window.showErrorMessage(`Review action failed to start: ${err.message}`);
      done();
    });
  });
}

// Re-show the last run's review when the panel opens again.
function restoreLastReview(context) {
  try {
    const dir = workbenchOutputDir(context);
    const statePath = dir && path.join(dir, "state.json");
    if (!statePath || !fs.existsSync(statePath)) return;
    const saved = JSON.parse(fs.readFileSync(statePath, "utf-8"));
    if (saved.codegen_review_path && fs.existsSync(saved.codegen_review_path)) {
      outDir = outDir || dir;
      lastReviewPath = saved.codegen_review_path;
      return runCodegenReview(context, "show", "", "");
    }
  } catch {
    // no saved run: nothing to restore
  }
}

async function openGeneratedPath(p) {
  // Only files of the generated model under review.
  if (!p || !lastReviewPath) return;
  const root = path.resolve(path.dirname(lastReviewPath));
  const target = path.resolve(p);
  if (!(target === root || target.startsWith(root + path.sep)) || !fs.existsSync(target)) return;
  const doc = await vscode.workspace.openTextDocument(target);
  await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
}

// ---------------------------------------------------------------------
// Run Pipeline: spawns engine/orchestrator_main.py, streams JSON events
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// Local model folders: pick one, and check it before running.
// ---------------------------------------------------------------------

function stripQuotes(s) {
  return String(s || "").trim().replace(/^["']+|["']+$/g, "").trim();
}

function looksLikePath(s) {
  return path.isAbsolute(s) || s.includes("\\") || s.startsWith(".") || s.startsWith("~") ||
    (fs.existsSync(s) && fs.statSync(s).isDirectory());
}

// Parameter count from a .safetensors header (8-byte length + JSON); reads
// only the header, never the weights.
function safetensorsParams(file) {
  let fd;
  try {
    fd = fs.openSync(file, "r");
    const lenBuf = Buffer.alloc(8);
    fs.readSync(fd, lenBuf, 0, 8, 0);
    const len = Number(lenBuf.readBigUInt64LE(0));
    if (!len || len > 100 * 1024 * 1024) return 0;
    const hdr = Buffer.alloc(len);
    fs.readSync(fd, hdr, 0, len, 8);
    let total = 0;
    for (const [k, v] of Object.entries(JSON.parse(hdr.toString("utf-8")))) {
      if (k === "__metadata__" || !v || !Array.isArray(v.shape)) continue;
      total += v.shape.reduce((a, b) => a * b, 1);
    }
    return total;
  } catch {
    return 0;
  } finally {
    if (fd !== undefined) fs.closeSync(fd);
  }
}

function inspectModelFolder(dir) {
  const full = path.resolve(dir);
  const info = { path: full, ok: true, problems: [], warnings: [], summary: "" };
  if (!fs.existsSync(full) || !fs.statSync(full).isDirectory()) {
    info.ok = false;
    info.problems.push("Folder not found.");
    return info;
  }
  const files = fs.readdirSync(full);
  if (!files.includes("config.json")) {
    info.ok = false;
    info.problems.push(files.some((f) => f.toLowerCase().endsWith(".gguf"))
      ? "This folder has only GGUF files, which are not supported yet -- use the Hugging Face (safetensors) version of the model."
      : "No config.json -- this is not a Hugging Face model folder.");
    return info;
  }
  let cfg;
  try {
    cfg = JSON.parse(fs.readFileSync(path.join(full, "config.json"), "utf-8"));
  } catch (e) {
    info.ok = false;
    info.problems.push(`config.json could not be read: ${e.message}`);
    return info;
  }
  const archs = (cfg.architectures || []).filter((a) => typeof a === "string");
  const arch = archs[0] || cfg.model_type || "unknown architecture";
  const causal = archs.some((a) => /(ForCausalLM|LMHeadModel)$/.test(a)) ||
    (archs.some((a) => /ForConditionalGeneration$/.test(a)) && !!cfg.text_config && !cfg.is_encoder_decoder);
  if (!causal) info.warnings.push(`${arch} is not a causal language model: the C++ generator will skip it.`);
  const st = files.filter((f) => f.endsWith(".safetensors"));
  const bins = files.filter((f) => /^pytorch_model.*\.bin$/.test(f));
  const bytes = [...st, ...bins].reduce((n, f) => n + fs.statSync(path.join(full, f)).size, 0);
  const params = st.reduce((n, f) => n + safetensorsParams(path.join(full, f)), 0);
  if (!st.length && !bins.length) {
    info.warnings.push("No weight files: tensor names come from transformers instead, and nothing can be converted.");
  } else if (!st.length) {
    info.warnings.push("Only pytorch_model*.bin weights (no safetensors): tensor shapes come from transformers instead.");
  }
  const tok = files.includes("tokenizer.json") || files.includes("tokenizer.model");
  if (!tok) info.warnings.push("No tokenizer.json (needed later to run the model).");
  const ramGB = (params * 4) / 1e9;
  const haveGB = os.totalmem() / 1e9;
  if (params && ramGB > haveGB * 0.6) {
    info.warnings.push(`A full Run builds the model in memory (~${ramGB.toFixed(0)} GB; this machine has ` +
      `${haveGB.toFixed(0)} GB): use Check only.`);
  }
  const gb = (bytes / 1e9).toFixed(bytes > 1e9 ? 1 : 2);
  info.summary = [
    arch,
    st.length || bins.length ? `${st.length || bins.length} weight file(s), ${gb} GB` : "no weights",
    params ? `${(params / 1e9).toFixed(2)}B parameters` : "",
    tok ? "tokenizer ✓" : "",
  ].filter(Boolean).join(" · ");
  return info;
}

async function browseModelFolder() {
  const picked = await vscode.window.showOpenDialog({
    canSelectFolders: true, canSelectFiles: false, canSelectMany: false,
    title: "Select a model folder (with config.json and *.safetensors)", openLabel: "Use this model",
  });
  if (!picked || !picked.length) return;
  post({ type: "modelFolderInfo", ...inspectModelFolder(picked[0].fsPath) });
}

let runStarting = false; // a run is being prepared (environment check): blocks a second start

async function runPipeline(context, modelNameArg, options) {
  const checkOnly = !!(options && options.checkOnly);
  openPanel(context);
  if (runStarting) {
    vscode.window.showInformationMessage("A pipeline run is already starting.");
    return;
  }

  let modelName = stripQuotes(modelNameArg);
  if (!modelName) {
    modelName = stripQuotes(await vscode.window.showInputBox({
      prompt: "Hugging Face model id or local model folder",
      placeHolder: "e.g. Qwen/Qwen3-0.6B or C:\\models\\Qwen3-0.6B",
    }));
  }
  if (!modelName) {
    return;
  }

  // A local folder is checked before anything starts (no Python needed).
  if (looksLikePath(modelName)) {
    const info = inspectModelFolder(modelName);
    post({ type: "modelFolderInfo", ...info });
    if (!info.ok) {
      vscode.window.showErrorMessage(`${modelName}: ${info.problems.join(" ")}`);
      post({ type: "pipelineNotStarted", model: modelName, reason: info.problems.join(" ") });
      return;
    }
    modelName = info.path;
    for (const w of info.warnings) output.appendLine(`[folder] ${w}`);
  }

  if (currentProc && currentProc.exitCode === null) {
    const choice = await vscode.window.showWarningMessage(
      `A pipeline is still running for ${runningModel}. Stop it and run ${modelName}?`,
      "Stop and run",
      "Cancel"
    );
    if (choice !== "Stop and run") return;
  }

  outDir = workbenchOutputDir(context);
  if (!outDir) {
    const reason = "Open the nntrainer folder (or set aiCompilerWorkbench.nntrainerRepoPath / outputDirectory) first -- " +
      "output is written under tools/agentic-ai-workbench.";
    vscode.window.showErrorMessage(reason);
    post({ type: "pipelineNotStarted", model: modelName, reason });
    return;
  }
  fs.mkdirSync(outDir, { recursive: true });

  runStarting = true;
  post({ type: "pipelinePreparing", model: modelName, checkOnly });
  let ready = false;
  try {
    ready = await ensureEnvironment(false);
  } finally {
    runStarting = false;
  }
  if (!ready) {
    post({ type: "pipelineNotStarted", model: modelName, reason: "The Python environment is not ready (see Check Env)." });
    return;
  }

  stopPipeline(true); // a previous run is replaced, never left running alongside

  const pythonPath = pythonFor();
  const causallmRoot =vscode.workspace.getConfiguration("aiCompilerWorkbench").get("causallmProjectRoot") || "";
  const installGenerated = !!vscode.workspace.getConfiguration("aiCompilerWorkbench").get("installGeneratedFiles");
  const generatedHeaderDir = vscode.workspace.getConfiguration("aiCompilerWorkbench").get("generatedHeaderDirectory") || "include/generated";
  const generatedSourceDir = vscode.workspace.getConfiguration("aiCompilerWorkbench").get("generatedSourceDirectory") || "src/generated";
  const engineDir = path.join(context.extensionPath, "engine");
  const scriptPath = path.join(engineDir, "orchestrator_main.py");

  const args = [scriptPath, modelName, "--out", outDir];
  if (causallmRoot) args.push("--causallm-root", causallmRoot);
  if (installGenerated) args.push("--install-generated");
  args.push("--generated-header-dir", generatedHeaderDir, "--generated-source-dir", generatedSourceDir);
  if (checkOnly) args.push("--check-only");
  // engineEnv carries the optional API key (never argv: argv is readable by
  // other users via `ps` / /proc), the nntrainer checkout and the Claude CLI settings.
  const spawnEnv = engineEnv(context);

  output.show(true);
  output.appendLine(`\n--- ${checkOnly ? "Checking" : "Running pipeline for"} "${modelName}" (output: ${outDir}) ---`);
  lastReviewPath = null;
  if (reviewProc) {
    killTree(reviewProc); // a review of the previous run must not repaint the new one
    reviewProc = null;
  }

  const proc = spawn(pythonPath, args, { cwd: engineDir, env: spawnEnv, detached: process.platform !== "win32" });
  currentProc = proc;
  runningModel = modelName;
  const startedAt = Date.now();
  post({ type: "pipelineStarted", model: modelName, startedAt, checkOnly });
  // Non-JSON stdout noise stays in the output channel only.
  forwardEngineLines(proc, () => proc === currentProc);

  proc.on("close", (code) => {
    if (proc !== currentProc) return; // a replaced run
    currentProc = null;
    if (code !== 0 && code !== null) {
      post({ type: "engineEvent", event: { event: "error", stage: "process", message: `Engine exited with code ${code}` } });
    }
    post({ type: "pipelineFinished", model: modelName, ok: code === 0, code, seconds: Math.round((Date.now() - startedAt) / 1000) });
  });

  proc.on("error", (err) => {
    if (proc !== currentProc) return;
    currentProc = null;
    vscode.window.showErrorMessage(`Could not start Python (${pythonPath}). Run 'Check Env'.`);
    post({ type: "engineEvent", event: { event: "error", stage: "process", message: err.message } });
    post({ type: "pipelineFinished", model: modelName, ok: false, code: null, seconds: 0 });
  });
}

function stopPipeline(replacing) {
  if (currentProc) {
    const proc = currentProc;
    currentProc = null; // its remaining output is ignored from here on
    killTree(proc);
    if (!replacing) post({ type: "pipelineStopped" });
  }
}

// ---------------------------------------------------------------------
// Chat: spawns agents/chat_agent.py against the last saved state.json
// ---------------------------------------------------------------------

let chatProc = null;

async function sendChat(context, message) {
  if (chatProc) {
    vscode.window.showInformationMessage("Claude is still answering the previous question.");
    return;
  }
  // Chat works before any run too (it then has no run to talk about).
  const chatDir = outDir || workbenchOutputDir(context) || path.join(context.globalStorageUri.fsPath, "chat");
  fs.mkdirSync(chatDir, { recursive: true });

  const pythonPath = pythonFor();
  const engineDir = path.join(context.extensionPath, "engine");

  post({ type: "engineEvent", event: { event: "chat", role: "user", content: message } });
  post({ type: "chatBusy" });

  // The optional key goes through the environment only (argv is visible to other processes).
  const chatEnv = engineEnv(context);
  chatProc = spawn(pythonPath, ["-m", "agents.chat_agent", chatDir, message], { cwd: engineDir, env: chatEnv });
  forwardEngineLines(chatProc);
  const done = () => {
    chatProc = null;
    post({ type: "chatIdle" });
  };
  chatProc.on("close", done);
  chatProc.on("error", (err) => {
    post({ type: "engineEvent", event: { event: "chat", role: "assistant", content: `Chat failed to start: ${err.message}` } });
    done();
  });
}

// ---------------------------------------------------------------------
// Python environment. Works the same on every machine:
//   1. If aiCompilerWorkbench.pythonPath is set, that interpreter is used.
//   2. Otherwise the workbench uses its own virtual environment, created on
//      first use from a real Python 3.10+ it finds (VS Code's selected
//      interpreter, the Windows `py` launcher, standard install folders,
//      PATH). The Microsoft Store "python3" placeholder is skipped.
// The venv lives outside the repo (about 1 GB, shared by every checkout, and
// short on Windows, where torch's deep header paths can exceed 260 chars).
// ---------------------------------------------------------------------

const DEFAULT_PYTHON_SETTINGS = new Set(["", "python3", "python"]);
const CORE_IMPORTS = "import torch, transformers, huggingface_hub, safetensors";
const TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu";
let discoveredPython = null;

function configuredPython() {
  return (vscode.workspace.getConfiguration("aiCompilerWorkbench").get("pythonPath") || "").trim();
}

function managedVenvDir() {
  const custom = (vscode.workspace.getConfiguration("aiCompilerWorkbench").get("venvPath") || "").trim();
  if (custom) return custom;
  if (process.platform === "win32") {
    const local = process.env.LOCALAPPDATA || path.join(os.homedir(), "AppData", "Local");
    return path.join(local, "ai-compiler-workbench", "venv");
  }
  const data = process.env.XDG_DATA_HOME || path.join(os.homedir(), ".local", "share");
  return path.join(data, "ai-compiler-workbench", "venv");
}

const VENV_MARKER = ".ai-compiler-workbench-venv";

// Only a folder the workbench created (marker file) or an empty one may be
// deleted -- never a folder someone pointed venvPath at.
function removeManagedVenv(dir) {
  if (!fs.existsSync(dir)) return true;
  let entries = [];
  try {
    entries = fs.readdirSync(dir);
  } catch {
    return false;
  }
  if (entries.length && !entries.includes(VENV_MARKER)) return false;
  fs.rmSync(dir, { recursive: true, force: true });
  return true;
}

function venvPythonPath(dir) {
  return process.platform === "win32" ? path.join(dir, "Scripts", "python.exe") : path.join(dir, "bin", "python");
}

// The interpreter every engine process is started with.
function pythonFor() {
  const configured = configuredPython();
  if (!DEFAULT_PYTHON_SETTINGS.has(configured)) return configured;
  const managed = venvPythonPath(managedVenvDir());
  if (fs.existsSync(managed)) return managed;
  if (discoveredPython) return discoveredPython.exe;
  return process.platform === "win32" ? "python" : "python3";
}

// Run a command, collecting its output; never rejects.
function runCommand(cmd, args, onData, timeoutMs) {
  return new Promise((resolve) => {
    let proc;
    let out = "";
    let timer = null;
    try {
      proc = spawn(cmd, args, { env: { ...process.env, PYTHONIOENCODING: "utf-8", PIP_DISABLE_PIP_VERSION_CHECK: "1" } });
    } catch {
      resolve({ ok: false, notFound: true, code: null, out });
      return;
    }
    const take = (d) => {
      const s = d.toString();
      out += s;
      if (onData) onData(s);
    };
    proc.stdout.on("data", take);
    proc.stderr.on("data", take);
    if (timeoutMs) timer = setTimeout(() => proc.kill(), timeoutMs);
    proc.on("error", (err) => {
      if (timer) clearTimeout(timer);
      resolve({ ok: false, notFound: err.code === "ENOENT", code: null, out });
    });
    proc.on("close", (code) => {
      if (timer) clearTimeout(timer);
      resolve({ ok: code === 0, notFound: false, code, out });
    });
  });
}

const PROBE =
  "import sys, json, importlib.util as u; print(json.dumps({'exe': sys.executable, " +
  "'v': list(sys.version_info[:2]), 'venv': u.find_spec('venv') is not None, " +
  "'ensurepip': u.find_spec('ensurepip') is not None}))";

async function probePython(cmd) {
  const r = await runCommand(cmd[0], [...cmd.slice(1), "-c", PROBE], null, 20000);
  if (!r.ok) return null;
  const line = r.out.trim().split(/\r?\n/).pop();
  try {
    const info = JSON.parse(line);
    if (!info.exe || /[\\/]WindowsApps[\\/]/i.test(info.exe)) return null;
    return info;
  } catch {
    return null;
  }
}

function listDirs(base, pattern) {
  try {
    return fs.readdirSync(base).filter((d) => pattern.test(d)).map((d) => path.join(base, d));
  } catch {
    return [];
  }
}

async function pythonCandidates() {
  const c = [];
  try {
    // The interpreter selected in VS Code's Python extension (conda, pyenv, ...).
    const ext = vscode.extensions.getExtension("ms-python.python");
    if (ext) {
      if (!ext.isActive) await ext.activate();
      const env = ext.exports && ext.exports.environments;
      const p = env && env.getActiveEnvironmentPath && env.getActiveEnvironmentPath();
      if (p && p.path) c.push([p.path]);
    }
  } catch {
    // no Python extension: fine
  }
  if (process.platform === "win32") {
    for (const v of ["3.13", "3.12", "3.11", "3.10", "3"]) c.push(["py", `-${v}`]);
    const local = process.env.LOCALAPPDATA || path.join(os.homedir(), "AppData", "Local");
    const roots = [
      path.join(local, "Programs", "Python"),
      process.env.ProgramFiles || "C:\\Program Files",
      process.env["ProgramFiles(x86)"] || "C:\\Program Files (x86)",
      "C:\\",
    ];
    for (const root of roots) {
      for (const d of listDirs(root, /^Python3\d+/i)) c.push([path.join(d, "python.exe")]);
    }
    c.push(["python"], ["python3"]);
  } else {
    for (const v of ["3.13", "3.12", "3.11", "3.10"]) c.push([`python${v}`]);
    for (const dir of ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"]) {
      for (const v of ["3.13", "3.12", "3.11", "3.10"]) c.push([path.join(dir, `python${v}`)]);
    }
    c.push(["python3"], ["python"]);
  }
  return c;
}

// Usable interpreters, best first: 3.10-3.13 (torch publishes builds for
// these on every platform) newest first, then newer versions.
async function findPythons() {
  const seen = new Map();
  const results = await Promise.all((await pythonCandidates()).map((cmd) => probePython(cmd)));
  for (const info of results) {
    if (!info) continue;
    const [major, minor] = info.v;
    if (major !== 3 || minor < 10) continue;
    const key = process.platform === "win32" ? info.exe.toLowerCase() : info.exe;
    if (key.startsWith(managedVenvDir().toLowerCase())) continue;
    if (!seen.has(key)) seen.set(key, info);
  }
  const rank = (i) => (i.v[1] <= 13 ? 0 : 1);
  return [...seen.values()].sort((a, b) => rank(a) - rank(b) || b.v[1] - a.v[1]);
}

async function importsOk(py) {
  const r = await runCommand(py, ["-c", CORE_IMPORTS], null, 120000);
  return r.ok;
}

function pipErrorSummary(out) {
  const lines = out.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  const err = lines.filter((l) => /^ERROR:|error:|No matching distribution|Could not find a version/i.test(l));
  let msg = (err.length ? err.slice(-2) : lines.slice(-2)).join(" ").slice(0, 400);
  if (/LongPaths|path too long|\[Errno 2\] No such file or directory/i.test(out)) {
    msg += " -- Windows path-length limit: enable long paths (Settings > System > For developers > " +
      "Enable long paths), or set aiCompilerWorkbench.venvPath to a short folder such as C:\\wbenv.";
  }
  if (/SSL|CERTIFICATE|ProxyError|Connection|timed out|getaddrinfo/i.test(out)) {
    msg += " -- network problem: check the internet connection / proxy (pip honours HTTPS_PROXY).";
  }
  return msg;
}

async function ensureEnvironment(explicit) {
  const configured = configuredPython();
  const py = pythonFor();
  if (await importsOk(py)) {
    if (explicit) vscode.window.showInformationMessage(`Python environment OK: ${py}`);
    return true;
  }

  if (!DEFAULT_PYTHON_SETTINGS.has(configured)) {
    const probe = await probePython([configured]);
    const choice = await vscode.window.showWarningMessage(
      probe
        ? `The Python in aiCompilerWorkbench.pythonPath (${configured}, ${probe.v.join(".")}) is missing torch/transformers. ` +
          "Install them into it, or let the workbench use its own environment?"
        : `aiCompilerWorkbench.pythonPath is "${configured}", which does not run. Let the workbench set up its own environment?`,
      ...(probe ? ["Install into this Python"] : []),
      "Use the workbench's own environment",
      "Open Settings"
    );
    if (choice === "Install into this Python") return installInto(configured);
    if (choice === "Open Settings") {
      vscode.commands.executeCommand("workbench.action.openSettings", "aiCompilerWorkbench.pythonPath");
      return false;
    }
    if (choice !== "Use the workbench's own environment") return false;
    await vscode.workspace.getConfiguration("aiCompilerWorkbench")
      .update("pythonPath", "", vscode.ConfigurationTarget.Global);
    return setupManagedEnvironment();
  }

  const venv = managedVenvDir();
  const choice = await vscode.window.showWarningMessage(
    `The workbench needs torch and transformers. Set up its own Python environment in ${venv}? ` +
      "(one-time, about 1 GB download; your other Python installs are not touched)",
    "Set up now",
    "Cancel"
  );
  if (choice !== "Set up now") return false;
  return setupManagedEnvironment();
}

function pipInstall(py, args, progress, label) {
  progress.report({ message: label });
  output.appendLine(`\n> ${py} -m pip install ${args.join(" ")}`);
  return runCommand(py, ["-m", "pip", "install", "--disable-pip-version-check", ...args], (s) => output.append(s));
}

async function installPackages(py, progress) {
  const engineDir = path.join(__dirname, "engine");
  // CPU build of torch: the workbench runs torch on the CPU only (tracing, and
  // Verify's tiny reference model). On Linux this avoids a multi-GB CUDA
  // download; Windows/macOS wheels from PyPI are CPU builds.
  let r;
  if (process.platform === "linux") {
    r = await pipInstall(py, ["torch", "--index-url", TORCH_CPU_INDEX], progress, "Installing torch (CPU build)...");
    if (!r.ok) r = await pipInstall(py, ["torch"], progress, "Installing torch...");
  } else {
    r = await pipInstall(py, ["torch"], progress, "Installing torch...");
  }
  if (!r.ok) return { ok: false, torchMissing: /No matching distribution found for torch|from versions: none/i.test(r.out), out: r.out };
  r = await pipInstall(py, ["-r", path.join(engineDir, "requirements.txt")], progress, "Installing transformers and friends...");
  if (!r.ok) return { ok: false, out: r.out };
  const opt = await pipInstall(py, ["-r", path.join(engineDir, "requirements-agentic.txt")], progress,
    "Installing optional agent packages (langchain, langgraph)...");
  if (!opt.ok) output.appendLine("\nOptional agent packages did not install; the pipeline runs its steps in order without them.");
  return { ok: true };
}

function setupManagedEnvironment() {
  return vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: "AI Compiler Workbench: Python environment", cancellable: false },
    async (progress) => {
      output.show(true);
      const venv = managedVenvDir();
      const vpy = venvPythonPath(venv);
      output.appendLine(`\n--- Setting up the workbench's Python environment in ${venv} ---`);

      // Reuse an existing environment that only lacks packages.
      if (fs.existsSync(vpy)) {
        const r = await installPackages(vpy, progress);
        if (r.ok && (await importsOk(vpy))) return envReady(vpy);
        output.appendLine("\nThe existing environment could not be completed; rebuilding it.");
      }

      progress.report({ message: "Looking for Python 3.10 or newer..." });
      const found = await findPythons();
      output.appendLine("Found: " + (found.map((p) => `${p.exe} (${p.v.join(".")})`).join(", ") || "none"));
      if (!found.length) {
        const pick = await vscode.window.showErrorMessage(
          "No Python 3.10 or newer was found. Install Python 3.12 (python.org; on Windows tick \"Add python.exe to PATH\"), " +
            "or point aiCompilerWorkbench.pythonPath at one, then run 'Check Env' again.",
          "Download Python",
          "Open Settings"
        );
        if (pick === "Download Python") vscode.env.openExternal(vscode.Uri.parse("https://www.python.org/downloads/"));
        if (pick === "Open Settings") vscode.commands.executeCommand("workbench.action.openSettings", "aiCompilerWorkbench.pythonPath");
        return false;
      }

      let lastError = "";
      for (const base of found.slice(0, 3)) {
        discoveredPython = base;
        output.appendLine(`\nBase interpreter: ${base.exe} (Python ${base.v.join(".")})`);
        if (!removeManagedVenv(venv)) {
          vscode.window.showErrorMessage(
            `${venv} already exists and was not created by the workbench. Set aiCompilerWorkbench.venvPath to an ` +
              "empty or new folder (or clear it to use the default)."
          );
          return false;
        }
        progress.report({ message: `Creating the environment with Python ${base.v.join(".")}...` });
        let r = await runCommand(base.exe, ["-m", "venv", venv], (s) => output.append(s));
        if (r.ok && fs.existsSync(vpy)) fs.writeFileSync(path.join(venv, VENV_MARKER), "created by the AI Compiler Workbench\n");
        if (!r.ok || !fs.existsSync(vpy)) {
          lastError = base.venv && base.ensurepip
            ? `could not create a virtual environment with ${base.exe}: ${pipErrorSummary(r.out)}`
            : `${base.exe} has no working venv module (on Debian/Ubuntu: sudo apt install python3-venv)`;
          output.appendLine(lastError);
          continue;
        }
        await pipInstall(vpy, ["--upgrade", "pip"], progress, "Updating pip...");
        const res = await installPackages(vpy, progress);
        if (res.ok && (await importsOk(vpy))) return envReady(vpy);
        lastError = pipErrorSummary(res.out || "");
        if (res.torchMissing) {
          output.appendLine(`torch has no build for Python ${base.v.join(".")} on this platform; trying another Python.`);
          continue;
        }
        break;
      }
      if (fs.existsSync(venv) && !(await importsOk(vpy))) removeManagedVenv(venv);
      vscode.window.showErrorMessage(`Python environment setup failed: ${lastError || "see the output panel"}`);
      return false;
    }
  );
}

function envReady(vpy) {
  output.appendLine(`\nPython environment ready: ${vpy}`);
  vscode.window.showInformationMessage("AI Compiler Workbench: Python environment ready.");
  return true;
}

// Install into an interpreter the user chose (aiCompilerWorkbench.pythonPath).
function installInto(py) {
  return vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: "AI Compiler Workbench: installing packages", cancellable: false },
    async (progress) => {
      output.show(true);
      output.appendLine(`\n--- Installing into ${py} ---`);
      const res = await installPackages(py, progress);
      if (res.ok && (await importsOk(py))) return envReady(py);
      const hint = /Permission|Access is denied|EnvironmentError: \[Errno 13\]/i.test(res.out || "")
        ? " -- no permission to install into this Python; use the workbench's own environment instead (clear aiCompilerWorkbench.pythonPath)."
        : "";
      vscode.window.showErrorMessage(`Installing into ${py} failed: ${pipErrorSummary(res.out || "")}${hint}`);
      return false;
    }
  );
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

function exportDefaultDir() {
  return outDir || (vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders[0].uri.fsPath) || ".";
}

async function exportFile(filename, content) {
  // The name comes from the webview; keep only the file name so it can't
  // point the default save location somewhere else.
  const name = path.basename(String(filename || "export.txt"));
  const uri = await vscode.window.showSaveDialog({
    defaultUri: vscode.Uri.file(path.join(exportDefaultDir(), name)),
  });
  if (!uri) return;
  fs.writeFileSync(uri.fsPath, content, "utf-8");
  vscode.window.showInformationMessage(`Exported ${name} to ${uri.fsPath}`);
}

// Several files for one pane (a CausalLM component's .h + .cpp): pick a
// folder once, confirm before overwriting, write each under its own name.
async function exportFiles(files) {
  const items = (Array.isArray(files) ? files : [])
    .filter((f) => f && f.filename)
    .map((f) => ({ name: path.basename(String(f.filename)), content: String(f.content || "") }));
  if (items.length === 0) return;

  const picked = await vscode.window.showOpenDialog({
    canSelectFiles: false,
    canSelectFolders: true,
    canSelectMany: false,
    defaultUri: vscode.Uri.file(exportDefaultDir()),
    openLabel: `Export ${items.map((i) => i.name).join(", ")} here`,
  });
  if (!picked || picked.length === 0) return;
  const dir = picked[0].fsPath;

  const existing = items.filter((i) => fs.existsSync(path.join(dir, i.name))).map((i) => i.name);
  if (existing.length) {
    const choice = await vscode.window.showWarningMessage(
      `${existing.join(", ")} already exist${existing.length === 1 ? "s" : ""} in ${dir}. Overwrite?`,
      { modal: true },
      "Overwrite"
    );
    if (choice !== "Overwrite") return;
  }

  for (const item of items) {
    fs.writeFileSync(path.join(dir, item.name), item.content, "utf-8");
  }
  vscode.window.showInformationMessage(`Exported ${items.map((i) => i.name).join(", ")} to ${dir}`);
}

// ---------------------------------------------------------------------
// Graph images. The webview draws them (it holds the graph data); the
// background, the format and where they go are picked here:
//   1. webview "graphImageRequest" {keys}  -> pick Dark / Light (+ format for both graphs)
//   2. extension "renderGraphImages"        -> the webview draws the images
//   3. webview "graphImages" {images}       -> save dialog (one) or folder (both)
// ---------------------------------------------------------------------
const GRAPH_KEYS = ["ini", "cpp"];

async function requestGraphImages(keys) {
  const wanted = (Array.isArray(keys) ? keys : []).filter((k) => GRAPH_KEYS.includes(k));
  if (!wanted.length || !panel) return;
  const both = wanted.length > 1;
  const theme = await vscode.window.showQuickPick(
    [
      { label: "Dark", description: "as on screen", value: "dark" },
      { label: "Light", description: "white background, for slides and documents", value: "light" },
    ],
    { placeHolder: `Background for the graph image${both ? "s" : ""}` });
  if (!theme) return;
  let format = "any";                     // one graph: PNG or SVG is picked in the save dialog
  if (both) {
    const pick = await vscode.window.showQuickPick(
      [
        { label: "PNG", description: "picture, opens anywhere", value: "png" },
        { label: "SVG", description: "vector, sharp at any zoom", value: "svg" },
        { label: "PNG + SVG", value: "both" },
      ],
      { placeHolder: "Image format" });
    if (!pick) return;
    format = pick.value;
  }
  post({ type: "renderGraphImages", keys: wanted, theme: theme.value, format });
}

function writeGraphImage(target, ext, img) {
  if (ext === "svg") fs.writeFileSync(target, img.svg, "utf-8");
  else fs.writeFileSync(target, Buffer.from(img.png, "base64"));
}

// A PNG of a very large graph is drawn below screen size: say so.
function scaledPngNote(imgs) {
  const small = imgs.filter((i) => typeof i.png === "string" && typeof i.scale === "number" && i.scale < 1);
  if (!small.length) return "";
  const pct = Math.round(Math.min(...small.map((i) => i.scale)) * 100);
  return ` The graph is large, so the PNG is drawn at ${pct}% of screen size; the SVG has full detail.`;
}

async function saveGraphImages(msg) {
  const images = (Array.isArray(msg.images) ? msg.images : [])
    .filter((i) => i && i.base && (typeof i.png === "string" || typeof i.svg === "string"))
    .map((i) => ({ ...i, base: path.basename(String(i.base)) }));
  if (!images.length) return;

  if (images.length === 1 && msg.format === "any") {
    const img = images[0];
    const filters = {};
    if (img.png) filters["PNG image"] = ["png"];
    if (img.svg) filters["SVG image"] = ["svg"];
    const uri = await vscode.window.showSaveDialog({
      defaultUri: vscode.Uri.file(path.join(exportDefaultDir(), `${img.base}.${img.png ? "png" : "svg"}`)),
      filters,
      saveLabel: "Save image",
    });
    if (!uri) return;
    let target = uri.fsPath;
    let ext = path.extname(target).slice(1).toLowerCase();
    if (ext !== "png" && ext !== "svg") {
      ext = img.png ? "png" : "svg";
      target += `.${ext}`;
    }
    if ((ext === "png" && !img.png) || (ext === "svg" && !img.svg)) {
      vscode.window.showErrorMessage(`No ${ext.toUpperCase()} could be made for this graph; save it as ${ext === "png" ? "SVG" : "PNG"}.`);
      return;
    }
    writeGraphImage(target, ext, img);
    const pick = await vscode.window.showInformationMessage(
      `Saved ${path.basename(target)} to ${path.dirname(target)}.${ext === "png" ? scaledPngNote([img]) : ""}`, "Open");
    if (pick === "Open") vscode.commands.executeCommand("vscode.open", vscode.Uri.file(target));
    return;
  }

  // Several images: pick a folder once, confirm before overwriting.
  const exts = msg.format === "png" ? ["png"] : msg.format === "svg" ? ["svg"] : ["png", "svg"];
  const items = [];
  for (const img of images) {
    for (const ext of exts) {
      if ((ext === "png" && img.png) || (ext === "svg" && img.svg)) items.push({ name: `${img.base}.${ext}`, ext, img });
    }
  }
  if (!items.length) return;
  const picked = await vscode.window.showOpenDialog({
    canSelectFiles: false,
    canSelectFolders: true,
    canSelectMany: false,
    defaultUri: vscode.Uri.file(exportDefaultDir()),
    openLabel: `Save ${items.map((i) => i.name).join(", ")} here`,
  });
  if (!picked || picked.length === 0) return;
  const dir = picked[0].fsPath;
  const existing = items.filter((i) => fs.existsSync(path.join(dir, i.name))).map((i) => i.name);
  if (existing.length) {
    const choice = await vscode.window.showWarningMessage(
      `${existing.join(", ")} already exist${existing.length === 1 ? "s" : ""} in ${dir}. Overwrite?`,
      { modal: true },
      "Overwrite"
    );
    if (choice !== "Overwrite") return;
  }
  for (const item of items) writeGraphImage(path.join(dir, item.name), item.ext, item.img);
  const missing = exts.includes("png") ? images.filter((i) => !i.png).map((i) => i.base) : [];
  vscode.window.showInformationMessage(
    `Saved ${items.map((i) => i.name).join(", ")} to ${dir}.` +
    (exts.includes("png") ? scaledPngNote(images) : "") +
    (missing.length ? ` No PNG could be made for ${missing.join(", ")}; use SVG.` : ""));
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
// The only native (non-webview) UI left is the one thing that
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
// Profile On-Device (the standalone command): asks for an nntrainer
// install prefix with dialogs if none is configured yet, then hands over
// to the same core as the unified pipeline above.
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
    const pythonPath = pythonFor();
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
// since that cannot be scripted from a background process.
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

  const pythonPath = pythonFor();
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

    const pythonPath = pythonFor();
    const engineDir = path.join(context.extensionPath, "engine");

    // The optional key goes through the environment, so the LLM fallback works
    // without it being stored in state.json or appearing in argv.
    const focusEnv = engineEnv(context);
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
