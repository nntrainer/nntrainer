# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file run_model.py
# @brief Run a verified generated model with its real weights on a device and profile it.
# @author prachi-t17 <prachi.t@samsung.com>

# This module holds one whole workbench step, kept together on purpose.
# pylint: disable=too-many-lines

"""
Run a verified generated model with its real weights on a device and profile it.

    weights   the model's Hugging Face safetensors, converted on the host by the
              *generated* weight_converter.py (verify_fixture.py --real-weights)
              -> generated/<slug>/run/nntr_<slug>_fp32.bin, reused while the
              converter and the weights are unchanged
    runner    run_<slug>.cpp, written here and modelled on upstream
              Applications/CausalLM/main.cpp: configs + chat template, the
              generated class constructed directly, initialize / load_weight /
              repack_weight / run, then one WORKBENCH_RUN line with
              getPerformanceMetrics() and getOutput()
    build     in the Docker build container (knowledge/docker_host.py) with the
              rules and libraries of its build (/work/src/build-wb in the volume)
    memory    before anything slow: the weights + KV cache estimated from the
              checkpoint's tensor shapes must fit Docker's memory (and --memory)
    load      target "pc": generated/<slug>/deploy/linux-x86_64 (the runner +
              every library ldd finds outside glibc) baked into a runtime image
              nntr-workbench-run:<slug>-<hash>; the Docker volume is mounted
              read-only and the run folder is read from /work/models/<slug>
    run       the run folder is staged in the Docker volume (/work/models/<slug>,
              the weights copied once per version: reading them through the
              Windows mount took ~1 min per run), then `docker run` R times
              (optionally --cpus / --memory to imitate a smaller device); the
              metrics go to review.json ("runs") and the Profiler dashboard
    layers    per_layer: one more run against the profiling build (nntrainer
              with -Denable-profile=true, build-wb-prof): every layer's time,
              the prompt's call apart from the generated tokens' calls, with a
              decoder-block map and the memory speed of the weight layers
              (record["layer_profile"])

The phone target (adb) is not here yet: it needs an Android build of nntrainer.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from . import verify
from .code_bank import workbench_dir
from .checklist import read_safetensors_header
from . import docker_host
from .preflight import memory_fix
from .codegen import CodegenError
from .codegen import load_review

Log = Callable[[str], None]
RUN_IMAGE_REPO = "nntr-workbench-run"
GLIBC = ("linux-vdso", "libc.so", "libm.so", "libpthread.so", "libdl.so", "librt.so", "ld-linux")
KEEP_RUNS = 20
MODEL_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "chat_template.jinja", "vocab.json", "merges.txt")


class RunError(RuntimeError):
    """A run of the generated model could not be done."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha(data: bytes, length: int = 12) -> str:
    return hashlib.sha256(data).hexdigest()[:length]


# ----------------------------------------------------------------- weights
def _has_weights(data: Path) -> bool:
    return data.is_dir() and (data / "config.json").is_file() and any(data.glob("*.safetensors"))


def short_weights_dir(wb_dir: Path, model: str) -> Path:
    """The pipeline's short weights folder, <workbench>/.w/<sha1 of the model
    id, 8 hex> -- the name agents/weight_download.py gives it (short because
    of the Windows path limit). The same name here, so weights the pipeline
    downloaded are found and a download from here lands in the same place."""
    return wb_dir / ".w" / hashlib.sha1(model.encode("utf-8")).hexdigest()[:8]


def find_model_dir(review: dict, hint: Optional[os.PathLike] = None) -> Optional[Path]:
    """The model's Hugging Face folder with real weights, if it is on disk:
    `hint` (the pipeline's weights folder), the folder the model was given as,
    the last run's weights folder for this model, or the workbench's weight
    caches (.ai_compiler_cache/<model>/weights, .w/<short id>)."""
    if hint and _has_weights(Path(hint)):
        return Path(hint)
    model = review.get("model", "")
    if model and os.path.isdir(model) and _has_weights(Path(model)):
        return Path(model)
    wb_dir = workbench_dir()
    if not wb_dir:
        return None
    state = wb_dir / "workbench_output" / "state.json"
    try:
        data = json.loads(state.read_text(encoding="utf-8"))
        if (
            data.get("model_name") == model
            and data.get("weights_path")
            and _has_weights(Path(data["weights_path"]))
        ):
            return Path(data["weights_path"])
    except (OSError, ValueError):
        pass
    for cand in (
        wb_dir / ".ai_compiler_cache" / model.replace("/", "__") / "weights",
        short_weights_dir(wb_dir, model),
        wb_dir / ".w" / _sha(model.encode("utf-8"), 8),
    ):  # the folder name of workbench 0.4.5-0.4.9
        if _has_weights(cand):
            return cand
    return None


def download_weights(review: dict, log: Log = print) -> Path:
    """Fetch the safetensors, config and tokenizer files into the workbench's cache."""
    model = review.get("model", "")
    if not model or os.path.isdir(model):
        raise RunError("the model is a local folder without *.safetensors weights")
    wb_dir = workbench_dir()
    if not wb_dir:
        raise RunError("cannot find tools/agentic-ai-workbench")
    dest = short_weights_dir(wb_dir, model)  # the pipeline's folder for this model
    # pylint: disable-next=import-outside-toplevel,import-error  # optional package
    from huggingface_hub import snapshot_download
    log(f"Downloading the weights of {model} into {dest} ...")
    snapshot_download(repo_id=model, local_dir=str(dest),
                      allow_patterns=["*.safetensors", "*.safetensors.index.json", *MODEL_FILES])
    if not _has_weights(dest):
        raise RunError(f"the download of {model} has no *.safetensors")
    return dest


def weights_size(model_dir: Path) -> int:
    """Total size of the model's safetensors files."""
    return sum(f.stat().st_size for f in model_dir.glob("*.safetensors"))


def _weights_key(review: dict, model_dir: Path) -> str:
    conv = Path(review["out_dir"]) / "weight_converter.py"
    parts = [conv.read_bytes() if conv.is_file() else b""]
    for path in sorted(model_dir.glob("*.safetensors")):
        status = path.stat()
        parts.append(f"{path.name}:{status.st_size}:{int(status.st_mtime)}".encode("utf-8"))
    return _sha(b"\0".join(parts), 16)


def convert_weights(review: dict, model_dir: Path, run_dir: Path, python: Optional[str] = None,
                    log: Log = print) -> Path:
    """Convert the real weights to nntrainer's format (the earlier result is reused when nothing
    changed)."""
    slug = review.get("slug") or Path(review["out_dir"]).name
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / f"nntr_{slug}_fp32.bin"
    stamp = run_dir / ".weights_key"
    key = _weights_key(review, model_dir)
    if out.is_file() and stamp.is_file() and stamp.read_text(encoding="utf-8").strip() == key:
        log(f"Reusing the converted weights ({out.stat().st_size / 1e9:.2f} GB).")
        return out
    log(f"Converting the real weights with the generated weight_converter.py "
        f"({weights_size(model_dir) / 1e9:.2f} GB of safetensors) ...")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONIOENCODING="utf-8")
    _code, text = verify.run_command(
        [
            python or sys.executable,
            "-E",
            str(verify.FIXTURE_SCRIPT),
            "--converter",
            str(Path(review["out_dir"]) / "weight_converter.py"),
            "--real-weights",
            str(model_dir),
            "--out",
            str(out),
        ],
        run_dir,
        env,
        3600,
    )
    summary = _last_json(text)
    if not summary or not summary.get("ok"):
        raise RunError(
            "weight conversion failed: " + ((summary or {}).get("error") or text[-2000:])
        )
    stamp.write_text(key, encoding="utf-8")
    log(f"Converted {summary.get('tensors')} tensors with {summary.get('converter_fn')} "
        f"-> {out.name} ({summary.get('bytes', 0) / 1e9:.2f} GB).")
    return out


# ------------------------------------------------------------------- memory
# Calibrated on Qwen2.5-0.5B (FP32 weights 1.98 GB, measured peak 2.60 GB):
# the estimate 1.15 x weights + KV cache + 0.4 GB = 2.69 GB stays just above it.
WEIGHT_FACTOR = 1.15
RUN_OVERHEAD = int(0.4 * 1e9)
TIGHT_MEMORY = 0.85               # above this share of the limit: warn, still run
_MEM_UNITS = {"": 1, "b": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3}


def _gib(size: int) -> str:
    return f"{size / 1024 ** 3:.1f} GB"


def checkpoint_bytes_fp32(model_dir: Path) -> int:
    """The checkpoint's parameters as FP32 (the workbench runs FP32), from the
    safetensors headers: nothing is loaded."""
    total = 0
    for path in sorted(Path(model_dir).glob("*.safetensors")):
        for shape in read_safetensors_header(path).values():
            if shape:
                num = 1
                for dim in shape:
                    num *= int(dim)
                total += num
    return total * 4


def kv_cache_bytes(config: dict, max_seq: int) -> int:
    """Memory the KV cache needs for `max_seq` tokens."""
    cfg = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    layers = int(cfg.get("num_hidden_layers") or 0)
    heads = int(cfg.get("num_attention_heads") or 0)
    kv_heads = int(cfg.get("num_key_value_heads") or heads)
    head_dim = int(cfg.get("head_dim") or ((cfg.get("hidden_size") or 0) // heads if heads else 0))
    return 2 * layers * kv_heads * head_dim * int(max_seq) * 4


def memory_needed(review: dict, model_dir: Path, tokens: int) -> int:
    """Estimated peak memory of a run in bytes (0 when it cannot be estimated)."""
    try:
        config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
        weights = checkpoint_bytes_fp32(model_dir)
    except (OSError, ValueError):
        return 0
    if not weights:
        return 0
    try:
        nntr = json.loads(
            (Path(review["out_dir"]) / "res" / "nntr_config.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, KeyError):
        nntr = {}
    init = int(nntr.get("init_seq_len") or 64)
    max_seq = max(
        int(nntr.get("max_seq_len") or 0), init + int(tokens)
    )  # as prepare_run_dir sets it
    return int(weights * WEIGHT_FACTOR + kv_cache_bytes(config, max_seq) + RUN_OVERHEAD)


def parse_docker_memory(text: Optional[str]) -> int:
    """`docker run --memory` values ('4g', '512m', '4096MB', bytes) in bytes; 0 if empty or
    unreadable."""
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([bkmg]?)b?\s*$", str(text or ""), re.I)
    return int(float(match.group(1)) * _MEM_UNITS[match.group(2).lower()]) if match else 0


def memory_limit(memory: Optional[str]) -> Tuple[int, str]:
    """(bytes, what sets it): Docker's memory, or the run's --memory when lower."""
    limit, where = docker_host.engine_resources()[1], "Docker"
    asked = parse_docker_memory(memory)
    if asked and (not limit or asked < limit):
        limit, where = asked, f"the run's memory limit ({memory}, aiCompilerWorkbench.runMemory)"
    return limit, where


def _memory_advice(memory: Optional[str]) -> str:
    if parse_docker_memory(memory):
        return "Raise aiCompilerWorkbench.runMemory or leave it empty."
    return memory_fix()


def check_memory(review: dict, model_dir: Path, tokens: int, memory: Optional[str],
                 log: Log = print) -> Tuple[int, int, str]:
    """Will the run fit? Raises RunError when the estimate exceeds the limit,
    logs a warning when it is above TIGHT_MEMORY of it. Returns (need, limit, where)."""
    need = memory_needed(review, model_dir, tokens)
    limit, where = memory_limit(memory)
    if need and limit:
        if need > limit:
            raise RunError(
                f"{review.get('model')} needs about {_gib(need)} of memory to run, but"
                f" {where} allows {_gib(limit)}. "
                + _memory_advice(memory)
            )
        if need > TIGHT_MEMORY * limit:
            log(
                f"Memory is tight: {review.get('model')} needs about {_gib(need)} and"
                f" {where} allows {_gib(limit)}; the run may be slow or stop."
            )
    return need, limit, where


def oom_error(need: int, limit: int, where: str, memory: Optional[str]) -> str:
    """The message for a run the out-of-memory killer stopped (exit 137)."""
    return (
        "the model ran out of memory"
        + (f": it needs about {_gib(need)}" if need else "")
        + (f" and {where} allows {_gib(limit)}" if limit else "")
        + ". "
        + _memory_advice(memory)
    )


def _last_json(text: str) -> Optional[dict]:
    for line in reversed(text.splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return None


# --------------------------------------------------------------- run folder
def prepare_run_dir(
    review: dict, model_dir: Path, run_dir: Path, bin_path: Path, tokens: int
) -> Path:
    """config / generation_config / tokenizer files from the model, nntr_config
    from the generated res/ with the weights, tokenizer and lengths filled in."""
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        src = model_dir / name
        if src.is_file():
            shutil.copyfile(src, run_dir / name)
    res = Path(review["out_dir"]) / "res" / "nntr_config.json"
    nntr = json.loads(res.read_text(encoding="utf-8")) if res.is_file() else {}
    nntr.update(
        {
            "model_file_name": bin_path.name,
            "tokenizer_file": "tokenizer.json",
            "batch_size": 1,
            "model_tensor_type": "FP32-FP32",
            "fc_layer_dtype": "FP32",
            "embedding_dtype": "FP32",
            "lmhead_dtype": "FP32",
            "fsu": False,
        }
    )
    init = int(nntr.get("init_seq_len") or 64)
    nntr["init_seq_len"] = init
    nntr["num_to_generate"] = int(tokens)
    nntr["max_seq_len"] = max(int(nntr.get("max_seq_len") or 0), init + int(tokens))
    (run_dir / "nntr_config.json").write_text(json.dumps(nntr, indent=2), encoding="utf-8")
    if not (run_dir / "generation_config.json").is_file():
        (run_dir / "generation_config.json").write_text("{}", encoding="utf-8")
    return run_dir / "nntr_config.json"


# ------------------------------------------------------------------ runner
RUNNER = """// Generated by the Agentic AI Workbench (knowledge/run_model.py) -- do not edit.
// Runs {cls} ({model}) with its real weights, modelled on upstream
// Applications/CausalLM/main.cpp, and prints one WORKBENCH_RUN line. Built
// against the workbench's profiling build (-DPROFILE=1) it also prints the
// time of every layer, split into the prompt and the generated tokens,
// between WORKBENCH_PROFILE_* lines.
#include "{header}"

#include <chat_template.h>
#ifdef PROFILE
#include <profiler.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {{

#ifdef PROFILE
// nntrainer times every layer (and every layer type) it runs. A run makes one
// inference call for the whole prompt, then one per generated token, and each
// call starts an "nn_forward" event: everything inside the first call is the
// prompt (phase 0), the rest is generation (phase 1). Memory events are not
// needed, so unlike GenericProfileListener nothing else is recorded.
class PhaseListener : public nntrainer::profile::ProfileListener {{
public:
  void notify(nntrainer::profile::PROFILE_EVENT event,
              const std::shared_ptr<nntrainer::profile::ProfileEventData> data) override {{
    if (event == nntrainer::profile::EVENT_TIME_START) {{
      if (data->event_str == "nn_forward")
        ++calls;
      return;
    }}
    if (event != nntrainer::profile::EVENT_TIME_END)
      return;
    auto it = index.find(data->event_str);
    if (it == index.end()) {{
      it = index.emplace(data->event_str, stats.size()).first;
      stats.emplace_back(data->event_str, Stat{{}});
    }}
    Stat &s = stats[it->second].second;
    const int p = calls <= 1 ? 0 : 1;
    const long long us = data->duration.count();
    s.n[p] += 1;
    s.sum[p] += us;
    s.mn[p] = s.n[p] == 1 ? us : std::min(s.mn[p], us);
    s.mx[p] = std::max(s.mx[p], us);
  }}
  void reset(const int, const std::string &) override {{}}
  const std::chrono::microseconds result(const int) override {{ return \
std::chrono::microseconds(0); }}
  void report(std::ostream &out) const override {{
    // calls <n>, then per key in first-run order:
    // L <key> <prompt calls> <prompt us> <gen calls> <gen us> <gen min us> <gen max us>
    out << "calls\\t" << calls << "\\n";
    for (const auto &[key, s] : stats)
      out << "L\\t" << key << '\\t' << s.n[0] << '\\t' << s.sum[0] << '\\t' << s.n[1] << '\\t' << \
s.sum[1]
          << '\\t' << s.mn[1] << '\\t' << s.mx[1] << "\\n";
  }}

private:
  struct Stat {{
    long long n[2] = {{0, 0}}, sum[2] = {{0, 0}}, mn[2] = {{0, 0}}, mx[2] = {{0, 0}};
  }};
  int calls = 0;
  std::unordered_map<std::string, size_t> index;
  std::vector<std::pair<std::string, Stat>> stats;
}};
#endif

std::string escape(const std::string &s) {{
  std::ostringstream os;
  for (unsigned char c : s) {{
    if (c == '"' || c == '\\\\')
      os << '\\\\' << c;
    else if (c == '\\n')
      os << "\\\\n";
    else if (c == '\\r')
      os << "\\\\r";
    else if (c == '\\t')
      os << "\\\\t";
    else if (c < 0x20)
      os << "\\\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else
      os << c;
  }}
  return os.str();
}}

double msSince(std::chrono::steady_clock::time_point t) {{
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}}

}} // namespace

int main(int argc, char *argv[]) {{
  if (argc < 2) {{
    std::cerr << "usage: " << argv[0] << " <model_dir> [--prompt TEXT] [--raw]\\n";
    return 2;
  }}
  const std::string dir = argv[1];
  std::string prompt;
  bool raw = false;
  for (int i = 2; i < argc; ++i) {{
    std::string a = argv[i];
    if (a == "--prompt" && i + 1 < argc)
      prompt = argv[++i];
    else if (a == "--raw")
      raw = true;
  }}
  try {{
    auto t0 = std::chrono::steady_clock::now();
    causallm::json cfg = causallm::LoadJsonFile(dir + "/config.json");
    causallm::json gen = causallm::json::object();
    if (std::filesystem::exists(dir + "/generation_config.json"))
      gen = causallm::LoadJsonFile(dir + "/generation_config.json");
    causallm::json nntr = causallm::LoadJsonFile(dir + "/nntr_config.json");
    nntr["tokenizer_file"] = dir + "/" + nntr["tokenizer_file"].get<std::string>();
    const std::string weights = dir + "/" + nntr["model_file_name"].get<std::string>();

    std::string input = prompt;
    if (input.empty())
      input = nntr.value("sample_input", std::string("Hello"));
    else if (!raw && causallm::ChatTemplate::Exists(dir)) {{
      causallm::json req = {{{{"messages", causallm::json::array({{{{{{"role", "user"}}, \
{{"content", input}}}}}})}}}};
      input = causallm::ChatTemplate::Load(dir).apply(req);
    }}

    auto model = std::make_unique<causallm::{cls}>(cfg, gen, nntr);
    model->initialize();
    double init_ms = msSince(t0);
    auto t1 = std::chrono::steady_clock::now();
    model->load_weight(weights);
    model->repack_weight();
    double load_ms = msSince(t1);

#ifdef PROFILE
    // Per-layer timing of this run only (loading is done).
    auto listener = std::make_shared<PhaseListener>();
    nntrainer::profile::Profiler::Global().subscribe(listener);
#endif
    std::cout << "WORKBENCH_TEXT_BEGIN" << std::endl;
    model->run(input, false, "", "", true);
    std::cout << std::endl << "WORKBENCH_TEXT_END" << std::endl;

    auto m = model->getPerformanceMetrics();
    std::cout << std::setprecision(9) << "WORKBENCH_RUN {{"
              << "\\"prefill_tokens\\":" << m.prefill_tokens
              << ",\\"prefill_ms\\":" << m.prefill_duration_ms
              << ",\\"generation_tokens\\":" << m.generation_tokens
              << ",\\"generation_ms\\":" << m.generation_duration_ms
              << ",\\"total_ms\\":" << m.total_duration_ms
              << ",\\"peak_memory_kb\\":" << m.peak_memory_kb
              << ",\\"init_ms\\":" << init_ms << ",\\"load_ms\\":" << load_ms
              << ",\\"prompt\\":\\"" << escape(input) << "\\""
              << ",\\"text\\":\\"" << escape(model->getOutput(0)) << "\\"}}" << std::endl;
#ifdef PROFILE
    nntrainer::profile::Profiler::Global().unsubscribe(listener);
    std::cout << "WORKBENCH_PROFILE_BEGIN" << std::endl;
    listener->report(std::cout);
    std::cout << "WORKBENCH_PROFILE_END" << std::endl;
#endif
  }} catch (const std::exception &e) {{
    std::cout << "WORKBENCH_RUN {{\\"error\\":\\"" << escape(e.what()) << "\\"}}" << std::endl;
    return 1;
  }}
  return 0;
}}
"""


def runner_source(review: dict) -> str:
    """The C++ runner that loads and runs the generated model."""
    cls = (review.get("classes") or {}).get("causallm")
    if not cls:
        raise RunError("review.json names no CausalLM class")
    header = verify.header_for(review, cls)
    if not header:
        raise RunError(f"no generated header declares class {cls}")
    return RUNNER.format(cls=cls, model=review.get("model", ""), header=header.name)


def parse_run(stdout: str) -> dict:
    """The WORKBENCH_RUN metrics (+ tokens/s) of one run, or {"error": ...}."""
    for line in reversed(stdout.splitlines()):
        if line.startswith("WORKBENCH_RUN "):
            try:
                data = json.loads(line[len("WORKBENCH_RUN "):])
            except ValueError:
                return {"error": "unreadable result line: " + line[:300]}
            for k in ("prefill", "generation"):
                millis, tokens = data.get(f"{k}_ms"), data.get(f"{k}_tokens")
                if millis and tokens:
                    data[f"{k}_tps"] = round(tokens * 1000.0 / millis, 2)
            return data
    tail = "\n".join(stdout.splitlines()[-15:])
    return {"error": "the runner printed no result" + (f":\n{tail}" if tail else "")}


_PROFILE_BLOCK = re.compile(r"WORKBENCH_PROFILE_BEGIN[^\n]*\n(.*?)\n?WORKBENCH_PROFILE_END", re.S)
FORWARD_KEY = "nn_forward"              # one whole inference call, not a layer
# nntrainer times each layer type ("fully_connected") and also each layer
# ("layer14_ffn_up:forward(fully_connected)"); the two overlap (a layer's
# time is inside its type's), so they are kept apart.
_LAYER_KEY = re.compile(r"^(?P<name>.+):(?P<op>[A-Za-z_]+)\((?P<type>[^()]+)\)$")
_BLOCK_NAME = re.compile(r"^layer(?P<block>\d+)_(?P<role>.+)$")
OUTLIER_FACTOR = 1.5  # a block's layer this much slower than the same layer's median
OUTLIER_MIN_SHARE = 1.0  # ... by at least this % of the phase's time per call: the prompt
# is one call per layer, so smaller gaps are mostly noise


def _share(part: float, whole: float) -> float:
    return round(100.0 * part / whole, 2) if whole else 0.0


# pylint: disable-next=too-many-locals
def parse_profile(stdout: str) -> Optional[dict]:
    """The runner's per-layer report (PhaseListener) from a profiling-build run.

    The first inference call of a run reads the whole prompt; every later call
    generates one token. Per layer type ("types", biggest first) and per layer
    ("layers", in the order they ran): the prompt call's time, the generated
    tokens' total, average, min and max per token, the total and the shares of
    each phase (microseconds). None when the run printed no report."""
    match = _PROFILE_BLOCK.search(stdout or "")
    if not match:
        return None
    calls, rows = 0, []
    for line in match.group(1).splitlines():
        parts = line.rstrip("\r").split("\t")
        if parts[0] == "calls" and len(parts) == 2 and parts[1].isdigit():
            calls = int(parts[1])
        elif parts[0] == "L" and len(parts) == 8:
            try:
                count0, us0, count1, us1, mn1, mx1 = (int(p) for p in parts[2:])
            except ValueError:
                continue
            rows.append(
                {
                    "key": parts[1],
                    "prompt_calls": count0,
                    "prompt_us": us0,
                    "generation_calls": count1,
                    "generation_us": us1,
                    "generation_min_us": mn1,
                    "generation_max_us": mx1,
                }
            )
    forward, types, layers = None, [], []
    for row in rows:
        row["total_us"] = row["prompt_us"] + row["generation_us"]
        row["generation_avg_us"] = (
            round(row["generation_us"] / row["generation_calls"], 1)
            if row["generation_calls"]
            else 0
        )
        if row["key"] == FORWARD_KEY:
            forward = row
            continue
        lm_head = _LAYER_KEY.match(row["key"])
        if not lm_head:
            types.append(row)
            continue
        builder = _BLOCK_NAME.match(lm_head.group("name"))
        layers.append(
            dict(
                row,
                name=lm_head.group("name"),
                op=lm_head.group("op"),
                type=lm_head.group("type"),
                block=int(builder.group("block")) if builder else None,
                role=builder.group("role") if builder else None,
            )
        )
    for group in (types, layers):
        tot = {k: sum(r[k] for r in group) for k in ("prompt_us", "generation_us", "total_us")}
        for row in group:
            row["pct"] = _share(row["total_us"], tot["total_us"])
            row["prompt_pct"] = _share(row["prompt_us"], tot["prompt_us"])
            row["generation_pct"] = _share(row["generation_us"], tot["generation_us"])
    types.sort(key=lambda r: -r["total_us"])
    totals = {k: sum(r[k] for r in types) for k in ("prompt_us", "generation_us", "total_us")}
    return {
        "calls": calls,
        "generation_calls": max(0, calls - 1),
        "forward": forward,
        "totals": totals,
        "types": types,
        "layers": layers,
    }


# pylint: disable-next=too-many-locals
def block_map(layers: List[dict]) -> dict:
    """Decoder blocks x the layers in each block ("wq", "ffn_up", ...), as
    indexes into `layers` (None where a block has no such layer), plus, per
    phase, the layers much slower than the same layer in the other blocks
    (1.5x its median, by at least 1% of the phase's time per call).
    nntrainer's own helper layers ("…/generated_out_0") are left out."""
    roles: List[str] = []
    cells: Dict[Tuple[int, str], int] = {}
    for i, layer in enumerate(layers):
        if layer.get("block") is None or "/" in (layer.get("role") or ""):
            continue
        if layer["role"] not in roles:
            roles.append(layer["role"])
        cells[(layer["block"], layer["role"])] = i
    blocks = sorted({b for b, _ in cells})
    grid = [[cells.get((b, role)) for role in roles] for b in blocks]
    outliers: Dict[str, List[int]] = {}
    for phase in PHASES:
        call_us = sum(per_call_us(r, phase) for r in layers)
        out: List[int] = []
        for role in roles:
            idx = [cells[(b, role)] for b in blocks if (b, role) in cells]
            vals = sorted(per_call_us(layers[i], phase) for i in idx)
            if len(vals) < 3:
                continue
            median = vals[len(vals) // 2]
            limit = max(OUTLIER_FACTOR * median, median + OUTLIER_MIN_SHARE / 100.0 * call_us)
            out += [i for i in idx if per_call_us(layers[i], phase) > limit]
        outliers[phase] = sorted(out)
    return {"blocks": blocks, "roles": roles, "grid": grid, "outliers": outliers}


PHASES = ("all", "prompt", "generation")


def per_call_us(res: dict, phase: str) -> float:
    """A layer's time per call: the prompt's one call, a generated token, or the run's average."""
    if phase == "prompt":
        return res["prompt_us"]
    if phase == "generation":
        return res["generation_avg_us"]
    return res["total_us"] / max(1, res["prompt_calls"] + res["generation_calls"])


# The checkpoint's tensors behind nntrainer's layer names, to turn a layer's
# time into the memory speed it reached (each generated token reads every
# weight once). The names follow upstream Applications/CausalLM/models.
_HF_BLOCK_ROLE = {"wq": "self_attn.q_proj", "wk": "self_attn.k_proj", "wv": "self_attn.v_proj",
                  "wo": "self_attn.o_proj", "attention_out": "self_attn.o_proj",
                  "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
_HF_TENSOR = re.compile(r"(?:^|\.)layers\.(?P<block>\d+)\.(?P<path>.+)\.(?:weight|bias)$")
BYTES_PER_PARAM = 4                     # the workbench runs FP32 (prepare_run_dir)
MIN_WEIGHT_BYTES = 1 << 20              # below this a layer's time is not about reading weights


def layer_weight_bytes(model_dir: Optional[Path], layers: List[dict]) -> Dict[str, int]:
    """Bytes of weights each layer reads per call, by layer name, from the
    safetensors headers; layers without a known checkpoint tensor are left out."""
    if not model_dir:
        return {}
    numel: Dict[Tuple[int, str], int] = {}
    top: Dict[str, int] = {}
    for path in sorted(Path(model_dir).glob("*.safetensors")):
        try:
            header = read_safetensors_header(path)
        except (OSError, ValueError):
            continue
        for name, shape in header.items():
            if not shape:
                continue
            num = 1
            for dim in shape:
                num *= int(dim)
            match = _HF_TENSOR.search(name)
            if match:
                key = (int(match.group("block")), match.group("path"))
                numel[key] = numel.get(key, 0) + num
            elif name.endswith(("lm_head.weight", "embed_tokens.weight")):
                top[name.rsplit(".", 2)[-2]] = num
    out: Dict[str, int] = {}
    for layer in layers:
        num = 0
        if layer.get("block") is not None and layer.get("role") in _HF_BLOCK_ROLE:
            num = numel.get((layer["block"], _HF_BLOCK_ROLE[layer["role"]]), 0)
        elif (
            layer.get("type") in ("tie_word_embeddings", "lm_head")
            and layer.get("name") == "output_of_causallm"
        ):
            num = (
                top.get("lm_head") or top.get("embed_tokens") or 0
            )  # tied: the embedding is the head
        if num * BYTES_PER_PARAM >= MIN_WEIGHT_BYTES:
            out[layer["name"]] = num * BYTES_PER_PARAM
    return out


def add_memory_speed(prof: dict, model_dir: Optional[Path]) -> None:
    """weight_mb and gbps (GB/s while generating) on the layers that read weights."""
    sizes = layer_weight_bytes(model_dir, prof["layers"])
    for layer in prof["layers"]:
        builder = sizes.get(layer["name"])
        if builder:
            layer["weight_mb"] = round(builder / 1e6, 1)
            layer["gbps"] = (
                round(builder / layer["generation_avg_us"] / 1e3, 1)
                if layer["generation_avg_us"]
                else None
            )


# ---------------------------------------------------------------- the bundle
def parse_ldd(text: str) -> List[str]:
    """Shared libraries (paths) a binary needs, from `ldd` output."""
    libs = []
    for line in text.splitlines():
        match = re.match(r"\s*(\S+)\s+=>\s+(/\S+)\s+\(", line)
        if match and not any(match.group(1).startswith(g) for g in GLIBC):
            libs.append(match.group(2))
    return libs


RUNTIME_DOCKERFILE = """# Generated by the Agentic AI Workbench: {model} for this PC (linux-x86_64).
# The runner and every library it needs except glibc; the configs, tokenizer and
# weights come from the workbench's Docker volume (/work/models/<model>), mounted
# read-only when it runs. The base is the build image's pinned Ubuntu.
FROM {base}
COPY bin/ /app/bin/
COPY lib/ /app/lib/
ENV LD_LIBRARY_PATH=/app/lib
ENTRYPOINT ["/app/bin/{exe}"]
"""


# pylint: disable-next=too-many-locals
def _build_runner(review: dict, ex, deploy: Path, log: Log, timeout: int) -> Tuple[Path, List[str]]:
    """Compile + link run_<slug> into deploy/bin with the Docker build's rules."""
    root = ex.root
    status = verify.build_status(root, ex=ex)
    if status["ok"] and (status["missing"] or ex.kind == "docker"):
        status = verify.ensure_libraries(root, log, ex=ex)
    if not status["ok"]:
        raise RunError(status.get("why") or status.get("status", "the build is not ready"))
    bdir = Path(status["build_dir"])
    ninja = verify.Ninja(ex.ninja_file())
    out_dir = Path(review["out_dir"])
    slug = review.get("slug") or out_dir.name
    work = work_dir_for(deploy)
    src = work / f"run_{slug}.cpp"
    work.mkdir(parents=True, exist_ok=True)
    src.write_text(runner_source(review), encoding="utf-8", newline="\n")
    gen = [
        Path(p)
        for n, p in sorted(review["files"].items())
        if n.endswith(".cpp")
        and not os.path.basename(n).startswith("register_")
        and Path(p).is_file()
    ]
    jobs = [(s, work / "obj" / (s.stem + ".o")) for s in [src, *gen]]
    headers = [
        Path(p) for n, p in review["files"].items() if n.endswith(".h") and Path(p).is_file()
    ]
    passed, detail = ex.push([src, *gen, *headers])          # Docker: into its copy of the checkout
    if not passed:
        raise RunError(
            "copying the generated code into the build container failed: " + detail[-500:]
        )
    ex.mkdirs(sorted({o.parent for _s, o in jobs}))
    log(f"Compiling the runner and {len(gen)} generated source(s) in {ex.describe()} ...")
    errors, logs = [], []
    for src_file, obj_path in jobs:
        passed, errs, text = verify.compile_source(
            ninja, bdir, src_file, obj_path, [out_dir, work], ex, timeout
        )
        logs.append(text)
        errors += errs
    if errors:
        raise RunError(
            "the runner did not build:\n"
            + "\n".join(f"{e['file']}:{e['line']}: {e['message']}" for e in errors[:15])
        )
    exe = work / f"run_{slug}"
    passed, errs, text = verify.link_objects(ninja, bdir, [o for _s, o in jobs], exe, ex, timeout,
                                  verify.link_extras(ex, gtest=False))
    if not passed:
        raise RunError("the runner did not link:\n" + "\n".join(e["message"] for e in errs[:15]))
    return exe, ex.lib_paths()


PROFILE_SUFFIX = "-prof"  # the per-layer (profiling build) runner's folders and image


def work_dir_for(deploy: Path) -> Path:
    """The runner's build folder (run/build, or run/build-prof for the profiling
    build); it exists in the build container's copy too (commands run from it)."""
    return (
        deploy.parent.parent
        / "run"
        / ("build" + (PROFILE_SUFFIX if deploy.name.endswith(PROFILE_SUFFIX) else ""))
    )


def deploy_dir(review: dict, ex) -> Path:
    """The folder the runner bundle is built in."""
    prof = getattr(ex, "variant", "") == "prof"
    return Path(review["out_dir"]) / "deploy" / ("linux-x86_64" + (PROFILE_SUFFIX if prof else ""))


# pylint: disable-next=too-many-locals
def bundle(review: dict, ex, log: Log = print, timeout: int = 1800) -> Tuple[Path, str]:
    """Build the runner and collect it with its libraries into
    generated/<slug>/deploy/linux-x86_64 (…-prof for the profiling build);
    returns (deploy dir, content hash)."""
    deploy = deploy_dir(review, ex)
    key = _bundle_key(review, ex)
    stamp = deploy / ".bundle_key"
    if stamp.is_file() and stamp.read_text(encoding="utf-8").strip() == key:
        log("Reusing the built runner (nothing changed since the last build).")
        return deploy, _dir_digest(deploy)
    if deploy.exists():
        shutil.rmtree(deploy)
    exe, dirs = _build_runner(review, ex, deploy, log, timeout)
    code, text = ex.run(
        ["ldd", ex.path(exe)], work_dir_for(deploy), 120, env_extra=ex.lib_env(dirs)
    )
    if code != 0:
        raise RunError("ldd failed: " + text[-1000:])
    missing = [line.strip() for line in text.splitlines() if "not found" in line]
    if missing:
        raise RunError(
            "the runner needs libraries the build does not have: " + "; ".join(missing[:5])
        )
    libdir, bindir = deploy / "lib", deploy / "bin"
    libdir.mkdir(parents=True, exist_ok=True)
    bindir.mkdir(parents=True, exist_ok=True)
    libs = parse_ldd(text)
    # The runner and its libraries live in the container (its copy of the
    # checkout, system packages): copy them into the workbench folder.
    lib_live, bin_live = shlex.quote(ex.live(libdir)), shlex.quote(ex.live(bindir))
    copy_cmds = [f"cp -L {shlex.quote(ex.path(exe))} {bin_live}/"] + [
        f"cp -L {shlex.quote(p)} {lib_live}/" for p in libs
    ]
    code, text = ex.run(" && ".join(copy_cmds), work_dir_for(deploy), 300)
    if code != 0:
        raise RunError("collecting the runner and its libraries failed: " + text[-1000:])
    (deploy / "Dockerfile").write_text(
        RUNTIME_DOCKERFILE.format(
            model=review.get("model", ""), exe=exe.name, base=docker_host.base_image()
        ),
        encoding="utf-8",
        newline="\n",
    )
    log(
        f"Bundled the runner with {len(libs)} libraries"
        f" ({sum(f.stat().st_size for f in libdir.iterdir()) / 1e6:.0f} MB)."
    )
    digest = _dir_digest(deploy)
    stamp.write_text(key, encoding="utf-8")
    return deploy, digest


def _dir_digest(data: Path) -> str:
    hasher = hashlib.sha256()
    for handle in sorted(p for p in data.rglob("*") if p.is_file() and p.name != ".bundle_key"):
        hasher.update(
            handle.relative_to(data).as_posix().encode("utf-8") + b"\0" + handle.read_bytes()
        )
    return hasher.hexdigest()[:12]


def _bundle_key(review: dict, ex) -> str:
    """What the runner build depends on: its source, the generated code and the build's
    libraries."""
    parts = [runner_source(review).encode("utf-8"), ex.describe().encode("utf-8")]
    for name, path in sorted(review.get("files", {}).items()):
        if name.endswith((".h", ".cpp")) and Path(path).is_file():
            parts.append(name.encode("utf-8") + b"\0" + Path(path).read_bytes())
    status = verify.build_status(ex.root, ex=ex)
    nfile = ex.ninja_file()
    if nfile.is_file():
        parts.append(nfile.read_bytes())
    if ex.kind == "docker":  # the libraries follow the container's copy of the checkout
        mfile = ex.host.wb_dir / "docker" / "src_manifest.json"
        try:
            parts.append(
                json.loads(mfile.read_text(encoding="utf-8")).get("id", "").encode("utf-8")
            )
        except (OSError, ValueError):
            pass
    else:
        for lib in status.get("libs", []):
            handle = Path(status["build_dir"]) / lib
            if handle.exists():
                parts.append(
                    f"{lib}:{handle.stat().st_size}:{int(handle.stat().st_mtime)}".encode("utf-8")
                )
    return _sha(b"\0".join(parts), 16)


def load_image(review: dict, deploy: Path, digest: str, log: Log = print) -> str:
    """Load the bundle on this PC as a runtime image (…-prof-<hash> for the profiling build)."""
    slug = re.sub(r"[^a-z0-9_.-]", "-", (review.get("slug") or "model").lower())
    if deploy.name.endswith(PROFILE_SUFFIX):
        slug += PROFILE_SUFFIX
    tag = f"{RUN_IMAGE_REPO}:{slug}-{digest}"
    if docker_host.image_exists(tag):
        log(f"Runtime image {tag} is already loaded.")
        return tag
    log(f"Loading the model's runtime image {tag} on this PC ...")
    code, out = docker_host.stream(["build", "-t", tag, str(deploy)], 1800, log)
    if code != 0:
        raise RunError("the runtime image did not build:\n" + out[-2000:])
    return tag


def stage_model(ex, run_dir: Path, slug: str, log: Log = print) -> str:
    """Copy the run folder into the Docker volume (/work/models/<slug>): the
    configs every run, the weights only when they changed."""
    dest = f"/work/models/{slug}"
    src = shlex.quote(ex.live(run_dir))
    script = (
        f"set -e; mkdir -p {dest}; for f in {src}/*; do case \"$f\" in *.bin) ;; *) cp -f \"$f\""
        f" {dest}/ ;; esac; done; if ! cmp -s {src}/.weights_key {dest}/.weights_key 2>/dev/null;"
        f" then echo COPYING_WEIGHTS; rm -f {dest}/*.bin; cp -f {src}/*.bin {dest}/; cp -f"
        f" {src}/.weights_key {dest}/; fi"
    )
    lines: List[str] = []

    def note(line: str) -> None:
        lines.append(line)
        if line.strip() == "COPYING_WEIGHTS":
            log(
                "Copying the converted weights into Docker's storage (once per weights version) ..."
            )

    code, out = ex.host.exec(script, timeout=3600, log=note)
    if code != 0:
        raise RunError("copying the model into Docker failed: " + out[-1000:])
    return dest


# --------------------------------------------------------------------- run
def profile_executor(root: Path, log: Log = print):
    """The executor of the profiling build (nntrainer's profiler compiled in),
    set up in the Docker volume on first use."""
    ex = verify.DockerExecutor(root, "prof")
    if not ex.ninja_file().is_file():
        log(
            "Per-layer timing needs a second nntrainer build with its profiler on; setting it up in"
            " Docker (first time only: this takes about as long as the first normal build) ..."
        )
        status = docker_host.setup(root, log, variant="prof")
        if not status.get("ok"):
            raise RunError(
                "the profiling build did not set up: "
                + str(status.get("log") or status.get("missing") or status.get("status"))[-1500:]
            )
    return ex


# pylint: disable-next=too-many-arguments
def run_args(
    ex, tag: str, model_c: str, prompt: str, cpus: Optional[str], memory: Optional[str]
) -> List[str]:
    """`docker run` for one run. The model files come from the Docker volume (local
    storage, like on a device): reading 2 GB of weights through the Windows mount takes ~1 min."""
    args = ["run", "--rm", "--mount", f"type=volume,source={ex.host.volume},target=/work,readonly"]
    if cpus:
        args += ["--cpus", str(cpus)]
    if memory:
        args += ["--memory", str(memory)]
    return args + [tag, model_c] + (["--prompt", prompt] if prompt else [])


def _without_report(log: Log) -> Log:
    """`log` for a profiled run's output, without the report's (hundreds of) lines."""
    inside = [False]

    def out(line: str) -> None:
        if line.startswith("WORKBENCH_PROFILE_BEGIN"):
            inside[0] = True
        elif line.startswith("WORKBENCH_PROFILE_END"):
            inside[0] = False
        elif not inside[0]:
            log(line)
    return out


PROFILE_COLD_FACTOR = (
    3  # a profiled prompt this much slower than the measured runs read the weights from disk
)


# pylint: disable-next=too-many-locals,too-many-arguments
def profile_layers(
    review: dict,
    root: Path,
    run_dir: Path,
    slug: str,
    prompt: str,
    cpus: Optional[str],
    memory: Optional[str],
    model_dir: Optional[Path],
    prefill_ms: Optional[float],
    log: Log = print,
    timeout: int = 1800,
) -> dict:
    """One run against the profiling build, right after the measured runs: the
    time of every layer split into the prompt and the generated tokens
    (parse_profile), the block map and the memory speed of the layers that read
    weights. {"error": ...} when the run gives no report."""
    ex = profile_executor(root, log)
    deploy, digest = bundle(review, ex, log, timeout)
    tag = load_image(review, deploy, digest, log)
    args = run_args(ex, tag, stage_model(ex, run_dir, slug, log), prompt, cpus, memory)
    for attempt in (1, 2):
        log("Per-layer timing: one more run, with nntrainer's profiler on ...")
        _code, out = docker_host.stream(args, timeout, _without_report(log))
        metrics = parse_run(out)
        if metrics.get("error"):
            return {"error": "the profiled run failed: " + str(metrics["error"])[:500]}
        prof = parse_profile(out)
        if not prof or not prof["layers"]:
            return {"error": "the runner printed no per-layer report"}
        cold = (
            bool(prefill_ms)
            and (metrics.get("prefill_ms") or 0) > PROFILE_COLD_FACTOR * prefill_ms + 100
        )
        if not (cold and attempt == 1):
            break
        log(
            "Its prompt took much longer than in the measured runs (the weights were read from"
            " disk); timing it once more ..."
        )
    add_memory_speed(prof, model_dir)
    prof["map"] = block_map(prof["layers"])
    prof["run"] = {
        k: metrics.get(k)
        for k in (
            "prefill_tokens",
            "prefill_ms",
            "prefill_tps",
            "generation_tokens",
            "generation_ms",
            "generation_tps",
        )
    }
    prof["image"] = tag
    prof["note"] = (
        "Measured in the profiling build, which runs slower than the normal build (the speeds "
        "above come from the normal build): compare layers by their shares, not their ms."
    )
    totals = prof["totals"]
    log(
        f"Per-layer timing: the prompt took {totals['prompt_us'] / 1000:.0f} ms in layers (1 call),"
        f" the {prof['generation_calls']} generated tokens {totals['generation_us'] / 1000:.0f} ms."
    )
    log("  Per layer type: " + ", ".join(f"{r['key']} {r['pct']:.0f}%" for r in prof["types"][:4]))
    top = sorted(prof["layers"], key=lambda r: -r["total_us"])[:3]
    log("  Slowest layers: " + ", ".join(f"{r['name']} ({r['type']}) {r['pct']:.1f}%" for r in top))
    return prof


# pylint: disable-next=too-many-locals,too-many-arguments,too-many-branches,too-many-statements
def run_generated(review_path: os.PathLike, prompt: str = "", tokens: int = 64, repeat: int = 3,
                  cpus: Optional[str] = None, memory: Optional[str] = None, target: str = "pc",
                  nntrainer_root: Optional[os.PathLike] = None, python: Optional[str] = None,
                  log: Log = print, timeout: int = 1800, per_layer: bool = False,
                  model_dir: Optional[os.PathLike] = None, warmup: bool = True) -> dict:
    """Build the verified model in Docker, load it on `target`, run it `repeat`
    times (after one warm-up run that is not counted) and record the metrics
    in review.json. per_layer: then run it once more against the profiling
    build for the time of every layer (record["layer_profile"]); the speeds
    still come from the normal build. model_dir: where the weights are, if
    known (the pipeline's download)."""

    review_path = Path(review_path)
    review = load_review(review_path)
    verify_res = review.get("verify") or {}
    if verify_res.get("status") != "passed" or verify_res.get(
        "files_hash"
    ) != verify.current_verify_hash(review):
        raise CodegenError(
            "Not run: verify the model against Hugging Face first (Review tab -> Verify)"
        )
    if target != "pc":
        raise RunError(
            "only target 'pc' (this computer, through Docker) exists so far; phones come next"
        )
    passed, detail = docker_host.daemon_status()
    if not passed:
        raise RunError(detail)
    root = verify.resolve_root(nntrainer_root)
    out_dir = Path(review["out_dir"])
    slug = review.get("slug") or out_dir.name
    run_dir = out_dir / "run" / "model"

    model_dir = find_model_dir(review, model_dir)
    if model_dir is None:
        raise RunError(
            f"NO_WEIGHTS: the weights of {review.get('model')} are not on this computer yet"
        )
    need, limit, where = check_memory(
        review, model_dir, tokens, memory, log
    )  # before anything slow
    ex = verify.DockerExecutor(root)
    bin_path = convert_weights(review, model_dir, run_dir, python, log)
    prepare_run_dir(review, model_dir, run_dir, bin_path, tokens)
    deploy, digest = bundle(review, ex, log, timeout)
    tag = load_image(review, deploy, digest, log)
    model_c = stage_model(ex, run_dir, slug, log)
    args = run_args(ex, tag, model_c, prompt, cpus, memory)
    runs = []
    if warmup:
        # The first run after a (re)load reads the weights from disk and warms the caches: on
        # Qwen2.5-0.5B it took 21 s of prefill against 0.2 s for the next. It is run, checked,
        # and not counted.
        log("Warm-up run (not counted) ...")
        code, out = docker_host.stream(args, timeout, log)
        weight = parse_run(out)
        if weight.get("error"):
            weight["exit_code"] = code
            if verify.out_of_memory(out, code):
                weight["error"] = oom_error(need, limit, where, memory)
            runs.append(weight)
    for i in range(0 if runs else max(1, int(repeat))):
        log(f"Run {i + 1}/{repeat} on this PC ({'--cpus ' + str(cpus) if cpus else 'all CPUs'}"
            f"{', --memory ' + str(memory) if memory else ''}) ...")
        code, out = docker_host.stream(args, timeout, log)
        metrics = parse_run(out)
        metrics["exit_code"] = code
        if metrics.get("error") and verify.out_of_memory(
            out, code
        ):  # 137: killed by the OOM killer
            metrics["error"] = oom_error(need, limit, where, memory)
        runs.append(metrics)
        if metrics.get("error"):
            break
    ok_runs = [r for r in runs if not r.get("error")]

    def avg(key):
        vals = [r[key] for r in ok_runs if r.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    record = {
        "at": _now(),
        "target": "pc",
        "device": f"this PC ({platform.processor() or platform.machine()}, "
        f"{os.cpu_count()} logical CPUs) in Docker",
        "image": tag,
        "cpus": cpus,
        "memory": memory,
        "tokens": tokens,
        "repeat": repeat,
        "prompt": prompt or "(sample_input)",
        "weights": str(bin_path),
        "host_label": ex.describe(),
        "ok": bool(ok_runs) and len(ok_runs) == len(runs),
        "text": (ok_runs[-1].get("text") if ok_runs else ""),
        "runs": runs,
        "summary": {
            k: avg(k)
            for k in (
                "prefill_tps",
                "generation_tps",
                "prefill_ms",
                "generation_ms",
                "total_ms",
                "load_ms",
            )
        },
        "peak_memory_kb": max((r.get("peak_memory_kb") or 0 for r in ok_runs), default=None),
        "error": next((r["error"] for r in runs if r.get("error")), ""),
        "per_layer": bool(per_layer),
        "warmup": bool(warmup),
    }
    if per_layer and record["ok"]:
        try:
            record["layer_profile"] = profile_layers(
                review,
                root,
                run_dir,
                slug,
                prompt,
                cpus,
                memory,
                model_dir,
                record["summary"].get("prefill_ms"),
                log,
                timeout,
            )
        except (RunError, OSError, ValueError) as exc:
            record["layer_profile"] = {"error": str(exc)}
        if record["layer_profile"].get("error"):
            log(f"Per-layer timing did not run: {record['layer_profile']['error']}")
    (out_dir / "run" / "report.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    review = load_review(review_path)
    review["run"] = record
    review["runs"] = (
        [{k: v for k, v in record.items() if k not in ("runs", "text", "layer_profile")}]
        + (review.get("runs") or [])
    )[:KEEP_RUNS]
    review_path.write_text(json.dumps(review, indent=2, default=str), encoding="utf-8")
    return review


def profile_event(record: dict, model: str) -> dict:
    """The Profiler dashboard's view of a run."""
    return {
        "kind": "llm_run",
        "ran": record.get("ok", False),
        "for": model,
        "device": record.get("device"),
        "image": record.get("image"),
        "cpus": record.get("cpus"),
        "memory": record.get("memory"),
        "summary": record.get("summary"),
        "peak_memory_kb": record.get("peak_memory_kb"),
        "runs": [
            {
                k: r.get(k)
                for k in (
                    "prefill_tokens",
                    "prefill_ms",
                    "prefill_tps",
                    "generation_tokens",
                    "generation_ms",
                    "generation_tps",
                    "total_ms",
                    "peak_memory_kb",
                    "load_ms",
                    "error",
                )
            }
            for r in record.get("runs", [])
        ],
        "text": record.get("text", ""),
        "reason": record.get("error", ""),
        "per_layer": record.get("per_layer", False),
        "layer_profile": record.get("layer_profile"),
    }
