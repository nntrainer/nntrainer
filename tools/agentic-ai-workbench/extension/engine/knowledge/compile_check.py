# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file compile_check.py
# @brief Syntax-only compile check of generated CausalLM code against nntrainer headers.
# @author prachi-t17 <prachi.t@samsung.com>

"""
Compile check for generated CausalLM code: each generated .cpp is compiled in
syntax-only mode against the nntrainer checkout's real headers. That catches
what LLM-written C++ usually gets wrong -- unknown members, wrong signatures,
type errors, missing includes -- without building nntrainer or linking.

Flags, best first:
  1. compile_commands.json from an nntrainer build (build*/ in the checkout):
     the exact flags meson used for a hand-written CausalLM model file.
  2. Otherwise every header folder of the checkout (Applications/CausalLM,
     its layers/models/third_party, api/, nntrainer/**) with -std=c++17.

Compiler: $CXX / aiCompilerWorkbench.cxxCompiler, else clang++, g++, c++ on
PATH (and LLVM's default folder on Windows), else MSVC: `cl` on PATH (a
Developer prompt) or any Visual Studio / Build Tools install that vswhere
finds. MSVC outside a Developer prompt runs with the environment that
vcvars64.bat sets (INCLUDE, LIB, PATH), captured once per process. This is a
compile check, not a build; knowledge/verify.py builds and runs the model.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DIAG_RE = re.compile(
    r"^(?P<file>.+?)[:(](?P<line>\d+)[:,]?(?P<col>\d+)?\)?:?\s*(?P<kind>fatal error|error|warning)"
    r"(?: [A-Z]+\d+)?:\s*(?P<msg>.*)$"
)


_WIN_GNU_PATHS = (r"C:\Program Files\LLVM\bin\clang++.exe", r"C:\msys64\ucrt64\bin\g++.exe",
                  r"C:\msys64\mingw64\bin\g++.exe")


def find_compiler(explicit: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """(path, kind) with kind "gnu" (g++/clang++) or "msvc", or None."""
    for cand in (explicit, os.environ.get("CXX")):
        if cand:
            exe_path = shutil.which(cand) or (cand if os.path.isfile(cand) else None)
            if exe_path:
                return exe_path, (
                    "msvc" if os.path.basename(exe_path).lower() in ("cl", "cl.exe") else "gnu"
                )
    for name in ("clang++", "g++", "c++"):
        exe_path = shutil.which(name)
        if exe_path:
            return exe_path, "gnu"
    if os.name == "nt":
        for exe_path in _WIN_GNU_PATHS:
            if os.path.isfile(exe_path):
                return exe_path, "gnu"
        exe_path = shutil.which("cl")
        if exe_path:
            return exe_path, "msvc"
        for inst in vs_installations():
            bat = _vcvars_for(inst)
            env = msvc_env(bat) if bat else None
            cl_path = shutil.which("cl", path=env_get(env, "PATH")) if env else None
            if cl_path:
                return cl_path, "msvc"
    return None


# ------------------------------------------------------------------ MSVC
_MSVC_ENV: Dict[str, Optional[Dict[str, str]]] = {}


def env_get(env: Optional[Dict[str, str]], key: str) -> Optional[str]:
    """Case-insensitive lookup: Windows `set` prints Path, PATH or path."""
    if not env:
        return None
    key = key.upper()
    return next((v for k, v in env.items() if k.upper() == key), None)


def _vswhere() -> Optional[str]:
    base = os.environ.get("ProgramFiles(x86)") or r"C:\Program Files (x86)"
    path = os.path.join(base, "Microsoft Visual Studio", "Installer", "vswhere.exe")
    return path if os.path.isfile(path) else None


def vs_installations() -> List[Path]:
    """Visual Studio / Build Tools folders that have the x64 C++ tools (newest first)."""
    vswhere = _vswhere() if os.name == "nt" else None
    if not vswhere:
        return []
    try:
        proc = subprocess.run(
            [
                vswhere,
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
                "-utf8",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [Path(line.strip()) for line in (proc.stdout or "").splitlines() if line.strip()]


def _vcvars_for(path: Path) -> Optional[Path]:
    """vcvars64.bat (vcvarsarm64.bat on an ARM64 host) of the install that holds `path`."""
    name = (
        "vcvarsarm64.bat" if platform.machine().upper() in ("ARM64", "AARCH64") else "vcvars64.bat"
    )
    path = Path(path)
    for data in (path, *path.parents):
        bat = data / "VC" / "Auxiliary" / "Build" / name
        if bat.is_file():
            return bat
    return None


def msvc_env(vcvars: Path) -> Optional[Dict[str, str]]:
    """The environment vcvars sets up, captured once per process (it takes seconds)."""
    key = str(vcvars)
    if key not in _MSVC_ENV:
        _MSVC_ENV[key] = _capture_env(Path(vcvars))
    return _MSVC_ENV[key]


def _capture_env(vcvars: Path) -> Optional[Dict[str, str]]:
    env = dict(os.environ)
    vswhere = _vswhere()
    if vswhere:      # vcvars calls vswhere itself and complains when it is not on PATH
        env["PATH"] = os.path.dirname(vswhere) + os.pathsep + env.get("PATH", "")
    try:
        # /s: cmd strips only the outer quotes, so the quoted path (spaces,
        # parentheses in "Program Files (x86)") survives.
        proc = subprocess.run(
            f'cmd /d /s /c "call "{vcvars}" >nul && set"',
            capture_output=True,
            text=True,
            encoding="oem",
            errors="replace",
            timeout=180,
            env=env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired, LookupError):
        return None
    out: Dict[str, str] = {}
    for line in (proc.stdout or "").splitlines():
        k, sep, value = line.partition("=")
        if sep and k and not k.startswith((" ", "=")):
            out[k] = value
    return out if proc.returncode == 0 and env_get(out, "INCLUDE") else None


def env_for(cxx: str, kind: str) -> Optional[Dict[str, str]]:
    """Environment to run `cxx` in, or None for the current one. MSVC needs
    INCLUDE/LIB from vcvars unless this process already runs in a Developer prompt."""
    if kind != "msvc" or os.environ.get("INCLUDE"):
        return None
    bat = _vcvars_for(Path(cxx))
    if bat is None:
        bat = next((b for b in (_vcvars_for(i) for i in vs_installations()) if b), None)
    return msvc_env(bat) if bat else None


# Short on purpose: meson's longest object path is ~180 characters, and with
# Windows long paths disabled the whole path must stay under 260.
BUILD_DIR_NAME = "nb"


def build_dir(root: Path) -> Path:
    """The workbench's own nntrainer build: tools/agentic-ai-workbench/nb (or
    $WORKBENCH_NNTR_BUILD). Kept under the workbench folder: the build never
    writes outside it (it is configured with enable-test=false, see setup_build)."""
    env = os.environ.get("WORKBENCH_NNTR_BUILD")
    return (
        Path(env).expanduser()
        if env
        else Path(root) / "tools" / "agentic-ai-workbench" / BUILD_DIR_NAME
    )


def compile_commands_files(root: Path) -> List[Path]:
    """compile_commands.json of the workbench build first, then any build*/ in the checkout."""
    root = Path(root)
    cands = [
        build_dir(root) / "compile_commands.json",
        *sorted(root.glob("build*/compile_commands.json")),
    ]
    seen, out = set(), []
    for cand in cands:
        if cand.is_file() and str(cand.resolve()) not in seen:
            seen.add(str(cand.resolve()))
            out.append(cand)
    return out


def entry_args(entry: dict) -> List[str]:
    """argv of a compile_commands entry. Meson writes a quoted `command` string
    for MSVC; shlex in non-POSIX mode keeps the quotes, so they are stripped here."""
    if entry.get("arguments"):
        return list(entry["arguments"])
    cmd = entry.get("command", "")
    if os.name != "nt":
        return shlex.split(cmd)
    out = []
    for tok in shlex.split(cmd, posix=False):
        if len(tok) >= 2 and tok[0] == tok[-1] == '"':
            tok = tok[1:-1]
        out.append(tok.replace('\\"', '"'))
    return out


def _entry_rank(entry: dict) -> Tuple[int, int]:
    """Prefer a model source of the causallm library itself (its include path
    covers Applications/CausalLM and every model folder) over a layer DLL that
    happens to live under models/ (e.g. gpt_oss/gptoss_moe_layer)."""
    where = (
        entry.get("output", "")
        + " "
        + entry.get("command", "")
        + " "
        + " ".join(entry.get("arguments") or [])
    ).lower()
    return (
        0 if "libcausallm" in where else 1,
        0 if entry.get("file", "").endswith("_causallm.cpp") else 1,
    )


def find_entry(root: Path, needle: str) -> Optional[Tuple[dict, Path]]:
    """(entry, compile_commands.json) of the best .cpp whose path contains `needle`."""
    for compile_commands_file in compile_commands_files(root):
        try:
            entries = json.loads(compile_commands_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        found = [e for e in entries if needle in e.get("file", "").replace("\\", "/")
                 and e.get("file", "").endswith(".cpp")]
        if found:
            return min(found, key=_entry_rank), compile_commands_file
    return None


_MSVC_KEEP = (
    "/std:",
    "-std:",
    "/EH",
    "-EH",
    "/Zc:",
    "-Zc:",
    "/utf-8",
    "-utf-8",
    "/permissive",
    "-permissive",
    "/arch:",
    "-arch:",
    "/bigobj",
    "-bigobj",
    "/FI",
    "-FI",
)


def _from_compile_commands(root: Path, kind: str = "gnu") -> Optional[List[str]]:
    """Include/define/std flags meson used for a CausalLM model file."""
    found = find_entry(Path(root), "Applications/CausalLM/models/")
    if not found:
        return None
    entry, _cc = found
    args = entry_args(entry)
    keep, items_iter, directory = [], iter(args[1:]), Path(entry.get("directory", root))

    def absol(val: str) -> str:
        return val if os.path.isabs(val) else str((directory / val).resolve())

    for attn in items_iter:
        if attn in ("-I", "-isystem", "-D", "-include", "-iquote", "/I", "/D"):
            val = next(items_iter, "")
            if attn in ("-I", "-isystem", "-iquote", "/I"):
                val = absol(val)
            keep += [attn, val]
        elif attn.startswith(("-isystem", "-iquote")):
            flag = next(f for f in ("-isystem", "-iquote") if attn.startswith(f))
            keep.append(flag + absol(attn[len(flag):]))
        elif attn.startswith(("-I", "/I")):
            keep.append(attn[:2] + absol(attn[2:]))
        elif attn.startswith(("-D", "/D")):
            keep.append(attn)
        elif kind == "msvc":
            if attn.startswith(_MSVC_KEEP):
                keep.append(attn)
        elif attn.startswith(("-std=", "-f", "-m", "-W")) and not attn.startswith("-fdiagnostics"):
            keep.append(attn)
    return keep


def _header_dirs(root: Path) -> List[str]:
    cl_path = root / "Applications" / "CausalLM"
    dirs = [
        cl_path,
        cl_path / "models",
        cl_path / "layers",
        cl_path / "third_party",
        cl_path / "third_party" / "minja" / "include",
        root / "api",
        root / "api" / "ccapi" / "include",
        root / "api" / "capi" / "include",
    ]
    for base in (root / "nntrainer",):
        for data, _subs, files in os.walk(base):
            if any(f.endswith((".h", ".hpp")) for f in files) and "test" not in Path(data).parts:
                dirs.append(Path(data))
    seen, out = set(), []
    for data in dirs:
        if data.is_dir() and str(data) not in seen:
            seen.add(str(data))
            out.append(str(data))
    return out


def flags_for(root: Path, kind: str, extra_includes: List[str]) -> Tuple[List[str], str]:
    """Compiler flags for the checkout (from compile_commands.json, or from its header folders)."""
    from_compile_commands = _from_compile_commands(root, kind)
    if from_compile_commands is not None:
        mode, flags = "compile_commands.json", from_compile_commands
        if kind == "msvc":
            flags = ["/nologo", *flags]
    else:
        mode = "checkout headers"
        incs = _header_dirs(root)
        # nntrainer's Windows build is C++20 (configurations/windows-native.ini).
        flags = (
            (
                [
                    "/nologo",
                    "/std:c++20",
                    "/EHsc",
                    "/utf-8",
                    "/permissive-",
                    "/Zc:__cplusplus",
                    "/D_USE_MATH_DEFINES",
                    "/DNOMINMAX",
                ]
                + [f"/I{d}" for d in incs]
            )
            if kind == "msvc"
            else (["-std=c++17"] + [f"-I{d}" for d in incs])
        )
    inc = [(f"/I{d}" if kind == "msvc" else f"-I{d}") for d in extra_includes]
    return inc + flags, mode


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# pylint: disable-next=too-many-locals
def check(sources: List[Path], nntrainer_root: Path, compiler: Optional[str] = None,
          timeout: int = 300) -> dict:
    """Syntax-check each source. Returns {status, ok, compiler, mode, errors, log, files, at}."""
    found = find_compiler(compiler)
    base = {"at": _now(), "files": [p.name for p in sources]}
    if not found:
        return dict(
            base,
            status="no_compiler",
            ok=False,
            compiler="",
            mode="",
            errors=[],
            log="No C++ compiler found. Install Visual Studio Build Tools with the C++ workload "
            "(Windows: winget install Microsoft.VisualStudio.2022.BuildTools), LLVM/clang "
            "(winget install LLVM.LLVM) or g++, or set aiCompilerWorkbench.cxxCompiler, "
            "then compile again.",
        )
    cxx, kind = found
    env = env_for(cxx, kind)
    if kind == "msvc" and env is None and not os.environ.get("INCLUDE"):
        return dict(base, status="no_compiler", ok=False, compiler=cxx, mode="", errors=[],
                    log=f"Found {cxx}, but could not set up its environment (vcvars64.bat). "
                        "Run VS Code from a Developer prompt, or repair the Build Tools install.")
    if not (Path(nntrainer_root) / "Applications" / "CausalLM").is_dir():
        return dict(
            base,
            status="no_checkout",
            ok=False,
            compiler=cxx,
            mode="",
            errors=[],
            log=f"{nntrainer_root} is not an nntrainer checkout (no Applications/CausalLM).",
        )
    dirs = sorted({str(p.parent) for p in sources})
    flags, mode = flags_for(Path(nntrainer_root), kind, dirs)
    only = (
        ["/Zs"]
        if kind == "msvc"
        else [
            "-fsyntax-only",
            (
                "-fno-color-diagnostics"
                if "clang" in os.path.basename(cxx).lower()
                else "-fdiagnostics-color=never"
            ),
        ]
    )
    errors: List[dict] = []
    logs: List[str] = []
    prefix = (
        ["cmd", "/c", cxx] if os.name == "nt" and cxx.lower().endswith((".cmd", ".bat")) else [cxx]
    )
    # Flags go in a response file: the checkout's header folders can exceed
    # the Windows command-line limit, and the file documents the exact flags.
    rsp = Path(dirs[0]) / ".codegen" / "compile_flags.rsp" if dirs else None
    flag_args = flags
    if rsp is not None:
        rsp.parent.mkdir(parents=True, exist_ok=True)
        def quote_arg(arg: str) -> str:
            arg = (
                arg.replace("\\", "/") if kind == "gnu" else arg
            )  # backslash escapes in gcc/clang .rsp
            return f'"{arg}"' if any(ch.isspace() for ch in arg) else arg
        rsp.write_text("\n".join(quote_arg(f) for f in flags) + "\n", encoding="utf-8")
        flag_args = [f"@{rsp}"]
    for src in sources:
        cmd = [*prefix, *flag_args, *only, str(src)]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            errors.append(
                {"file": src.name, "line": 0, "message": f"compiler timed out after {timeout}s"}
            )
            continue
        except OSError as exc:
            return dict(base, status="no_compiler", ok=False, compiler=cxx, mode=mode, errors=[],
                        log=f"could not run {cxx}: {exc}")
        out = (proc.stdout or "") + (proc.stderr or "")
        logs.append(
            f"$ {os.path.basename(cxx)} ... {src.name}  (exit {proc.returncode})\n{out.strip()}"
        )
        before = len(errors)
        for line in out.splitlines():
            match = DIAG_RE.match(line.strip())
            if match and match.group("kind") in ("error", "fatal error"):
                errors.append(
                    {
                        "file": os.path.basename(match.group("file")),
                        "line": int(match.group("line")),
                        "message": match.group("msg").strip(),
                    }
                )
        if proc.returncode != 0 and len(errors) == before:
            errors.append(
                {
                    "file": src.name,
                    "line": 0,
                    "message": out.strip()[-300:] or f"exit {proc.returncode}",
                }
            )
    passed = not errors
    return dict(
        base,
        status="ok" if passed else "failed",
        ok=passed,
        compiler=cxx,
        mode=mode,
        errors=errors[:40],
        log="\n\n".join(logs)[-20000:],
    )
