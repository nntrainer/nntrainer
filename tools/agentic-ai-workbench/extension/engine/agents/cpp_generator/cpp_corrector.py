"""
CppCorrector: runs the system C++ compiler against a temp file and
summarises any errors for the LLM feedback loop.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any

from config import get_config

from ..nntrainer_env import discover_flags

logger = logging.getLogger(__name__)


class CppCorrector:
    """
    Compiles C++ code and produces a tidy error summary.
    Uses a per-call TemporaryDirectory so nothing leaks between runs,
    even if the process is killed mid-compilation.
    """

    def __init__(self) -> None:
        self._cfg = get_config()

    # -------------------------------------------------------------- compile
    def compile(self, code: str, nntrainer_prefix: str) -> Dict[str, Any]:
        """
        Write code to a temp file, compile it, return a result dict.

        Returns
        -------
        {
          "success":  bool,
          "stdout":   str,
          "stderr":   str,
          "summary":  str   # human-readable error summary for LLM
        }
        """
        timeout = self._cfg.get("cpp_generator", "compilation_timeout_sec", 60)

        # Discover include/lib flags through the shared helper so this
        # (legacy) path links the ccapi and stays in lockstep with the
        # live Compiler Agent instead of drifting (it used to link only
        # -lnntrainer, missing the ccapi symbols the generated code needs).
        cflags, libs, _ = discover_flags(prefix=str(nntrainer_prefix))
        if cflags is None:
            # Preserve the explicit-prefix contract this method was called with.
            prefix = Path(nntrainer_prefix)
            cflags = ["-I", str(prefix / "include")]
            libs = ["-L", str(prefix / "lib"), "-lccapi-nntrainer", "-lnntrainer"]

        # Using a context-managed temp dir guarantees cleanup even on exception
        with tempfile.TemporaryDirectory(prefix="nntrainer_build_") as tmpdir:
            src = Path(tmpdir) / "generated_model.cpp"
            out = Path(tmpdir) / "generated_model"

            src.write_text(code, encoding="utf-8")

            cmd = ["g++", "-std=c++17"] + cflags + [str(src), "-o", str(out)] + libs

            logger.debug("Compile: %s", " ".join(cmd))

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                msg = f"Compilation timed out after {timeout}s"
                logger.warning(msg)
                return {"success": False, "stdout": "", "stderr": msg, "summary": msg}
            except Exception as exc:
                msg = f"Compiler invocation failed: {exc}"
                logger.error(msg)
                return {"success": False, "stdout": "", "stderr": msg, "summary": msg}

            if proc.returncode == 0:
                return {"success": True, "stdout": proc.stdout, "stderr": proc.stderr, "summary": ""}

            summary = self._summarise(proc.stderr)
            logger.debug("Compilation failed:\n%s", proc.stderr[:400])
            return {
                "success": False,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "summary": summary,
            }

    # ---------------------------------------------------------- error summary
    @staticmethod
    def _summarise(stderr: str) -> str:
        """
        Extract the first N distinct error messages from compiler output.
        Keeps the summary short enough for the LLM to act on without being
        overwhelmed by cascading secondary errors.
        """
        error_pattern = re.compile(r"error:\s*(.+?)(?:\s*\[-|$)", re.IGNORECASE)
        seen: list[str] = []
        for line in stderr.splitlines():
            m = error_pattern.search(line)
            if m:
                msg = m.group(1).strip()
                if msg not in seen:
                    seen.append(msg)
                if len(seen) >= 5:
                    break

        if seen:
            return "Compilation errors:\n" + "\n".join(f"  • {e}" for e in seen)
        # Fallback: return first 400 chars of raw stderr
        return "Compiler output:\n" + stderr[:400]

    def __repr__(self) -> str:
        return "CppCorrector()"
