"""
Shared nntrainer discovery for the compile paths, so the Compiler Agent
and the (legacy) CppCorrector can't drift on include/lib flags.

Discovery order (first that resolves wins):
  1. explicit prefix arg   -> {prefix}/include, {prefix}/lib
  2. NNTRAINER_INCLUDE_DIR  (+ optional NNTRAINER_LIB_DIR); the include dir
     may be an os.pathsep-separated list of directories
  3. pkg-config            -> ccapi-nntrainer, then nntrainer

Returns (cflags, libs, source):
  cflags/libs : lists ready to splice into a g++ command
  source      : short human-readable label, or None if nntrainer wasn't found

The generated code uses the ccapi (model.h / layer.h, `using namespace
ml::train`), so -lccapi-nntrainer is always linked ahead of -lnntrainer.
BLAS is intentionally not added here: a shared nntrainer pulls its own
deps, and adding it in exactly one path is what caused drift before. If a
*static* nntrainer needs it, add `-lopenblas` once, here, so both paths stay
in lockstep.
"""
import os
import shutil
import subprocess

_PKGCONFIG_CANDIDATES = ["ccapi-nntrainer", "nntrainer"]
_LINK_LIBS = ["-lccapi-nntrainer", "-lnntrainer"]


def _resolve_lib_dir(prefix):
    """Return the dir under {prefix}/lib that actually holds libnntrainer.*,
    accounting for Debian/Ubuntu multiarch (lib/x86_64-linux-gnu/, etc.).
    Falls back to {prefix}/lib if nothing more specific is found."""
    lib = os.path.join(prefix, "lib")
    have = lambda d: (os.path.exists(os.path.join(d, "libnntrainer.so"))
                      or os.path.exists(os.path.join(d, "libnntrainer.a")))
    if have(lib):
        return lib
    if os.path.isdir(lib):
        for sub in sorted(os.listdir(lib)):
            cand = os.path.join(lib, sub)
            if os.path.isdir(cand) and have(cand):
                return cand
    return lib


def _from_prefix(prefix):
    if not prefix:
        return None
    inc = os.path.join(prefix, "include")
    if not os.path.isdir(inc):
        return None
    lib = _resolve_lib_dir(prefix)
    return ["-I", inc], ["-L", lib] + _LINK_LIBS, f"prefix {prefix}"


def _from_env():
    include = os.environ.get("NNTRAINER_INCLUDE_DIR")
    if not include:
        return None
    cflags = []
    for inc in include.split(os.pathsep):
        inc = inc.strip()
        if inc:
            cflags += ["-I", inc]
    libs = []
    lib_dir = os.environ.get("NNTRAINER_LIB_DIR")
    if lib_dir:
        libs += ["-L", lib_dir]
    libs += _LINK_LIBS
    return cflags, libs, "NNTRAINER_INCLUDE_DIR"


def _from_pkgconfig():
    if not shutil.which("pkg-config"):
        return None
    for pkg in _PKGCONFIG_CANDIDATES:
        try:
            if subprocess.run(["pkg-config", "--exists", pkg], timeout=15).returncode != 0:
                continue
            cflags = subprocess.run(
                ["pkg-config", "--cflags", pkg],
                capture_output=True, text=True, timeout=15,
            ).stdout.split()
            libs = subprocess.run(
                ["pkg-config", "--libs", pkg],
                capture_output=True, text=True, timeout=15,
            ).stdout.split()
            return cflags, libs, f"pkg-config ({pkg})"
        except Exception:
            continue
    return None


def discover_flags(prefix=None):
    """Return (cflags, libs, source). (None, None, None) if not found."""
    for finder in (lambda: _from_prefix(prefix), _from_env, _from_pkgconfig):
        result = finder()
        if result:
            return result
    return None, None, None
