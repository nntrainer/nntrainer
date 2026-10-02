#!/usr/bin/env python3
"""Dependency-free test runner (no pytest needed). Discovers test_*.py in this
folder, runs every test_* function, prints a summary, exits non-zero on failure.
`pytest` also works if installed."""
import importlib.util
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _load(path):
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    passed = failed = 0
    failures = []
    for fname in sorted(os.listdir(HERE)):
        if not (fname.startswith("test_") and fname.endswith(".py")):
            continue
        mod = _load(os.path.join(HERE, fname))
        for attr in sorted(dir(mod)):
            if not attr.startswith("test_"):
                continue
            fn = getattr(mod, attr)
            if not callable(fn):
                continue
            try:
                fn()
                passed += 1
                print(f"  PASS  {fname}::{attr}")
            except Exception:
                failed += 1
                failures.append(f"{fname}::{attr}\n{traceback.format_exc()}")
                print(f"  FAIL  {fname}::{attr}")
    print(f"\n{passed} passed, {failed} failed")
    for f in failures:
        print("\n" + "=" * 60 + "\n" + f)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
