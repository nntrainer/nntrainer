#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file run_all.py
# @brief Dependency-free test runner (no pytest needed).
# @author prachi-t17 <prachi.t@samsung.com>

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
    """Run every test_* function in this folder and print a summary; 1 when any failed."""
    passed = failed = 0
    failures = []
    for fname in sorted(os.listdir(HERE)):
        if not (fname.startswith("test_") and fname.endswith(".py")):
            continue
        mod = _load(os.path.join(HERE, fname))
        for attr in sorted(dir(mod)):
            if not attr.startswith("test_"):
                continue
            func = getattr(mod, attr)
            if not callable(func):
                continue
            try:
                func()
                passed += 1
                print(f"  PASS  {fname}::{attr}")
            # a test failure of any kind is counted, including code under test calling sys.exit
            except (Exception, SystemExit):  # pylint: disable=broad-except
                failed += 1
                failures.append(f"{fname}::{attr}\n{traceback.format_exc()}")
                print(f"  FAIL  {fname}::{attr}")
    print(f"\n{passed} passed, {failed} failed")
    for failure in failures:
        print("\n" + "=" * 60 + "\n" + failure)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
