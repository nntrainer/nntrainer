# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file test_docker_e2e.py
# @brief Opt-in end-to-end test of the real Docker self-test.
# @author prachi-t17 <prachi.t@samsung.com>

# Tests call private helpers and import the engine from the folder that run_all.py
# puts on sys.path at run time (some only after installing stand-ins), which pylint
# cannot follow when it checks this file on its own.
# pylint: disable=import-error,import-outside-toplevel

"""
Opt-in end-to-end test: the real Docker self-test (knowledge/selftest.py) --
system check, Docker build set up, full build, upstream's tiny Qwen3
reference test. Runs only with WORKBENCH_DOCKER_E2E=1 (Docker must be
running; the first run takes a while). Run it before packaging a VSIX:

    WORKBENCH_DOCKER_E2E=1 python -B tests/run_all.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(_HERE), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def test_docker_self_test_end_to_end():
    """Docker self test end to end."""
    if os.environ.get("WORKBENCH_DOCKER_E2E") != "1":
        print("    (skipped: set WORKBENCH_DOCKER_E2E=1 to run the real Docker self-test)")
        return
    from knowledge import selftest
    result = selftest.run_selftest(log=print)
    assert result["ok"], result["summary"]
