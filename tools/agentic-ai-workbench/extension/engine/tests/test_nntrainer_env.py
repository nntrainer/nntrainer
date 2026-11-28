# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file test_nntrainer_env.py
# @brief Tests for finding the nntrainer build flags.
# @author prachi-t17 <prachi.t@samsung.com>

"""Tests for finding the nntrainer build flags."""

import os
import tempfile
from pathlib import Path

from _util import load

nenv = load("nntrainer_env_test", "agents/nntrainer_env.py")  # stdlib only


def test_prefix_discovery():
    """Prefix discovery."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.makedirs(os.path.join(tmp_dir, "include"))
        cflags, libs, source = nenv.discover_flags(prefix=tmp_dir)
        assert cflags[:1] == ["-I"]
        assert "-lccapi-nntrainer" in libs and "-lnntrainer" in libs
        assert source.startswith("prefix")


def test_env_discovery_multi_include():
    """Env discovery multi include."""
    os.environ["NNTRAINER_INCLUDE_DIR"] = os.pathsep.join(["/opt/a/include", "/opt/b/include"])
    os.environ["NNTRAINER_LIB_DIR"] = "/opt/a/lib"
    try:
        cflags, libs, source = nenv.discover_flags()
        assert cflags.count("-I") == 2
        assert "-L" in libs
        assert libs[-2:] == ["-lccapi-nntrainer", "-lnntrainer"]  # ccapi first
        assert source == "NNTRAINER_INCLUDE_DIR"
    finally:
        del os.environ["NNTRAINER_INCLUDE_DIR"]
        del os.environ["NNTRAINER_LIB_DIR"]


def test_not_found_returns_none():
    """Not found returns none."""
    # no prefix, no env; pkg-config almost certainly has no nntrainer here
    for k in ("NNTRAINER_INCLUDE_DIR", "NNTRAINER_LIB_DIR"):
        os.environ.pop(k, None)
    assert nenv.discover_flags() == (None, None, None)


def test_multiarch_lib_dir_resolved():
    """Multiarch lib dir resolved."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.makedirs(os.path.join(tmp_dir, "include"))
        arch = os.path.join(tmp_dir, "lib", "aarch64-linux-gnu")
        os.makedirs(arch)
        Path(os.path.join(arch, "libnntrainer.so")).write_text("", encoding="utf-8")
        _cflags, libs, _ = nenv.discover_flags(prefix=tmp_dir)
        assert libs[libs.index("-L") + 1] == arch  # multiarch dir, not plain lib/
