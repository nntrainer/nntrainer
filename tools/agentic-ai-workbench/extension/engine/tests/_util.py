# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file _util.py
# @brief Loads a single engine module by file path for the tests.
# @author prachi-t17 <prachi.t@samsung.com>

"""Load a single engine module by file path, so tests don't drag in torch,
langchain, or package __init__ side effects."""
import importlib.util
import os

_ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(module_name, relpath):
    """Load one engine module from its file, without importing its package."""
    path = os.path.join(_ENGINE, relpath)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
