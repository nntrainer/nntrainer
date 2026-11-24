# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file qwen2.py
# @brief Qwen2 adapter.
# @author prachi-t17 <prachi.t@samsung.com>

"""Qwen2 adapter (no Q/K norm, no sliding window by default -- both
still detected at runtime if a checkpoint enables them anyway)."""
from api.adapters.llama_family import LlamaFamilyAdapter


class Qwen2Adapter(LlamaFamilyAdapter):
    """Adapter for Qwen2 models."""
    model_types = ("qwen2",)
