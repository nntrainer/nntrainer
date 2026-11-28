# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file llama.py
# @brief Llama / Llama-2 / Llama-3 adapter.
# @author prachi-t17 <prachi.t@samsung.com>

"""Llama / Llama-2 / Llama-3 adapter."""
from api.adapters.llama_family import LlamaFamilyAdapter


class LlamaAdapter(LlamaFamilyAdapter):
    """Adapter for Llama models."""
    model_types = ("llama",)
