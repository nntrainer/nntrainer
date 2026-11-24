# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file qwen3.py
# @brief Qwen3 adapter.
# @author prachi-t17 <prachi.t@samsung.com>

"""Qwen3 adapter. Structurally identical to LlamaFamilyAdapter's
generic handling (Q/K RMSNorm and GQA are both detected at runtime),
so this class exists purely to make `model_type == "qwen3"` an
explicit, visible match rather than falling through to the generic
Llama-family handling silently."""
from api.adapters.llama_family import LlamaFamilyAdapter


class Qwen3Adapter(LlamaFamilyAdapter):
    """Adapter for Qwen3 models (per-head Q/K norm)."""
    model_types = ("qwen3",)
