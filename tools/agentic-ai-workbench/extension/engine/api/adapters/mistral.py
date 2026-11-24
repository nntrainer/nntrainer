# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file mistral.py
# @brief Mistral adapter.
# @author prachi-t17 <prachi.t@samsung.com>

"""Mistral adapter. Sliding-window attention is read from
config.sliding_window at runtime by the shared base class -- listed
here only so `model_type == "mistral"` matches explicitly."""
from api.adapters.llama_family import LlamaFamilyAdapter


class MistralAdapter(LlamaFamilyAdapter):
    """Adapter for Mistral models."""
    model_types = ("mistral",)
