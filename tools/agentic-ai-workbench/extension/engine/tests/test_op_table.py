# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file test_op_table.py
# @brief Tests for the table that maps model operations to nntrainer layers.
# @author prachi-t17 <prachi.t@samsung.com>

"""Tests for the table that maps model operations to nntrainer layers."""

from _util import load
op_table = load("op_table", "api/compatibility/op_table.py")
classify_op = op_table.classify_op


def test_linear_maps_to_fully_connected():
    """Linear maps to fully connected."""
    entry = classify_op("Linear")
    assert (
        entry["nntrainer_type"] == "fully_connected" and entry["supported"] and entry["emits_layer"]
    )


def test_gpt2_conv1d_is_linear():
    """GPT-2 Conv1D is linear."""
    entry = classify_op("Conv1D")
    assert entry["nntrainer_type"] == "fully_connected" and entry["supported"]


def test_real_conv1d_is_not_matched_as_linear():
    """Real Conv1D is not matched as linear."""
    # nn.Conv1d must NOT collide with GPT-2's Conv1D
    entry = classify_op("Conv1d")
    assert entry["nntrainer_type"] != "fully_connected"


def test_hf_gelu_activation():
    """HF GELU activation."""
    entry = classify_op("GELUActivation")
    assert entry["nntrainer_type"] == "activation"
    assert entry["attributes"]["activation"] == "gelu" and entry["supported"]


def test_hf_silu_activation():
    """HF SiLU activation."""
    entry = classify_op("SiLUActivation")
    assert entry["attributes"]["activation"] == "swish"


def test_dropout_emits_no_layer():
    """Dropout emits no layer."""
    entry = classify_op("Dropout")
    assert entry["emits_layer"] is False and entry["supported"]


def test_rmsnorm_family_recognized_but_manual():
    """RMSNorm family recognized but manual."""
    for name in ("RMSNorm", "LlamaRMSNorm", "Qwen2RMSNorm", "GemmaRMSNorm"):
        entry = classify_op(name)
        assert entry["emits_layer"] is True
        assert entry["supported"] is False
        assert entry["nntrainer_type"] is None       # never a guessed type string
        assert "RMS" in entry["reason"]


def test_quick_gelu_not_silently_approximated():
    """Quick GELU not silently approximated."""
    # deliberately unmapped -> stays unknown rather than pretending it's gelu
    entry = classify_op("QuickGELUActivation")
    assert entry["supported"] is False


def test_unknown_op():
    """Unknown op."""
    entry = classify_op("SomeMadeUpOp")
    assert entry["supported"] is False and entry["emits_layer"] is True
    assert "unknown operator" in entry["reason"]
