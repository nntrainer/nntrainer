# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file test_edge_resolver.py
# @brief Tests for resolving the edges between traced modules.
# @author prachi-t17 <prachi.t@samsung.com>

"""Tests for resolving the edges between traced modules."""

from _util import load
R = load("edge_resolver", "api/parsers/edge_resolver.py").resolve_traced_edges


def test_linear_chain():
    """Linear chain."""
    edges, orphans = R(["a", "b", "c"], {"a", "b", "c"}, {"b": ["a"], "c": ["b"]})
    assert sorted(edges) == [("a", "b"), ("b", "c")] and not orphans


def test_diamond_branch_recovered():
    """Diamond branch recovered."""
    edges, orphans = R(
        ["root", "a", "b", "merge"], {"root", "a", "b", "merge"},
        {"a": ["root"], "b": ["root"], "merge": ["a", "b"]},
    )
    assert sorted(edges) == [("a", "merge"), ("b", "merge"), ("root", "a"), ("root", "b")]
    assert not orphans


def test_edge_walks_through_passthrough_leaf():
    """Edge walks through passthrough leaf."""
    # 'drop' is NOT emitted; edge b->c must be recovered through it
    edges, orphans = R(
        ["a", "b", "drop", "c"], {"a", "b", "c"},
        {"b": ["a"], "drop": ["b"], "c": ["drop"]},
    )
    assert sorted(edges) == [("a", "b"), ("b", "c")] and not orphans


def test_functional_gap_signals_fallback():
    """Functional gap signals fallback."""
    # c's producer is invisible (functional residual) -> orphan -> fallback
    _edges, orphans = R(["a", "b", "c"], {"a", "b", "c"}, {"b": ["a"]})
    assert orphans == ["c"]


def test_empty_trace_signals_fallback():
    """Empty trace signals fallback."""
    edges, orphans = R(["a", "b"], {"a", "b"}, {})
    assert orphans == ["b"] and edges == []
