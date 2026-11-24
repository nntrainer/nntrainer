# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file test_layout.py
# @brief Tests for the vertical layout of the graph views.
# @author prachi-t17 <prachi.t@samsung.com>

# Tests call private helpers and import the engine from the folder that run_all.py
# puts on sys.path at run time (some only after installing stand-ins), which pylint
# cannot follow when it checks this file on its own.
# pylint: disable=protected-access

"""Tests for the vertical layout of the graph views."""

from _util import load
gv = load("graph_views_test", "agents/graph_views.py")  # imports only `re`


def _nodes(ids):
    return [{"id": i, "label": i, "type": "x", "status": "mapped"} for i in ids]


def test_diamond_is_centered_and_layered():
    """Diamond is centered and layered."""
    nodes = _nodes(["root", "a", "b", "m"])
    edges = [{"source": "root", "target": "a"}, {"source": "root", "target": "b"},
             {"source": "a", "target": "m"}, {"source": "b", "target": "m"}]
    gv._layout_vertical(nodes, edges, [n["id"] for n in nodes])
    positions = {n["id"]: (n["x"], n["y"]) for n in nodes}
    # layered vertically: root above a/b above m
    assert positions["root"][1] < positions["a"][1] < positions["m"][1]
    assert positions["a"][1] == positions["b"][1]
    # root and merge centered over the two middle nodes
    assert positions["root"][0] == positions["m"][0] == (positions["a"][0] + positions["b"][0]) // 2


def test_empty_graph_is_safe():
    """Empty graph is safe."""
    nodes, edges = [], []
    gv._layout_vertical(nodes, edges, [])   # must not raise
    assert isinstance(nodes, list) and not nodes
