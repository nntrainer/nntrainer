from _util import load
gv = load("graph_views_test", "agents/graph_views.py")  # imports only `re`


def _nodes(ids):
    return [{"id": i, "label": i, "type": "x", "status": "mapped"} for i in ids]


def test_diamond_is_centered_and_layered():
    nodes = _nodes(["root", "a", "b", "m"])
    edges = [{"source": "root", "target": "a"}, {"source": "root", "target": "b"},
             {"source": "a", "target": "m"}, {"source": "b", "target": "m"}]
    gv._layout_vertical(nodes, edges, [n["id"] for n in nodes])
    xy = {n["id"]: (n["x"], n["y"]) for n in nodes}
    # layered vertically: root above a/b above m
    assert xy["root"][1] < xy["a"][1] < xy["m"][1]
    assert xy["a"][1] == xy["b"][1]
    # root and merge centered over the two middle nodes
    assert xy["root"][0] == xy["m"][0] == (xy["a"][0] + xy["b"][0]) // 2


def test_empty_graph_is_safe():
    nodes, edges = [], []
    gv._layout_vertical(nodes, edges, [])   # must not raise
    assert nodes == []
