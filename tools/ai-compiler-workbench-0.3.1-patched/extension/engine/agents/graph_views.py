"""
Builds the two graph views the webview renders side by side: one
parsed straight out of model.ini, one parsed straight out of
generated_model.cpp. Deliberately parses the *actual generated text*
rather than reusing the in-memory IR a second time -- that way the
graph shown is a genuine reflection of what got written to disk (and,
incidentally, a cheap cross-check that the .ini and .cpp agree with
each other), including each node's weight info for the inspector panel.

Both graphs use a vertical, top-to-bottom layered layout: one row per
topological depth, nodes within a row spread out left-to-right.
"""
import re

ROW_HEIGHT = 130
COL_WIDTH = 200

WEIGHT_RE = re.compile(
    r'weight:\s*name=(\S+)\s*shape=(\[[^\]]*\])\s*dtype=(\S+)\s*params=(\d+)',
    re.IGNORECASE,
)


def _parse_weight_comment(line: str):
    m = WEIGHT_RE.search(line)
    if not m:
        return None
    name, shape_str, dtype, params = m.groups()
    try:
        shape = [int(x.strip()) for x in shape_str.strip("[]").split(",") if x.strip()]
    except ValueError:
        shape = []
    return {"name": name, "shape": shape, "dtype": dtype, "params": int(params)}


def build_ini_graph(ini_content: str) -> dict:
    if not ini_content:
        return {"nodes": [], "edges": []}

    nodes = []
    edges = []
    order = []

    # Split into [section] blocks
    blocks = re.split(r"^\[(.+?)\]\s*$", ini_content, flags=re.MULTILINE)
    # re.split with a capturing group returns [pre, name1, body1, name2, body2, ...]
    it = iter(blocks[1:])
    for name, body in zip(it, it):
        if name == "Model":
            continue
        attrs = {}
        input_layers = []
        layer_type = "unknown"
        weight_info = None
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(";"):
                w = _parse_weight_comment(line)
                if w:
                    weight_info = w
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key.lower() == "type":
                layer_type = value
            elif key.lower() == "input_layers":
                input_layers = [v.strip() for v in value.split(",") if v.strip()]
            else:
                attrs[key] = value

        order.append(name)
        node = {
            "id": name,
            "label": name,
            "type": layer_type,
            "status": "mapped",
            "attributes": attrs,
        }
        if weight_info:
            node["weightInfo"] = weight_info
        nodes.append(node)
        for src in input_layers:
            edges.append({"source": src, "target": name})

    _layout_vertical(nodes, edges, order)
    return {"nodes": nodes, "edges": edges}


def build_cpp_graph(cpp_code: str) -> dict:
    if not cpp_code:
        return {"nodes": [], "edges": []}

    nodes = []
    edges = []
    order = []

    var_type = {}      # C++ variable name -> layer type
    var_attrs = {}      # C++ variable name -> {key: value}
    var_name = {}       # C++ variable name -> layer name= value
    var_weight = {}      # C++ variable name -> weight info dict

    create_re = re.compile(r'auto\s+(\w+)\s*=\s*createLayer\("([^"]+)"\);')
    setprop_re = re.compile(r'(\w+)->setProperty\(\{"([^=]+)=([^"]*)"\}\);')
    todo_re = re.compile(r'//\s*TODO\(unsupported\):\s*(\S+)\s*\[([^\]]+)\]\s*--\s*(.*)')

    lines = cpp_code.splitlines()
    current_unsupported = None  # name of the TODO block currently being read, if any

    for line in lines:
        m = todo_re.search(line)
        if m:
            unsup_name, unsup_type, reason = m.group(1), m.group(2), m.group(3)
            current_unsupported = unsup_name
            order.append(unsup_name)
            nodes.append({
                "id": unsup_name,
                "label": unsup_name,
                "type": unsup_type,
                "status": "unmapped",
                "attributes": {},
                "reason": reason,
            })
            continue

        w = _parse_weight_comment(line)
        if w:
            if current_unsupported:
                for n in nodes:
                    if n["id"] == current_unsupported:
                        n["weightInfo"] = w
                        break
            continue

        m = create_re.search(line)
        if m:
            var, layer_type = m.group(1), m.group(2)
            var_type[var] = layer_type
            var_attrs[var] = {}
            current_unsupported = None
            continue

        m = setprop_re.search(line)
        if m:
            var, key, value = m.group(1), m.group(2), m.group(3)
            if key == "name":
                var_name[var] = value
            elif key == "input_layers":
                var_attrs.setdefault(var, {})["__input_layers__"] = value
            else:
                var_attrs.setdefault(var, {})[key] = value
            continue

    # Weight comments for *supported* layers appear after their setProperty
    # calls but reference the most recently declared variable -- re-scan
    # to attach them since we need the var context, not just node id.
    current_var = None
    for line in lines:
        m = create_re.search(line)
        if m:
            current_var = m.group(1)
            continue
        if todo_re.search(line):
            current_var = None
            continue
        w = _parse_weight_comment(line)
        if w and current_var:
            var_weight[current_var] = w

    for var, layer_type in var_type.items():
        name = var_name.get(var, var)
        attrs = {k: v for k, v in var_attrs.get(var, {}).items() if not k.startswith("__")}
        order.append(name)
        node = {
            "id": name,
            "label": name,
            "type": layer_type,
            "status": "mapped",
            "attributes": attrs,
        }
        if var in var_weight:
            node["weightInfo"] = var_weight[var]
        nodes.append(node)

        input_layers_str = var_attrs.get(var, {}).get("__input_layers__", "")
        if input_layers_str:
            for src in input_layers_str.split(","):
                src = src.strip()
                if src:
                    edges.append({"source": src, "target": name})

    _layout_vertical(nodes, edges, order)
    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Two-graph views: "Model Graph" (from the semantic IR) and "nntrainer Graph"
# (from the lowered target IR). Both feed the same webview tabs; see
# extension/webview/main.html. Only these two tabs exist -- a third
# "C++ Audit" / "Mapping" tab was considered and dropped: mapping is
# handled by click-to-highlight using `build_node_mappings` below instead
# of a dedicated view, per the simplified two-tab design.
# ---------------------------------------------------------------------------

#: Layers beyond this count get the first/last layer fully expanded and
#: everything in between collapsed into one node per layer -- keeps a
#: 28-layer model's graph readable without hiding any data (collapsed
#: nodes still carry every member id in source_node_ids for expand-on-click).
_EXPAND_THRESHOLD = 3


def build_model_graph_view(model_ir) -> dict:
    """Architecture-level source graph, built directly from the semantic
    IR (api.semantic.model.CausalLMIR) -- never by re-parsing generated
    text. See agents/nntrainer_lowering.py."""
    nodes, edges, order = [], [], []

    def node(id_, label, type_, semantic_type, group_id="", source_ids=None, attrs=None):
        n = {
            "id": id_, "label": label, "type": type_, "status": "mapped",
            "semanticType": semantic_type, "group": group_id,
            "sourceNodeIds": source_ids or [], "attributes": attrs or {},
        }
        nodes.append(n)
        order.append(id_)
        return n

    prev_id = node(
        "embedding", "Embedding", "embedding", "embedding",
        source_ids=[model_ir.embedding_name],
    )["id"]

    n_layers = len(model_ir.decoder_layers)
    for layer in model_ir.decoder_layers:
        i = layer.index
        expand = n_layers <= _EXPAND_THRESHOLD or i in (0, n_layers - 1)
        if expand:
            entry_id, exit_id = _build_expanded_layer(node, edges, layer, i)
        else:
            gid = f"decoder_{i}"
            member_source_ids = [
                layer.input_norm.source_name, layer.attention.source_name,
                layer.post_attention_norm.source_name, layer.mlp.source_name,
            ]
            n = node(
                f"{gid}_collapsed", f"Decoder Layer {i}", "decoder_layer", "decoder_layer",
                group_id=gid, source_ids=member_source_ids,
                attrs={"collapsed": True, "repeat_index": i},
            )
            entry_id = exit_id = n["id"]
        edges.append({"source": prev_id, "target": entry_id})
        prev_id = exit_id

    final_id = node(
        "final_norm", "Final Norm", model_ir.final_norm.norm_type, "normalization",
        source_ids=[model_ir.final_norm.source_name],
    )["id"]
    edges.append({"source": prev_id, "target": final_id})

    lm_head_id = node(
        "lm_head", "LM Head", "fully_connected", "lm_head",
        source_ids=[model_ir.lm_head_name],
    )["id"]
    edges.append({"source": final_id, "target": lm_head_id})

    _layout_vertical(nodes, edges, order)
    return {"nodes": nodes, "edges": edges}


def _build_expanded_layer(node_fn, edges, layer, index):
    gid = f"decoder_{index}"
    a, m = layer.attention, layer.mlp

    input_norm = node_fn(f"{gid}.input_norm", "Input RMSNorm", layer.input_norm.norm_type,
                          "normalization", gid, [layer.input_norm.source_name])["id"]

    q = node_fn(f"{gid}.wq", "Q Projection", "fully_connected", "attention", gid, [a.q_proj.source_name])["id"]
    k = node_fn(f"{gid}.wk", "K Projection", "fully_connected", "attention", gid, [a.k_proj.source_name])["id"]
    v = node_fn(f"{gid}.wv", "V Projection", "fully_connected", "attention", gid, [a.v_proj.source_name])["id"]
    edges += [{"source": input_norm, "target": q}, {"source": input_norm, "target": k}, {"source": input_norm, "target": v}]

    q_in, k_in = q, k
    if a.q_norm is not None:
        q_in = node_fn(f"{gid}.q_norm", "Q/K Norm", "reshaped_rms_norm", "normalization", gid, [a.q_norm.source_name])["id"]
        edges.append({"source": q, "target": q_in})
    if a.k_norm is not None:
        k_in = node_fn(f"{gid}.k_norm", "Q/K Norm", "reshaped_rms_norm", "normalization", gid, [a.k_norm.source_name])["id"]
        edges.append({"source": k, "target": k_in})

    attn_out = node_fn(f"{gid}.attention", "Attention Output", "mha_core", "attention", gid, [a.source_name])["id"]
    for src in (q_in, k_in, v):
        edges.append({"source": src, "target": attn_out})

    o_proj = node_fn(f"{gid}.wo", "Output Projection", "fully_connected", "attention", gid, [a.o_proj.source_name])["id"]
    edges.append({"source": attn_out, "target": o_proj})

    res1 = node_fn(f"{gid}.attention_residual", "Residual Add", "addition", "residual", gid, [])["id"]
    edges.append({"source": o_proj, "target": res1})

    post_norm = node_fn(f"{gid}.post_attention_norm", "Post-Attention RMSNorm", layer.post_attention_norm.norm_type,
                         "normalization", gid, [layer.post_attention_norm.source_name])["id"]
    edges.append({"source": res1, "target": post_norm})

    mlp_label = "Gated MLP" if m.gated else "MLP"
    if m.gated:
        gate = node_fn(f"{gid}.gate", "Gate Projection", "fully_connected", "mlp", gid, [m.gate_proj.source_name])["id"]
        up = node_fn(f"{gid}.up", "Up Projection", "fully_connected", "mlp", gid, [m.up_proj.source_name])["id"]
        edges += [{"source": post_norm, "target": gate}, {"source": post_norm, "target": up}]
        act = node_fn(f"{gid}.activation", m.activation, m.activation, "mlp", gid, [])["id"]
        edges.append({"source": gate, "target": act})
        mul = node_fn(f"{gid}.gate_mul", "Multiply", "multiply", "mlp", gid, [])["id"]
        edges += [{"source": act, "target": mul}, {"source": up, "target": mul}]
        mlp_out_src = mul
    else:
        up = node_fn(f"{gid}.up", "Up Projection", "fully_connected", "mlp", gid, [m.up_proj.source_name])["id"]
        edges.append({"source": post_norm, "target": up})
        act = node_fn(f"{gid}.activation", m.activation, m.activation, "mlp", gid, [])["id"]
        edges.append({"source": up, "target": act})
        mlp_out_src = act

    down = node_fn(f"{gid}.down", "Down Projection", "fully_connected", "mlp", gid, [m.down_proj.source_name])["id"]
    edges.append({"source": mlp_out_src, "target": down})

    res2 = node_fn(f"{gid}.mlp_residual", "Residual Add", "addition", "residual", gid, [])["id"]
    edges.append({"source": down, "target": res2})

    return input_norm, res2


def build_nntrainer_graph_view(nntrainer_graph_ir: dict) -> dict:
    """Renders the exact lowered target graph -- the same
    nntrainer_graph_ir the C++ generator consumes -- collapsing every
    decoder-layer group except the first and last so a 28-layer model
    stays readable. Nothing is discarded: a collapsed node's
    sourceNodeIds lists every member node id."""
    raw_nodes = {n["id"]: n for n in nntrainer_graph_ir.get("nodes", [])}
    raw_edges = nntrainer_graph_ir.get("edges", [])

    groups: dict[str, list] = {}
    for n in raw_nodes.values():
        if n.get("group_id", "").startswith("decoder_"):
            groups.setdefault(n["group_id"], []).append(n)

    layer_indices = sorted(int(g.split("_", 1)[1]) for g in groups)
    collapse = set(f"decoder_{i}" for i in layer_indices[1:-1]) if len(layer_indices) > _EXPAND_THRESHOLD else set()

    # id -> id it was folded into (itself if not collapsed)
    redirect: dict[str, str] = {}
    view_nodes = []
    order = []

    for n in raw_nodes.values():
        gid = n.get("group_id", "")
        if gid in collapse:
            continue
        redirect[n["id"]] = n["id"]
        view_nodes.append({
            "id": n["id"], "label": n["name"], "type": n["node_type"] or "passthrough",
            "status": n.get("status", "supported"),
            "attributes": n.get("attributes", {}),
            "group": gid, "sourceNodeIds": n.get("source_node_ids", []),
            "weightInfo": (
                {"name": n["weight_name"], "shape": list(n["weight_shape"] or []),
                 "dtype": n["weight_dtype"], "params": n["parameter_count"]}
                if n.get("weight_name") else None
            ),
        })
        order.append(n["id"])

    for gid in collapse:
        members = groups[gid]
        collapsed_id = f"{gid}_collapsed"
        for n in members:
            redirect[n["id"]] = collapsed_id
        entry = next((n for n in members if n["template_id"].endswith(".input_norm")), members[0])
        view_nodes.append({
            "id": collapsed_id, "label": f"Decoder Layer {gid.split('_', 1)[1]} (collapsed)",
            "type": "decoder_layer", "status": "supported",
            "attributes": {"collapsed": True, "member_count": len(members)},
            "group": gid, "sourceNodeIds": sorted({sid for n in members for sid in n.get("source_node_ids", [])}),
            "weightInfo": None,
        })
        order.append(collapsed_id)

    seen_edges = set()
    view_edges = []
    for e in raw_edges:
        src = redirect.get(e["source"])
        tgt = redirect.get(e["target"])
        if not src or not tgt or src == tgt:
            continue
        key = (src, tgt)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        view_edges.append({"source": src, "target": tgt})

    _layout_vertical(view_nodes, view_edges, order)
    return {"nodes": view_nodes, "edges": view_edges}


def build_node_mappings(model_ir, nntrainer_graph) -> list:
    """One mapping record per decoder layer (plus embedding/final-norm/
    lm-head) so the webview can highlight the matching nodes in the
    *other* graph on click, without a dedicated third "mapping" tab."""
    mappings = []
    by_group: dict[str, list] = {}
    ungrouped = []
    for n in nntrainer_graph.get_nodes():
        if n.group_id.startswith("decoder_"):
            by_group.setdefault(n.group_id, []).append(n)
        else:
            ungrouped.append(n)

    for gid, members in sorted(by_group.items(), key=lambda kv: int(kv[0].split("_", 1)[1])):
        source_ids = sorted({sid for n in members for sid in n.source_node_ids})
        mappings.append({
            "sourceIds": source_ids,
            "targetIds": [n.id for n in members],
            "mappingType": "many_to_one" if len(source_ids) < len(members) else "one_to_one",
            "description": f"Decoder layer {gid.split('_', 1)[1]}",
        })

    for n in ungrouped:
        if n.source_node_ids:
            mappings.append({
                "sourceIds": list(n.source_node_ids),
                "targetIds": [n.id],
                "mappingType": "one_to_one",
                "description": n.semantic_type or n.name,
            })

    return mappings


BARYCENTER_SWEEPS = 4


def _layout_vertical(nodes: list, edges: list, order: list):
    """Sugiyama-style layered layout, assigning x/y in place.

    Three stages:
      1. Longest-path layering -> each node's row (y = depth * ROW_HEIGHT).
      2. Barycenter crossing reduction -> the order of nodes *within* each
         row is repeatedly re-sorted by the average position of their
         parents (down sweep) and children (up sweep), which pulls
         connected nodes into vertical alignment and untangles crossings.
      3. Row centering -> narrower rows are centered against the widest
         row, so a single parent sits roughly above the midpoint of its
         children instead of everything being left-packed.

    Nodes with no resolvable depth fall back to row 0; declaration order
    (`order`, then node order) is the tie-breaker so layout is stable and
    deterministic across runs.
    """
    ids = [n["id"] for n in nodes]
    if not ids:
        return
    id_set = set(ids)
    by_id = {n["id"]: n for n in nodes}

    # --- stage 1: longest-path layering ---
    depth = {nid: 0 for nid in ids}
    for _ in range(len(ids) + 1):
        changed = False
        for e in edges:
            src, tgt = e["source"], e["target"]
            if src in id_set and tgt in id_set and depth[tgt] < depth[src] + 1:
                depth[tgt] = depth[src] + 1
                changed = True
        if not changed:
            break

    parents = {nid: [] for nid in ids}
    children = {nid: [] for nid in ids}
    for e in edges:
        src, tgt = e["source"], e["target"]
        if src in id_set and tgt in id_set:
            children[src].append(tgt)
            parents[tgt].append(src)

    # Stable declaration order: honour `order` first, then any stragglers.
    decl = {nid: i for i, nid in enumerate(order) if nid in id_set}
    for i, nid in enumerate(ids):
        decl.setdefault(nid, len(order) + i)

    max_depth = max(depth.values())
    rows = {d: [] for d in range(max_depth + 1)}
    for nid in sorted(ids, key=lambda x: decl[x]):
        rows[depth[nid]].append(nid)

    # --- stage 2: barycenter crossing reduction ---
    pos = {}

    def reindex():
        for d in rows:
            for i, nid in enumerate(rows[d]):
                pos[nid] = i

    reindex()

    def barycenter(nid, neighbours):
        ns = neighbours[nid]
        # No neighbours in the adjacent row -> keep current slot (stable).
        return sum(pos[n] for n in ns) / len(ns) if ns else pos[nid]

    for _ in range(BARYCENTER_SWEEPS):
        for d in range(1, max_depth + 1):                 # down: align to parents
            rows[d].sort(key=lambda nid: (barycenter(nid, parents), decl[nid]))
            reindex()
        for d in range(max_depth - 1, -1, -1):            # up: align to children
            rows[d].sort(key=lambda nid: (barycenter(nid, children), decl[nid]))
            reindex()

    # --- stage 3: assign coordinates, centering each row ---
    widest = max((len(rows[d]) for d in rows), default=1)
    for d in range(max_depth + 1):
        row = rows[d]
        offset = (widest - len(row)) / 2.0
        for i, nid in enumerate(row):
            by_id[nid]["y"] = d * ROW_HEIGHT
            by_id[nid]["x"] = int(round((offset + i) * COL_WIDTH))
