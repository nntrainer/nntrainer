# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file graph_views.py
# @brief Graph Views - Builds Model Graph and nntrainer Graph for webview.
# @author prachi-t17 <prachi.t@samsung.com>

"""
Graph Views - Builds Model Graph and nntrainer Graph for webview.

Every node id these builders emit is the id the webview pane renders, and
the node-mapping table uses those same ids on both sides, so a click in
one pane can find its counterparts in the other. Keep this module
stdlib-only: tests load it standalone (see tests/test_layout.py).
"""
import re
from typing import Optional

ROW_HEIGHT = 130
COL_WIDTH = 200


WEIGHT_RE = re.compile(
    r'weight:\s*name=(\S+)\s*shape=(\[[^\]]*\])\s*dtype=(\S+)\s*params=(\d+)',
    re.IGNORECASE,
)


def _parse_weight_comment(line: str):
    match = WEIGHT_RE.search(line)
    if not match:
        return None
    name, shape_str, dtype, params = match.groups()
    try:
        shape = [int(x.strip()) for x in shape_str.strip("[]").split(",") if x.strip()]
    except ValueError:
        shape = []
    return {"name": name, "shape": shape, "dtype": dtype, "params": int(params)}


# pylint: disable-next=too-many-locals,too-many-branches
def build_ini_graph(ini_content: str) -> dict:
    """Graph view (nodes, edges) of a model.ini file."""
    if not ini_content:
        return {"nodes": [], "edges": []}

    nodes = []
    edges = []
    order = []

    blocks = re.split(r"^\[(.+?)\]\s*$", ini_content, flags=re.MULTILINE)
    items_iter = iter(blocks[1:])
    for name, body in zip(items_iter, items_iter):
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
                weight_comment = _parse_weight_comment(line)
                if weight_comment:
                    weight_info = weight_comment
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


# pylint: disable-next=too-many-locals,too-many-branches,too-many-statements
def build_cpp_graph(cpp_code: str) -> dict:
    """Graph view (nodes, edges) of a generated_model.cpp file."""
    if not cpp_code:
        return {"nodes": [], "edges": []}

    nodes = []
    edges = []
    order = []

    var_type = {}
    var_attrs = {}
    var_name = {}
    var_weight = {}

    create_re = re.compile(r'auto\s+(\w+)\s*=\s*createLayer\("([^"]+)"\);')
    setprop_re = re.compile(r'(\w+)->setProperty\(\{"([^=]+)=([^"]*)"\}\);')
    todo_re = re.compile(r'//\s*TODO\(unsupported\):\s*(\S+)\s*\[([^\]]+)\]\s*--\s*(.*)')

    lines = cpp_code.splitlines()
    current_unsupported = None

    for line in lines:
        match = todo_re.search(line)
        if match:
            unsup_name, unsup_type, reason = match.group(1), match.group(2), match.group(3)
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

        weight_comment = _parse_weight_comment(line)
        if weight_comment:
            if current_unsupported:
                for view_node in nodes:
                    if view_node["id"] == current_unsupported:
                        view_node["weightInfo"] = weight_comment
                        break
            continue

        match = create_re.search(line)
        if match:
            var, layer_type = match.group(1), match.group(2)
            var_type[var] = layer_type
            var_attrs[var] = {}
            current_unsupported = None
            continue

        match = setprop_re.search(line)
        if match:
            var, key, value = match.group(1), match.group(2), match.group(3)
            if key == "name":
                var_name[var] = value
            elif key == "input_layers":
                var_attrs.setdefault(var, {})["__input_layers__"] = value
            else:
                var_attrs.setdefault(var, {})[key] = value
            continue

    current_var = None
    for line in lines:
        match = create_re.search(line)
        if match:
            current_var = match.group(1)
            continue
        if todo_re.search(line):
            current_var = None
            continue
        weight_comment = _parse_weight_comment(line)
        if weight_comment and current_var:
            var_weight[current_var] = weight_comment

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


_EXPAND_THRESHOLD = 3

#: Placeholder for the token-count dimension in display shapes.
SEQ = "seq"


def collapsed_layer_indices(indices) -> set:
    """Decoder layers drawn as a single collapsed box: every layer except
    the first and the last, once there are more than _EXPAND_THRESHOLD.
    Both view builders and the mapping table use this, so the two panes
    always collapse exactly the same layers."""
    ordered = sorted(indices)
    if len(ordered) <= _EXPAND_THRESHOLD:
        return set()
    return set(ordered[1:-1])


def _weight_info(name, shape, params, dtype=None, note=None) -> Optional[dict]:
    if not name:
        return None
    info = {"name": name, "shape": list(shape or []), "dtype": dtype, "params": int(params or 0)}
    if note:
        info["note"] = note
    return info


def _proj_shape_params(proj):
    """HF nn.Linear weight layout [out, in] and its parameter count; just
    [out] and 0 when the input size isn't known."""
    if proj.input_size is None:
        return [proj.output_size], 0
    params = proj.output_size * proj.input_size + (proj.output_size if proj.bias else 0)
    return [proj.output_size, proj.input_size], params


def _proj_attrs(proj) -> dict:
    attrs = {"hf_module": proj.source_name}
    if proj.input_size is not None:
        attrs["in_features"] = proj.input_size
    attrs["out_features"] = proj.output_size
    attrs["bias"] = proj.bias
    return attrs


def _norm_attrs(norm) -> dict:
    return {
        "hf_module": norm.source_name, "norm_type": norm.norm_type,
        "epsilon": norm.epsilon, "feature_size": norm.feature_size,
    }


def _layer_param_count(layer) -> int:
    attn, mlp = layer.attention, layer.mlp
    projections = [attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj, mlp.up_proj, mlp.down_proj]
    if mlp.gate_proj is not None:
        projections.append(mlp.gate_proj)
    norms = [layer.input_norm, layer.post_attention_norm, attn.q_norm, attn.k_norm]
    return (sum(_proj_shape_params(p)[1] for p in projections)
            + sum(n.feature_size for n in norms if n is not None))


# pylint: disable-next=too-many-locals
def build_model_graph_view(model_ir, default_dtype: Optional[str] = None) -> dict:
    """The "Model Graph" pane for architectures with a semantic adapter.
    `default_dtype` is the checkpoint's declared dtype (HF config
    torch_dtype); the weight-preview step replaces it with each tensor's
    real dtype once the weights are on disk."""
    nodes, edges, order = [], [], []
    hidden = model_ir.hidden_size

    # pylint: disable-next=too-many-arguments
    def node(id_, label, type_, semantic_type, group_id="", source_ids=None, attrs=None,
             in_shape=None, out_shape=None, weight=None):
        node_info = {
            "id": id_, "label": label, "type": type_, "status": "mapped",
            "semanticType": semantic_type, "group": group_id,
            "sourceNodeIds": source_ids or [], "attributes": attrs or {},
            "inputShape": list(in_shape) if in_shape else None,
            "outputShape": list(out_shape) if out_shape else None,
            "weightInfo": weight,
        }
        nodes.append(node_info)
        order.append(id_)
        return node_info

    def weight(name, shape, params, note=None):
        return _weight_info(name, shape, params, default_dtype, note)

    embedding_weight = f"{model_ir.embedding_name}.weight"
    prev_id = node(
        "embedding",
        "Embedding",
        "embedding",
        "embedding",
        source_ids=[model_ir.embedding_name],
        attrs={
            "hf_module": model_ir.embedding_name,
            "vocab_size": model_ir.vocab_size,
            "hidden_size": hidden,
        },
        in_shape=[SEQ],
        out_shape=[SEQ, hidden],
        weight=weight(
            embedding_weight, [model_ir.vocab_size, hidden], model_ir.vocab_size * hidden
        ),
    )["id"]

    collapsed = collapsed_layer_indices(layer.index for layer in model_ir.decoder_layers)
    for layer in model_ir.decoder_layers:
        i = layer.index
        if i not in collapsed:
            entry_id, exit_id = _build_expanded_layer(node, weight, edges, layer, i, hidden)
        else:
            gid = f"decoder_{i}"
            member_source_ids = [
                layer.input_norm.source_name, layer.attention.source_name,
                layer.post_attention_norm.source_name, layer.mlp.source_name,
            ]
            new_node = node(
                f"{gid}_collapsed",
                f"Decoder Layer {i}",
                "decoder_layer",
                "decoder_layer",
                group_id=gid,
                source_ids=member_source_ids,
                attrs={
                    "collapsed": True,
                    "repeat_index": i,
                    "parameters": _layer_param_count(layer),
                },
                in_shape=[SEQ, hidden],
                out_shape=[SEQ, hidden],
            )
            entry_id = exit_id = new_node["id"]
        edges.append({"source": prev_id, "target": entry_id})
        prev_id = exit_id

    final = model_ir.final_norm
    final_id = node(
        "final_norm", "Final Norm", final.norm_type, "normalization",
        source_ids=[final.source_name], attrs=_norm_attrs(final),
        in_shape=[SEQ, hidden], out_shape=[SEQ, hidden],
        weight=weight(f"{final.source_name}.weight", [final.feature_size], final.feature_size),
    )["id"]
    edges.append({"source": prev_id, "target": final_id})

    tied = model_ir.tied_embeddings
    lm_head_id = node(
        "lm_head", "LM Head", "fully_connected", "lm_head",
        source_ids=[model_ir.lm_head_name],
        attrs={"hf_module": model_ir.lm_head_name, "in_features": hidden,
               "out_features": model_ir.vocab_size, "tied_embeddings": tied},
        in_shape=[SEQ, hidden], out_shape=[SEQ, model_ir.vocab_size],
        weight=weight(
            embedding_weight if tied else f"{model_ir.lm_head_name}.weight",
            [model_ir.vocab_size, hidden],
            0 if tied else model_ir.vocab_size * hidden,
            note="shared with the embedding (tied)" if tied else None,
        ),
    )["id"]
    edges.append({"source": final_id, "target": lm_head_id})

    _layout_vertical(nodes, edges, order)
    return {"nodes": nodes, "edges": edges}


# pylint: disable-next=too-many-locals,too-many-arguments,too-many-statements
def _build_expanded_layer(node_fn, weight_fn, edges, layer, index, hidden):
    gid = f"decoder_{index}"
    attn, mlp = layer.attention, layer.mlp
    hidden_shape = [SEQ, hidden]

    def proj_node(tag, label, semantic_type, proj):
        shape, params = _proj_shape_params(proj)
        in_size = proj.input_size if proj.input_size is not None else hidden
        return node_fn(
            f"{gid}.{tag}", label, "fully_connected", semantic_type, gid, [proj.source_name],
            attrs=_proj_attrs(proj), in_shape=[SEQ, in_size], out_shape=[SEQ, proj.output_size],
            weight=weight_fn(proj.weight_name or f"{proj.source_name}.weight", shape, params),
        )["id"]

    # pylint: disable-next=too-many-arguments
    def norm_node(tag, label, type_, norm, shape, per_head=False):
        attrs = _norm_attrs(norm)
        if per_head:
            attrs["per_head"] = True
        return node_fn(
            f"{gid}.{tag}", label, type_, "normalization", gid, [norm.source_name],
            attrs=attrs, in_shape=shape, out_shape=shape,
            weight=weight_fn(f"{norm.source_name}.weight", [norm.feature_size], norm.feature_size),
        )["id"]

    input_norm = norm_node("input_norm", "Input RMSNorm", layer.input_norm.norm_type,
                           layer.input_norm, hidden_shape)

    q_node = proj_node("wq", "Q Projection", "attention", attn.q_proj)
    k_node = proj_node("wk", "K Projection", "attention", attn.k_proj)
    v_node = proj_node("wv", "V Projection", "attention", attn.v_proj)
    edges += [
        {"source": input_norm, "target": q_node},
        {"source": input_norm, "target": k_node},
        {"source": input_norm, "target": v_node},
    ]

    q_in, k_in = q_node, k_node
    if attn.q_norm is not None:
        q_in = norm_node("q_norm", "Q Norm", "reshaped_rms_norm", attn.q_norm,
                         [SEQ, attn.q_proj.output_size], per_head=True)
        edges.append({"source": q_node, "target": q_in})
    if attn.k_norm is not None:
        k_in = norm_node("k_norm", "K Norm", "reshaped_rms_norm", attn.k_norm,
                         [SEQ, attn.k_proj.output_size], per_head=True)
        edges.append({"source": k_node, "target": k_in})

    attn_attrs = {
        "hf_module": attn.source_name,
        "num_heads": attn.num_heads,
        "num_kv_heads": attn.num_kv_heads,
        "head_dim": attn.head_dim,
        "rope_theta": attn.rope_theta,
        "max_position_embeddings": attn.max_position_embeddings,
        "causal": attn.causal,
    }
    if attn.sliding_window is not None:
        attn_attrs["sliding_window"] = attn.sliding_window
    attn_out = node_fn(
        f"{gid}.attention", "Attention Output", "mha_core", "attention", gid, [attn.source_name],
        attrs=attn_attrs, in_shape=[SEQ, attn.q_proj.output_size],
        out_shape=[SEQ, attn.num_heads * attn.head_dim],
    )["id"]
    for src in (q_in, k_in, v_node):
        edges.append({"source": src, "target": attn_out})

    o_proj = proj_node("wo", "Output Projection", "attention", attn.o_proj)
    edges.append({"source": attn_out, "target": o_proj})

    res1 = node_fn(f"{gid}.attention_residual", "Residual Add", "addition", "residual", gid, [],
                   attrs={"adds": "layer input + attention output"},
                   in_shape=hidden_shape, out_shape=hidden_shape)["id"]
    edges.append({"source": o_proj, "target": res1})

    post_norm = norm_node(
        "post_attention_norm",
        "Post-Attention RMSNorm",
        layer.post_attention_norm.norm_type,
        layer.post_attention_norm,
        hidden_shape,
    )
    edges.append({"source": res1, "target": post_norm})

    intermediate_shape = [SEQ, mlp.up_proj.output_size]
    if mlp.gated:
        gate = proj_node("gate", "Gate Projection", "mlp", mlp.gate_proj)
        up_proj = proj_node("up", "Up Projection", "mlp", mlp.up_proj)
        edges += [{"source": post_norm, "target": gate}, {"source": post_norm, "target": up_proj}]
        act = node_fn(
            f"{gid}.activation",
            mlp.activation,
            mlp.activation,
            "mlp",
            gid,
            [],
            attrs={"activation": mlp.activation},
            in_shape=[SEQ, mlp.gate_proj.output_size],
            out_shape=[SEQ, mlp.gate_proj.output_size],
        )["id"]
        edges.append({"source": gate, "target": act})
        mul = node_fn(f"{gid}.gate_mul", "Multiply", "multiply", "mlp", gid, [],
                      attrs={"multiplies": f"{mlp.activation}(gate) * up"},
                      in_shape=intermediate_shape, out_shape=intermediate_shape)["id"]
        edges += [{"source": act, "target": mul}, {"source": up_proj, "target": mul}]
        mlp_out_src = mul
    else:
        up_proj = proj_node("up", "Up Projection", "mlp", mlp.up_proj)
        edges.append({"source": post_norm, "target": up_proj})
        act = node_fn(f"{gid}.activation", mlp.activation, mlp.activation, "mlp", gid, [],
                      attrs={"activation": mlp.activation},
                      in_shape=intermediate_shape, out_shape=intermediate_shape)["id"]
        edges.append({"source": up_proj, "target": act})
        mlp_out_src = act

    down = proj_node("down", "Down Projection", "mlp", mlp.down_proj)
    edges.append({"source": mlp_out_src, "target": down})

    res2 = node_fn(f"{gid}.mlp_residual", "Residual Add", "addition", "residual", gid, [],
                   attrs={"adds": "attention residual + MLP output"},
                   in_shape=hidden_shape, out_shape=hidden_shape)["id"]
    edges.append({"source": down, "target": res2})

    return input_norm, res2


def _decoder_groups(group_ids) -> dict:
    """{"decoder_<i>": i} for every decoder-layer group id in `group_ids`."""
    return {g: int(g.split("_", 1)[1]) for g in group_ids if g.startswith("decoder_")}


def _collapsed_groups(layer_of: dict) -> set:
    collapsed = collapsed_layer_indices(layer_of.values())
    return {g for g, i in layer_of.items() if i in collapsed}


def _shape(value) -> Optional[list]:
    # The IR holds tuples until it is serialised; the views always use lists.
    return list(value) if value else None


# pylint: disable-next=too-many-locals
def build_nntrainer_graph_view(
    nntrainer_graph_ir: dict, default_dtype: Optional[str] = None
) -> dict:
    """The nntrainer Graph pane's view of the lowered graph."""
    raw_nodes = {n["id"]: n for n in nntrainer_graph_ir.get("nodes", [])}
    raw_edges = nntrainer_graph_ir.get("edges", [])

    groups: dict[str, list] = {}
    for raw_node in raw_nodes.values():
        if raw_node.get("group_id", "").startswith("decoder_"):
            groups.setdefault(raw_node["group_id"], []).append(raw_node)

    layer_of = _decoder_groups(groups)
    collapse = _collapsed_groups(layer_of)

    redirect: dict[str, str] = {}
    view_nodes = []
    order = []

    for raw_node in raw_nodes.values():
        gid = raw_node.get("group_id", "")
        if gid in collapse:
            continue
        redirect[raw_node["id"]] = raw_node["id"]

        tied_lm_head = (
            raw_node.get("semantic_type") == "lm_head" and raw_node.get("status") == "fused"
        )
        weight_info = _weight_info(
            raw_node.get("weight_name"),
            raw_node.get("weight_shape"),
            raw_node.get("parameter_count"),
            raw_node.get("weight_dtype") or default_dtype,
            note="shared with the embedding (tied)" if tied_lm_head else None,
        )

        view_nodes.append(
            {
                "id": raw_node["id"],
                "label": raw_node["name"],
                "type": raw_node["node_type"] or "passthrough",
                "status": raw_node.get("status", "supported"),
                "semanticType": raw_node.get("semantic_type", ""),
                "attributes": raw_node.get("attributes", {}),
                "inputShape": _shape(raw_node.get("input_shape")),
                "outputShape": _shape(raw_node.get("output_shape")),
                "group": gid,
                "sourceNodeIds": raw_node.get("source_node_ids", []),
                "weightInfo": weight_info,
            }
        )
        order.append(raw_node["id"])

    for gid in sorted(collapse, key=layer_of.get):
        members = groups[gid]
        collapsed_id = f"{gid}_collapsed"
        for raw_node in members:
            redirect[raw_node["id"]] = collapsed_id
        entry = next((n for n in members if n["template_id"].endswith(".input_norm")), members[0])
        exit_ = next(
            (n for n in members if n["template_id"].endswith(".mlp_residual")), members[-1]
        )
        view_nodes.append(
            {
                "id": collapsed_id,
                "label": f"Decoder Layer {layer_of[gid]} (collapsed)",
                "type": "decoder_layer",
                "status": "supported",
                "semanticType": "decoder_layer",
                "attributes": {
                    "collapsed": True,
                    "member_count": len(members),
                    "parameters": sum(n.get("parameter_count") or 0 for n in members),
                },
                "inputShape": _shape(entry.get("input_shape")),
                "outputShape": _shape(exit_.get("output_shape")),
                "group": gid,
                "sourceNodeIds": sorted(
                    {sid for n in members for sid in n.get("source_node_ids", [])}
                ),
                "weightInfo": None,
            }
        )
        order.append(collapsed_id)

    seen_edges = set()
    view_edges = []
    for raw_edge in raw_edges:
        src = redirect.get(raw_edge["source"])
        tgt = redirect.get(raw_edge["target"])
        if not src or not tgt or src == tgt:
            continue
        key = (src, tgt)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        view_edges.append({"source": src, "target": tgt})

    _layout_vertical(view_nodes, view_edges, order)
    return {"nodes": view_nodes, "edges": view_edges}


#: Lowerer template tag -> Model Graph node-id suffix, where the two differ.
#: The KV cache has no box of its own in the Model Graph; it belongs to
#: the attention box.
_MODEL_VIEW_TAGS = {
    "mha_core": "attention",
    "kv_cache": "attention",
    "act": "activation",
    "mul": "gate_mul",
}


def _model_view_id(node, collapsed_groups) -> Optional[str]:
    """The Model Graph pane id for a lowered nntrainer node (see
    build_model_graph_view for how those ids are built)."""
    gid = node.group_id
    if gid.startswith("decoder_"):
        if gid in collapsed_groups:
            return f"{gid}_collapsed"
        tag = node.template_id.split(".", 1)[1] if "." in node.template_id else ""
        return f"{gid}.{_MODEL_VIEW_TAGS.get(tag, tag)}" if tag else None
    if node.semantic_type == "embedding":
        return "embedding"
    if gid == "final":
        return "final_norm"
    if gid == "lm_head":
        return "lm_head"
    return None


def build_node_mappings(nntrainer_graph, model_view: Optional[dict] = None) -> list:
    """Click-to-highlight table between the two panes: one entry per Model
    Graph box, listing the nntrainer Graph boxes it lowers to. Both sides
    use the ids the panes render (Model Graph ids such as "decoder_0.wq",
    the nntrainer pane's node ids, "decoder_<i>_collapsed" for collapsed
    layers), so the webview can match them directly. Pass `model_view` to
    drop any entry whose Model Graph box doesn't exist."""
    layer_of = _decoder_groups({n.group_id for n in nntrainer_graph.get_nodes()})
    collapsed = _collapsed_groups(layer_of)
    known = {n["id"] for n in model_view["nodes"]} if model_view else set()

    targets_by_source: dict[str, list] = {}
    for node in nntrainer_graph.get_nodes():
        source = _model_view_id(node, collapsed)
        if source is None or (model_view and source not in known):
            continue
        target = f"{node.group_id}_collapsed" if node.group_id in collapsed else node.id
        targets = targets_by_source.setdefault(source, [])
        if target not in targets:
            targets.append(target)

    return [
        {
            "sourceIds": [source],
            "targetIds": targets,
            "mappingType": "one_to_many" if len(targets) > 1 else "one_to_one",
            "description": source,
        }
        for source, targets in targets_by_source.items()
    ]


def apply_weight_previews(view: Optional[dict], previews: dict) -> int:
    """Merge per-tensor statistics (see agents/weight_preview.py) into a
    pane's weightInfo blocks in place. Returns how many nodes got one."""
    count = 0
    for node in (view or {}).get("nodes", []):
        info = node.get("weightInfo")
        stats = previews.get(info["name"]) if info else None
        if stats:
            info.update(stats)
            count += 1
    return count


BARYCENTER_SWEEPS = 4


# pylint: disable-next=too-many-locals,too-many-branches
def _layout_vertical(nodes: list, edges: list, order: list):
    ids = [n["id"] for n in nodes]
    if not ids:
        return
    id_set = set(ids)
    by_id = {n["id"]: n for n in nodes}

    depth = {nid: 0 for nid in ids}
    for _ in range(len(ids) + 1):
        changed = False
        for edge in edges:
            src, tgt = edge["source"], edge["target"]
            if src in id_set and tgt in id_set and depth[tgt] < depth[src] + 1:
                depth[tgt] = depth[src] + 1
                changed = True
        if not changed:
            break

    parents = {nid: [] for nid in ids}
    children = {nid: [] for nid in ids}
    for edge in edges:
        src, tgt = edge["source"], edge["target"]
        if src in id_set and tgt in id_set:
            children[src].append(tgt)
            parents[tgt].append(src)

    decl = {nid: i for i, nid in enumerate(order) if nid in id_set}
    for i, nid in enumerate(ids):
        decl.setdefault(nid, len(order) + i)

    max_depth = max(depth.values())
    rows = {d: [] for d in range(max_depth + 1)}
    for nid in sorted(ids, key=lambda x: decl[x]):
        rows[depth[nid]].append(nid)

    pos = {}

    def reindex():
        for data in rows:
            for i, nid in enumerate(rows[data]):
                pos[nid] = i

    reindex()

    def barycenter(nid, neighbours):
        neighbour_ids = neighbours[nid]
        return (
            sum(pos[n] for n in neighbour_ids) / len(neighbour_ids) if neighbour_ids else pos[nid]
        )

    for _ in range(BARYCENTER_SWEEPS):
        for data in range(1, max_depth + 1):
            rows[data].sort(key=lambda nid: (barycenter(nid, parents), decl[nid]))
            reindex()
        for data in range(max_depth - 1, -1, -1):
            rows[data].sort(key=lambda nid: (barycenter(nid, children), decl[nid]))
            reindex()

    widest = max((len(rows[d]) for d in rows), default=1)
    for data in range(max_depth + 1):
        row = rows[data]
        offset = (widest - len(row)) / 2.0
        for i, nid in enumerate(row):
            by_id[nid]["y"] = data * ROW_HEIGHT
            by_id[nid]["x"] = int(round((offset + i) * COL_WIDTH))
