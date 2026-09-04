"""
Validates the lowered nntrainer graph before it reaches the C++
generator. Each decoder layer's checks are independent of every other
layer's -- validating layer 7 needs nothing from layer 3 -- so they
run concurrently on a thread pool instead of a plain for-loop. Results
are collected via `executor.map`, which preserves input order, so
diagnostics come back in layer order without needing a lock: each
worker only ever reads its own layer's nodes and returns a list, it
never touches shared mutable state.

Fails (returns ok=False) rather than letting the C++ generator emit
TODO-filled code silently -- catching a structurally broken graph here
is much cheaper than debugging wrong-but-compiling C++ later.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from api.graph.graph import Graph
from api.semantic.model import CausalLMIR

_MAX_WORKERS = 8


@dataclass
class Diagnostic:
    severity: str  # "error" | "warning"
    layer_index: int | None
    message: str

    def to_dict(self) -> dict:
        return {"severity": self.severity, "layer_index": self.layer_index, "message": self.message}


class ValidationResult:
    def __init__(self, diagnostics: list[Diagnostic]):
        self.diagnostics = diagnostics

    @property
    def ok(self) -> bool:
        return not any(d.severity == "error" for d in self.diagnostics)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "diagnostics": [d.to_dict() for d in self.diagnostics]}


def _nodes_by_group(graph: Graph, group_id: str):
    return [n for n in graph.get_nodes() if n.group_id == group_id]


def _validate_layer(graph: Graph, layer_index: int) -> list[Diagnostic]:
    group_id = f"decoder_{layer_index}"
    nodes = _nodes_by_group(graph, group_id)
    by_type = {}
    for n in nodes:
        by_type.setdefault(n.node_type, []).append(n)
    diags: list[Diagnostic] = []

    def require(node_type, what):
        if node_type not in by_type:
            diags.append(Diagnostic("error", layer_index, f"decoder layer {layer_index} missing {what}"))

    require("fully_connected", "linear projection(s)")
    require("mha_core", "attention core")
    require("addition", "residual add(s)")

    wq = [n for n in nodes if n.template_id.endswith(".wq")]
    wk = [n for n in nodes if n.template_id.endswith(".wk")]
    wv = [n for n in nodes if n.template_id.endswith(".wv")]
    if wq and wk and wv:
        q_in, k_in, v_in = wq[0].inputs, wk[0].inputs, wv[0].inputs
        if not (q_in == k_in == v_in):
            diags.append(Diagnostic(
                "error", layer_index,
                f"decoder layer {layer_index}: Q, K and V projections must share the same input",
            ))
    else:
        diags.append(Diagnostic("error", layer_index, f"decoder layer {layer_index} missing Q/K/V branches"))

    residual_adds = [n for n in nodes if n.node_type == "addition"]
    if len(residual_adds) != 2:
        diags.append(Diagnostic(
            "error", layer_index,
            f"decoder layer {layer_index} has {len(residual_adds)} residual add(s), expected 2",
        ))

    norms = [n for n in nodes if n.node_type in ("rms_norm", "reshaped_rms_norm")]
    input_norms = [n for n in norms if n.template_id.endswith(".input_norm")]
    post_norms = [n for n in norms if n.template_id.endswith(".post_attention_norm")]
    if not input_norms:
        diags.append(Diagnostic("error", layer_index, f"decoder layer {layer_index} missing input norm"))
    if not post_norms:
        diags.append(Diagnostic("error", layer_index, f"decoder layer {layer_index} missing post-attention norm"))

    wo = [n for n in nodes if n.template_id.endswith(".wo")]
    if not wo:
        diags.append(Diagnostic("error", layer_index, f"decoder layer {layer_index} missing attention output projection"))

    if any(n.template_id.endswith(".gate") for n in nodes):
        gate = [n for n in nodes if n.template_id.endswith(".gate")]
        up = [n for n in nodes if n.template_id.endswith(".up")]
        mul = [n for n in nodes if n.node_type == "multiply"]
        if not (gate and up and mul):
            diags.append(Diagnostic(
                "error", layer_index,
                f"decoder layer {layer_index}: gated MLP missing gate/up/multiply branch",
            ))

    return diags


def _validate_acyclic(graph: Graph) -> list[Diagnostic]:
    """Excludes host_managed (KV-cache) edges, which are explicitly
    stateful across timesteps and not part of the single-pass DAG."""
    adjacency = {n.id: [] for n in graph.get_nodes()}
    for e in graph.get_edges():
        src_node = graph.get_node(e.source)
        if src_node and src_node.status == "host_managed":
            continue
        adjacency.setdefault(e.source, []).append(e.target)

    visiting, visited = set(), set()

    def dfs(node_id):
        if node_id in visited:
            return False
        if node_id in visiting:
            return True
        visiting.add(node_id)
        for nxt in adjacency.get(node_id, []):
            if dfs(nxt):
                return True
        visiting.discard(node_id)
        visited.add(node_id)
        return False

    for n in graph.get_nodes():
        if dfs(n.id):
            return [Diagnostic("error", None, "target graph contains a cycle outside of KV-cache edges")]
    return []


def validate(graph: Graph, model_ir: CausalLMIR) -> ValidationResult:
    diagnostics: list[Diagnostic] = []

    if not model_ir.decoder_layers:
        diagnostics.append(Diagnostic("error", None, "model has no decoder layers"))
    else:
        with ThreadPoolExecutor(
            max_workers=min(_MAX_WORKERS, len(model_ir.decoder_layers)),
            thread_name_prefix="nntrainer-validate-layer",
        ) as pool:
            per_layer = pool.map(lambda i: _validate_layer(graph, i), range(len(model_ir.decoder_layers)))
            for layer_diags in per_layer:
                diagnostics.extend(layer_diags)

    if not any(n.template_id == "final.final_norm" for n in graph.get_nodes()):
        diagnostics.append(Diagnostic("error", None, "missing final norm"))
    if not any(n.semantic_type == "lm_head" for n in graph.get_nodes()):
        diagnostics.append(Diagnostic("error", None, "missing lm head"))

    diagnostics.extend(_validate_acyclic(graph))

    unsupported = [n for n in graph.get_nodes() if n.status == "unsupported"]
    for n in unsupported:
        diagnostics.append(Diagnostic("error", None, f"unsupported node reached target graph: {n.name}"))

    return ValidationResult(diagnostics)
