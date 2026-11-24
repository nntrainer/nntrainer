# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file generic_fx_parser.py
# @brief GenericFxParser (module-tree walker with best-effort dataflow tracing).
# @author prachi-t17 <prachi.t@samsung.com>

"""
GenericFxParser (module-tree walker with best-effort dataflow tracing).

Builds the Compiler IR by walking the model's module tree directly,
instead of relying on torch.fx.symbolic_trace. This avoids the
data-dependent control flow limitation and works with BERT, GPT,
and similar models that have dynamic branches (if statements, etc.).

Edges between layers are recovered in two tiers:

  1. Best-effort dataflow tracing. During the shape-inference forward
     pass, per-leaf forward hooks record which tensor each leaf module
     consumes and produces (by object identity). After the pass, a
     consumer whose input tensor was produced by another leaf becomes a
     real edge -- walking back through passthrough leaves (Dropout, etc.)
     to the nearest emitted ancestor. This recovers genuine branches
     (module-to-module skips) that a linear walk cannot see.

  2. Linear fallback. Hook tracing only sees tensors that flow directly
     module -> module. Functional ops between modules (a residual `x +
     sublayer(x)`, a `torch.cat`, an activation applied as a function)
     produce fresh tensors no module "owns", so those edges are invisible
     to hooks. Rather than ship a *partial* graph (less connected than a
     linear chain), tracing is adopted ONLY when it fully connects the
     emitted nodes (every emitted node except the single execution-order
     root has a resolved predecessor). Otherwise, or on any tracing error,
     the nodes are wired as a linear chain in declaration order. Net
     effect: never worse than the linear chain, strictly better when
     tracing is complete.

Fully recovering functional-op dataflow (transformer residuals/concats)
needs tensor-level interception (a TorchDispatch / __torch_function__
mode), which is a larger change not made here.
"""
import logging

# torch is an optional engine package: this parser runs only when the model is loaded with it
import torch  # pylint: disable=import-error
from api.adapters.registry import KNOWN_MODEL_TYPES
from api.compatibility.op_table import classify_op
from api.graph.graph import Graph
from api.graph.node import GraphNode
from api.parsers.edge_resolver import resolve_traced_edges

logger = logging.getLogger(__name__)

#: Tokens in the dummy forward pass used to infer shapes and dataflow.
SHAPE_PROBE_SEQ_LEN = 16


class TracingError(Exception):
    """Raised when we can't build a graph from the model."""


class TransformerSemanticLoweringRequired(TracingError):
    """
    Raised instead of falling back to a linear chain when the model is a
    *known* transformer causal-LM architecture (one with a registered
    ArchitectureAdapter) and dataflow tracing didn't fully connect the
    emitted nodes.

    A linear q_proj -> k_proj -> v_proj -> o_proj chain is a plausible-
    looking but mathematically wrong graph for these architectures --
    it would silently generate C++ that compiles and runs but computes
    the wrong thing. For known architectures the correct move is to
    not treat declaration order as dataflow and hand off to the
    semantic adapter, which reads the real Q/K/V/attention/MLP
    structure straight from the module tree instead of guessing it
    from hook-observed tensor identity.

    This is deliberately narrow: unknown/non-transformer architectures
    still get the conservative sequential fallback below.
    """


# pylint: disable-next=too-many-instance-attributes,too-few-public-methods
class GenericFxParser:
    """Walks a PyTorch model's module tree into a compiler IR graph."""
    def __init__(self, model, model_name: str = "model"):
        self.model = model
        self.model_name = model_name
        self.module_to_node = {}  # module -> GraphNode
        self.input_shapes = {}    # module_name -> input shape
        self.output_shapes = {}   # module_name -> output shape

        # --- dataflow tracing state (populated by hooks) ---
        self.hooks = []
        self._leaf_names = set()          # names of leaf modules
        self._producer = {}               # id(tensor) -> producing leaf name
        self._keepalive = []              # refs to keep output tensor ids stable
        self._traced_inputs = {}          # leaf name -> [producer leaf names]
        self._exec_order = []             # leaf names in hook-fire order (first seen)
        self.shape_inference_error = None
        self.edge_source = "linear"       # "traced" or "linear", for diagnostics

    # ------------------------------------------------------------------ parse
    def parse(self) -> Graph:
        """Walk the module tree and build a graph."""
        graph = Graph()
        graph.model_name = self.model_name
        graph.architecture = type(self.model).__name__ or "unknown"

        try:
            self._register_hooks()
            self._infer_shapes()
        finally:
            self._unregister_hooks()

        # Build nodes for leaf modules that emit a layer. Collect them in
        # creation order first; wiring strategy is decided afterwards.
        emitted_nodes = []      # GraphNode, in creation order
        name_to_node = {}       # leaf name -> GraphNode

        for name, module in self.model.named_modules():
            if not name:  # Skip the root module itself
                continue
            if any(True for _ in module.children()):
                continue  # not a leaf -- it's a container, skip it

            module_type = type(module).__name__
            capability = classify_op(module_type)
            if not capability.get("emits_layer", False):
                continue

            node = GraphNode(
                name=name,
                node_type=capability.get("nntrainer_type") or module_type,
                attributes=dict(capability.get("attributes", {})),
                metadata={"module_type": module_type},
            )
            node.supported = capability.get("supported", False)
            node.compatibility_reason = capability.get("reason", "")

            if name in self.input_shapes:
                node.input_shape = self.input_shapes[name]
            if name in self.output_shapes:
                node.output_shape = self.output_shapes[name]

            params = list(module.named_parameters(recurse=False))
            if params:
                main_name, main_param = max(params, key=lambda kv: kv[1].numel())
                node.weight_name = main_name
                node.weight_shape = tuple(main_param.shape)
                node.weight_dtype = str(main_param.dtype)
                node.parameter_count = sum(p.numel() for _, p in params)

            graph.add_node(node)
            self.module_to_node[name] = node
            name_to_node[name] = node
            emitted_nodes.append(node)

        self._wire_edges(graph, emitted_nodes, name_to_node)
        return graph

    # ------------------------------------------------------------- edge wiring
    def _wire_edges(self, graph, emitted_nodes, name_to_node):
        """Adopt traced edges if they fully connect the emitted nodes;
        otherwise fall back to the linear chain. Never leaves the graph
        less connected than the linear chain."""
        emitted_names = set(name_to_node.keys())

        traced_ok = False
        if self._exec_order and self._traced_inputs:
            try:
                edges, orphans = resolve_traced_edges(
                    self._exec_order, emitted_names, self._traced_inputs
                )
                if edges and not orphans:
                    for src, tgt in edges:
                        graph.connect(name_to_node[src], name_to_node[tgt])
                    traced_ok = True
                    self.edge_source = "traced"
                    logger.info(
                        "Dataflow tracing recovered %d edges (fully connected)",
                        len(edges),
                    )
                else:
                    logger.info(
                        "Dataflow tracing incomplete (%d orphan node(s)) -- "
                        "falling back to linear chain",
                        len(orphans),
                    )
            # tracing must never break the build
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Dataflow tracing failed (%s) -- linear chain", exc)

        if not traced_ok:
            if self._is_known_transformer_architecture():
                raise TransformerSemanticLoweringRequired(
                    f"Incomplete dataflow tracing for '{self.model_name}' "
                    f"(model_type={self._model_type()!r}): the hook-based tracer "
                    "can't see functional-op edges (residual adds, concats), and "
                    "a declaration-order linear chain would be a mathematically "
                    "incorrect graph for a known transformer architecture. "
                    "Semantic adapter lowering is required -- see api/adapters."
                )
            self._wire_conservative_sequential_fallback(graph, emitted_nodes)

    def _wire_conservative_sequential_fallback(self, graph, emitted_nodes):
        """Declaration-order linear chain. Only used for architectures with
        no registered semantic adapter -- never used for a known
        transformer architecture (see TransformerSemanticLoweringRequired)."""
        self.edge_source = "linear"
        prev = None
        for node in emitted_nodes:
            if prev is not None:
                graph.connect(prev, node)
            prev = node

    def _model_type(self):
        config = getattr(self.model, "config", None)
        return getattr(config, "model_type", None)

    def _is_known_transformer_architecture(self) -> bool:
        return self._model_type() in KNOWN_MODEL_TYPES

    # ------------------------------------------------------------------- hooks
    @staticmethod
    def _iter_tensors(obj):
        """Yield torch.Tensors from a possibly-nested input/output structure."""
        if isinstance(obj, torch.Tensor):
            yield obj
        elif isinstance(obj, (tuple, list)):
            for elem in obj:
                yield from GenericFxParser._iter_tensors(elem)
        elif isinstance(obj, dict):
            for elem in obj.values():
                yield from GenericFxParser._iter_tensors(elem)

    def _register_hooks(self):
        """Register forward hooks to capture (a) input/output shapes and
        (b) module-to-module dataflow for leaf modules."""
        self.hooks = []
        self._leaf_names = {
            name for name, m in self.model.named_modules()
            if name and not any(True for _ in m.children())
        }

        def make_hook(name):
            is_leaf = name in self._leaf_names

            def hook(_module, inp, output):
                # (a) shapes -- unchanged behavior
                if isinstance(inp, tuple) and len(inp) > 0:
                    first_in = inp[0]
                    if isinstance(first_in, torch.Tensor):
                        self.input_shapes[name] = tuple(first_in.shape)
                if isinstance(output, torch.Tensor):
                    self.output_shapes[name] = tuple(output.shape)
                elif isinstance(output, tuple) and len(output) > 0:
                    if isinstance(output[0], torch.Tensor):
                        self.output_shapes[name] = tuple(output[0].shape)

                # (b) dataflow -- leaf modules only, best-effort
                if not is_leaf:
                    return
                try:
                    if name not in self._exec_order:
                        self._exec_order.append(name)
                    producers = self._traced_inputs.setdefault(name, [])
                    for tensor in self._iter_tensors(inp):
                        src = self._producer.get(id(tensor))
                        if src and src != name and src not in producers:
                            producers.append(src)
                    for tensor in self._iter_tensors(output):
                        # keep a reference so this id isn't reused mid-pass
                        self._producer[id(tensor)] = name
                        self._keepalive.append(tensor)
                except Exception:  # pylint: disable=broad-except
                    # tracing is best-effort; a bad hook must not abort the pass
                    pass

            return hook

        for name, module in self.model.named_modules():
            if name:
                self.hooks.append(module.register_forward_hook(make_hook(name)))

    def _unregister_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self._keepalive.clear()   # release retained activations

    def _infer_shapes(self):
        """Run a dummy forward pass to infer shapes and dataflow."""
        self.model.eval()
        with torch.no_grad():
            try:
                if hasattr(self.model, "config"):
                    config = self.model.config
                    # A few tokens are enough to see every module's dataflow;
                    # the graph does not depend on the sequence length. Using
                    # max_position_embeddings here (40960 for Qwen3, 131072
                    # for Llama 3) ran a forward pass that took hours.
                    seq_len = min(
                        int(getattr(config, "max_position_embeddings", None) or 16),
                        SHAPE_PROBE_SEQ_LEN,
                    )
                    batch_size = 1
                    if hasattr(config, "vocab_size"):
                        dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))
                    else:
                        hidden_size = getattr(config, "hidden_size", 768)
                        dummy_input = torch.randn(batch_size, seq_len, hidden_size)
                    _ = self.model(dummy_input)
                else:
                    dummy_input = torch.randn(1, 512, 768)
                    _ = self.model(dummy_input)
            except Exception as exc:  # pylint: disable=broad-except
                # Shape/dataflow inference is best-effort. Record why so a graph
                # with no shapes (and a linear-chain fallback) is diagnosable
                # instead of silently mysterious.
                self.shape_inference_error = str(exc)
                logger.warning("Shape inference forward pass failed: %s", exc)
