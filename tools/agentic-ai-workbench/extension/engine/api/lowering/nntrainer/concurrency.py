"""
Thread-safety primitives for the nntrainer lowering stage.

Within one decoder layer, the Q/K/V projections (and, in the MLP, the
gate/up projections) are independent of each other -- they all read
the same input tensor and don't depend on each other's output. The
lowerer builds each of those independent branches concurrently on a
small thread pool: real work (attribute lookups, dataclass
construction, dict building) rather than a decorative pool, and it
mirrors the actual data dependency graph instead of parallelising
something that's secretly sequential.

Decoder layers themselves are NOT lowered concurrently -- layer i+1's
input is layer i's output, so that dependency is real and threading
across layers would just add synchronization overhead for no benefit.

`Graph`/`GraphNode` (api.graph.graph.Graph) were written for
single-threaded use (plain dict + list, no locking). Rather than
sprinkle locks through that shared, well-tested module, this wraps it:
one `threading.Lock` guards every mutation, so concurrent branch
lowering can safely call `add_node`/`connect` from worker threads
without corrupting `graph.nodes` / `graph.edges`.
"""
from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, TypeVar

from api.graph.graph import Graph
from api.graph.node import GraphNode

T = TypeVar("T")

#: Q/K/V (and gate/up) branches per layer -- never needs to be large.
_MAX_BRANCH_WORKERS = 4


class ThreadSafeGraphBuilder:
    """Guards a single api.graph.graph.Graph with a lock so multiple
    worker threads can build independent branches of the same layer
    concurrently and merge their nodes/edges safely."""

    def __init__(self, graph: Graph):
        self._graph = graph
        self._lock = threading.Lock()

    @property
    def graph(self) -> Graph:
        return self._graph

    def add_node(self, node: GraphNode) -> GraphNode:
        with self._lock:
            self._graph.add_node(node)
        return node

    def connect(self, source: GraphNode, target: GraphNode) -> None:
        with self._lock:
            self._graph.connect(source, target)

    def run_concurrent_branches(self, branch_fns: list[Callable[[], T]]) -> list[T]:
        """Runs each zero-arg callable in `branch_fns` on a worker thread
        and returns their results in the same order they were given
        (NOT completion order), so callers can index results
        positionally (e.g. results[0] is always the Q branch)."""
        if len(branch_fns) <= 1:
            return [fn() for fn in branch_fns]

        with ThreadPoolExecutor(
            max_workers=min(_MAX_BRANCH_WORKERS, len(branch_fns)),
            thread_name_prefix="nntrainer-lower-branch",
        ) as pool:
            futures: list[Future] = [pool.submit(fn) for fn in branch_fns]
            # .result() re-raises inside the caller's thread, so a failed
            # branch surfaces as a normal exception, not a silently
            # swallowed worker-thread crash.
            return [f.result() for f in futures]
