"""
Aggregates the per-node supported/reason flags that GenericFxParser
already set (via op_table.classify_op) into the report the extension
shows the user, and the decision point ("continue anyway?") hangs off.
"""
from api.graph.graph import Graph


class OpLevelCompatibilityChecker:
    def analyze(self, graph: Graph) -> dict:
        report = []
        supported = 0
        unsupported = 0

        for node in graph.get_nodes():
            entry = {
                "id": node.id,
                "name": node.name,
                "type": node.node_type,
                "supported": node.supported,
                "reason": node.compatibility_reason,
            }
            report.append(entry)

            if node.supported:
                supported += 1
            else:
                unsupported += 1

        total = len(graph)

        return {
            "summary": {
                "supported_nodes": supported,
                "unsupported_nodes": unsupported,
                "compatibility": round(supported / total * 100, 2) if total else 0,
            },
            "nodes": report,
            "unsupported": [n for n in report if not n["supported"]],
        }
