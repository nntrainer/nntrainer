"""
Pure edge-resolution logic for the dataflow tracer, kept torch-free so it
can be unit-tested without importing torch or the rest of the engine.
"""


def resolve_traced_edges(exec_order, emitted_names, traced_inputs):
    """Resolve edges among emitted nodes from raw per-leaf dataflow.

    Args:
      exec_order:     leaf module names in the order their hooks fired.
      emitted_names:  set of leaf names that became graph nodes.
      traced_inputs:  leaf_name -> list of direct producer leaf names.

    Returns (edges, orphans):
      edges:   list of (src, tgt) among emitted names, walking back through
               non-emitted passthrough leaves to the nearest emitted ancestor.
      orphans: emitted names (excluding the first emitted node in exec order)
               with no resolved predecessor -- the signal that tracing was
               incomplete and the caller should fall back to the linear chain.
    """
    emitted = set(emitted_names)

    def resolve(name, seen):
        out = []
        for p in traced_inputs.get(name, []):
            if p in seen:
                continue
            seen.add(p)
            if p in emitted:
                out.append(p)
            else:
                out.extend(resolve(p, seen))
        return out

    edges = []
    seen_edges = set()
    preds_count = {}
    for tgt in exec_order:
        if tgt not in emitted:
            continue
        preds_count.setdefault(tgt, 0)
        for src in resolve(tgt, set()):
            if src == tgt:
                continue
            key = (src, tgt)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            edges.append(key)
            preds_count[tgt] += 1

    emitted_in_order = [n for n in exec_order if n in emitted]
    first = emitted_in_order[0] if emitted_in_order else None
    orphans = [n for n in emitted_in_order if n != first and preds_count.get(n, 0) == 0]
    return edges, orphans
