#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# @file   gen_golden.py
# @date   05 June 2026
# @brief  Dump DDTree golden vectors for the buildTree parity unit tests.
# @author Seunghui Lee <shsh1004.lee@samsung.com>
#
# A faithful, dependency-free transcription of ``build_ddtree_tree`` from the
# DDTree Python reference. It avoids torch/numpy so the golden vectors can be
# regenerated in any environment, replacing the two torch ops with exact
# standard-library equivalents:
#
#   * ``torch.topk``      -> sort by (logit desc, index asc)  [matches CPU topk]
#   * ``torch.logsumexp`` -> m + log(sum(exp(x - m)))
#
# The best-first heap uses Python's native tuple ordering, exactly as the
# reference does via ``heapq`` on ``(-logw, ranks, parent, depth, rank, logw)``
# -- so the tie-break semantics under test are reproduced verbatim. Inputs use
# well-separated logits so fp32-vs-fp64 rounding never flips an ordering (except
# where an exact tie is intended to exercise the tie-break).
#
# This file locks tree structure / visibility; end-to-end token parity is
# covered separately by the runtime trace-replay test.
import heapq
import json
import math
import os
import struct


def f32(x):
    """Round a Python float to float32 precision, matching ddtree.py's
    top_log_probs.to(torch.float32) and the C++ core's fp32 storage (spec 6.2)."""
    return struct.unpack("f", struct.pack("f", x))[0]


def build_ddtree_tree(draft_logits, budget):
    """Transcription of ddtree.py build_ddtree_tree (lines 92-174)."""
    depth_limit = len(draft_logits)
    vocab = len(draft_logits[0]) if depth_limit else 0

    if budget <= 0 or depth_limit == 0:
        return [], [], [-1], [{}], [[1]]

    topk = min(budget, vocab)

    # Per-row top-k log-probs (fp-equivalent of torch.topk + logsumexp).
    top_log_probs = []
    top_token_ids = []
    for row in draft_logits:
        order = sorted(range(vocab), key=lambda i: (-row[i], i))[:topk]
        m = max(row)
        # logsumexp accumulated in fp64 then rounded to fp32 (C++ casts the
        # double log_z to float); top_log_probs stored fp32.
        lse = f32(m + math.log(sum(math.exp(x - m) for x in row)))
        top_token_ids.append([order[r] for r in range(topk)])
        top_log_probs.append([f32(f32(row[order[r]]) - lse) for r in range(topk)])

    first_logw = float(top_log_probs[0][0])
    heap = [(-first_logw, (0,), 0, 1, 0, first_logw)]

    node_token_ids = []
    node_depths = []
    parents = [-1]
    child_maps = [dict()]
    node_count = 0

    while heap and node_count < budget:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)

        token_id = int(top_token_ids[depth - 1][rank])
        current_index = node_count + 1
        node_token_ids.append(token_id)
        node_depths.append(depth)
        parents.append(parent_index)
        child_maps.append(dict())
        child_maps[parent_index][token_id] = current_index
        node_count += 1

        if rank + 1 < topk:
            sibling_ranks = ranks[:-1] + (rank + 1,)
            sibling_logw = (logw - float(top_log_probs[depth - 1][rank])
                            + float(top_log_probs[depth - 1][rank + 1]))
            heapq.heappush(
                heap,
                (-sibling_logw, sibling_ranks, parent_index, depth, rank + 1,
                 sibling_logw))

        if depth < depth_limit:
            child_ranks = ranks + (0,)
            child_logw = logw + float(top_log_probs[depth][0])
            heapq.heappush(
                heap,
                (-child_logw, child_ranks, current_index, depth + 1, 0,
                 child_logw))

    current_length = 1 + node_count
    vis = [[0] * current_length for _ in range(current_length)]
    vis[0][0] = 1
    for index in range(1, current_length):
        parent_index = parents[index]
        for j in range(index):
            vis[index][j] = vis[parent_index][j]
        vis[index][index] = 1

    return node_token_ids, node_depths, parents, child_maps, vis


def dump_build(name, logits, budget):
    ntids, ndepths, parents, _child_maps, vis = build_ddtree_tree(logits, budget)
    return {
        "name": name,
        "logits": logits,
        "budget": budget,
        "depth": len(logits),
        "vocab": len(logits[0]),
        "node_token_ids": [int(x) for x in ntids],
        "node_depths": [int(x) for x in ndepths],
        "parents": [int(x) for x in parents],
        "visibility": vis,
    }


if __name__ == "__main__":
    cases = []
    cases.append(dump_build("small", [[2.0, 1.0, 0.0], [0.0, -1.0, 1.0]], 3))
    # equal logits force exact-tie heap entries -> exercises tie-break order.
    cases.append(dump_build("tie", [[1.0, 1.0, 0.0], [0.5, 0.5, 0.5]], 4))
    # budget > vocab clamps topk to vocab. The logits are widely separated so
    # every heap-ordering decision has a large margin (~0.4): log-sum-exp is
    # computed in double then stored fp32 (parity spec), so a near-tie here
    # could let x86/ARM64 libm rounding flip the fp32 logZ and reorder nodes,
    # breaking the bit-exact golden comparison cross-architecture.
    cases.append(
        dump_build("budget_gt_vocab", [[8.0, 5.0, 2.0], [6.0, 5.0, 4.0]], 100))
    cases.append(
        dump_build("full_fill",
                   [[3.0, 2.0, 1.0, 0.0], [3.0, 2.0, 1.0, 0.0],
                    [3.0, 2.0, 1.0, 0.0]], 8))

    here = os.path.dirname(os.path.abspath(__file__))

    # Human-readable JSON (for inspection).
    out_json = os.path.join(here, "build_golden.json")
    with open(out_json, "w") as f:
        json.dump(cases, f, indent=2)

    # Flat, whitespace-delimited format the C++ test parses with ifstream >>.
    # Per case:
    #   CASE <name>
    #   DIMS <depth> <vocab> <budget>
    #   LOGITS <depth*vocab floats, row-major>
    #   NODES <count> <token ids...>
    #   DEPTHS <count> <depths...>
    #   PARENTS <len> <parents...>
    #   VIS <L> <L*L ints, row-major>
    # File starts with: NCASES <n>
    lines = ["NCASES %d" % len(cases)]
    for c in cases:
        flat_logits = [x for row in c["logits"] for x in row]
        flat_vis = [x for row in c["visibility"] for x in row]
        L = len(c["visibility"])
        lines.append("CASE %s" % c["name"])
        lines.append("DIMS %d %d %d" % (c["depth"], c["vocab"], c["budget"]))
        lines.append("LOGITS " + " ".join(repr(x) for x in flat_logits))
        lines.append("NODES %d %s" % (len(c["node_token_ids"]),
                                      " ".join(str(x) for x in c["node_token_ids"])))
        lines.append("DEPTHS %d %s" % (len(c["node_depths"]),
                                       " ".join(str(x) for x in c["node_depths"])))
        lines.append("PARENTS %d %s" %
                     (len(c["parents"]), " ".join(str(x) for x in c["parents"])))
        lines.append("VIS %d %s" % (L, " ".join(str(x) for x in flat_vis)))
    flat = "\n".join(lines) + "\n"

    out_txt = os.path.join(here, "build_golden.txt")
    with open(out_txt, "w") as f:
        f.write(flat)

    # Embedded C++ header: the unit test parses this string instead of opening a
    # file. Cross-compiled binaries run on a separate test runner with a
    # different workspace drive, so a build-time absolute path to the data file
    # does not resolve at run time -- embedding keeps the test self-contained.
    out_h = os.path.join(here, "build_golden.h")
    header = (
        "// SPDX-License-Identifier: Apache-2.0\n"
        "/**\n"
        " * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>\n"
        " *\n"
        " * @brief  Embedded DDTree buildTree golden vectors (auto-generated)\n"
        " * @file   build_golden.h\n"
        " * @date   05 June 2026\n"
        " * @see    https://github.com/nntrainer/nntrainer\n"
        " * @author Seunghui Lee <shsh1004.lee@samsung.com>\n"
        " * @bug    No known bugs except for NYI items\n"
        " *\n"
        " * Generated by gen_golden.py -- do not edit by hand. Regenerate with\n"
        " * `python3 gen_golden.py`. Format documented above in gen_golden.py.\n"
        " */\n"
        "#ifndef __NNTRAINER_DDTREE_BUILD_GOLDEN_H__\n"
        "#define __NNTRAINER_DDTREE_BUILD_GOLDEN_H__\n"
        "\n"
        "static const char *const kDDTreeBuildGolden = R\"GOLDEN(\n"
        + flat +
        ")GOLDEN\";\n"
        "\n"
        "#endif // __NNTRAINER_DDTREE_BUILD_GOLDEN_H__\n")
    with open(out_h, "w") as f:
        f.write(header)

    print("wrote", out_json + ",", out_txt, "and", out_h, "with", len(cases),
          "cases")
