#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tokenize a text file into an int32 LE token-id file for the hexagon harnesses.

Usage: make_tokens.py <model-dir> <text> <out.tokens.i32> [--limit N]
       --limit N also writes <out>.txt, the detokenized prefix of N tokens, so
       the CPU baselines can be evaluated on exactly the same tokens.
To read ids back: AutoTokenizer.from_pretrained(d).decode(np.fromfile(f, "<i4"))
"""
import argparse
import sys

import numpy as np
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("text")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    with open(args.text, encoding="utf-8") as f:
        ids = tok.encode(f.read(), add_special_tokens=False)
    if args.limit:
        ids = ids[: args.limit]
        base = args.out.removesuffix(".tokens.i32")
        with open(base + ".txt", "w", encoding="utf-8") as f:
            f.write(tok.decode(ids))
    np.asarray(ids, dtype="<i4").tofile(args.out)
    print(f"tokens={len(ids)} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
