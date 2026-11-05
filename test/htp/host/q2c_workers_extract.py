#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Writes typedefs and static functions of a DSP source to one file, for a
host check to include: the check runs the production code, not a copy of
it. With no names: q2c_ctx and the q2 calibration workers of
nntr_hvx_attn_q.c (q2c_calib_check.c). A name ending in ';' is a
typedef struct (`} name;`), any other a static function."""
import re
import sys

Q2C = ("q2c_ctx;", "q2c_worker", "q2c_worker_kd", "q2c_worker_kd_q")


def block(text, start):
    i = text.index("{", start) + 1
    depth = 1
    while depth:
        depth += (text[i] == "{") - (text[i] == "}")
        i += 1
    return i


def main(src, out, names):
    text = open(src, encoding="utf-8").read()
    parts = []
    for name in names or Q2C:
        if name.endswith(";"):
            end = text.index("} " + name) + len(name) + 2
            parts.append(text[text.rindex("typedef struct {", 0, end):end])
            continue
        m = re.search(r"^static [^;{]*?\b" + name + r"\(", text, re.M)
        if not m:
            raise SystemExit("missing production function " + name)
        parts.append(text[m.start():block(text, m.start())])
    open(out, "w", encoding="utf-8").write("\n\n".join(parts) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
