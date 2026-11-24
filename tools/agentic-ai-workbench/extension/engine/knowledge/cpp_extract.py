# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file cpp_extract.py
# @brief Small, dependency-free C++ reader for the CausalLM sources.
# @author prachi-t17 <prachi.t@samsung.com>

"""
Small, dependency-free C++ reader for the CausalLM sources.

It finds out-of-class member definitions (`Tensor Qwen3Transformer::
createAttention(...) { ... }`), class declarations with their bases, and the
`type` string a custom layer registers under. That is all the code bank's
capture step needs; it is not a general C++ parser.

Braces and parentheses are matched on a masked copy of the source in which
comments and string/char literals are blanked out, so a `{` inside a string
or comment never throws the matching off.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class CppDefinition:
    """A C++ method definition: class, name, signature and body."""
    class_name: str
    method: str
    signature: str          # declaration text up to (not including) the body
    body: str               # the body, including its outer braces
    text: str               # leading comment + signature + body, as written
    start_line: int         # 1-based line of the signature start
    end_line: int           # 1-based line of the closing brace


@dataclass
class CppClass:
    """A C++ class and its base classes."""
    name: str
    bases: List[str] = field(default_factory=list)


def mask_non_code(src: str) -> str:
    """Return `src` with comments and string/char literal contents replaced
    by spaces (newlines kept), so offsets and line numbers still line up."""
    out = list(src)
    i, size = 0, len(src)

    def blank(start: int, stop: int) -> None:
        for k in range(start, min(stop, size)):
            if out[k] != "\n":
                out[k] = " "

    while i < size:
        char = src[i]
        if char == "/" and i + 1 < size and src[i + 1] == "/":
            j = src.find("\n", i)
            j = size if j == -1 else j
            blank(i, j)
            i = j
        elif char == "/" and i + 1 < size and src[i + 1] == "*":
            j = src.find("*/", i + 2)
            j = size if j == -1 else j + 2
            blank(i, j)
            i = j
        elif (
            char == "R"
            and src.startswith('R"', i)
            and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_"))
        ):
            match = re.match(r'R"([^(\s]{0,16})\(', src[i:])
            if not match:
                i += 1
                continue
            end = src.find(")" + match.group(1) + '"', i + match.end())
            end = size if end == -1 else end + len(match.group(1)) + 2
            blank(i + 2, end - 1)
            i = end
        elif char in "\"'":
            j = i + 1
            while j < size and src[j] != char:
                if src[j] == "\\":
                    j += 1
                elif src[j] == "\n":   # unterminated -- stop at line end
                    break
                j += 1
            blank(i + 1, j)
            i = j + 1
        else:
            i += 1
    return "".join(out)


def _match_close(masked: str, open_idx: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    for k in range(open_idx, len(masked)):
        char = masked[k]
        if char == open_ch:
            depth += 1
        elif char == close_ch:
            depth -= 1
            if depth == 0:
                return k
    return -1


def _code_depths(masked: str) -> List[int]:
    """Brace depth at each offset, not counting `namespace {` / `extern "C" {`
    braces, so a definition inside `namespace causallm { ... }` is at 0."""
    depths = [0] * (len(masked) + 1)
    stack: List[bool] = []          # True = transparent (namespace) brace
    depth = 0
    for k, char in enumerate(masked):
        depths[k] = depth
        if char == "{":
            head = masked[max(0, k - 200):k]
            transparent = bool(re.search(r'(namespace(\s+[\w:]+)?|extern\s+"?\s*"?)\s*$', head))
            stack.append(transparent)
            if not transparent:
                depth += 1
        elif char == "}":
            if stack and not stack.pop():
                depth -= 1
    depths[len(masked)] = depth
    return depths


_QUALIFIED = re.compile(r"\b([A-Za-z_]\w*)\s*::\s*(~?[A-Za-z_]\w*)\s*\(")
_AFTER_PARAMS = re.compile(
    r"\s*(?:const\b|override\b|final\b|noexcept\b(?:\s*\([^)]*\))?|&&?"
    r"|->\s*[\w:<>,\s\*&]+?(?=\s*\{))*\s*"
)


def _leading_comment_start(src: str, sig_start: int) -> int:
    """Include a comment block that sits directly above the signature."""
    lines_before = src[:sig_start].split("\n")
    # lines_before[-1] is the (possibly empty) text on the signature's line.
    keep = len(lines_before) - 1
    k = keep - 1
    in_block = False
    while k >= 0:
        text = lines_before[k].strip()
        if in_block:
            if text.startswith("/*"):
                in_block = False
            k -= 1
            keep = k + 1
            continue
        if text.startswith("//") or text.startswith("*") or text.startswith("/*"):
            if text.endswith("*/") and not text.startswith("/*"):
                in_block = True
            k -= 1
            keep = k + 1
            continue
        break
    return len("\n".join(lines_before[:keep])) + (1 if keep else 0)


def _body_start(masked: str, paren_close: int) -> int:
    """Index of the `{` that opens a function body after its parameter list
    (skipping const/override/noexcept/trailing return and a constructor's
    initialiser list), or -1 if this is a declaration or a call."""
    k = paren_close + 1
    match = _AFTER_PARAMS.match(masked, k)
    if match:
        k = match.end()
    if (
        k < len(masked) and masked[k] == ":" and masked[k + 1 : k + 2] != ":"
    ):  # constructor init list
        j = k + 1
        while j < len(masked):
            char = masked[j]
            if char == "(":
                j = _match_close(masked, j, "(", ")")
            elif char == "{":
                # `member{value}` in an init list vs. the body: the body's
                # brace follows `)` or `}`.
                prev = masked[:j].rstrip()
                if prev.endswith(")") or prev.endswith("}"):
                    break
                j = _match_close(masked, j, "{", "}")
            elif char == ";":
                return -1
            if j == -1:
                return -1
            j += 1
        k = j
    if k >= len(masked) or masked[k] != "{":
        return -1
    return k


def _statement_start(masked: str, src: str, pos: int, floor: int = 0) -> int:
    """Start of the declaration that contains offset `at`: after the previous
    `;`, `{` or `}` (or a preprocessor line), skipping blanked comments."""
    j = pos - 1
    while (
        j >= floor
        and masked[j] not in ";{}"
        and not (masked[j] == "\n" and masked[j + 1 : j + 2] == "#")
    ):
        j -= 1
    start = j + 1
    while start < pos and masked[start].isspace():
        start += 1
    while True:
        line_end = src.find("\n", start)
        if src[start:start + 1] == "#" and 0 <= line_end < pos:
            start = line_end + 1
            while start < pos and masked[start].isspace():
                start += 1
        else:
            return start


# pylint: disable-next=too-many-arguments
def _definition(src: str, masked: str, cls: str, method: str, name_start: int, body_open: int,
                floor: int = 0) -> Optional[CppDefinition]:
    body_close = _match_close(masked, body_open, "{", "}")
    if body_close == -1:
        return None
    sig_start = _statement_start(masked, src, name_start, floor)
    # Access specifiers in a class body (`public:`) are not part of a signature.
    match = re.match(r"(?:(?:public|protected|private)\s*:\s*)+", masked[sig_start:name_start])
    if match:
        sig_start += match.end()
        while sig_start < name_start and masked[sig_start].isspace():
            sig_start += 1
    text_start = _leading_comment_start(src, sig_start)
    return CppDefinition(
        class_name=cls, method=method,
        signature=src[sig_start:body_open].strip(),
        body=src[body_open:body_close + 1],
        text=src[text_start:body_close + 1].strip("\n"),
        start_line=src.count("\n", 0, sig_start) + 1,
        end_line=src.count("\n", 0, body_close) + 1,
    )


def find_definitions(src: str) -> List[CppDefinition]:
    """Out-of-class member definitions: `Ret Class::method(...) { ... }`."""
    masked = mask_non_code(src)
    depths = _code_depths(masked)
    found: List[CppDefinition] = []
    pos = 0
    while True:
        match = _QUALIFIED.search(masked, pos)
        if not match:
            break
        pos = match.end()
        if depths[match.start()] != 0:
            continue
        paren_close = _match_close(masked, match.end() - 1, "(", ")")
        if paren_close == -1:
            continue
        k = _body_start(masked, paren_close)
        if k == -1:
            continue                                   # a declaration or a call
        definition = _definition(src, masked, match.group(1), match.group(2), match.start(), k)
        if definition:
            found.append(definition)
            pos = k + len(definition.body)
    return found


_KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "return",
    "sizeof",
    "catch",
    "static_assert",
    "decltype",
    "alignof",
    "alignas",
    "defined",
    "noexcept",
    "throw",
    "new",
    "delete",
    "operator",
    "requires",
}
_NAME_PAREN = re.compile(r"(~?[A-Za-z_]\w*)\s*\(")


def find_inline_methods(src: str) -> List[CppDefinition]:
    """Member functions defined inside a class body (typically in headers):
    `void setupParameters(...) override { ... }`, inline constructors, ..."""
    masked = mask_non_code(src)
    found: List[CppDefinition] = []
    for match in _CLASS.finditer(masked):
        if re.search(r"\benum\s+$", masked[max(0, match.start() - 8):match.start()]):
            continue
        open_idx = match.end() - 1
        close = _match_close(masked, open_idx, "{", "}")
        if close == -1:
            continue
        cls = match.group(1)
        pos = open_idx + 1
        while pos < close:
            char = masked[pos]
            if char == "{":                       # nested class/enum/initialiser: skip
                end = _match_close(masked, pos, "{", "}")
                pos = (end if end != -1 else close) + 1
                continue
            if char == "(":                       # a parameter list we did not start at
                end = _match_close(masked, pos, "(", ")")
                pos = (end if end != -1 else close) + 1
                continue
            mod = _NAME_PAREN.match(masked, pos)
            if not mod or (pos > 0 and (masked[pos - 1].isalnum() or masked[pos - 1] in "_:.>")):
                pos += 1
                continue
            name = mod.group(1)
            paren_close = _match_close(masked, mod.end() - 1, "(", ")")
            if paren_close == -1 or name in _KEYWORDS:
                pos = mod.end()
                continue
            k = _body_start(masked, paren_close)
            if k == -1 or k > close:
                pos = paren_close + 1
                continue
            definition = _definition(src, masked, cls, name, mod.start(), k, floor=open_idx + 1)
            if definition:
                found.append(definition)
                pos = k + len(definition.body)
            else:
                pos = paren_close + 1
    return found


def calls_in(body: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    """(qualified calls [(Class, method)], unqualified/this-> calls [method])
    in a function body, ignoring comments and strings. Calls through another
    object (`obj.f()`, `p->f()`) are not this object's methods and are skipped."""
    masked = mask_non_code(body)
    qualified = re.findall(r"\b([A-Z]\w*)::(~?\w+)\s*\(", masked)
    plain = []
    for match in re.finditer(r"(?:\bthis\s*->\s*)?\b([A-Za-z_]\w*)\s*\(", masked):
        start = match.start()
        before = masked[max(0, start - 2):start]
        if not match.group(0).startswith("this") and (before.endswith(".") or before.endswith("->")
                                                  or before.endswith("::")):
            continue
        if match.group(1) not in _KEYWORDS:
            plain.append(match.group(1))
    return qualified, plain


def string_literals(body: str) -> List[str]:
    """The identifier-like string literals in a C++ body."""
    return re.findall(r'"([A-Za-z0-9_]+)"', body)


_CLASS = re.compile(
    r"\b(?:class|struct)\s+(?:[A-Z_][A-Z0-9_]*\s+)*([A-Za-z_]\w*)\s*(?:final\s*)?"
    r"(?::\s*([^{;]+))?\{"
)


def find_classes(src: str) -> List[CppClass]:
    """The classes defined in a C++ source."""
    masked = mask_non_code(src)
    classes = []
    for match in _CLASS.finditer(masked):
        if re.search(r"\benum\s+$", masked[max(0, match.start() - 8):match.start()]):
            continue                                   # `enum class X {`
        bases = []
        for part in (match.group(2) or "").split(","):
            words = [w for w in re.split(r"\s+", part.strip()) if w]
            words = [w for w in words if w not in ("public", "protected", "private", "virtual")]
            if words:
                bases.append(re.sub(r"^.*::", "", words[-1]))
        classes.append(CppClass(match.group(1), bases))
    return classes


_LAYER_TYPE = re.compile(r'\btype\s*=\s*"([A-Za-z0-9_]+)"')


def find_layer_types(src: str) -> List[str]:
    """`static constexpr const char *type = "qwen_moe";` -> ["qwen_moe"]."""
    return sorted(set(_LAYER_TYPE.findall(src)))


_CREATE_LAYER = re.compile(r'createLayer\s*\(\s*"([A-Za-z0-9_]+)"')


def layer_types_used(src: str) -> List[str]:
    """The layer types a C++ source creates with createLayer(...)."""
    return sorted(set(_CREATE_LAYER.findall(src)))


def final_overriders(cls: str, classes: Dict[str, List[str]],
                     defined: Dict[str, set]) -> Dict[str, str]:
    """Map method -> class that provides the final overrider for `cls`.

    `classes` maps class -> direct bases; `defined` maps class -> methods it
    defines. Follows the C++ rule: of all classes in the hierarchy that
    define the method, the one no other defining class derives from wins
    (virtual bases make the Transformer/CausalLM diamond resolve this way).
    """
    def ancestors(class_name: str, seen=None) -> set:
        seen = set() if seen is None else seen
        for builder in classes.get(class_name, []):
            if builder not in seen:
                seen.add(builder)
                ancestors(builder, seen)
        return seen

    hierarchy = {cls} | ancestors(cls)
    result: Dict[str, str] = {}
    methods = set().union(*(defined.get(c, set()) for c in hierarchy))
    for meth in methods:
        definers = [c for c in hierarchy if meth in defined.get(c, set())]
        finals = [c for c in definers
                  if not any(c in ancestors(o) for o in definers if o != c)]
        if len(finals) == 1:
            result[meth] = finals[0]
    return result
