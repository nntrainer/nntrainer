"""
LLM-driven CausalLM model generation: plan -> skeleton -> generate -> validate.

Pipeline
--------
    semantic_ir
        |  model_plan.build_plan          (deterministic)
    ModelPlan
        |  skeleton.emit_skeletons        (deterministic)
    .h (complete) + .cpp (holes)
        |  prompts_causallm + LLM         <-- the only non-deterministic step
    method bodies
        |  _splice_bodies                 (deterministic)
    .cpp (filled)
        |  plan_validator.validate        (deterministic)
    accept, or correction_prompt and retry

Two design choices worth calling out:

**Body splicing, not file replacement.** The prompt asks for both files in
full, but the output is *not* trusted to supply the scaffolding. Method bodies
are extracted from whatever the model returns and spliced into our own
skeleton. So the include guard, SPDX banner, namespace, the virtual-base
constructor diamond, and every method signature are always exactly what
`skeleton.py` emitted, no matter how badly the model strayed. A weak model that
drops the header, mangles the constructor, or renames a method still produces a
usable file as long as one recognisable body came back. This converts a large
class of total failures into partial successes.

**Validation before compilation.** The failures that matter here compile
cleanly -- a wrong layer name silently loads zero weights, a missing swiglu
index remap silently computes the wrong function. `plan_validator` catches
those; the compiler cannot. Findings feed `correction_prompt` as structured
`fix` strings, which small models act on far more reliably than a raw compiler
dump.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..events import bus
from . import model_plan, skeleton
from .plan_validator import format_report, validate
# Import common base components (always available)
from .prompts_causallm import (
    SYSTEM_PROMPT,
    correction_prompt as base_correction_prompt,
    render_catalog,
)

# Model-specific prompt modules are imported dynamically based on architecture

DEFAULT_MAX_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Model-specific prompt routing
# ---------------------------------------------------------------------------
def _get_prompt_module(architecture: str):
    """
    Return the model-specific prompt module based on architecture name.
    
    Falls back to common prompts_causallm if no model-specific module exists.
    
    Args:
        architecture: The architecture name (e.g., "Gemma3ForCausalLM", "Qwen3ForCausalLM")
    
    Returns:
        Module with generation_prompt() and correction_prompt() functions
    """
    arch_lower = architecture.lower() if architecture else ""
    
    # Map architecture names to prompt modules
    # Gemma family
    if "gemma" in arch_lower:
        if "gemma4" in arch_lower:
            try:
                from . import prompts_gemma4_causallm as prompts
                bus.log(f"Using Gemma4-specific prompt template for {architecture}", "info")
                return prompts
            except ImportError:
                bus.log("Gemma4 prompt module not found, trying Gemma3", "warn")
        try:
            from . import prompts_gemma3_causallm as prompts
            bus.log(f"Using Gemma3-specific prompt template for {architecture}", "info")
            return prompts
        except ImportError:
            bus.log("Gemma3 prompt module not found, falling back to base", "warn")
    
    # Qwen family
    elif "qwen" in arch_lower:
        if "qwen2" in arch_lower:
            try:
                from . import prompts_qwen2_causallm as prompts
                bus.log(f"Using Qwen2-specific prompt template for {architecture}", "info")
                return prompts
            except ImportError:
                bus.log("Qwen2 prompt module not found, trying Qwen3", "warn")
        try:
            from . import prompts_qwen3_causallm as prompts
            bus.log(f"Using Qwen3-specific prompt template for {architecture}", "info")
            return prompts
        except ImportError:
            bus.log("Qwen3 prompt module not found, falling back to base", "warn")
    
    # BERT family
    elif "bert" in arch_lower or "roberta" in arch_lower or "xlm" in arch_lower:
        try:
            from . import prompts_bert_causallm as prompts
            bus.log(f"Using BERT-specific prompt template for {architecture}", "info")
            return prompts
        except ImportError:
            bus.log("BERT prompt module not found, falling back to base", "warn")
    
    # Fallback to common base
    from . import prompts_causallm as prompts
    bus.log(f"Using base prompt template for {architecture}", "info")
    return prompts


def _get_generation_prompt(prompt_module, plan: Dict, skeleton_h: str, skeleton_cpp: str) -> str:
    """
    Get the generation prompt from the appropriate module.
    
    Args:
        prompt_module: The prompt module (model-specific or base)
        plan: The resolved ModelPlan
        skeleton_h: The header skeleton content
        skeleton_cpp: The source skeleton content
    
    Returns:
        The formatted generation prompt string
    """
    # Try model-specific generation prompt first (order matters: most specific first)
    
    # Gemma family
    if hasattr(prompt_module, 'gemma4_generation_prompt'):
        return prompt_module.gemma4_generation_prompt(plan, skeleton_h, skeleton_cpp)
    if hasattr(prompt_module, 'gemma3_generation_prompt'):
        return prompt_module.gemma3_generation_prompt(plan, skeleton_h, skeleton_cpp)
    
    # Qwen family
    if hasattr(prompt_module, 'qwen2_generation_prompt'):
        return prompt_module.qwen2_generation_prompt(plan, skeleton_h, skeleton_cpp)
    if hasattr(prompt_module, 'qwen3_generation_prompt'):
        return prompt_module.qwen3_generation_prompt(plan, skeleton_h, skeleton_cpp)
    
    # BERT family
    if hasattr(prompt_module, 'bert_generation_prompt'):
        return prompt_module.bert_generation_prompt(plan, skeleton_h, skeleton_cpp)
    
    # Fallback to base
    if hasattr(prompt_module, 'generation_prompt'):
        return prompt_module.generation_prompt(plan, skeleton_h, skeleton_cpp)
    
    raise ValueError("No generation_prompt function found in prompt module")


def _get_correction_prompt(prompt_module, plan: Dict, previous_code: str, 
                           findings: List[Dict], attempt: int, max_attempts: int) -> str:
    """
    Get the correction prompt from the appropriate module.
    
    Args:
        prompt_module: The prompt module (model-specific or base)
        plan: The resolved ModelPlan
        previous_code: The previously generated code
        findings: Validation findings/errors
        attempt: Current attempt number
        max_attempts: Maximum attempts allowed
    
    Returns:
        The formatted correction prompt string
    """
    # Try model-specific correction prompt first, then fall back to base
    if hasattr(prompt_module, 'correction_prompt'):
        return prompt_module.correction_prompt(
            plan, previous_code, findings, attempt, max_attempts)
    else:
        return base_correction_prompt(
            plan, previous_code, findings, attempt, max_attempts)

# Paths for compile-checking generated files.
_CAUSALLM_ROOT = "/storage_data/snap/Prachi/nntrainer/Applications/CausalLM"
_NNTRAINER_INCLUDE = "/usr/local/include/nntrainer"


@dataclass
class ComponentFiles:
    """
    Mirrors converters.cpp_generator.ComponentFiles so the existing
    `_finish_causallm_component` tail in cpp_generator_agent.py can consume
    LLM output unchanged -- writing files, populating state, emitting
    bus.code() events, and writing manifests all stay in one place.
    """
    header: str
    source: str
    header_filename: str
    source_filename: str
    transformer_class: str
    causal_lm_class: str
    architecture: str


# ---------------------------------------------------------------------------
# LLM backend selection
# ---------------------------------------------------------------------------
def _make_llm(state: Dict):
    """
    Return a callable (system, user) -> str, or None if nothing is configured.

    Backends are tried in order:
    1. Claude Code CLI using Opus (the default for C++ generation)
    2. Anthropic API (when an API key is configured)
    3. Cline/OpenAI-compatible endpoint, including Ollama (fallback only)

    All are wrapped so the caller never sees a provider-specific type.
    """
    # Claude is deliberately tried before a persisted local Ollama setting.
    # Existing user settings such as deepseek-coder must not silently replace
    # the requested Opus C++ generator.
    try:
        from .claude_cli_backend import _make_claude_cli_llm
        llm = _make_claude_cli_llm(state)
        if llm:
            bus.log("C++ generation using Claude CLI (preferred backend)")
            return llm
    except Exception as exc:
        bus.log(f"Claude CLI backend unavailable ({exc}); trying Anthropic API", "warn")

    api_key = state.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            from core.llm_manager import LLMManager
            mgr = LLMManager(api_key=api_key,
                             model=state.get("anthropic_model") or "claude-opus-4-6",
                             max_tokens=8192)
            bus.log("C++ generation using Anthropic Opus API")
            return mgr.invoke
        except Exception as exc:
            bus.log(f"Anthropic API unavailable ({exc}); trying Cline/Ollama fallback", "warn")

    cline_token = state.get("cline_ums_token") or os.environ.get("CLINE_UMS_TOKEN") or "ollama"
    cline_base = state.get("cline_api_base") or "http://localhost:11434/v1"
    cline_model = state.get("cline_model") or "codellama:13b"
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
        client = ChatOpenAI(model=cline_model, max_tokens=int(state.get("cline_max_tokens") or 8192),
                            openai_api_key=cline_token, openai_api_base=cline_base,
                            temperature=0, cache=False)
        def _invoke(system: str, user: str) -> str:
            return _as_text(client.invoke([SystemMessage(content=system), HumanMessage(content=user)]).content)
        bus.log(f"C++ generation using Cline/Ollama fallback at {cline_base} (model {cline_model})")
        return _invoke
    except Exception as exc:
        bus.log(f"Cline/Ollama fallback unavailable ({exc})", "warn")
        return None


def _as_text(content) -> str:
    """Normalise LangChain content, which may be str or a list of blocks."""
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content
                       if isinstance(b, dict))
    return content or ""


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------
_FENCE = re.compile(r"```(?:cpp|c\+\+|C\+\+)?\s*\n(.*?)```", re.DOTALL)


def _strip_fences(raw: str) -> str:
    """Concatenate fenced code blocks, or return the text if there are none."""
    blocks = _FENCE.findall(raw)
    return "\n\n".join(blocks) if blocks else raw


def _find_body(code: str, cls: str, method: str) -> Optional[str]:
    """
    Extract the body of `cls::method` from `code` by brace matching.

    Regex alone cannot do this -- bodies contain nested braces from property
    initialiser lists like `{withKey(...), withKey(...)}` -- so the opening
    brace is located by pattern and the matching close found by counting, with
    string and comment spans skipped so a brace inside a literal or a `//`
    comment cannot unbalance the count.
    """
    pat = re.compile(
        rf'\b\w[\w:<>,\s\*&]*?\b{re.escape(cls)}::{re.escape(method)}\s*\(',
        re.DOTALL,
    )
    m = pat.search(code)
    if not m:
        return None

    # Walk to the '{' that opens the body, past the parameter list.
    i = code.find(")", m.end())
    if i == -1:
        return None
    depth_paren = 1
    i = m.end()
    while i < len(code) and depth_paren:
        if code[i] == "(":
            depth_paren += 1
        elif code[i] == ")":
            depth_paren -= 1
        i += 1
    brace = code.find("{", i)
    if brace == -1:
        return None

    depth, j = 1, brace + 1
    in_str = in_chr = in_line = in_block = False
    while j < len(code) and depth:
        c, nxt = code[j], code[j + 1:j + 2]
        if in_line:
            if c == "\n":
                in_line = False
        elif in_block:
            if c == "*" and nxt == "/":
                in_block = False
                j += 1
        elif in_str:
            if c == "\\":
                j += 1
            elif c == '"':
                in_str = False
        elif in_chr:
            if c == "\\":
                j += 1
            elif c == "'":
                in_chr = False
        elif c == "/" and nxt == "/":
            in_line = True
            j += 1
        elif c == "/" and nxt == "*":
            in_block = True
            j += 1
        elif c == '"':
            in_str = True
        elif c == "'":
            in_chr = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return code[brace + 1:j]
        j += 1
    return None


_MARKER = re.compile(
    r"[ \t]*// ===FILL:(\d+)===.*?// ===END:\1===[ \t]*\n?",
    re.DOTALL,
)


def _splice_bodies(skeleton_src: str, raw: str, plan: Dict,
                   holes: List[Dict]) -> Tuple[str, List[str]]:
    """
    Replace each FILL marker in the skeleton with the corresponding body from
    the model output.

    Returns (spliced_source, unfilled_hooks). Any hole whose body could not be
    located keeps its marker, so a partial result is still a valid file and the
    correction pass knows exactly what is missing.
    """
    code = _strip_fences(raw)
    tcls = plan["transformer_class"]
    by_id = {h["id"]: h for h in holes}
    unfilled: List[str] = []

    def _sub(m: re.Match) -> str:
        hole = by_id.get(int(m.group(1)))
        if not hole:
            return m.group(0)
        body = _find_body(code, tcls, hole["hook"])
        if body is None:
            # Also accept a definition written against the CausalLM class --
            # a common near-miss that is otherwise perfectly good code.
            body = _find_body(code, plan["class_name"], hole["hook"])
        if body is None:
            unfilled.append(hole["hook"])
            return m.group(0)
        return body.strip("\n") + "\n"

    return _MARKER.sub(_sub, skeleton_src), unfilled


# ---------------------------------------------------------------------------
# Compile checking
# ---------------------------------------------------------------------------
def _compile_syntax_check(header_path: str, source_path: str) -> Tuple[int, str]:
    """
    Invoke g++ -fsyntax-only on the generated pair.

    Returns (return_code, stderr). For a successful syntax check, return_code
    is 0 and stderr is empty or contains only warnings. For failure, return_code
    is 1 and stderr contains compiler errors.

    The generated files are placed in `header_path`'s directory and that
    directory is included FIRST, so `#include <x_causallm.h>` resolves to the
    generated header, not a same-named hand-written one that would shadow it.
    """
    import subprocess
    gen_dir = os.path.dirname(header_path)
    cmd = [
        "g++", "-std=c++17", "-fsyntax-only",
        f"-I{gen_dir}",
        f"-I{_NNTRAINER_INCLUDE}",
        f"-I{_CAUSALLM_ROOT}",
        f"-I{_CAUSALLM_ROOT}/models",
        f"-I{_CAUSALLM_ROOT}/layers",
        f"-I{_CAUSALLM_ROOT}/third_party/json/include",
        source_path,
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return p.returncode, p.stderr
    except subprocess.TimeoutExpired:
        return 1, "g++ timed out (>30s)"
    except FileNotFoundError:
        return 1, "g++ not found in PATH"
    except Exception as exc:
        return 1, f"compile check failed: {exc}"


def _parse_compile_errors(stderr: str, source: str) -> List[Dict]:
    """
    Parse g++ stderr into structured findings for correction_prompt.

    Extracts file:line:col errors from g++ output and includes the actual
    source line, so the LLM can pinpoint the exact location to fix.
    """
    source_lines = source.split("\n")
    findings = []
    seen_lines = set()

    for line in stderr.split("\n"):
        # Pattern: path/file.cpp:123:45: error: message
        m = re.match(r"^([^:]+):(\d+):(\d+):\s+(error|warning):\s+(.+)$", line)
        if not m:
            continue
        _, line_num_str, col_str, level, detail = m.groups()
        if level == "warning":
            continue

        line_num = int(line_num_str)
        col_num = int(col_str)

        # Avoid duplicate findings for the same line
        if line_num in seen_lines:
            continue
        seen_lines.add(line_num)

        # Get the actual source line (1-indexed)
        source_line = ""
        if 1 <= line_num <= len(source_lines):
            source_line = source_lines[line_num - 1]

        # Point at the column with a caret
        caret = " " * (col_num - 1) + "^"

        findings.append({
            "kind": "compile-error",
            "severity": "error",
            "detail": detail,
            "source_context": f"Line {line_num}, column {col_num}:\n"
                              f"  {source_line}\n"
                              f"  {caret}",
            "fix": f"Fix the error at line {line_num}: {detail}. "
                   f"The marked column is where the problem was detected.",
        })

    return findings[:10]  # Limit to first 10 to keep it focused


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def generate(state: Dict) -> Optional[ComponentFiles]:
    """
    Generate a CausalLM model file pair via LLM, or return None if generation
    is not possible (no semantic_ir, no LLM configured) so the caller can fall
    back. Raises nothing -- every failure is logged and reported as None.
    """
    try:
        plan = model_plan.build_plan(state)
    except ValueError as exc:
        bus.log(f"C++ plan resolution failed: {exc}", "warn")
        return None

    bus.log(f"Plan: {model_plan.plan_summary(plan)}")
    state["cpp_model_plan"] = plan

    sk = skeleton.emit_skeletons(plan)
    holes = sk["holes"]

    # Nothing to generate: the architecture matches the base graph, so the
    # deterministic skeleton IS the finished file. No LLM call needed, and no
    # opportunity for one to introduce an error.
    if not holes:
        bus.log("Architecture matches the base graph exactly -- emitting the "
                "skeleton as final, no LLM call needed.")
        return _finalize(plan, sk, sk["source"], state, method="skeleton_only")

    llm = _make_llm(state)
    if llm is None:
        bus.log(
            "No LLM configured (set aiCompilerWorkbench.clineUmsToken or "
            "anthropicApiKey). Emitting the skeleton with unfilled markers so "
            "it can be completed by hand.", "warn"
        )
        return _finalize(plan, sk, sk["source"], state,
                         method="skeleton_unfilled")

    max_attempts = int(state.get("cpp_max_attempts") or DEFAULT_MAX_ATTEMPTS)
    # Prompt with plan["holes"] narrowed to the holes that are actually marked
    # in the source -- registerCustomLayers is emitted deterministically and
    # must not be requested, or the model writes a competing definition.
    prompt_plan = dict(plan, holes=holes)

    # Get the model-specific prompt module based on architecture
    prompt_module = _get_prompt_module(plan.get("architecture", ""))
    
    # Use per-hole generation (split task) for better reliability with smaller models
    # Each hole is filled in a separate LLM call, reducing timeout risk and complexity
    bus.log(f"Using per-hole generation for {len(holes)} hole(s)")
    
    filled_bodies: Dict[int, str] = {}
    last_source: Optional[str] = None
    last_result: Optional[Dict] = None
    all_findings: List[Dict] = []

    for hole_idx, hole in enumerate(holes):
        bus.log(f"Filling hole {hole_idx + 1}/{len(holes)}: {hole['hook']}")
        
        # Create a single-hole prompt for this specific method
        single_hole_prompt = _create_single_hole_prompt(
            prompt_module, plan, sk["header"], sk["source"], hole, hole_idx
        )
        
        hole_filled = False
        for attempt in range(1, max_attempts + 1):
            bus.log(f"  Attempt {attempt}/{max_attempts} for {hole['hook']}")
            try:
                raw = llm(SYSTEM_PROMPT, single_hole_prompt)
            except Exception as exc:
                bus.log(f"  LLM call failed: {exc}", "error")
                break

            # Extract just this hole's body
            body = _find_body(raw, plan["transformer_class"], hole["hook"])
            if body is None:
                body = _find_body(raw, plan["class_name"], hole["hook"])
            
            if body:
                filled_bodies[hole["id"]] = body.strip("\n")
                hole_filled = True
                bus.log(f"  Successfully filled hole {hole['hook']}")
                break
            else:
                bus.log(f"  Failed to extract body for {hole['hook']}", "warn")
        
        if not hole_filled:
            bus.log(f"  Could not fill hole {hole['hook']} after {max_attempts} attempts", "error")
            all_findings.append({
                "kind": "unfilled-hole",
                "severity": "error",
                "detail": f"No body for {hole['hook']}() could be generated.",
                "fix": f"Manually implement: Tensor {plan['transformer_class']}::{hole['hook']}(...)",
            })

    # Splice all filled bodies into the skeleton
    source = _splice_filled_bodies(sk["source"], filled_bodies, holes)
    last_source = source

    # Validate the complete file
    result = validate(source, prompt_plan, strict=True)
    last_result = result
    findings = list(result["findings"])
    findings.extend(all_findings)

    # Check for unfilled holes
    unfilled = [h["hook"] for h in holes if h["id"] not in filled_bodies]
    for hook in unfilled:
        findings.append({
            "kind": "unfilled-hole",
            "severity": "error",
            "detail": f"No body for {hook}() could be found in the output.",
            "fix": f"Emit a complete definition: "
                   f"Tensor {plan['transformer_class']}::{hook}(...) "
                   f"{{ ... }}",
        })

    # Semantic validation passed -- now check syntax by running the compiler.
    if not [f for f in findings if f.get("severity") == "error"]:
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="wb_compile_")
        hdr_path = os.path.join(tmpdir, sk["header_filename"])
        src_path = os.path.join(tmpdir, sk["source_filename"])
        with open(hdr_path, "w") as f:
            f.write(sk["header"])
        with open(src_path, "w") as f:
            f.write(source)
        rc, err = _compile_syntax_check(hdr_path, src_path)
        if rc == 0:
            bus.log(f"C++ generation validated. {format_report(result)}")
            state["cpp_validation"] = result
            state["cpp_gaps"] = result["gaps"]
            return _finalize(plan, sk, source, state, method="llm")
        # Compile failed: parse errors and retry
        compile_errors = _parse_compile_errors(err, source)
        if compile_errors:
            findings.extend(compile_errors)
            bus.log(f"Passed semantic validation but failed compilation: {len(compile_errors)} error(s)",
                    "warn")

    errors = [f for f in findings if f.get("severity") == "error"]
    if not errors:
        bus.log(f"C++ generation validated. {format_report(result)}")
        state["cpp_validation"] = result
        state["cpp_gaps"] = result["gaps"]
        return _finalize(plan, sk, source, state, method="llm")

    bus.log(f"Validation issues: {len(errors)} error(s)", "warn")
    for f in errors[:6]:
        bus.log(f"  {f['kind']}: {f['detail']}", "warn")

    # Emit the best source we have rather than nothing
    if last_source is not None:
        bus.log("C++ generation did not fully validate -- emitting the best "
                "attempt with findings recorded for review.", "warn")
        if last_result:
            state["cpp_validation"] = last_result
            state["cpp_gaps"] = last_result["gaps"]
        return _finalize(plan, sk, last_source, state,
                         method="llm_unvalidated")

    # Out of attempts. Emit the best source we have rather than nothing: the
    # scaffolding is ours and correct, so the result is a reviewable starting
    # point with the specific problems recorded in state for the UI.
    if last_source is not None:
        bus.log("C++ generation did not fully validate -- emitting the best "
                "attempt with findings recorded for review.", "warn")
        if last_result:
            state["cpp_validation"] = last_result
            state["cpp_gaps"] = last_result["gaps"]
        return _finalize(plan, sk, last_source, state,
                         method="llm_unvalidated")
    return None


# ---------------------------------------------------------------------------
# Helper functions for per-hole generation
# ---------------------------------------------------------------------------
def _create_single_hole_prompt(prompt_module, plan: Dict, skeleton_h: str, 
                                skeleton_cpp: str, hole: Dict, hole_idx: int) -> str:
    """
    Create a focused prompt for filling a single hole.
    
    This is much smaller than the full prompt, reducing timeout risk and
    improving quality by focusing the model on one method at a time.
    """
    hook = hole["hook"]
    tcls = plan["transformer_class"]
    
    # Extract just the relevant method signature from skeleton
    sig_match = re.search(
        rf'(Tensor|void|std::pair<Tensor,\s*Tensor>)\s+{re.escape(tcls)}::{re.escape(hook)}\s*\([^)]*\)',
        skeleton_cpp, re.DOTALL
    )
    signature = sig_match.group(0) if sig_match else f"Tensor {tcls}::{hook}(...)"
    
    # Get the specific layers this hole needs
    layers_info = ""
    if hole.get("layers"):
        layers_info = f"\n   Layers needed: {', '.join(hole['layers'])}"
    
    # Get relevant section from the plan
    section = {
        "createAttention": "attention",
        "createMlp": "mlp", 
        "createTransformerDecoderBlock": "block",
        "constructModel": "construct",
        "setupParameters": "setup_extra",
    }.get(hook, hook)
    
    plan_section = plan.get("resolved", {}).get(section, plan.get(section, {}))
    
    return f"""\
Fill in ONE method body in the C++ skeleton below.

Model: {plan.get('model_id', '<unknown>')}
Architecture: {plan.get('architecture', '<unknown>')}
Method to implement: {hook}

============================ 1. METHOD SIGNATURE =============================
Implement this exact signature. Do not change the signature:

```cpp
{signature}
```

============================ 2. THE PLAN =====================================
Relevant section for {hook}:
{json.dumps(plan_section, indent=2)}
{layers_info}

============================ 3. LAYER MENU ===================================
Only these layers are available. Copy type strings exactly:
{render_catalog(only=plan.get('layer_types'))}

============================ 4. WORKED EXAMPLE ===============================
This is `Qwen3Transformer::createAttention` verbatim:

```cpp
Tensor Qwen3Transformer::createAttention(const int layer_id, int seq_len,
                                         int n_heads, int head_dim,
                                         Tensor query, Tensor key,
                                         Tensor value) {{

  // Q layer
  LayerHandle wq(createLayer(
    "fully_connected",
    {{withKey("name", "layer" + std::to_string(layer_id) + "_wq"),
      withKey("unit", head_dim * n_heads), withKey("disable_bias", "true"),
      withKey("weight_initializer", "ones")}}));
  Tensor q = wq(query);

  // ... rest of layers ...
  return wo(a);
}}
```

============================ 5. RULES ========================================
1. Layer names MUST match the naming convention in the plan
2. Use only layers from the menu above
3. Return the correct type (Tensor/void/std::pair)
4. Use withKey() for properties
5. Call createKVCachePlaceholders() if attention needs KV cache

============================ 6. OUTPUT FORMAT ================================
Output ONLY the method body (the code between {{ and }}), nothing else.
Do not include the signature, no fences, no explanations.
Just the implementation code.
"""

def _splice_filled_bodies(skeleton_src: str, filled_bodies: Dict[int, str], 
                          holes: List[Dict]) -> str:
    """
    Replace each FILL marker in the skeleton with the corresponding pre-filled body.
    
    Args:
        skeleton_src: The skeleton source with FILL markers
        filled_bodies: Dict mapping hole_id -> filled body text
        holes: List of hole definitions
    
    Returns:
        The complete source with all available bodies spliced in
    """
    by_id = {h["id"]: h for h in holes}
    
    def _sub(m: re.Match) -> str:
        hole_id = int(m.group(1))
        hole = by_id.get(hole_id)
        if not hole:
            return m.group(0)
        
        body = filled_bodies.get(hole_id)
        if body is None:
            # Keep the marker for unfilled holes
            return m.group(0)
        
        return body.strip("\n") + "\n"

    return _MARKER.sub(_sub, skeleton_src)


def _finalize(plan: Dict, sk: Dict, source: str, state: Dict,
              method: str) -> ComponentFiles:
    """Package the result and record plan-level gaps for the UI."""
    gaps = list(state.get("cpp_gaps") or [])
    for g in plan.get("gaps", []):
        if g not in gaps:
            gaps.append(g)
    if gaps:
        state["cpp_gaps"] = gaps
        bus.log(f"{len(gaps)} unmapped op(s) recorded as contributor tasks:",
                "warn")
        for g in gaps:
            bus.log(f"  gap: {g.get('op')} -- {g.get('reason', '')}", "warn")

    state["cpp_generation_method"] = method
    return ComponentFiles(
        header=sk["header"],
        source=source,
        header_filename=sk["header_filename"],
        source_filename=sk["source_filename"],
        transformer_class=plan["transformer_class"],
        causal_lm_class=plan["class_name"],
        architecture=plan.get("arch_key") or plan.get("architecture", ""),
    )
