"""
Auto-Fix Agent for compiler errors.

This agent uses Cline (Gauss models only) with UMS token to fix C++ compilation errors.
It supports:
- Configurable max iterations
- Cline API (OpenAI-compatible) with Gauss models
- Error categorization (fixable vs unfixable)
- Smarter prompt engineering with error analysis
- Rule-based corrections via CppCorrector integration
- Fix history for learning from past attempts
"""
import os
import re
import subprocess
import sys
from typing import Optional, Dict, List, Any, Tuple

from ..events import bus
from ..cpp_generator.cpp_corrector import CppCorrector

# Default configuration - All LLM calls use Cline with UMS token (Gauss models only)
# Capped at 2: this is a bounded, mechanical-fix-first loop, not an
# open-ended retry loop (see _apply_rule_based_fixes / balance_braces).
DEFAULT_MAX_FIX_ITERATIONS = 2
DEFAULT_CLINE_API_BASE = "http://localhost:6543/v1"
DEFAULT_CLINE_MODEL = "gauss-4-5"  # Default Gauss model
DEFAULT_MAX_TOKENS = 40


# -----------------------------------------------------------------------------
# Error Categorization
# -----------------------------------------------------------------------------

FIXABLE_ERROR_PATTERNS = [
    (r"error:\s*.*was not declared in this scope", "missing_declaration"),
    (r"error:\s*expected.*before.*token", "syntax_error"),
    (r"error:\s*no matching function for call", "wrong_signature"),
    (r"error:\s*no member named.*in", "missing_member"),
    (r"error:\s*use of undeclared identifier", "undeclared_identifier"),
    (r"error:\s*cannot convert.*to.*in assignment", "type_mismatch"),
    (r"error:\s*invalid conversion.*to.*from", "invalid_conversion"),
    (r"error:\s*too few arguments to function", "missing_arguments"),
    (r"error:\s*too many arguments to function", "extra_arguments"),
    (r"error:\s*class.*has no member named", "missing_class_member"),
    (r"error:\s*variable or field.*declared void", "void_declaration"),
    (r"error:\s*expected.*at end of input", "missing_brace"),
    (r"warning:\s*unused variable", "unused_variable"),
    (r"warning:\s*unused parameter", "unused_parameter"),
]

UNFIXABLE_ERROR_PATTERNS = [
    (r"fatal error:\s*.*: No such file or directory", "missing_header_file"),
    (r"fatal error:\s*Python\.h: No such file or directory", "missing_python_dev"),
    (r"undefined reference to", "linker_error"),
    (r"cannot find -l", "missing_library"),
    (r"ld returned 1 exit status", "linker_failure"),
    (r"collect2: error: ld returned", "linker_failure"),
    (r"error:.*nntrainer not found", "nntrainer_not_installed"),
    (r"error:.*NNTRAINER_INCLUDE_DIR", "nntrainer_path_not_set"),
]


def _categorize_errors(compile_log: str) -> Tuple[List[Dict], List[Dict], str]:
    """
    Categorize compiler errors into fixable and unfixable.
    
    Returns:
        (fixable_errors, unfixable_errors, raw_log)
    """
    fixable = []
    unfixable = []
    
    for line in compile_log.splitlines():
        # Check fixable patterns
        for pattern, error_type in FIXABLE_ERROR_PATTERNS:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                fixable.append({
                    "line": line.strip(),
                    "type": error_type,
                    "pattern": pattern,
                })
                break
        else:
            # Check unfixable patterns
            for pattern, error_type in UNFIXABLE_ERROR_PATTERNS:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    unfixable.append({
                        "line": line.strip(),
                        "type": error_type,
                        "pattern": pattern,
                    })
                    break
    
    return fixable, unfixable, compile_log


def _get_first_n_errors(fixable: List[Dict], n: int = 5) -> str:
    """Get a summary of the first N distinct fixable errors."""
    seen_types = set()
    distinct_errors = []
    
    for err in fixable:
        if err["type"] not in seen_types:
            seen_types.add(err["type"])
            distinct_errors.append(err)
        if len(distinct_errors) >= n:
            break
    
    if not distinct_errors:
        return "No distinct fixable errors identified."
    
    lines = ["Identified compilation errors:"]
    for i, err in enumerate(distinct_errors, 1):
        lines.append(f"  {i}. [{err['type']}] {err['line'][:100]}")
    
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# LLM Client
# -----------------------------------------------------------------------------

def _get_llm_client(
    cline_ums_token: str,
    cline_api_base: str = None,
    cline_model: str = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
):
    """
    Get LLM client - Cline with UMS token (Gauss models only).
    
    Args:
        cline_ums_token: Cline UMS token
        cline_api_base: Cline API base URL
        cline_model: Model name for Cline (Gauss model)
        max_tokens: Max tokens for response
    
    Returns:
        LLM client instance or None if unavailable
    """
    cline_api_base = cline_api_base or DEFAULT_CLINE_API_BASE
    cline_model = cline_model or DEFAULT_CLINE_MODEL
    
    # Use Cline's OpenAI-compatible API with UMS token (Gauss models only)
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        bus.log("langchain-openai not installed -- attempting auto-install...", "warn")
        bus.agent_status("auto_fix", "running", "installing langchain-openai...")
        try:
            python_path = sys.executable or "python3"
            subprocess.check_call([python_path, "-m", "pip", "install", "-q", "langchain-openai"])
            from langchain_openai import ChatOpenAI
            bus.log("langchain-openai installed successfully", "info")
        except Exception as e:
            bus.log(f"Failed to install langchain-openai: {e}", "error")
            bus.agent_status("auto_fix", "error", "pip install failed")
            return None
    
    bus.log(f"Using Cline API at {cline_api_base} with Gauss model {cline_model}", "info")
    
    return ChatOpenAI(
        model=cline_model,
        max_tokens=max_tokens,
        openai_api_key=cline_ums_token,
        openai_api_base=cline_api_base,
        temperature=0,  # Deterministic for code fixes
    )


# -----------------------------------------------------------------------------
# Prompt Engineering
# -----------------------------------------------------------------------------

def _build_prompt(
    compile_log: str,
    code: str,
    fixable_errors: List[Dict],
    unfixable_errors: List[Dict],
    fix_history: List[Dict] = None,
) -> str:
    """
    Build an optimized prompt for the LLM.
    
    Args:
        compile_log: Raw compiler output
        code: The C++ code to fix
        fixable_errors: List of categorized fixable errors
        unfixable_errors: List of categorized unfixable errors
        fix_history: Past fix attempts for learning
    
    Returns:
        Formatted prompt string
    """
    # Build error summary
    error_summary = _get_first_n_errors(fixable_errors, 5)
    
    # Build unfixable warning if present
    unfixable_warning = ""
    if unfixable_errors:
        unfixable_types = set(e["type"] for e in unfixable_errors)
        unfixable_warning = (
            f"\n\n⚠️ UNFIXABLE ERRORS DETECTED: {', '.join(unfixable_types)}\n"
            "These are system/environment issues (missing headers, linker errors).\n"
            "Focus ONLY on the fixable errors listed above.\n"
        )
    
    # Build learning from history
    history_context = ""
    if fix_history and len(fix_history) > 0:
        recent = fix_history[-2:]  # Last 2 attempts
        history_lines = ["\n\nPrevious fix attempts and outcomes:"]
        for i, attempt in enumerate(recent, 1):
            outcome = "✓ SUCCESS" if attempt.get("success") else "✗ FAILED"
            history_lines.append(f"  {i}. [{outcome}] Fixed: {attempt.get('error_type', 'unknown')}")
        history_context = "\n".join(history_lines)
    
    # Common issues reference
    common_issues = """
COMMON NNTRAINER C++ ISSUES AND FIXES:
1. Missing #include: Add `#include <nntrainer/layer.h>` or similar
2. Wrong namespace: Use `nntrainer::Layer` not just `Layer`
3. Missing semicolon: Check end of class/function declarations
4. Wrong method signature: Check nntrainer API for correct parameters
5. Undefined variable: Declare before use or check spelling
"""
    
    # Full prompt
    prompt = f"""You are an expert C++ developer fixing nntrainer compilation errors.

{error_summary}
{unfixable_warning}
{history_context}
{common_issues}

INSTRUCTIONS:
1. Focus on the FIRST error - others may be cascading effects
2. Fix ONLY what the compiler errors indicate - don't rewrite unrelated code
3. DO NOT invent new nntrainer APIs - use only existing, documented APIs
4. DO NOT remove TODO(unsupported) comment blocks
5. Preserve existing code structure and style
6. Return the COMPLETE corrected file - no markdown fences, no commentary

--- COMPILER OUTPUT ---
{compile_log[:4000]}

--- GENERATED_MODEL.CPP ---
{code}
"""
    
    return prompt


def _strip_markdown_fences(content: str) -> str:
    """Remove markdown code fences from LLM response."""
    content = content.strip()
    
    # Handle ```cpp or ``` at start
    if content.startswith("```"):
        lines = content.split("\n", 1)
        if len(lines) > 1:
            content = lines[1]
    
    # Handle ``` at end
    if content.endswith("```"):
        content = content.rsplit("```", 1)[0]
    
    return content.strip()


# -----------------------------------------------------------------------------
# Rule-Based Corrections (CppCorrector Integration)
# -----------------------------------------------------------------------------

def _apply_rule_based_fixes(code: str, compile_log: str) -> Tuple[str, List[str]]:
    """
    Apply rule-based corrections before LLM invocation.

    Returns:
        (corrected_code, list_of_applied_fixes)
    """
    corrections = []

    # Rule 0: Balance unmatched braces -- the "just fix the brackets, don't
    # touch the logic" mechanical fix. Only fires when the compile log
    # itself points at a brace-balance problem.
    code, brace_fixes = CppCorrector.balance_braces(code, compile_log)
    corrections.extend(brace_fixes)

    # Rule 1: Add missing nntrainer include if nntrainer types are used
    if "nntrainer::" in code and "#include <nntrainer/" not in code:
        code = "#include <nntrainer/nntrainer.h>\n" + code
        corrections.append("Added missing nntrainer include")
    
    # Rule 2: Fix common typo: "std::cout" without iostream
    if "std::cout" in code and "#include <iostream>" not in code:
        code = "#include <iostream>\n" + code
        corrections.append("Added missing iostream include")
    
    # Rule 3: Ensure using namespace std if std:: is used heavily
    if code.count("std::") > 3 and "using namespace std;" not in code:
        # Find position after last include
        lines = code.split("\n")
        insert_pos = 0
        for i, line in enumerate(lines):
            if line.startswith("#include"):
                insert_pos = i + 1
        lines.insert(insert_pos, "using namespace std;")
        code = "\n".join(lines)
        corrections.append("Added 'using namespace std;'")
    
    # Rule 4: Fix missing return type in main
    if "main()" in code and "int main()" not in code and "void main()" not in code:
        code = code.replace("main()", "int main()")
        corrections.append("Fixed main() return type")
    
    # Rule 5: Add return 0; to main if missing
    if "int main()" in code and "return 0;" not in code:
        # Find end of main function (simplified heuristic)
        lines = code.split("\n")
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == "}":
                lines.insert(i, "    return 0;")
                corrections.append("Added 'return 0;' to main")
                break
        code = "\n".join(lines)
    
    if corrections:
        bus.log(f"Applied {len(corrections)} rule-based correction(s): {', '.join(corrections)}", "info")
    
    return code, corrections


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def run(state: dict) -> dict:
    """
    Auto-Fix Agent: Fix compilation errors using Cline (Gauss models only).
    
    Args:
        state: Pipeline state dict with:
            - cline_ums_token: UMS token for Cline
            - cpp_code: Current C++ code
            - cpp_path: Path to write fixed code
            - compile_log: Compiler error output
            - fix_iterations: Number of fix attempts so far
            - max_fix_iterations: Maximum allowed attempts
            - cline_api_base: Cline API URL
            - cline_model: Cline model name (Gauss model)
            - cline_max_tokens: Max tokens for LLM
            - fix_history: Past fix attempts
    
    Returns:
        Updated state dict
    """
    # Extract configuration from state
    cline_ums_token = state.get("cline_ums_token")
    code = state.get("cpp_code")
    compile_log = state.get("compile_log", "")
    cpp_path = state.get("cpp_path")
    
    # Configuration with defaults (all Cline/Gauss)
    cline_api_base = state.get("cline_api_base", DEFAULT_CLINE_API_BASE)
    cline_model = state.get("cline_model", DEFAULT_CLINE_MODEL)
    max_tokens = state.get("cline_max_tokens", DEFAULT_MAX_TOKENS)
    max_fix_iterations = state.get("max_fix_iterations", DEFAULT_MAX_FIX_ITERATIONS)
    fix_history = state.get("fix_history", [])
    
    # Check prerequisites. Note: a Cline token is NOT required here -- the
    # mechanical rule-based fixes (brace balancing, missing includes, etc.)
    # run without any LLM dependency. The token is only checked later, right
    # before the LLM fallback, so "no manual intervention" doesn't silently
    # require a locally-running Cline proxy just to fix a missing brace.
    if not code or not compile_log:
        bus.log("No code or compile log available -- skipping auto-fix", "warn")
        return state
    
    current_attempt = state.get("fix_iterations", 0) + 1
    if current_attempt > max_fix_iterations:
        bus.log(f"Max fix iterations ({max_fix_iterations}) reached -- skipping auto-fix", "warn")
        return state
    
    bus.agent_status("auto_fix", "running", f"attempt {current_attempt}")
    
    # Step 1: Categorize errors
    fixable, unfixable, _ = _categorize_errors(compile_log)
    
    if not fixable:
        if unfixable:
            unfixable_types = set(e["type"] for e in unfixable)
            bus.log(f"No fixable errors detected. Unfixable: {', '.join(unfixable_types)}", "warn")
        else:
            bus.log("No errors identified in compile log -- skipping auto-fix", "warn")
        return state
    
    bus.log(f"Identified {len(fixable)} fixable and {len(unfixable)} unfixable errors", "info")

    # Step 2: Apply rule-based corrections first. These are mechanical
    # (brace balancing, missing includes, etc.) and never rewrite logic --
    # prefer them over an LLM rewrite whenever they make any change at all,
    # and let the caller recompile/rebuild to see if that was enough before
    # spending an LLM call.
    code, rule_fixes = _apply_rule_based_fixes(code, compile_log)
    if rule_fixes:
        state["cpp_code"] = code
        if cpp_path:
            with open(cpp_path, "w", encoding="utf-8") as f:
                f.write(code)

        state["fix_iterations"] = current_attempt
        fix_history.append({
            "attempt": current_attempt,
            "error_types": [e["type"] for e in fixable[:3]],
            "rule_fixes": rule_fixes,
            "used_llm": False,
            "timestamp": __import__("time").time(),
        })
        state["fix_history"] = fix_history[-5:]

        bus.code("generated_model.cpp", code)
        bus.log(f"Rule-based fix applied ({', '.join(rule_fixes)}) -- recompiling without LLM")
        bus.agent_status("auto_fix", "done", f"attempt {current_attempt} (rule-based)")
        return state

    # Step 2b: No mechanical fix applied. Only fall back to an LLM rewrite
    # if explicitly opted in -- by default auto-fix should not touch code
    # it can't fix mechanically, since a full-file LLM rewrite risks
    # changing logic the generator already got right.
    if not state.get("auto_fix_use_llm", False):
        bus.log("No mechanical fix available and LLM fallback is disabled -- stopping auto-fix", "warn")
        bus.agent_status("auto_fix", "done", "no mechanical fix available")
        return state

    if not cline_ums_token:
        bus.log("No mechanical fix available and no Cline UMS token configured -- stopping auto-fix", "warn")
        bus.agent_status("auto_fix", "done", "no token for LLM fallback")
        return state

    # Step 3: Build optimized prompt
    prompt = _build_prompt(
        compile_log=compile_log,
        code=code,
        fixable_errors=fixable,
        unfixable_errors=unfixable,
        fix_history=fix_history,
    )
    
    # Step 4: Get LLM client (Cline with UMS token, Gauss models only)
    llm = _get_llm_client(
        cline_ums_token=cline_ums_token,
        cline_api_base=cline_api_base,
        cline_model=cline_model,
        max_tokens=max_tokens,
    )
    
    if llm is None:
        bus.agent_status("auto_fix", "error", "LLM client not available")
        return state
    
    # Step 5: Call LLM
    try:
        from langchain_core.messages import HumanMessage
        
        resp = llm.invoke([HumanMessage(content=prompt)])
        fixed = _strip_markdown_fences(resp.content or "")
        
        if not fixed:
            bus.log("LLM returned empty response", "warn")
            bus.agent_status("auto_fix", "error", "empty response")
            return state
        
        # Step 6: Update state
        state["cpp_code"] = fixed
        if cpp_path:
            with open(cpp_path, "w", encoding="utf-8") as f:
                f.write(fixed)
        
        state["fix_iterations"] = current_attempt
        
        # Step 7: Record fix history
        fix_entry = {
            "attempt": current_attempt,
            "error_types": [e["type"] for e in fixable[:3]],
            "rule_fixes": rule_fixes,
            "timestamp": __import__("time").time(),
        }
        fix_history.append(fix_entry)
        state["fix_history"] = fix_history[-5:]  # Keep last 5 attempts
        
        # Step 8: Notify webview
        bus.code("generated_model.cpp", fixed)
        bus.log(f"Auto-fix attempt {current_attempt}/{max_fix_iterations} applied -- recompiling")
        bus.agent_status("auto_fix", "done", f"attempt {current_attempt}")
        
    except Exception as exc:
        bus.log(f"Auto-fix request failed: {exc}", "warn")
        bus.agent_status("auto_fix", "error", str(exc))
    
    return state
