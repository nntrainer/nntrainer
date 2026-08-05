"""
Auto-Fix Agent (LLM).

Runs only when the Compiler Agent reports real g++ errors against the
generated file (not for unsupported-op TODOs -- those are a known,
already-explained gap, not a bug to "fix" by guessing). Sends the
compiler's stderr plus the offending source to an LLM via LangChain
and asks for the corrected file back, verbatim. Bounded to
MAX_FIX_ITERATIONS so a model that can't converge on a fix doesn't
loop forever; each attempt is a fresh compile, so a bad patch just
falls through to "leave the TODOs / errors for the user" rather than
silently shipping something unverified.
"""
from .events import bus

MAX_FIX_ITERATIONS = 2


def run(state: dict) -> dict:
    api_key = state.get("anthropic_api_key")
    code = state.get("cpp_code")
    compile_log = state.get("compile_log", "")

    if not api_key or not code or not compile_log:
        return state

    bus.agent_status("auto_fix", "running", f"attempt {state.get('fix_iterations', 0) + 1}")

    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage
    except ImportError:
        bus.log("langchain-anthropic not installed -- cannot auto-fix compile errors", "warn")
        bus.agent_status("auto_fix", "error", "langchain-anthropic not installed")
        return state

    prompt = (
        "The following nntrainer C++ file failed to compile. Fix ONLY what the "
        "compiler errors point to -- do not invent new nntrainer APIs, do not "
        "remove the TODO(unsupported) comment blocks. Return the complete "
        "corrected file and nothing else (no markdown fences, no commentary).\n\n"
        f"--- compiler output ---\n{compile_log[:4000]}\n\n"
        f"--- generated_model.cpp ---\n{code}\n"
    )

    try:
        llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=4000, api_key=api_key)
        resp = llm.invoke([HumanMessage(content=prompt)])
        fixed = (resp.content or "").strip()
        if fixed.startswith("```"):
            fixed = fixed.split("\n", 1)[1].rsplit("```", 1)[0]

        state["cpp_code"] = fixed
        with open(state["cpp_path"], "w", encoding="utf-8") as f:
            f.write(fixed)
        state["fix_iterations"] = state.get("fix_iterations", 0) + 1

        bus.code("generated_model.cpp", fixed)
        bus.log(f"Auto-fix attempt {state['fix_iterations']} applied -- recompiling")
        bus.agent_status("auto_fix", "done", f"attempt {state['fix_iterations']}")
    except Exception as exc:
        bus.log(f"Auto-fix request failed: {exc}", "warn")
        bus.agent_status("auto_fix", "error", str(exc))

    return state
