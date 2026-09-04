"""Auto-Fix Agent for compiler errors."""
import subprocess
import sys
from .events import bus

MAX_FIX_ITERATIONS = 2


def _ensure_langchain_installed(api_key: str) -> bool:
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage
        return True
    except ImportError:
        pass
    
    bus.log("langchain-anthropic not installed -- attempting auto-install...", "warn")
    bus.agent_status("auto_fix", "running", "installing langchain-anthropic...")
    
    try:
        python_path = sys.executable or "python3"
        pip_args = ["-m", "pip", "install", "-q", "langchain-anthropic", "langchain-core"]
        subprocess.check_call([python_path] + pip_args)
        bus.log("langchain-anthropic installed successfully", "info")
        return True
    except subprocess.CalledProcessError as e:
        bus.log(f"Failed to install langchain-anthropic: {e}", "error")
        bus.agent_status("auto_fix", "error", "pip install failed")
        return False
    except Exception as e:
        bus.log(f"Failed to install langchain-anthropic: {e}", "error")
        bus.agent_status("auto_fix", "error", str(e))
        return False


def run(state: dict) -> dict:
    api_key = state.get("anthropic_api_key")
    code = state.get("cpp_code")
    compile_log = state.get("compile_log", "")

    if not api_key or not code or not compile_log:
        return state

    bus.agent_status("auto_fix", "running", f"attempt {state.get('fix_iterations', 0) + 1}")

    if not _ensure_langchain_installed(api_key):
        return state

    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage
    except ImportError:
        bus.log("langchain-anthropic still not available after install attempt", "warn")
        bus.agent_status("auto_fix", "error", "langchain-anthropic not available")
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
