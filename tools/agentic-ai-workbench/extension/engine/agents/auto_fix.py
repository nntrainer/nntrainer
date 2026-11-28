# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file auto_fix.py
# @brief Auto-Fix Agent for compiler errors.
# @author prachi-t17 <prachi.t@samsung.com>

"""Auto-Fix Agent for compiler errors."""
from .events import bus
from .optional_deps import API_MODEL, install_hint, missing


def run(state: dict) -> dict:
    """Pipeline step: ask Claude (API key) to fix the compile errors in generated_model.cpp."""
    api_key = state.get("anthropic_api_key")
    code = state.get("cpp_code")
    compile_log = state.get("compile_log", "")

    if not api_key or not code or not compile_log:
        return state

    bus.agent_status("auto_fix", "running", f"attempt {state.get('fix_iterations', 0) + 1}")

    gone = missing("langchain_anthropic", "langchain_core")
    if gone:
        bus.log(
            f"Auto-Fix needs the optional LangChain packages ({', '.join(gone)} not installed). "
            + install_hint(),
            "warn",
        )
        bus.agent_status("auto_fix", "skipped", "LangChain not installed")
        return state

    # pylint: disable-next=import-outside-toplevel,import-error  # optional package
    from langchain_anthropic import ChatAnthropic
    # pylint: disable-next=import-outside-toplevel,import-error  # optional package
    from langchain_core.messages import HumanMessage

    prompt = (
        "The following nntrainer C++ file failed to compile. Fix ONLY what the "
        "compiler errors point to -- do not invent new nntrainer APIs, do not "
        "remove the TODO(unsupported) comment blocks. Return the complete "
        "corrected file and nothing else (no markdown fences, no commentary).\n\n"
        f"--- compiler output ---\n{compile_log[:4000]}\n\n"
        f"--- generated_model.cpp ---\n{code}\n"
    )

    try:
        llm = ChatAnthropic(model=API_MODEL, max_tokens=4000, api_key=api_key)
        resp = llm.invoke([HumanMessage(content=prompt)])
        fixed = (resp.content or "").strip()
        if fixed.startswith("```"):
            fixed = fixed.split("\n", 1)[1].rsplit("```", 1)[0]

        state["cpp_code"] = fixed
        with open(state["cpp_path"], "w", encoding="utf-8") as handle:
            handle.write(fixed)
        state["fix_iterations"] = state.get("fix_iterations", 0) + 1

        bus.code("generated_model.cpp", fixed)
        bus.log(f"Auto-fix attempt {state['fix_iterations']} applied -- recompiling")
        bus.agent_status("auto_fix", "done", f"attempt {state['fix_iterations']}")
    # the fix is optional; any failure is only reported
    except Exception as exc:  # pylint: disable=broad-except
        bus.log(f"Auto-fix request failed: {exc}", "warn")
        bus.agent_status("auto_fix", "error", str(exc))

    return state
