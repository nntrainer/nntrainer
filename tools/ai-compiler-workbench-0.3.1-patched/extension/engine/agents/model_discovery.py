"""
Model Discovery Agent (no LLM).

Discovers and extracts model metadata/architecture from the Hugging
Face Hub (or a local path) -- config.json, tokenizer info, declared
architecture class -- without downloading weights yet. Mirrors the
"Model Discovery Agent" box in the architecture diagram.
"""
from .events import bus


def run(state: dict) -> dict:
    bus.agent_status("model_discovery", "running")
    model_name = state["model_name"]

    try:
        from transformers import AutoConfig
    except ImportError:
        bus.agent_status("model_discovery", "error", "transformers not installed")
        state.setdefault("errors", []).append(
            "transformers/torch not found -- run 'AI Compiler Workbench: Check Environment'"
        )
        return state

    try:
        config = AutoConfig.from_pretrained(model_name)
        hf_config = config.to_dict()
    except Exception as exc:
        bus.log(f"Could not load config for '{model_name}': {exc}", "error")
        bus.agent_status("model_discovery", "error", str(exc))
        state.setdefault("errors", []).append(f"model_discovery: {exc}")
        return state

    state["hf_config"] = hf_config
    state["architecture"] = (hf_config.get("architectures") or [type(config).__name__])[0]

    bus.log(f"Discovered architecture: {state['architecture']}")
    bus.log(
        f"hidden_size={hf_config.get('hidden_size', '?')} "
        f"num_layers={hf_config.get('num_hidden_layers', hf_config.get('n_layer', '?'))} "
        f"vocab_size={hf_config.get('vocab_size', '?')}"
    )
    bus.agent_status("model_discovery", "done", state["architecture"])
    return state
