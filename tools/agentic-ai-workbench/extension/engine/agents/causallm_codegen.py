# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file causallm_codegen.py
# @brief CausalLM C++ Generator (Claude CLI) -- pipeline step after cpp_generator.
# @author prachi-t17 <prachi.t@samsung.com>

"""
CausalLM C++ Generator (Claude CLI) -- pipeline step after cpp_generator.

For the model being converted:
  1. reads its config.json and tensor names/shapes without downloading
     weights (local snapshot -> Hub safetensors headers -> transformers model
     on the meta device);
  2. ticks the checklist and plans against the code bank (verified code is
     inherited or copied; only new parts are written);
  3. has the Claude CLI write the .h/.cpp (+ converter), validates the reply
     and retries with the errors (knowledge/codegen.py);
  4. writes everything to tools/agentic-ai-workbench/generated/<slug>/ and
     publishes the review (Review tab) and the files (C++ pane, Export).

The bank is filled from Applications/CausalLM on first use.
Settings arrive as environment variables from the extension:
  NNTRAINER_ROOT, CLAUDE_PATH, CLAUDE_MODEL, CLAUDE_CLI_TIMEOUT,
  WORKBENCH_CODEGEN (0 = off), WORKBENCH_CODEGEN_ATTEMPTS.
"""
import importlib
import inspect
import json
import os
import re

from agents.cpp.claude_cli_backend import ClaudeCLI, ClaudeCLIError
from knowledge import checklist as ck
from knowledge import codegen as cg
from knowledge.capture_causallm import capture, ensure_core_layer_types, find_nntrainer_root
from knowledge.code_bank import CodeBank, CodeBankError
from knowledge.codegen import CodegenError

from .events import bus

AGENT = "causallm_codegen"
_HF_CLASS = re.compile(
    r"(Attention|MLP|Mlp|MoeBlock|SparseMoeBlock|MoE|Experts|Router|TopKRouter|FeedForward|"
    r"DecoderLayer|RMSNorm|LayerNorm|RotaryEmbedding|ScaledWordEmbedding|Model)$"
)
_HF_FUNCS = ("rotate_half", "apply_rotary_pos_emb", "repeat_kv", "eager_attention_forward")


_DECODER_SUFFIXES = ("ForCausalLM", "LMHeadModel")


def not_causal_lm_reason(cfg):
    """None for a decoder-only (causal) language model, else why it is not one.
    The generator writes CausalLM classes only; encoders (BERT, RoBERTa,
    DeBERTa, ...) and encoder-decoders (T5, BART, ...) would come out as
    confidently wrong code."""
    cfg = cfg or {}
    archs = [a for a in (cfg.get("architectures") or []) if isinstance(a, str)]
    if any(a.endswith(_DECODER_SUFFIXES) for a in archs):
        return None
    text = cfg.get("text_config")
    if any(a.endswith("ForConditionalGeneration") for a in archs) and isinstance(text, dict) and \
            not cfg.get("is_encoder_decoder"):
        return None                                   # multimodal model with a decoder LLM inside
    if not archs and cfg.get("is_decoder") and not cfg.get("is_encoder_decoder"):
        return None
    what = archs[0] if archs else f"model_type '{cfg.get('model_type', '?')}'"
    kind = (
        "an encoder-decoder model"
        if cfg.get("is_encoder_decoder")
        else "not a causal (decoder-only) language model"
    )
    return (
        f"{what} is {kind}. The Claude CLI generator writes nntrainer CausalLM (decoder) code only,"
        " so it was skipped. Use a ...ForCausalLM model (e.g. Qwen/Qwen3-0.6B)."
    )


def make_llm():
    """(system, user) -> reply using the Claude CLI; raises CodegenError on failure."""
    try:
        cli = ClaudeCLI(
            os.environ.get("CLAUDE_PATH") or None,
            os.environ.get("CLAUDE_MODEL") or None,
            os.environ.get("CLAUDE_CLI_TIMEOUT") or None,
        )
    except ClaudeCLIError as exc:
        raise CodegenError(str(exc)) from exc
    bus.log(f"Claude CLI: {cli.describe()}")

    def llm(system: str, user: str) -> str:
        bus.log(
            f"Claude CLI request: {len(system) + len(user):,} characters (the prompt goes through"
            " stdin)"
        )
        try:
            return cli.run(system, user)
        except ClaudeCLIError as exc:
            raise CodegenError(str(exc)) from exc
    return llm


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _facts(state):
    """(facts, raw config.json, generation_config.json or None)."""
    model = state["model_name"]
    raw, gen, facts = None, None, None
    if os.path.isdir(model):
        raw = _read_json(os.path.join(model, "config.json"))
        gen = _read_json(os.path.join(model, "generation_config.json"))
        # Each lookup below is best effort: whatever fails, the next source is tried
        # (local folder, the Hub, then transformers' model on the meta device).
        try:
            facts = ck.facts_from_dir(model)
        except Exception as exc:  # pylint: disable=broad-except
            bus.log(f"Could not read the local snapshot's tensor list: {exc}", "warn")
    else:
        try:
            # pylint: disable-next=import-outside-toplevel,import-error  # optional package
            from huggingface_hub import hf_hub_download
            raw = _read_json(hf_hub_download(model, "config.json"))
            try:
                gen = _read_json(hf_hub_download(model, "generation_config.json"))
            except Exception:  # pylint: disable=broad-except  # the file is optional
                gen = None
        except Exception as exc:  # pylint: disable=broad-except
            bus.log(f"Could not fetch config.json from the Hub: {exc}", "warn")
        try:
            facts = ck.facts_from_hub(model)
            bus.log(
                f"Read {len(facts.tensors)} tensor patterns from the Hub's safetensors headers (no"
                " download)"
            )
        except Exception as exc:  # pylint: disable=broad-except
            bus.log(
                f"Safetensors headers not available ({exc}); using transformers' model on the meta"
                " device",
                "warn",
            )
    raw = raw or state.get("hf_config") or {}
    if facts is None or not facts.tensors_known:
        facts = _facts_from_meta_model(model, raw)
    return facts, raw, gen


def _facts_from_meta_model(model, raw):
    try:
        # pylint: disable-next=import-outside-toplevel,import-error  # optional package
        import torch
        # pylint: disable-next=import-outside-toplevel,import-error  # optional package
        from transformers import AutoConfig, AutoModelForCausalLM
        cfg = AutoConfig.from_pretrained(model)
        with torch.device("meta"):
            tiny_model = AutoModelForCausalLM.from_config(cfg)
        tied = bool(getattr(cfg, "tie_word_embeddings", False))
        tensors = {
            k: list(v.shape)
            for k, v in tiny_model.state_dict().items()
            if not (tied and k.endswith("lm_head.weight"))
        }
        bus.log(
            f"Read {len(tensors)} tensors from transformers' {type(tiny_model).__name__} on the"
            " meta device"
        )
        return ck.make_facts(
            model, raw, tensors, f"transformers {type(tiny_model).__name__} (meta device)"
        )
    except Exception as exc:  # pylint: disable=broad-except  # best effort: fall back to the config
        bus.log(f"Could not build the tensor list: {exc}", "warn")
        return ck.make_facts(model, raw, {}, "config only")


def hf_sources(raw):
    """Source of the Hugging Face modelling classes the C++ has to match."""
    cfg = raw.get("text_config") if isinstance(raw.get("text_config"), dict) else raw
    types = [t for t in (cfg.get("model_type"), raw.get("model_type")) if t]
    mod = None
    for model_type in types + [re.sub(r"_text$", "", name) for name in types]:
        try:
            mod = importlib.import_module(f"transformers.models.{model_type}.modeling_{model_type}")
            break
        # a modelling module can fail to import in many ways (missing extras, version skew)
        except Exception:  # pylint: disable=broad-except
            continue
    if mod is None:
        return {}
    out, total = {}, 0
    items = sorted(vars(mod).items(), key=lambda kv: (not kv[0].endswith("DecoderLayer"), kv[0]))
    for name, obj in items:
        keep = (
            inspect.isclass(obj)
            and getattr(obj, "__module__", "") == mod.__name__
            and _HF_CLASS.search(name)
            and "PreTrained" not in name
        ) or (inspect.isfunction(obj) and name in _HF_FUNCS)
        if not keep:
            continue
        try:
            src = inspect.getsource(obj)
        except (OSError, TypeError):
            continue
        if total + len(src) > 60000:
            break
        out[name] = src[:16000]
        total += len(out[name])
    return out


# pylint: disable-next=too-many-return-statements
def run(state: dict) -> dict:
    """Pipeline step: generate nntrainer CausalLM C++ for the model with the Claude CLI."""
    bus.agent_status(AGENT, "running")
    if os.environ.get("WORKBENCH_CODEGEN", "1").lower() in ("0", "false", "off", "no"):
        bus.agent_status(AGENT, "skipped", "disabled in settings")
        return state
    reason = not_causal_lm_reason(state.get("hf_config"))
    if reason:
        bus.log(reason, "warn")
        bus.chat("assistant", reason, AGENT)
        bus.agent_status(AGENT, "skipped", "not a causal LM")
        return state

    try:
        bank = CodeBank()
    except CodeBankError as exc:
        bus.error(AGENT, str(exc))
        bus.agent_status(AGENT, "error", "no code bank")
        return state
    root = os.environ.get("NNTRAINER_ROOT") or find_nntrainer_root()
    if bank.stats()["total"] == 0:
        if not root:
            bus.error(
                AGENT,
                "the code bank is empty and the nntrainer checkout was not found (set"
                " NNTRAINER_ROOT)",
            )
            bus.agent_status(AGENT, "error", "empty code bank")
            return state
        bus.log(f"Filling the code bank from {root}/Applications/CausalLM (first run)")
        summary = capture(root, bank)
        bus.log(f"Code bank: {summary['stats']['total']} entries ({summary['stats']['by_status']})")
    else:
        ensure_core_layer_types(bank, root)            # banks captured before this list existed

    facts, raw, gen = _facts(state)
    report = ck.run_checklist(facts, bank.learned_features())
    plan = ck.match_bank(report, bank)
    for line in ck.format_report(report, plan).splitlines():
        bus.log(line)
    if not facts.tensors_known:
        bus.error(AGENT, "no tensor list for this checkpoint, so the checklist cannot run")
        bus.agent_status(AGENT, "error", "no tensor list")
        return state

    llm = None
    if (plan.get("whole_model") or {}).get("state") != "reuse":
        try:
            llm = make_llm()
        except cg.CodegenError as exc:
            bus.error(AGENT, str(exc))
            bus.agent_status(AGENT, "error", "Claude CLI not available")
            return state
    try:
        review = cg.generate(
            facts,
            bank,
            llm,
            raw_config=raw,
            generation_config=gen,
            hf_sources=hf_sources(raw),
            model_id=state["model_name"],
            max_attempts=int(os.environ.get("WORKBENCH_CODEGEN_ATTEMPTS") or 3),
            log=bus.log,
        )
    except cg.CodegenError as exc:
        bus.error(AGENT, str(exc))
        bus.agent_status(AGENT, "error", "generation failed")
        return state

    publish(review, state)
    summary = review["summary"]
    detail = (f"{'checks passed' if review['validation']['ok'] else 'checks FAILED'}, "
              f"{summary['reuse']} verified reused, {summary['to_review']} to review")
    bus.chat(
        "assistant",
        f"Generated nntrainer CausalLM code for **{review['model']}** in `{review['out_dir']}` "
        f"({detail}). Open the Review tab to check the new parts.",
        AGENT,
    )
    bus.agent_status(AGENT, "done" if review["validation"]["ok"] else "error", detail)
    return state


def publish(review: dict, state: dict = None) -> None:
    """Send the review and the generated files to the webview; point the
    pipeline's C++ outputs at them."""
    bus.codegen(review)
    files = review.get("files", {})
    shown = [n for n in files if n.endswith((".h", ".cpp")) and not n.startswith("register_")]
    for name in sorted(shown, key=lambda n: (not n.endswith(".h"), n)):
        try:
            with open(files[name], encoding="utf-8") as handle:
                content = handle.read()
        except OSError:
            continue
        bus.file_content("nntrainer", name, content)
        bus.code(name, content)
    if state is None:
        return
    header = next((n for n in shown if n.endswith(".h")), None)
    source = next((n for n in shown if n.endswith(".cpp")), None)
    if header and source:
        with open(files[header], encoding="utf-8") as handle:
            state["causallm_header"] = handle.read()
        with open(files[source], encoding="utf-8") as handle:
            state["causallm_source"] = handle.read()
        state["causallm_header_path"], state["causallm_source_path"] = files[header], files[source]
        state["cpp_emission_mode"] = "causallm_component"
        state["requires_causallm_build"] = True
        state["cpp_code"] = state["causallm_header"] + "\n\n" + state["causallm_source"]
        state["cpp_path"] = files[source]
    state["codegen_review_path"] = review["review_path"]
    state["codegen_dir"] = review["out_dir"]
    state["codegen_ok"] = review["validation"]["ok"]
