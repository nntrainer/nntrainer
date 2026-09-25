"""
Prompt Harness dispatcher -- the main entry point for adding a new model.

Routing:
  1. REFERENCE PATH (preferred): Applications/CausalLM/models/<model>/<model>_causallm.cpp
     already exists (shipping, tested code) -> extract patterns directly.
  2. LLM PATH (fallback): no reference implementation -> pull architecture from
     HuggingFace config, ask an LLM to draft the missing method bodies, extract
     patterns from that draft.

Either path ends the same way: PromptGenerator renders a new
prompts_<model>_causallm.py under prompts_generated/, and PromptValidator
checks it before it's marked usable in the registry.

This module is purely additive -- it does not modify llm_codegen.py's
_get_prompt_module() routing, nor any existing prompts_*_causallm.py file.
Wiring a newly-generated prompt into the live pipeline is a separate,
deliberate step.
"""
import os
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass, field

from .reference_extractor import ReferenceExtractor, ExtractedPattern
from .hf_architecture import HFArchitecture, extract_hf_architecture
from .prompt_generator import PromptGenerator, PromptGenerationInput
from .prompt_validator import PromptValidator, ValidationResult, format_validation_report
from .registry import ModelRegistry, ModelEntry

# Repo root: .../tools/agentic-ai-workbench/extension/engine/agents/cpp/harness/prompt_harness
# -> up 8 levels to repo root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, *([".."] * 8)))
_MODELS_DIR = os.path.join(_REPO_ROOT, "Applications", "CausalLM", "models")
_BASE_TRANSFORMER_CPP = os.path.join(_MODELS_DIR, "transformer.cpp")

METHOD_KEYS = ["createAttention", "createMlp", "createTransformerDecoderBlock"]


@dataclass
class DispatchResult:
    model_name: str
    prompt_path: str
    source: str  # 'reference_implementation' | 'hf_llm_generated'
    validation: ValidationResult
    registered: bool = False


class PromptDispatcher:
    """
    Smart router: reference implementation first, LLM draft as fallback.

    llm_callable, if provided, must be a (system: str, user: str) -> str
    function -- the same interface used by claude_cli_backend._make_claude_cli_llm.
    Without it, models with no reference implementation cannot be processed
    (dispatch() raises instead of silently producing an empty prompt).
    """

    def __init__(
        self,
        nntrainer_root: Optional[str] = None,
        llm_callable: Optional[Callable[[str, str], str]] = None,
        registry: Optional[ModelRegistry] = None,
    ):
        self.nntrainer_root = nntrainer_root or _REPO_ROOT
        self.models_dir = os.path.join(self.nntrainer_root, "Applications", "CausalLM", "models")
        self.llm_callable = llm_callable
        self.registry = registry or ModelRegistry()

    def find_reference_implementation(self, model_name: str) -> Optional[str]:
        """Return path to <model>_causallm.cpp if it exists, else None."""
        candidate = os.path.join(self.models_dir, model_name, f"{model_name}_causallm.cpp")
        return candidate if os.path.exists(candidate) else None

    def dispatch(self, model_name: str, hf_id: Optional[str] = None) -> DispatchResult:
        """
        Generate (and validate) a prompt module for `model_name`.

        Does NOT overwrite an already-registered, validated entry unless
        force is handled by the caller -- this method always regenerates,
        callers should check `self.registry.get(model_name)` first if they
        want to skip already-ready models.
        """
        ref_cpp = self.find_reference_implementation(model_name)

        if ref_cpp:
            gen_input = self._build_input_from_reference(model_name, ref_cpp)
            source = "reference_implementation"
        else:
            if not hf_id:
                raise ValueError(
                    f"No reference implementation found for '{model_name}' "
                    f"(looked in {self.models_dir}/{model_name}/) and no "
                    f"hf_id provided to fall back to the LLM path."
                )
            gen_input = self._build_input_from_llm(model_name, hf_id)
            source = "hf_llm_generated"

        generator = PromptGenerator(gen_input)
        prompt_path = generator.write()

        validator = PromptValidator(model_name, prompt_path)
        result = validator.validate_static()

        entry = ModelEntry(
            name=model_name,
            prompt_file=os.path.relpath(prompt_path, self.nntrainer_root),
            source=source,
            status="ready" if result.ok else "failed",
            hf_id=hf_id,
            reference_cpp=os.path.relpath(ref_cpp, self.nntrainer_root) if ref_cpp else None,
            validated=result.ok,
            validation_errors=list(result.errors),
        )
        self.registry.register(entry)

        return DispatchResult(
            model_name=model_name,
            prompt_path=prompt_path,
            source=source,
            validation=result,
            registered=True,
        )

    # -- Reference path ----------------------------------------------------

    def _build_input_from_reference(self, model_name: str, ref_cpp: str) -> PromptGenerationInput:
        extractor = ReferenceExtractor(ref_cpp)
        patterns = extractor.extract_all_patterns()
        constants = extractor.extract_constants()
        notes_raw = extractor.extract_architecture_notes()
        notes = [line.strip("/ ").strip() for line in notes_raw.splitlines() if line.strip()]

        # Model files commonly override only createAttention and inherit
        # createMlp / createTransformerDecoderBlock from the shared base
        # Transformer class (this is exactly the qwen3 case). Fill any
        # pattern the model-specific file didn't override from there, so
        # the generated prompt still has a complete worked example set.
        missing = [k for k in METHOD_KEYS if k not in patterns]
        if missing and os.path.exists(_BASE_TRANSFORMER_CPP):
            base_extractor = ReferenceExtractor(_BASE_TRANSFORMER_CPP)
            for key in missing:
                base_pattern = base_extractor.extract_method(key)
                if base_pattern:
                    base_pattern.description = (
                        (base_pattern.description + " " if base_pattern.description else "")
                        + f"(inherited from base Transformer -- {model_name} does not override this)"
                    )
                    patterns[key] = base_pattern

        return PromptGenerationInput(
            model_name=model_name,
            class_prefix=model_name.capitalize(),
            architecture_notes=notes,
            patterns=patterns,
            constants=constants,
            source="reference",
            source_path=os.path.relpath(ref_cpp, self.nntrainer_root),
        )

    # -- LLM fallback path ---------------------------------------------------

    def _build_input_from_llm(self, model_name: str, hf_id: str) -> PromptGenerationInput:
        if self.llm_callable is None:
            raise RuntimeError(
                "No llm_callable configured on PromptDispatcher -- cannot use "
                "the LLM fallback path. Pass one (e.g. claude_cli_backend."
                "_make_claude_cli_llm(state)) when constructing PromptDispatcher."
            )

        arch = extract_hf_architecture(hf_id)
        draft_cpp = self._llm_draft_implementation(model_name, arch)

        # Extract patterns from the LLM draft the same way we would from a
        # real reference file -- write it to a temp path so ReferenceExtractor
        # (which works off file content) can be reused unchanged.
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as tmp:
            tmp.write(draft_cpp)
            tmp_path = tmp.name

        try:
            extractor = ReferenceExtractor(tmp_path)
            patterns = extractor.extract_all_patterns()
            constants = extractor.extract_constants()
        finally:
            os.unlink(tmp_path)

        notes = self._architecture_to_notes(arch)

        return PromptGenerationInput(
            model_name=model_name,
            class_prefix=model_name.capitalize(),
            architecture_notes=notes,
            patterns=patterns,
            constants=constants,
            source="llm_generated",
            source_path=None,
        )

    def _llm_draft_implementation(self, model_name: str, arch: HFArchitecture) -> str:
        """Ask the configured LLM to draft createAttention/createMLP bodies
        for a model with no NNTrainer reference implementation, grounded in
        the same layer catalog used for real prompt generation."""
        from knowledge.causallm_kb import render_catalog

        system = (
            "You are an NNTrainer C++ expert. You write causallm::Transformer "
            "method overrides using ONLY createLayer(type, {withKey(...)}) "
            "calls with layer types from the provided catalog. Never invent "
            "layer types or HF-style dotted names."
        )
        user = f"""\
Draft createAttention and createMlp (and createTransformerDecoderBlock if
the pattern isn't a standard pre-norm block) for {model_name}, following
the exact style of NNTrainer CausalLM models (see LayerHandle / createLayer
pattern).

Architecture (from {arch.hf_id}):
  hidden_size: {arch.hidden_size}
  num_attention_heads: {arch.num_attention_heads}
  num_heads_kv: {arch.num_heads_kv}
  intermediate_size: {arch.intermediate_size}
  norm_type: {arch.norm_type}
  attention_variant: {arch.attention_variant}
  mlp_variant: {arch.mlp_variant}
  hidden_act: {arch.hidden_act}

Layer catalog (use ONLY these types):
{render_catalog()}

Output plain C++ method definitions only, no prose.
"""
        return self.llm_callable(system, user)

    def _architecture_to_notes(self, arch: HFArchitecture) -> List[str]:
        notes = [
            f"hidden_size={arch.hidden_size}, num_attention_heads={arch.num_attention_heads}",
            f"normalization: {arch.norm_type}",
            f"attention variant: {arch.attention_variant}"
            + (f" (num_heads_kv={arch.num_heads_kv})" if arch.num_heads_kv else ""),
            f"mlp variant: {arch.mlp_variant}",
        ]
        if arch.sliding_window:
            notes.append(f"sliding_window: {arch.sliding_window}")
        if arch.rope_scaling:
            notes.append(f"rope_scaling: {arch.rope_scaling}")
        notes.append(
            "SOURCE: extracted from an LLM-drafted implementation, not a "
            "shipping reference -- verify against the model's real weights "
            "before trusting this in production."
        )
        return notes
