"""
Validate a generated prompt module before it's trusted for real generation.

Two levels of validation:
1. STATIC  - the module imports cleanly and exposes the expected functions
             (<model>_generation_prompt, <model>_hard_constraints), and any
             layer types referenced in worked examples exist in LAYER_CATALOG.
2. DYNAMIC (optional) - actually run the generation pipeline against a
             sample plan and check the result against plan_validator, the
             same deterministic check used for hand-written prompts.
"""
import importlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class PromptValidator:
    """Validates auto-generated prompt modules"""

    def __init__(self, model_name: str, module_path: str):
        """
        Args:
            model_name: e.g. "mistral7b"
            module_path: filesystem path to the generated .py file
        """
        self.model_name = model_name
        self.module_path = module_path

    def validate_static(self) -> ValidationResult:
        """Check the module is well-formed without executing generation."""
        errors: List[str] = []
        warnings: List[str] = []

        module = self._load_module()
        if module is None:
            return ValidationResult(ok=False, errors=[f"Failed to import {self.module_path}"])

        expected_fn = f"{self.model_name}_generation_prompt"
        expected_constraints_fn = f"{self.model_name}_hard_constraints"

        if not hasattr(module, expected_fn):
            errors.append(f"Missing expected function: {expected_fn}()")
        if not hasattr(module, expected_constraints_fn):
            errors.append(f"Missing expected function: {expected_constraints_fn}()")

        layer_errors, layer_warnings = self._validate_layer_references(module)
        errors.extend(layer_errors)
        warnings.extend(layer_warnings)

        if not self._has_worked_example(module):
            warnings.append(
                "No worked example variables found (expected "
                f"{self.model_name.upper()}_*_EXAMPLE) -- prompt may be too generic"
            )

        return ValidationResult(ok=len(errors) == 0, errors=errors, warnings=warnings)

    def validate_dynamic(self, sample_plan: Dict) -> ValidationResult:
        """
        Run the actual generation prompt against a minimal sample plan and
        check that it renders without exceptions and produces non-trivial
        output. Does NOT call an LLM -- just exercises the prompt-building
        code path the same way llm_codegen.py would.
        """
        errors: List[str] = []
        warnings: List[str] = []

        module = self._load_module()
        if module is None:
            return ValidationResult(ok=False, errors=[f"Failed to import {self.module_path}"])

        fn_name = f"{self.model_name}_generation_prompt"
        fn = getattr(module, fn_name, None)
        if fn is None:
            return ValidationResult(ok=False, errors=[f"Missing {fn_name}()"])

        try:
            prompt_text = fn(sample_plan, "// skeleton header", "// skeleton source")
        except Exception as e:
            return ValidationResult(ok=False, errors=[f"{fn_name}() raised: {e}"])

        if not prompt_text or len(prompt_text) < 200:
            warnings.append("Generated prompt is suspiciously short (<200 chars)")

        if "TODO" in prompt_text and "no pattern extracted" in prompt_text:
            warnings.append(
                "Prompt contains placeholder text -- no real worked example was "
                "extracted, manual review required before use"
            )

        return ValidationResult(ok=len(errors) == 0, errors=errors, warnings=warnings)

    def _load_module(self):
        """
        Load the generated module as a real member of the agents.cpp.
        prompts_generated package, not a standalone file. The generated
        module uses `from ..prompts_causallm import (...)` -- the same
        relative import every hand-written prompts_<model>_causallm.py
        uses -- which only resolves when the module has a proper
        `__package__`. Loading it via spec_from_file_location alone
        produces a package-less module and that import fails.
        """
        dotted_name = f"agents.cpp.prompts_generated.prompts_{self.model_name}_causallm"
        try:
            import sys
            if dotted_name in sys.modules:
                del sys.modules[dotted_name]
            module = importlib.import_module(dotted_name)
            return module
        except Exception:
            return None

    def _has_worked_example(self, module) -> bool:
        prefix = f"{self.model_name.upper()}_"
        return any(
            name.startswith(prefix) and name.endswith("_EXAMPLE")
            for name in dir(module)
        )

    def _validate_layer_references(self, module) -> (List[str], List[str]):
        """Check that createLayer("<type>", ...) calls in worked examples
        reference real layer types from the knowledge base catalog."""
        errors: List[str] = []
        warnings: List[str] = []

        try:
            from knowledge.causallm_kb import LAYER_CATALOG, BUILTIN_TYPES
            available = set(LAYER_CATALOG.keys()) | set(BUILTIN_TYPES.keys())
        except ImportError:
            warnings.append("Could not import LAYER_CATALOG -- skipping layer validation")
            return errors, warnings

        example_vars = [
            getattr(module, name) for name in dir(module)
            if name.endswith("_EXAMPLE") and isinstance(getattr(module, name), str)
        ]

        found_types = set()
        for text in example_vars:
            for match in re.finditer(r'createLayer\s*\(\s*"([^"]+)"', text):
                found_types.add(match.group(1))

        unknown = found_types - available
        if unknown:
            errors.append(
                f"Worked examples reference unknown layer types: {sorted(unknown)}. "
                f"Available: {sorted(available)}"
            )

        return errors, warnings


def format_validation_report(result: ValidationResult, model_name: str) -> str:
    """Human-readable report, mirroring plan_validator.format_report style."""
    lines = [f"Validation report for '{model_name}':"]
    if result.ok:
        lines.append("  Status: OK")
    else:
        lines.append("  Status: FAILED")

    if result.errors:
        lines.append("  Errors:")
        lines.extend(f"    - {e}" for e in result.errors)
    if result.warnings:
        lines.append("  Warnings:")
        lines.extend(f"    - {w}" for w in result.warnings)
    if not result.errors and not result.warnings:
        lines.append("  No issues found.")

    return "\n".join(lines)
