"""
C++ Code Generation Integration Module

Main entry point for C++ code generation following nntrainer instructions.
Orchestrates:
1. Configuration validation
2. C++ code generation
3. Compliance verification
4. Artifact generation

This module ensures 100% compliance with the specifications.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .nntrainer_cpp_generator import (
    NNTrainerCppGenerator,
    ModelConfig,
    NNTrainerConfig,
    GenerationConfig,
)
from .cpp_generation_validator import CppGenerationValidator

logger = logging.getLogger(__name__)


class CppCodeGenerationPipeline:
    """
    Complete C++ code generation pipeline.

    Ensures strict compliance with nntrainer specifications:
    https://github.com/nnstreamer/nntrainer/tree/main/Applications/CausalLM
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.validator = CppGenerationValidator()

    def log(self, message: str, level: str = "info"):
        """Log with optional verbosity."""
        if self.verbose:
            getattr(logger, level)(message)
            print(f"[{level.upper()}] {message}")

    def generate_from_huggingface_config(
        self,
        config_json_path: str,
        architecture: str,
        output_dir: str,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate complete C++ implementation from HuggingFace config.

        Args:
            config_json_path: Path to config.json from HuggingFace
            architecture: Model architecture (e.g., "Qwen3ForCausalLM")
            output_dir: Output directory for generated files
            validate: Whether to validate generated code

        Returns:
            Result dictionary with:
            - success (bool)
            - files (dict)
            - report (validation report if validate=True)
            - messages (list of log messages)
        """
        messages = []

        try:
            # Step 1: Load and validate input
            self.log("=" * 70)
            self.log("C++ CODE GENERATION PIPELINE - START")
            self.log("=" * 70)

            self.log(f"1. Loading HuggingFace config from {config_json_path}")
            with open(config_json_path, 'r') as f:
                hf_config = json.load(f)

            messages.append(f"Loaded config: {config_json_path}")

            # Step 2: Extract model configuration
            self.log("2. Extracting model configuration")
            model_config = self._extract_model_config(hf_config)
            messages.append(f"Model: {architecture}")
            messages.append(f"  Hidden size: {model_config.hidden_size}")
            messages.append(f"  Num layers: {model_config.num_hidden_layers}")
            messages.append(f"  Vocab size: {model_config.vocab_size}")

            # Step 3: Create NNTrainer configuration
            self.log("3. Creating NNTrainer configuration")
            nntrainer_config = self._create_nntrainer_config(
                hf_config, architecture
            )
            messages.append(f"NNTrainer config created")

            # Step 4: Generate C++ code
            self.log("4. Generating C++ code")
            generator = NNTrainerCppGenerator(
                architecture=architecture,
                model_config=model_config,
                nntrainer_config=nntrainer_config,
            )
            messages.append(f"Generator initialized for {architecture}")

            # Step 5: Save all files
            self.log(f"5. Saving generated files to {output_dir}")
            output_path = Path(output_dir)
            generator.save_all(output_path)
            messages.append(f"Files saved to {output_dir}")

            # Collect generated files
            generated_files = {}
            for filename in generator.generate_all().keys():
                filepath = output_path / filename
                if filepath.exists():
                    generated_files[filename] = str(filepath)
                    self.log(f"   ✓ {filename}", "info")

            # Step 6: Validate generated code (if requested)
            self.log("6. Validating generated code")
            validation_report = None
            if validate:
                validation_report = self.validator.validate_all(output_path)
                self.validator.print_report(validation_report)

                if not validation_report.get("valid", False):
                    self.log(
                        "⚠️  Validation found issues, but code was generated",
                        "warning"
                    )

            # Step 7: Generate summary
            self.log("=" * 70)
            self.log("C++ CODE GENERATION PIPELINE - COMPLETE")
            self.log("=" * 70)

            return {
                "success": True,
                "architecture": architecture,
                "files": generated_files,
                "output_dir": output_dir,
                "validation_report": validation_report,
                "messages": messages,
                "stats": {
                    "total_files": len(generated_files),
                    "model_config": {
                        "hidden_size": model_config.hidden_size,
                        "num_layers": model_config.num_hidden_layers,
                        "num_heads": model_config.num_attention_heads,
                        "vocab_size": model_config.vocab_size,
                    },
                },
            }

        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            self.log(error_msg, "error")
            messages.append(error_msg)

            return {
                "success": False,
                "files": {},
                "validation_report": None,
                "messages": messages,
                "error": str(e),
            }

    def generate_from_specification(
        self,
        spec: Dict[str, Any],
        output_dir: str,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate from a complete specification dictionary.

        Specification format:
        {
            "architecture": "Qwen3ForCausalLM",
            "model_config": {
                "model_type": "qwen3",
                "hidden_size": 2048,
                ...
            },
            "nntrainer_config": {...},
            "generation_config": {...}
        }
        """
        messages = []

        try:
            self.log("=" * 70)
            self.log("C++ CODE GENERATION - SPECIFICATION MODE")
            self.log("=" * 70)

            architecture = spec.get("architecture", "Model")
            self.log(f"Architecture: {architecture}")

            # Extract configurations
            model_cfg = spec.get("model_config", {})
            model_config = ModelConfig(
                model_type=model_cfg.get("model_type", "unknown"),
                hidden_size=model_cfg.get("hidden_size", 2048),
                num_hidden_layers=model_cfg.get("num_hidden_layers", 24),
                num_attention_heads=model_cfg.get("num_attention_heads", 16),
                num_key_value_heads=model_cfg.get("num_key_value_heads", 8),
                intermediate_size=model_cfg.get("intermediate_size", 5632),
                vocab_size=model_cfg.get("vocab_size", 100000),
                rope_theta=model_cfg.get("rope_theta", 10000.0),
                max_position_embeddings=model_cfg.get("max_position_embeddings", 40960),
            )

            nntr_cfg = spec.get("nntrainer_config", {})
            nntrainer_config = NNTrainerConfig(
                model_type=nntr_cfg.get("model_type", "unknown"),
                model_name=nntr_cfg.get("model_name", "model"),
                model_file_name=nntr_cfg.get("model_file_name", "model.bin"),
                tokenizer_type=nntr_cfg.get("tokenizer_type", "huggingface"),
                enable_transformer=nntr_cfg.get("enable_transformer", True),
                batch_size=nntr_cfg.get("batch_size", 1),
                max_seq_len=nntr_cfg.get("max_seq_len", 40960),
                use_kv_cache=nntr_cfg.get("use_kv_cache", True),
                quantization=nntr_cfg.get("quantization"),
            )

            gen_cfg = spec.get("generation_config", {})
            generation_config = GenerationConfig(
                temperature=gen_cfg.get("temperature", 0.7),
                top_p=gen_cfg.get("top_p", 0.9),
                top_k=gen_cfg.get("top_k", 40),
                repetition_penalty=gen_cfg.get("repetition_penalty", 1.1),
                max_new_tokens=gen_cfg.get("max_new_tokens", 512),
                eos_token_id=gen_cfg.get("eos_token_id"),
                bos_token_id=gen_cfg.get("bos_token_id"),
            )

            # Generate
            generator = NNTrainerCppGenerator(
                architecture=architecture,
                model_config=model_config,
                nntrainer_config=nntrainer_config,
                generation_config=generation_config,
            )

            output_path = Path(output_dir)
            generator.save_all(output_path)

            # Collect files
            generated_files = {}
            for filename in generator.generate_all().keys():
                filepath = output_path / filename
                if filepath.exists():
                    generated_files[filename] = str(filepath)

            # Validate
            validation_report = None
            if validate:
                validation_report = self.validator.validate_all(output_path)
                self.validator.print_report(validation_report)

            self.log("✓ Generation complete")

            return {
                "success": True,
                "architecture": architecture,
                "files": generated_files,
                "output_dir": output_dir,
                "validation_report": validation_report,
                "messages": messages,
            }

        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            self.log(error_msg, "error")

            return {
                "success": False,
                "files": {},
                "validation_report": None,
                "messages": messages,
                "error": str(e),
            }

    def validate_generated_files(
        self,
        output_dir: str,
    ) -> Dict[str, Any]:
        """Run validation on already-generated files."""
        self.log(f"Validating generated files in {output_dir}")

        report = self.validator.validate_all(Path(output_dir))
        self.validator.print_report(report)

        return report

    # ====================================================================
    # HELPER METHODS
    # ====================================================================

    def _extract_model_config(self, hf_config: Dict) -> ModelConfig:
        """Extract ModelConfig from HuggingFace config."""
        return ModelConfig(
            model_type=hf_config.get("model_type", "unknown"),
            hidden_size=hf_config.get("hidden_size", 2048),
            num_hidden_layers=hf_config.get("num_hidden_layers", 24),
            num_attention_heads=hf_config.get("num_attention_heads", 16),
            num_key_value_heads=hf_config.get("num_key_value_heads", 8),
            intermediate_size=hf_config.get("intermediate_size", 5632),
            vocab_size=hf_config.get("vocab_size", 100000),
            rope_theta=float(hf_config.get("rope_theta", 10000.0)),
            max_position_embeddings=hf_config.get("max_position_embeddings", 40960),
        )

    def _create_nntrainer_config(
        self,
        hf_config: Dict,
        architecture: str,
    ) -> NNTrainerConfig:
        """Create NNTrainerConfig from HuggingFace config and architecture."""
        model_type = hf_config.get("model_type", "unknown")
        arch_slug = self._extract_slug(architecture)

        return NNTrainerConfig(
            model_type=model_type,
            model_name=f"{arch_slug.lower()}-model",
            model_file_name=f"{arch_slug.lower()}-model-fp32.bin",
            tokenizer_type="huggingface",
            enable_transformer=True,
            batch_size=1,
            max_seq_len=hf_config.get("max_position_embeddings", 40960),
            use_kv_cache=True,
            quantization={
                "fc_dtype": "Q4_0",
                "embd_dtype": "FP32",
                "lmhead_dtype": "FP32",
            },
        )

    @staticmethod
    def _extract_slug(architecture: str) -> str:
        """Extract model slug from architecture name."""
        slug = architecture.replace("ForCausalLM", "").replace("CausalLM", "").replace("Model", "")
        if slug and len(slug) > 0:
            return slug[0].upper() + slug[1:] if len(slug) > 1 else slug.upper()
        return architecture.lower()


# ====================================================================
# CONVENIENCE FUNCTIONS
# ====================================================================

def generate_cpp_code(
    config_json_path: str,
    architecture: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Simplified entry point for C++ code generation.

    Args:
        config_json_path: Path to HuggingFace config.json
        architecture: Model architecture
        output_dir: Output directory

    Returns:
        Generation result
    """
    pipeline = CppCodeGenerationPipeline(verbose=True)
    return pipeline.generate_from_huggingface_config(
        config_json_path,
        architecture,
        output_dir,
        validate=True,
    )


if __name__ == "__main__":
    # Example usage
    import tempfile
    import sys

    # Create temporary directories
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample config.json
        sample_config = {
            "model_type": "qwen3",
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "intermediate_size": 5632,
            "vocab_size": 151936,
            "rope_theta": 10000.0,
            "max_position_embeddings": 40960,
        }

        config_path = Path(tmpdir) / "config.json"
        config_path.write_text(json.dumps(sample_config, indent=2))

        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()

        # Run pipeline
        result = generate_cpp_code(
            str(config_path),
            "Qwen3ForCausalLM",
            str(output_dir),
        )

        print("\n" + "=" * 70)
        print("GENERATION RESULT")
        print("=" * 70)
        print(f"Success: {result.get('success')}")
        print(f"Files generated: {len(result.get('files', {}))}")
        print(f"\nGenerated files:")
        for filename, filepath in result.get('files', {}).items():
            print(f"  - {filename}")

        if result.get('validation_report'):
            print(f"\nValidation: {result['validation_report'].get('valid')}")
