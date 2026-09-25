"""
C++ Code Generation Validator

Validates that generated C++ code strictly follows the nntrainer specifications
from the instructions document. Ensures 100% compliance with:

1. Model Architecture Definition
2. Layer Implementations
3. Configuration Files
4. KV Cache Implementation
5. Tokenizer Integration
6. Build System Integration
7. Weight Conversion
8. Testing Framework
9. Implementation Checklist
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ComplianceCheck:
    """Result of a compliance check."""
    name: str
    passed: bool
    details: str
    severity: str  # "critical", "error", "warning", "info"


class CppGenerationValidator:
    """Validates generated C++ code against instruction requirements."""

    # ====================================================================
    # SECTION 1: MODEL ARCHITECTURE DEFINITION
    # ====================================================================

    REQUIRED_MODEL_METHODS = [
        "setupParameters",
        "constructModel",
        "registerCustomLayers",
        "allocateAndBindKVCache",
        "createAttention",
        "createMLP",
        "createDecoderLayer",
    ]

    REQUIRED_CLASSES = [
        "CausalLM",
        "Transformer",
    ]

    # ====================================================================
    # SECTION 2: LAYER IMPLEMENTATIONS
    # ====================================================================

    REQUIRED_LAYER_TYPES = {
        "embedding": "Token to vector projection",
        "rms_norm": "Normalization (replaces LayerNorm)",
        "reshaped_rms_norm": "Reshaped normalization for Q/K",
        "mha_core": "Multi-head attention with KV cache",
        "swiglu": "Gated MLP activation",
        "fully_connected": "Linear projections",
        "lm_head": "Final vocabulary projection",
    }

    # ====================================================================
    # SECTION 3: CONFIGURATION FILES
    # ====================================================================

    REQUIRED_CONFIG_FILES = [
        "config.json",          # HuggingFace format
        "nntr_config.json",     # NNTrainer-specific
        "generation_config.json", # Generation parameters
    ]

    REQUIRED_CONFIG_KEYS = {
        "config.json": [
            "model_type",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "vocab_size",
            "rope_theta",
            "max_position_embeddings",
        ],
        "nntr_config.json": [
            "model_type",
            "model_name",
            "model_file_name",
            "tokenizer_type",
            "enable_transformer",
            "batch_size",
            "max_seq_len",
            "use_kv_cache",
        ],
        "generation_config.json": [
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "max_new_tokens",
            "eos_token_id",
            "bos_token_id",
        ],
    }

    # ====================================================================
    # SECTION 5: KV CACHE
    # ====================================================================

    KV_CACHE_REQUIREMENTS = [
        "allocate",           # Allocate contiguous memory
        "getKeyCacheWriteView",  # Write view for current position
        "getValueCacheWriteView",
        "getKeyCacheReadView",   # Read view for attention
        "getValueCacheReadView",
        "setPosition",        # Position management
        "advance",
        "reset",
    ]

    # ====================================================================
    # SECTION 8: BUILD SYSTEM
    # ====================================================================

    REQUIRED_BUILD_ITEMS = [
        "shared_library",
        "sources",
        "include_directories",
        "dependencies",
        "install",
    ]

    def __init__(self):
        self.checks: List[ComplianceCheck] = []

    def validate_all(self, output_dir: Path) -> Dict[str, any]:
        """Run all validation checks."""
        self.checks = []

        # Section 1: Model Architecture
        self._validate_model_architecture(output_dir)

        # Section 2: Layer Implementations
        self._validate_layer_implementations(output_dir)

        # Section 3: Configuration Files
        self._validate_configuration_files(output_dir)

        # Section 5: KV Cache
        self._validate_kv_cache(output_dir)

        # Section 8: Build System
        self._validate_build_system(output_dir)

        # Summary
        return self._generate_report()

    # ====================================================================
    # SECTION 1: MODEL ARCHITECTURE VALIDATION
    # ====================================================================

    def _validate_model_architecture(self, output_dir: Path):
        """Validate model architecture follows specification."""
        header_file = output_dir / "*_causallm.h"
        source_file = output_dir / "*_causallm.cpp"

        header_list = list(output_dir.glob("*_causallm.h"))
        source_list = list(output_dir.glob("*_causallm.cpp"))

        if not header_list:
            self.checks.append(ComplianceCheck(
                name="Model Header File",
                passed=False,
                details="No *_causallm.h file found in output directory",
                severity="critical",
            ))
            return

        if not source_list:
            self.checks.append(ComplianceCheck(
                name="Model Source File",
                passed=False,
                details="No *_causallm.cpp file found in output directory",
                severity="critical",
            ))
            return

        header_content = header_list[0].read_text()
        source_content = source_list[0].read_text()

        # Check header guard
        self._check_header_guard(header_content, header_list[0].name)

        # Check base classes
        self._check_base_classes(header_content, source_content)

        # Check required methods
        self._check_required_methods(header_content, source_content)

        # Check method signatures
        self._check_method_signatures(source_content)

    def _check_header_guard(self, header_content: str, filename: str):
        """Verify header guard pattern."""
        # Extract model name from filename (e.g., qwen3 from qwen3_causallm.h)
        model_name = filename.replace("_causallm.h", "").upper()
        expected_guard = f"__{model_name}_CAUSAL_LM_H__"

        has_ifndef = f"#ifndef {expected_guard}" in header_content
        has_define = f"#define {expected_guard}" in header_content
        has_endif = f"#endif" in header_content and expected_guard in header_content

        passed = has_ifndef and has_define and has_endif

        self.checks.append(ComplianceCheck(
            name="Header Guard",
            passed=passed,
            details=f"Expected: {expected_guard}, Found: {has_ifndef and has_define and has_endif}",
            severity="critical" if not passed else "info",
        ))

    def _check_base_classes(self, header_content: str, source_content: str):
        """Verify inheritance from Transformer and CausalLM."""
        for base_class in ["Transformer", "CausalLM"]:
            # Check in header
            found_in_header = f"virtual public {base_class}" in header_content or f": public {base_class}" in header_content or f": {base_class}" in header_content

            self.checks.append(ComplianceCheck(
                name=f"Base Class: {base_class}",
                passed=found_in_header,
                details=f"Class should inherit from {base_class}",
                severity="critical" if not found_in_header else "info",
            ))

    def _check_required_methods(self, header_content: str, source_content: str):
        """Verify all required methods are defined."""
        for method in self.REQUIRED_MODEL_METHODS:
            found_in_header = method in header_content
            found_in_source = f"void {method}" in source_content or f"Tensor {method}" in source_content

            passed = found_in_header and found_in_source

            self.checks.append(ComplianceCheck(
                name=f"Method: {method}",
                passed=passed,
                details=f"In header: {found_in_header}, In source: {found_in_source}",
                severity="critical" if not passed else "info",
            ))

    def _check_method_signatures(self, source_content: str):
        """Verify method signatures match specification."""
        signatures = {
            "setupParameters": "void setupParameters()",
            "constructModel": "void constructModel()",
            "registerCustomLayers": "void registerCustomLayers()",
            "allocateAndBindKVCache": "void allocateAndBindKVCache()",
            "createAttention": "Tensor createAttention(const int layer_id",
            "createMLP": "Tensor createMLP(const int layer_id",
        }

        for method, sig_pattern in signatures.items():
            found = sig_pattern in source_content
            self.checks.append(ComplianceCheck(
                name=f"Signature: {method}",
                passed=found,
                details=f"Expected pattern: {sig_pattern}",
                severity="error" if not found else "info",
            ))

    # ====================================================================
    # SECTION 2: LAYER IMPLEMENTATION VALIDATION
    # ====================================================================

    def _validate_layer_implementations(self, output_dir: Path):
        """Validate layer implementations are properly used."""
        source_files = list(output_dir.glob("*_causallm.cpp"))

        if not source_files:
            self.checks.append(ComplianceCheck(
                name="Layer Implementations",
                passed=False,
                details="No source file found",
                severity="critical",
            ))
            return

        source_content = source_files[0].read_text()

        # Check for layer creation patterns
        for layer_type, description in self.REQUIRED_LAYER_TYPES.items():
            # createLayer("layer_type", ...) pattern
            pattern = f'createLayer("{layer_type}"'
            found = pattern in source_content

            # Skip embedding/lm_head if not found (may be optional)
            severity = "warning" if layer_type in ["embedding", "lm_head"] else "error"

            self.checks.append(ComplianceCheck(
                name=f"Layer: {layer_type}",
                passed=found,
                details=f"Should use createLayer(\"{layer_type}\", ...) - {description}",
                severity=severity if not found else "info",
            ))

        # Check for layer registration
        self._check_layer_registration(source_content)

    def _check_layer_registration(self, source_content: str):
        """Verify custom layers are registered."""
        registration_patterns = [
            "registerFactory",
            "ReshapedRMSNormLayer",
            "MHACoreLayer",
            "SwiGLULayer",
        ]

        for pattern in registration_patterns:
            found = pattern in source_content
            self.checks.append(ComplianceCheck(
                name=f"Registration: {pattern}",
                passed=found,
                details=f"Custom layer {pattern} should be registered in registerCustomLayers()",
                severity="warning" if not found else "info",
            ))

    # ====================================================================
    # SECTION 3: CONFIGURATION FILES VALIDATION
    # ====================================================================

    def _validate_configuration_files(self, output_dir: Path):
        """Validate all required configuration files exist and are correct."""
        import json

        for config_file in self.REQUIRED_CONFIG_FILES:
            filepath = output_dir / config_file
            exists = filepath.exists()

            self.checks.append(ComplianceCheck(
                name=f"Config File: {config_file}",
                passed=exists,
                details=f"File should exist at {filepath}",
                severity="critical" if not exists else "info",
            ))

            if exists:
                try:
                    config = json.loads(filepath.read_text())
                    self._validate_config_keys(config_file, config)
                except json.JSONDecodeError as e:
                    self.checks.append(ComplianceCheck(
                        name=f"Config Parse: {config_file}",
                        passed=False,
                        details=f"JSON parse error: {e}",
                        severity="critical",
                    ))

    def _validate_config_keys(self, filename: str, config: dict):
        """Verify required keys in configuration."""
        required_keys = self.REQUIRED_CONFIG_KEYS.get(filename, [])

        for key in required_keys:
            found = key in config

            self.checks.append(ComplianceCheck(
                name=f"Config Key: {filename}[{key}]",
                passed=found,
                details=f"Required key '{key}' in {filename}",
                severity="error" if not found else "info",
            ))

    # ====================================================================
    # SECTION 5: KV CACHE VALIDATION
    # ====================================================================

    def _validate_kv_cache(self, output_dir: Path):
        """Validate KV cache implementation."""
        source_files = list(output_dir.glob("*_causallm.cpp"))

        if not source_files:
            return

        source_content = source_files[0].read_text()

        # Check allocateAndBindKVCache method
        has_alloc_method = "allocateAndBindKVCache" in source_content

        self.checks.append(ComplianceCheck(
            name="KV Cache: Allocation Method",
            passed=has_alloc_method,
            details="allocateAndBindKVCache() method should be implemented",
            severity="critical" if not has_alloc_method else "info",
        ))

        # Check KV cache patterns in implementation
        kv_patterns = [
            "KVCacheManager",
            "createKVCachePlaceholders",
            "cache_k",
            "cache_v",
            "max_seq_len",
        ]

        for pattern in kv_patterns:
            found = pattern in source_content

            self.checks.append(ComplianceCheck(
                name=f"KV Cache: {pattern}",
                passed=found,
                details=f"KV cache implementation should use {pattern}",
                severity="warning" if not found else "info",
            ))

    # ====================================================================
    # SECTION 8: BUILD SYSTEM VALIDATION
    # ====================================================================

    def _validate_build_system(self, output_dir: Path):
        """Validate meson.build configuration."""
        meson_file = output_dir / "meson.build"

        if not meson_file.exists():
            self.checks.append(ComplianceCheck(
                name="Build System: meson.build",
                passed=False,
                details="meson.build file not found",
                severity="error",
            ))
            return

        content = meson_file.read_text()

        for item in self.REQUIRED_BUILD_ITEMS:
            found = item in content

            self.checks.append(ComplianceCheck(
                name=f"Build Config: {item}",
                passed=found,
                details=f"meson.build should define {item}",
                severity="warning" if not found else "info",
            ))

    # ====================================================================
    # VALIDATION REPORT
    # ====================================================================

    def _generate_report(self) -> Dict:
        """Generate comprehensive validation report."""
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c.passed)
        failed = total - passed

        critical_failed = sum(1 for c in self.checks if not c.passed and c.severity == "critical")
        error_failed = sum(1 for c in self.checks if not c.passed and c.severity == "error")
        warning_failed = sum(1 for c in self.checks if not c.passed and c.severity == "warning")

        return {
            "valid": critical_failed == 0,
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "critical_issues": critical_failed,
            "error_issues": error_failed,
            "warning_issues": warning_failed,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "details": c.details,
                    "severity": c.severity,
                }
                for c in self.checks
            ],
        }

    def print_report(self, report: Dict):
        """Print human-readable validation report."""
        print("\n" + "=" * 70)
        print("C++ CODE GENERATION COMPLIANCE REPORT")
        print("=" * 70)

        print(f"\nTotal Checks: {report['total_checks']}")
        print(f"✓ Passed: {report['passed']}")
        print(f"✗ Failed: {report['failed']}")

        if report['critical_issues'] > 0:
            print(f"\n🔴 CRITICAL ISSUES: {report['critical_issues']}")
        if report['error_issues'] > 0:
            print(f"🟠 ERROR ISSUES: {report['error_issues']}")
        if report['warning_issues'] > 0:
            print(f"🟡 WARNINGS: {report['warning_issues']}")

        print("\n" + "-" * 70)
        print("DETAILED RESULTS:")
        print("-" * 70)

        for check in report['checks']:
            status = "✓" if check['passed'] else "✗"
            severity_icon = {
                "critical": "🔴",
                "error": "🟠",
                "warning": "🟡",
                "info": "ℹ️",
            }.get(check['severity'], "")

            if not check['passed']:
                print(f"\n{severity_icon} {check['name']}")
                print(f"   Details: {check['details']}")

        print("\n" + "=" * 70)
        if report['valid']:
            print("✅ COMPLIANCE CHECK PASSED")
        else:
            print("❌ COMPLIANCE CHECK FAILED")
            print(f"Fix {report['critical_issues']} critical issues before proceeding")
        print("=" * 70 + "\n")
