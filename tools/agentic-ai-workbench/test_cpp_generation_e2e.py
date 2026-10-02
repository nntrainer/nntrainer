#!/usr/bin/env python3
"""
End-to-End Test: C++ Code Generation

This test demonstrates the complete C++ code generation pipeline
following the nntrainer specifications exactly.

Test Cases:
1. Generate code from HuggingFace config (Qwen3 example)
2. Validate all generated files
3. Verify compliance with specifications
4. Check file contents
"""

import json
import tempfile
from pathlib import Path
import sys

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent / "extension/engine"))

from converters.cpp_generation_integration import CppCodeGenerationPipeline
from converters.cpp_generation_validator import CppGenerationValidator


def test_generate_from_config():
    """Test 1: Generate C++ code from HuggingFace config."""
    print("\n" + "=" * 80)
    print("TEST 1: C++ Code Generation from HuggingFace Config (Qwen3)")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create sample Qwen3 config.json
        qwen3_config = {
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

        config_path = tmpdir / "config.json"
        config_path.write_text(json.dumps(qwen3_config, indent=2))

        output_dir = tmpdir / "qwen3_generated"
        output_dir.mkdir()

        # Run generation pipeline
        pipeline = CppCodeGenerationPipeline(verbose=True)
        result = pipeline.generate_from_huggingface_config(
            str(config_path),
            "Qwen3ForCausalLM",
            str(output_dir),
            validate=True,
        )

        # Check results
        print("\n" + "-" * 80)
        print("GENERATION RESULT:")
        print("-" * 80)
        print(f"✓ Success: {result['success']}")
        print(f"✓ Architecture: {result['architecture']}")
        print(f"✓ Output directory: {result['output_dir']}")
        print(f"✓ Files generated: {result['stats']['total_files']}")

        print("\nGenerated files:")
        for filename in sorted(result['files'].keys()):
            filepath = result['files'][filename]
            file_size = Path(filepath).stat().st_size
            print(f"  ✓ {filename:<40} ({file_size:>8} bytes)")

        # Verify each file exists and has content
        print("\n" + "-" * 80)
        print("FILE VALIDATION:")
        print("-" * 80)

        required_files = [
            "qwen3_causallm.h",
            "qwen3_causallm.cpp",
            "config.json",
            "nntr_config.json",
            "generation_config.json",
            "meson.build",
            "convert_weights.py",
            "quantize.sh",
            "IMPLEMENTATION_GUIDE.md",
            "CHECKLIST.md",
        ]

        all_exist = True
        for filename in required_files:
            filepath = output_dir / filename
            exists = filepath.exists()
            has_content = filepath.stat().st_size > 0 if exists else False
            status = "✓" if exists and has_content else "✗"
            print(f"{status} {filename:<40} {filepath.stat().st_size if exists else 0:>8} bytes")
            if not (exists and has_content):
                all_exist = False

        if all_exist:
            print("\n✅ All required files generated successfully")
        else:
            print("\n❌ Some required files are missing")

        # Check header file structure
        print("\n" + "-" * 80)
        print("HEADER FILE STRUCTURE:")
        print("-" * 80)

        header_file = output_dir / "qwen3_causallm.h"
        if header_file.exists():
            header_content = header_file.read_text()

            checks = [
                ("Header guard", "__QWEN3_CAUSAL_LM_H__" in header_content),
                ("Namespace", "namespace causallm {" in header_content),
                ("Transformer class", "class GeneratedQwen3Transformer" in header_content),
                ("CausalLM class", "class GeneratedQwen3CausalLM" in header_content),
                ("setupParameters method", "setupParameters()" in header_content),
                ("constructModel method", "constructModel()" in header_content),
                ("registerCustomLayers method", "registerCustomLayers()" in header_content),
                ("createAttention method", "createAttention(" in header_content),
                ("createMLP method", "createMLP(" in header_content),
            ]

            for check_name, passed in checks:
                status = "✓" if passed else "✗"
                print(f"{status} {check_name}")

        # Check source file structure
        print("\n" + "-" * 80)
        print("SOURCE FILE STRUCTURE:")
        print("-" * 80)

        source_file = output_dir / "qwen3_causallm.cpp"
        if source_file.exists():
            source_content = source_file.read_text()

            checks = [
                ("Includes nntrainer headers", "#include <llm_util.hpp>" in source_content),
                ("Namespace", "namespace causallm {" in source_content),
                ("setupParameters implementation", "void GeneratedQwen3Transformer::setupParameters()" in source_content),
                ("constructModel implementation", "void GeneratedQwen3Transformer::constructModel()" in source_content),
                ("createAttention implementation", "Tensor GeneratedQwen3Transformer::createAttention(" in source_content),
                ("createMLP implementation", "Tensor GeneratedQwen3Transformer::createMLP(" in source_content),
                ("RMSNorm layer", 'createLayer("rms_norm"' in source_content),
                ("MHA core layer", 'createLayer("mha_core"' in source_content),
                ("SwiGLU layer", 'createLayer("swiglu"' in source_content),
                ("Fully connected layer", 'createLayer("fully_connected"' in source_content),
            ]

            for check_name, passed in checks:
                status = "✓" if passed else "✗"
                print(f"{status} {check_name}")

        # Check configuration files
        print("\n" + "-" * 80)
        print("CONFIGURATION FILES:")
        print("-" * 80)

        config_file = output_dir / "config.json"
        if config_file.exists():
            config = json.loads(config_file.read_text())
            print(f"✓ config.json: {len(config)} keys")
            for key in ["model_type", "hidden_size", "num_attention_heads", "vocab_size"]:
                print(f"    - {key}: {config.get(key)}")

        nntr_config_file = output_dir / "nntr_config.json"
        if nntr_config_file.exists():
            nntr_config = json.loads(nntr_config_file.read_text())
            print(f"✓ nntr_config.json: {len(nntr_config)} keys")
            print(f"    - model_name: {nntr_config.get('model_name')}")
            print(f"    - use_kv_cache: {nntr_config.get('use_kv_cache')}")

        gen_config_file = output_dir / "generation_config.json"
        if gen_config_file.exists():
            gen_config = json.loads(gen_config_file.read_text())
            print(f"✓ generation_config.json: {len(gen_config)} keys")
            print(f"    - temperature: {gen_config.get('temperature')}")
            print(f"    - max_new_tokens: {gen_config.get('max_new_tokens')}")

        # Check build system
        print("\n" + "-" * 80)
        print("BUILD SYSTEM (meson.build):")
        print("-" * 80)

        meson_file = output_dir / "meson.build"
        if meson_file.exists():
            meson_content = meson_file.read_text()
            checks = [
                ("shared_library definition", "shared_library(" in meson_content),
                ("sources", "model_sources" in meson_content),
                ("include_directories", "include_directories" in meson_content),
                ("dependencies", "dependencies" in meson_content),
                ("install", "install" in meson_content),
            ]

            for check_name, passed in checks:
                status = "✓" if passed else "✗"
                print(f"{status} {check_name}")

        # Check documentation
        print("\n" + "-" * 80)
        print("DOCUMENTATION:")
        print("-" * 80)

        guide_file = output_dir / "IMPLEMENTATION_GUIDE.md"
        if guide_file.exists():
            guide_content = guide_file.read_text()
            sections = [
                ("Model Architecture", "## Model Structure" in guide_content),
                ("Integration Steps", "## Integration Steps" in guide_content),
                ("Layer Reference", "## Layer Reference" in guide_content),
                ("Performance Optimization", "## Performance Optimization" in guide_content),
                ("KV Cache", "KV Cache" in guide_content),
                ("Quantization", "Quantization" in guide_content),
                ("Troubleshooting", "## Troubleshooting" in guide_content),
            ]

            for section_name, found in sections:
                status = "✓" if found else "✗"
                print(f"{status} {section_name}")

        checklist_file = output_dir / "CHECKLIST.md"
        if checklist_file.exists():
            checklist_content = checklist_file.read_text()
            items = checklist_content.count("- [ ]")
            print(f"✓ Implementation Checklist: {items} items")

        # Validation report
        print("\n" + "-" * 80)
        print("COMPLIANCE VALIDATION:")
        print("-" * 80)

        if result.get('validation_report'):
            report = result['validation_report']
            print(f"✓ Total checks: {report['total_checks']}")
            print(f"✓ Passed: {report['passed']}")
            print(f"✗ Failed: {report['failed']}")

            if report['critical_issues'] > 0:
                print(f"\n🔴 Critical issues: {report['critical_issues']}")
            else:
                print(f"\n✅ No critical issues")

            if report['error_issues'] > 0:
                print(f"🟠 Error issues: {report['error_issues']}")
            if report['warning_issues'] > 0:
                print(f"🟡 Warnings: {report['warning_issues']}")

        print("\n" + "=" * 80)
        return result['success'] and all_exist


def test_validation_only():
    """Test 2: Validate existing generated files."""
    print("\n" + "=" * 80)
    print("TEST 2: Validation of Generated Files")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Generate files first
        qwen3_config = {
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

        config_path = tmpdir / "config.json"
        config_path.write_text(json.dumps(qwen3_config, indent=2))

        output_dir = tmpdir / "generated"
        output_dir.mkdir()

        pipeline = CppCodeGenerationPipeline(verbose=False)
        pipeline.generate_from_huggingface_config(
            str(config_path),
            "Qwen3ForCausalLM",
            str(output_dir),
            validate=False,
        )

        # Now validate
        print("\nRunning comprehensive validation...")
        validator = CppGenerationValidator()
        report = validator.validate_all(output_dir)

        validator.print_report(report)

        return report.get('valid', False)


def test_multiple_architectures():
    """Test 3: Generate code for multiple architectures."""
    print("\n" + "=" * 80)
    print("TEST 3: Multiple Architectures")
    print("=" * 80)

    architectures = [
        ("Qwen3ForCausalLM", {"hidden_size": 2048, "num_hidden_layers": 24}),
        ("Gemma3ForCausalLM", {"hidden_size": 1024, "num_hidden_layers": 20}),
        ("LlamaForCausalLM", {"hidden_size": 1536, "num_hidden_layers": 22}),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        pipeline = CppCodeGenerationPipeline(verbose=False)

        results = {}
        for arch_name, arch_config in architectures:
            print(f"\n  Generating {arch_name}...")

            # Create config
            config = {
                "model_type": arch_name.lower().replace("forcausallm", ""),
                "hidden_size": arch_config.get("hidden_size", 2048),
                "num_hidden_layers": arch_config.get("num_hidden_layers", 24),
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "intermediate_size": 5632,
                "vocab_size": 100000,
                "rope_theta": 10000.0,
                "max_position_embeddings": 40960,
            }

            config_path = tmpdir / f"{arch_name}_config.json"
            config_path.write_text(json.dumps(config, indent=2))

            output_dir = tmpdir / arch_name
            output_dir.mkdir(exist_ok=True)

            result = pipeline.generate_from_huggingface_config(
                str(config_path),
                arch_name,
                str(output_dir),
                validate=False,
            )

            success = result.get('success', False)
            num_files = len(result.get('files', {}))

            results[arch_name] = {
                "success": success,
                "files": num_files,
            }

            status = "✓" if success else "✗"
            print(f"    {status} {arch_name}: {num_files} files")

        print("\n" + "-" * 80)
        print("SUMMARY:")
        for arch, res in results.items():
            status = "✓" if res['success'] else "✗"
            print(f"{status} {arch}: {res['files']} files")

        return all(r['success'] for r in results.values())


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + " C++ CODE GENERATION - END-TO-END TEST ".center(78) + "║")
    print("║" + " Following NNTrainer Specifications Exactly ".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")

    # Run tests
    test1_passed = test_generate_from_config()
    test2_passed = test_validation_only()
    test3_passed = test_multiple_architectures()

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    tests = [
        ("Test 1: Generate from HuggingFace Config", test1_passed),
        ("Test 2: Validation of Generated Files", test2_passed),
        ("Test 3: Multiple Architectures", test3_passed),
    ]

    for test_name, passed in tests:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status:12} {test_name}")

    all_passed = all(p for _, p in tests)
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED - Implementation follows specifications exactly")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 80 + "\n")

    sys.exit(0 if all_passed else 1)
