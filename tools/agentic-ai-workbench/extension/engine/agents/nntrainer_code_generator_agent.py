"""
NNTrainer Code Generation Agent

Orchestrates comprehensive C++ code generation following nntrainer best practices.
Generates:
- Header and source files with proper class definitions
- Configuration files (config.json, nntr_config.json, generation_config.json)
- Build system integration (meson.build)
- Helper scripts for weight conversion and quantization
- Implementation guide and checklist
- Test templates

This agent is the main entry point for C++ code generation from model specifications.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..converters.nntrainer_cpp_generator import (
    NNTrainerCppGenerator,
    ModelConfig,
    NNTrainerConfig,
    GenerationConfig,
    create_generator_from_config,
)
from .events import bus

logger = logging.getLogger(__name__)


class NNTrainerCodeGeneratorAgent:
    """Generate complete NNTrainer model implementation."""

    def __init__(self):
        self.logger = logger

    def log(self, message: str, level: str = "info"):
        """Log a message and emit to event bus."""
        getattr(self.logger, level)(message)
        bus.log(message, level)

    def generate_from_huggingface_config(
        self,
        hf_config_path: str,
        architecture: str,
        output_dir: str,
        nntrainer_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate C++ code from a HuggingFace config.json.

        Args:
            hf_config_path: Path to HuggingFace config.json
            architecture: Model architecture (e.g., "Qwen3ForCausalLM")
            output_dir: Output directory for generated files
            nntrainer_overrides: Optional overrides for nntrainer config

        Returns:
            Dictionary with:
            - success (bool)
            - files_generated (dict: filename -> path)
            - stats (dict with model info)
            - messages (list of log messages)
        """
        messages = []
        try:
            self.log(f"Loading HuggingFace config from {hf_config_path}")

            # Load HuggingFace config
            with open(hf_config_path, 'r') as f:
                hf_config = json.load(f)

            # Create model config from HF format
            model_config = ModelConfig(
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

            # Create NNTrainer config with defaults
            arch_slug = self._extract_slug(architecture)
            nntrainer_config = NNTrainerConfig(
                model_type=hf_config.get("model_type", "unknown"),
                model_name=f"{arch_slug.lower()}-model",
                model_file_name=f"{arch_slug.lower()}-model-fp32.bin",
                tokenizer_type="huggingface",
                enable_transformer=True,
                batch_size=1,
                max_seq_len=model_config.max_position_embeddings,
                use_kv_cache=True,
                quantization={
                    "fc_dtype": "Q4_0",
                    "embd_dtype": "FP32",
                    "lmhead_dtype": "FP32",
                },
            )

            # Apply overrides if provided
            if nntrainer_overrides:
                for key, value in nntrainer_overrides.items():
                    if hasattr(nntrainer_config, key):
                        setattr(nntrainer_config, key, value)

            # Create generator
            generator = NNTrainerCppGenerator(
                architecture=architecture,
                model_config=model_config,
                nntrainer_config=nntrainer_config,
            )

            self.log(f"Generating C++ code for {architecture}")
            self.log(f"  Hidden size: {model_config.hidden_size}")
            self.log(f"  Num layers: {model_config.num_hidden_layers}")
            self.log(f"  Vocab size: {model_config.vocab_size}")

            # Generate all files
            output_path = Path(output_dir)
            generator.save_all(output_path)

            # Collect generated files
            generated_files = {}
            for filename in generator.generate_all().keys():
                filepath = output_path / filename
                if filepath.exists():
                    generated_files[filename] = str(filepath)

            self.log(f"✓ Generated {len(generated_files)} files")

            return {
                "success": True,
                "files_generated": generated_files,
                "stats": {
                    "architecture": architecture,
                    "hidden_size": model_config.hidden_size,
                    "num_layers": model_config.num_hidden_layers,
                    "num_heads": model_config.num_attention_heads,
                    "vocab_size": model_config.vocab_size,
                    "max_position_embeddings": model_config.max_position_embeddings,
                },
                "messages": messages,
            }

        except Exception as e:
            error_msg = f"Failed to generate code: {str(e)}"
            self.log(error_msg, "error")
            return {
                "success": False,
                "files_generated": {},
                "stats": {},
                "messages": [error_msg],
            }

    def generate_from_specification(
        self,
        spec: Dict[str, Any],
        output_dir: str,
    ) -> Dict[str, Any]:
        """Generate from a complete specification dict.

        Args:
            spec: Dictionary with keys:
                - architecture (str)
                - model_config (dict)
                - nntrainer_config (dict)
                - generation_config (dict, optional)
            output_dir: Output directory

        Returns:
            Generation result dictionary
        """
        try:
            generator = create_generator_from_config(spec)
            self.log(f"Generating from specification: {spec.get('architecture')}")

            output_path = Path(output_dir)
            generator.save_all(output_path)

            generated_files = {}
            for filename in generator.generate_all().keys():
                filepath = output_path / filename
                if filepath.exists():
                    generated_files[filename] = str(filepath)

            self.log(f"✓ Generated {len(generated_files)} files")

            return {
                "success": True,
                "files_generated": generated_files,
                "stats": {
                    "architecture": spec.get("architecture"),
                    "hidden_size": spec.get("model_config", {}).get("hidden_size"),
                    "num_layers": spec.get("model_config", {}).get("num_hidden_layers"),
                },
                "messages": [],
            }

        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            self.log(error_msg, "error")
            return {
                "success": False,
                "files_generated": {},
                "stats": {},
                "messages": [error_msg],
            }

    def validate_generated_code(self, cpp_file: str) -> Dict[str, Any]:
        """Perform basic validation on generated C++ code.

        Args:
            cpp_file: Path to generated .cpp file

        Returns:
            Validation result with issues and warnings
        """
        issues = []
        warnings = []

        try:
            with open(cpp_file, 'r') as f:
                content = f.read()

            # Check for required includes
            required_includes = [
                "#include <nntrainer/llm_util.hpp>",
                "#include <nntrainer/model.h>",
                "#include <nntrainer/tensor.h>",
            ]
            for inc in required_includes:
                if inc not in content:
                    warnings.append(f"Missing include: {inc}")

            # Check for required methods
            required_methods = [
                "setupParameters",
                "constructModel",
                "registerCustomLayers",
                "createAttention",
                "createMLP",
            ]
            for method in required_methods:
                if f"void {method}" not in content and f"Tensor {method}" not in content:
                    issues.append(f"Missing method: {method}")

            # Check for namespace
            if "namespace causallm" not in content:
                warnings.append("Missing namespace causallm")

            # Check for header guards
            if "#ifndef" not in content or "#endif" not in content:
                warnings.append("Missing or incomplete header guards")

            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "warnings": warnings,
                "file": cpp_file,
            }

        except Exception as e:
            return {
                "valid": False,
                "issues": [f"Validation error: {str(e)}"],
                "warnings": [],
                "file": cpp_file,
            }

    def generate_test_template(
        self,
        architecture: str,
        output_dir: str,
    ) -> str:
        """Generate a unit test template for the model.

        Args:
            architecture: Model architecture
            output_dir: Output directory

        Returns:
            Path to generated test file
        """
        arch_slug = self._extract_slug(architecture)
        test_filename = f"test_{arch_slug}_causallm.cpp"
        test_path = Path(output_dir) / test_filename

        content = f'''/**
 * Unit tests for {architecture} CausalLM model
 * Auto-generated test template
 */

#include <gtest/gtest.h>
#include <memory>
#include <json.hpp>

#include "{arch_slug}_causallm.h"
#include "test_util.h"

namespace causallm {{
namespace test {{

using json = nlohmann::json;

class {arch_slug}CausalLMTest : public ::testing::Test {{
 protected:
  void SetUp() override {{
    // Load configuration
    config_ = load_json("config.json");
    nntr_config_ = load_json("nntr_config.json");
    gen_config_ = load_json("generation_config.json");
  }}

  json config_;
  json nntr_config_;
  json gen_config_;
}};

TEST_F({arch_slug}CausalLMTest, LoadAndInitialize) {{
  auto model = std::make_unique<Generated{arch_slug}CausalLM>(
      config_, gen_config_, nntr_config_);

  EXPECT_NE(model, nullptr);
  EXPECT_NO_THROW(model->initialize());
}}

TEST_F({arch_slug}CausalLMTest, ConfigurationLoaded) {{
  auto model = std::make_unique<Generated{arch_slug}CausalLM>(
      config_, gen_config_, nntr_config_);

  model->setupParameters();

  // Verify configuration values
  EXPECT_EQ(config_["hidden_size"], {self._get_default_hidden_size()});
  EXPECT_EQ(config_["num_hidden_layers"], {self._get_default_num_layers()});
}}

TEST_F({arch_slug}CausalLMTest, RegisterCustomLayers) {{
  auto model = std::make_unique<Generated{arch_slug}CausalLM>(
      config_, gen_config_, nntr_config_);

  EXPECT_NO_THROW(model->registerCustomLayers());
}}

TEST_F({arch_slug}CausalLMTest, ForwardPass) {{
  auto model = std::make_unique<Generated{arch_slug}CausalLM>(
      config_, gen_config_, nntr_config_);

  model->initialize();

  // Create dummy input tensor
  std::vector<int32_t> input_ids = {{0, 1, 2, 3}};

  // Run forward pass
  auto output = model->forward(input_ids);

  // Verify output shape
  EXPECT_EQ(output.size(), 4);  // Same as input length
}}

TEST_F({arch_slug}CausalLMTest, GenerationWithKVCache) {{
  auto model = std::make_unique<Generated{arch_slug}CausalLM>(
      config_, gen_config_, nntr_config_);

  model->initialize();
  model->allocateAndBindKVCache();

  // Test incremental generation
  std::vector<int32_t> input = {{0, 1}};
  auto output = model->generate(input, 5);  // Generate 5 tokens

  EXPECT_GT(output.size(), input.size());
}}

}}  // namespace test
}}  // namespace causallm

GTEST_API_ int main(int argc, char **argv) {{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}}
'''

        test_path.write_text(content, encoding="utf-8")
        self.log(f"Generated test template: {test_filename}")
        return str(test_path)

    def generate_documentation(
        self,
        architecture: str,
        output_dir: str,
    ) -> str:
        """Generate comprehensive documentation.

        Args:
            architecture: Model architecture
            output_dir: Output directory

        Returns:
            Path to generated documentation
        """
        arch_slug = self._extract_slug(architecture)
        doc_filename = f"{arch_slug}_IMPLEMENTATION.md"
        doc_path = Path(output_dir) / doc_filename

        content = f"""# {architecture} Implementation Guide

## Quick Start

This document provides a step-by-step guide to integrate the {architecture} model into NNTrainer.

### Generated Files

The generation process created the following files:

1. **C++ Source**
   - `{arch_slug}_causallm.h` - Header with class definitions
   - `{arch_slug}_causallm.cpp` - Implementation

2. **Configuration**
   - `config.json` - HuggingFace model configuration
   - `nntr_config.json` - NNTrainer-specific settings
   - `generation_config.json` - Generation parameters

3. **Build System**
   - `meson.build` - Build integration

4. **Utilities**
   - `convert_weights.py` - Weight conversion script
   - `quantize.sh` - Quantization script

5. **Documentation**
   - `IMPLEMENTATION_GUIDE.md` - Detailed implementation guide
   - `CHECKLIST.md` - Implementation checklist

## Architecture Overview

### Model Structure

```
{architecture}
├── Embedding Layer
│   └── Token embedding (vocab_size → hidden_size)
├── Transformer Stack
│   └── {N} Decoder Layers
│       ├── Attention Block
│       │   ├── Q/K/V Projections
│       │   ├── RoPE (Rotary Position Embedding)
│       │   ├── Multi-Head Attention (with KV cache)
│       │   └── Output Projection
│       └── MLP Block
│           ├── Gate + Up Projections
│           ├── SwiGLU Activation
│           └── Down Projection
├── Final RMSNorm
└── LM Head
    └── Vocabulary projection
```

### Key Configurations

- **Hidden Size**: configurable from config.json
- **Number of Layers**: configurable from config.json
- **Attention Heads**: supports Grouped Query Attention (GQA)
- **Position Embeddings**: Rotary Position Embeddings (RoPE)
- **KV Cache**: Enabled for efficient incremental generation

## Integration Steps

### 1. File Organization

```bash
Applications/CausalLM/models/{arch_slug}/
├── {arch_slug}_causallm.h
├── {arch_slug}_causallm.cpp
├── config.json
├── nntr_config.json
├── generation_config.json
├── meson.build
├── convert_weights.py
└── quantize.sh
```

### 2. Build System Integration

Add to `Applications/CausalLM/models/meson.build`:

```meson
subdir('{arch_slug}')
```

### 3. Weight Preparation

```bash
# Convert weights from HuggingFace
python3 convert_weights.py model.safetensors model.bin

# Optional: Quantize
bash quantize.sh ./model Q4_0
```

### 4. Compilation

```bash
meson build -Denable-transformer=true
ninja -C build Applications/CausalLM/nntr_causallm
```

### 5. Testing

Run the unit tests:

```bash
ninja -C build test
```

## Layer Reference

The implementation uses the following layers from NNTrainer:

### Built-in Layers
- `embedding` - Token embedding
- `fully_connected` - Linear projections (Q, K, V, output, up/gate/down)
- `addition` - Residual connections
- `activation` - Non-linearities

### Custom Layers
- `reshaped_rms_norm` - Normalization with reshaping
- `mha_core` - Multi-head attention with RoPE and causal masking
- `swiglu` - SwiGLU activation (gate * swish(up))

All custom layers are registered in `registerCustomLayers()`.

## Performance Optimization

### KV Cache

The model uses an incremental KV cache for efficient autoregressive generation:

```
Memory per layer: 2 * batch * max_seq_len * num_heads_kv * head_dim
Total: num_layers * above
```

For default config:
- batch=1, max_seq_len=40960, num_heads_kv=8, head_dim=128
- Per layer: 2 * 1 * 40960 * 8 * 128 = ~84 MB
- Total (24 layers): ~2 GB

### Quantization

Post-conversion quantization reduces model size:

```bash
./build/Applications/CausalLM/nntr_quantize ./model \\
    --fc_dtype Q4_0 \\
    --embd_dtype FP32 \\
    --output_format bin
```

Expected size reduction: ~75% (4-bit quantization on FC layers)

### Platform-Specific

**Mobile (Android/Tizen)**
- Use FP16: 2x speed improvement
- Enable tensor pooling for memory reuse
- Consider expert caching for MoE models

**Desktop/Server**
- Use FP32 for accuracy
- Enable AVX2/AVX512 for CPU
- Profile with VTune or perf

## Troubleshooting

### Compilation Issues

- **Missing base class**: Ensure `CausalLM` and `Transformer` headers exist
- **Undefined layers**: Check custom layer registration
- **Version mismatch**: Verify NNTrainer version compatibility

### Runtime Issues

- **Segmentation fault**: Check weight shapes against layer definitions
- **Memory exhaustion**: Reduce max_seq_len or batch_size
- **Wrong output shape**: Verify config.json matches model

### Performance Issues

- Profile: `perf record ./build/Applications/CausalLM/nntr_causallm`
- Check KV cache allocation
- Consider quantization

## References

- [NNTrainer Repository](https://github.com/nnstreamer/nntrainer)
- [CausalLM Applications](https://github.com/nnstreamer/nntrainer/tree/main/Applications/CausalLM)
- [HuggingFace {architecture}](https://huggingface.co/models?search={arch_slug.lower()})
- [Generated Implementation Guide](IMPLEMENTATION_GUIDE.md)
- [Implementation Checklist](CHECKLIST.md)

## Support

For issues or questions:
1. Check the CHECKLIST.md for implementation status
2. Review IMPLEMENTATION_GUIDE.md for detailed steps
3. Study reference implementations in Applications/CausalLM/models/
4. Consult NNTrainer documentation and source code
"""

        doc_path.write_text(content, encoding="utf-8")
        self.log(f"Generated documentation: {doc_filename}")
        return str(doc_path)

    @staticmethod
    def _extract_slug(architecture: str) -> str:
        """Extract capitalized model name from architecture."""
        slug = architecture.replace("ForCausalLM", "").replace("CausalLM", "").replace("Model", "")
        if slug and len(slug) > 0:
            return slug[0].upper() + slug[1:] if len(slug) > 1 else slug.upper()
        return architecture.lower()

    @staticmethod
    def _get_default_hidden_size() -> int:
        return 2048

    @staticmethod
    def _get_default_num_layers() -> int:
        return 24

    def __repr__(self) -> str:
        return "NNTrainerCodeGeneratorAgent()"
