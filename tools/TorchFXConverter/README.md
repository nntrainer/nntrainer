# TorchFXConverter

Converts HuggingFace PyTorch models to NNTrainer C++ code using `torch.fx` graph tracing.

## Overview

The converter traces a PyTorch model's forward pass, maps the resulting FX graph nodes to NNTrainer layer definitions, detects high-level patterns (attention, FFN, transformer blocks), and emits C++ source, INI config, or JSON metadata.

Supported architectures:
- **Decoder-only** (CausalLM): Qwen3, LLaMA, Granite, LFM2
- **Encoder-only**: XLM-RoBERTa (multilingual-e5)
- **Encoder-decoder**: T5, Gemma2 (experimental)
- **Embedding models**: Qwen3, Gemma2, KaLM (base models without LM head)
- **Custom models**: GLiNER2 (NER extractor with span markers)

## Quick Start

```bash
# Convert a HuggingFace model to C++
python converter.py --model Qwen/Qwen3-0.6B --output ./out/ --format cpp

# Convert a local model directory
python converter.py --model ./my_model/ --output ./out/ --format all

# With weights
python converter.py --model Qwen/Qwen3-0.6B --output ./out/ --weights --dtype float16
```

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | HuggingFace model ID or local path (required) | — |
| `--output` | Output directory (required) | — |
| `--format` | Output format: `cpp`, `ini`, `json`, `all` | `all` |
| `--weights` | Convert and save model weights | off |
| `--dtype` | Weight dtype: `float32`, `float16` | `float32` |
| `--batch-size` | Batch size for INI config | `1` |
| `--seq-len` | Sequence length for tracing | `8` |
| `--model-name` | Override output file basename | derived from `--model` |
| `--quiet` | Suppress progress output | off |

## Tests

### Prerequisites

1. **NNTrainer must be built** with meson before running build-and-run tests:
   ```bash
   # From the nntrainer repo root
   meson setup builddir
   ninja -C builddir
   ```

2. **Python dependencies**: `torch`, `transformers`, `pytest`
   ```bash
   pip install torch transformers pytest
   ```

### Running Tests

All test commands should be run from the `tools/TorchFXConverter/` directory.

#### Full test suite (all unit tests)

```bash
python -m pytest tests/ -v
```

#### Unit tests only (no NNTrainer build required)

These tests validate the Python converter pipeline (tracing, mapping, pattern detection, code emission) without building or running C++:

```bash
# Tracer tests
python -m pytest tests/test_tracer_simple.py tests/test_tracer_qwen3.py -v

# Node mapper and decomposer tests
python -m pytest tests/test_node_mapper.py tests/test_decomposer.py -v

# Pattern detection tests
python -m pytest tests/test_pattern_detector.py tests/test_pattern_modules.py -v

# Emitter tests (C++, INI, JSON output generation)
python -m pytest tests/test_emitters.py tests/test_emitter_cpp_modules.py tests/test_emitter_ini_modules.py -v

# End-to-end conversion (no build)
python -m pytest tests/test_e2e.py tests/test_multi_arch.py -v
```

#### Build-and-run integration tests (requires NNTrainer build)

These tests run the full pipeline: convert model → generate C++ → build with meson/ninja → run the executable with NNTrainer:

```bash
# All build-and-run tests
python -m pytest tests/test_build_and_run.py -v

# Individual model tests
python -m pytest tests/test_build_and_run.py::TestConverterBuildAndRun::test_qwen3_tiny_build_and_run -v
python -m pytest tests/test_build_and_run.py::TestConverterBuildAndRun::test_gliner2 -v
python -m pytest tests/test_build_and_run.py::TestConverterBuildAndRun::test_multilingual_e5 -v
```

#### Test models in build-and-run suite

| Test | Architecture | Model Type |
|------|-------------|------------|
| `test_qwen3_tiny_build_and_run` | Decoder-only | Qwen3 CausalLM |
| `test_llama_tiny_build_and_run` | Decoder-only | LLaMA CausalLM |
| `test_qwen3_tied_embeddings` | Decoder-only | Qwen3 (tied embeddings) |
| `test_qwen3_06b` | Decoder-only | Qwen3-0.6B config |
| `test_qwen3_17b` | Decoder-only | Qwen3-1.7B config |
| `test_granite_40` | Decoder-only | GraniteMoeHybrid (dense) |
| `test_lfm_700m` | Decoder-only | LFM2 |
| `test_qwen3_embedding` | Embedding | Qwen3 (no LM head) |
| `test_embedding_gemma` | Embedding | Gemma2 |
| `test_kalm_embedding` | Embedding | KaLM (Qwen2-based) |
| `test_multilingual_e5` | Encoder-only | XLM-RoBERTa |
| `test_gliner2` | Custom | GLiNER2 NER extractor |
| `test_t5gemma2` | Encoder-decoder | T5-Gemma2 (skipped) |
| `test_prebuilt_qwen3_executable` | — | Pre-built binary check |

### How the build-and-run tests work

Each test follows this pipeline:

1. **Create model** — Generates a small synthetic model with random weights (tiny hidden dims, 1-2 layers) matching the target architecture's config
2. **Convert** — Runs `converter.py` to produce `.cpp` and `.h` files
3. **Copy to jni/** — Places generated files in `jni/` where `meson.build` can find them
4. **Meson reconfigure** — Runs `meson setup --reconfigure` to pick up new source files
5. **Ninja build** — Builds the test executable linking against `libnntrainer.so` and custom layer libraries
6. **Run** — Executes the binary, which calls `initialize()` (compile + init the NNTrainer graph) and `summarize()` to verify the model works end-to-end

Generated files are cleaned up after each test. The `jni/meson.build` conditionally includes source files only if they exist via `fs.exists()`.

## Project Structure

```
TorchFXConverter/
├── converter.py              # CLI entry point
├── custom_models.py          # Custom model loaders (GLiNER2, etc.)
├── tracer.py                 # torch.fx callback tracer
├── decomposer.py             # Multi-pass conversion pipeline
├── node_mapper.py            # FX node → NNTrainer layer mapping
├── module_mapper.py          # nn.Module mapping (Linear, Conv, etc.)
├── function_mapper.py        # torch.* function mapping
├── method_mapper.py          # Tensor.* method mapping
├── op_registry.py            # Operation lookup tables
├── mapper_helpers.py         # Name sanitization, scoping utilities
├── nntrainer_layers.py       # NNTrainer layer type constants
├── pattern_detector.py       # High-level pattern detection
├── weight_converter.py       # Weight format conversion
├── emitter_cpp/              # C++ code generation
│   ├── __init__.py           # emit_cpp(), emit_cpp_header(), emit_cpp_source()
│   ├── header.py             # Class header generation
│   ├── source_construct.py   # constructModel() for flat & transformer models
│   ├── source_attention.py   # Attention block generation
│   ├── source_ffn.py         # FFN/MLP block generation
│   ├── source_block.py       # Transformer block assembly
│   ├── source_custom.py      # Custom layer registration
│   └── helpers.py            # C++ generation utilities
├── emitter_ini/              # INI config generation
├── emitter_json.py           # JSON metadata generation
├── patterns/                 # Pattern detection modules
├── jni/                      # NNTrainer test harness
│   ├── main.cpp              # Generic test driver template
│   ├── meson.build           # Build config for test executables
│   └── test_model.*          # Pre-built Qwen3 reference model
├── tests/                    # Test suite
│   ├── test_build_and_run.py # Integration tests (convert→build→run)
│   ├── test_tracer_*.py      # Tracer unit tests
│   ├── test_node_mapper.py   # Mapper unit tests
│   ├── test_decomposer.py    # Decomposer unit tests
│   ├── test_emitters.py      # Emitter unit tests
│   └── ...
└── DESIGN.md                 # Architecture documentation
```
