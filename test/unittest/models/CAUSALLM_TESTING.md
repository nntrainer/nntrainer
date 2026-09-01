# CausalLM Model Unit Tests

Reference guide for adding tests when integrating a new model family (e.g. LLaMA, Phi).

## Test Structure

```
test/unittest/models/
├── causallm_test_utils.h                      # Common interfaces, adapter template, helpers
├── causallm_test_utils.cpp                    # Helper implementations
├── unittest_causallm_qwen3.cpp                # Tiny deterministic tests (example)
├── unittest_causallm_qwen3_reference.cpp      # Differential tests (example)
└── causallm_reference/
    └── qwen3_tiny/                            # Committed HF reference fixture
        ├── config.json
        ├── generation_config.json
        ├── nntr_config.json
        ├── tokenizer.json
        ├── nntr_qwen3_tiny_fp32.bin
        ├── input_ids.json
        ├── reference_logits.json
        ├── reference_tokens.json
        └── meta.json
```

Tests are organized into two layers:

| Layer | File pattern | Description |
|-------|-------------|-------------|
| **Tiny deterministic** | `unittest_causallm_<model>.cpp` | Verifies golden logits with deterministic weights. No fixture required. |
| **Differential reference** | `unittest_causallm_<model>_reference.cpp` | Compares nntrainer output against a committed HF reference. Fixture required. |

---

## Key Types

### `CausalLMTestAdapter<ModelBase>`

A generic adapter template shared by all models. Inherits from `ModelBase`
(e.g. `Qwen3CausalLM`) and implements the `TinyCausalLMRunner` interface once,
so per-model test files only differ in how weights are populated.

```cpp
// Per-model using aliases (add to causallm_test_utils.h)
using Qwen3Adapter  = causallm_test::CausalLMTestAdapter<causallm::Qwen3CausalLM>;
using Qwen2Adapter  = causallm_test::CausalLMTestAdapter<causallm::Qwen2CausalLM>;
using Gemma3Adapter = causallm_test::CausalLMTestAdapter<causallm::Gemma3CausalLM>;
```

**Note**: If a model needs to pre-process its config before passing it to the
`Transformer` base constructor (e.g. Gemma3's `sanitizeConfig()`), add a thin
subclass that initializes `Transformer` with the processed config first.

### `TinyCausalLMCase`

Describes one tiny deterministic test case.

```cpp
struct TinyCausalLMCase {
  std::string name;                              // gtest parameter name
  TinyCausalLMDataType data_type;               // FP32 / Q4_0-FP32, etc.
  TinyCausalLMExpectedLogits expected_logits;   // golden logit vector + tolerance
  std::function<causallm::json()> make_model_config;
  std::function<std::map<...>(const TinyCausalLMDataType &)> make_layer_dtype_map;
  std::function<std::unique_ptr<TinyCausalLMRunner>(json &, json &, json &)> create_model;
  std::function<void(TinyCausalLMRunner &)> setup_weights; // populate deterministic weights
};
```

### `DifferentialModel`

Describes one differential test entry.

```cpp
struct DifferentialModel {
  std::string fixture_name;  // sub-directory under causallm_reference/ (e.g. "qwen3_tiny")
  std::function<std::unique_ptr<TinyCausalLMRunner>(json &, json &, json &)> make_model;
};
```

---

## Adding a New Model

### Step 1: Tiny deterministic tests (`unittest_causallm_<model>.cpp`)

#### 1-1. Add an adapter alias

Add to `causallm_test_utils.h`:

```cpp
using MyModelAdapter = causallm_test::CausalLMTestAdapter<causallm::MyModelCausalLM>;
```

#### 1-2. Write the deterministic weight setup function

```cpp
// unittest_causallm_mymodel.cpp
void setupMyModelDeterministicWeights(TinyMyModelCausalLM &model) {
  model.forEachLayer([](ml::train::Layer &layer,
                        nntrainer::RunLayerContext &ctx, void *) {
    for (unsigned int i = 0; i < ctx.getNumWeights(); ++i) {
      auto &w = ctx.getWeight(i);
      if (w.getDataType() != ml::train::TensorDim::DataType::FP32)
        continue;
      w.setValue(0.0f);
      if (causallm_test::isRmsNormLayerType(layer.getType()))
        w.setValue(1.0f);
      // Initialize embedding or other layers as needed
    }
  });
}
```

Match RMS norm nodes with `causallm_test::isRmsNormLayerType()` rather than a
literal `getType() == "rms_norm"`. The same operator has one type string per
backend -- the application class reports `"rms_norm"`, its per-head variant
reports `"reshaped_rms_norm"`, and the OpenCL and CUDA device classes report
the backend-neutral `"rmsnorm"` -- so a literal comparison silently skips those
nodes on an `engine=gpu` or `engine=cuda` graph, saves a zero gamma, and the
model reloads as all-zero logits.

Weight visit order follows the **DFS topological sort** of the nntrainer graph
(not insertion order). For parallel branches where `wq`, `wk`, `wv` all consume
the same normed input, DFS visits them depth-first: `wq → q_norm → wk → k_norm → wv`.

#### 1-3. Write the `TinyCausalLMCase` factory

```cpp
causallm_test::TinyCausalLMCase
makeMyModelCase(const causallm_test::TinyCausalLMDataType &data_type) {
  return {
    "MyModel_" + data_type.name,
    data_type,
    {"hello tok4", makeExpectedMyModelLogits(),
     data_type.name == "FP32" ? 1e-4f : 1e-3f},
    makeTinyMyModelConfig,
    makeMyModelLayerDtypeMap,
    [](causallm::json &cfg, causallm::json &gen, causallm::json &nntr) {
      return std::make_unique<TinyMyModelCausalLM>(cfg, gen, nntr);
    },
    [](causallm_test::TinyCausalLMRunner &runner) {
      setupMyModelDeterministicWeights(
        static_cast<TinyMyModelCausalLM &>(runner));
    },
  };
}
```

#### 1-4. Register the parameterized test suite

```cpp
INSTANTIATE_TEST_SUITE_P(
  MyModel, CausalLMTinyModelTest,
  ::testing::Values(
    makeMyModelCase(causallm_test::makeTinyFp32DataType()),
    makeMyModelCase(causallm_test::makeTinyQ40Fp32DataType())),
  [](const ::testing::TestParamInfo<causallm_test::TinyCausalLMCase> &info) {
    return info.param.name;
  });
```

This automatically generates the following three tests for each variant (FP32 / Q4_0):
- `GreedyGenerationSelectsArgmaxLogit`
- `WeightRoundTripProducesSameLogits`
- `PromptProducesExpectedLogits`

### Step 2: Tiny config guidelines

The dimensions in `makeTiny<Model>Config()` must **exactly match** the C++ model
constructor. A mismatch causes a weight bin shape error at load time.

Recommended tiny dimensions:

| Field | Value |
|-------|-------|
| `hidden_size` | 64 |
| `intermediate_size` | 64 |
| `num_hidden_layers` | 1–2 |
| `num_attention_heads` | 8 |
| `num_key_value_heads` | 4 (GQA) |
| `head_dim` | 8 |
| `vocab_size` | 32 |
| `max_position_embeddings` | 8 |

The shared nntrainer config already sets `max_seq_len = 8`, `init_seq_len = 4`,
`num_to_generate = 1`.

**Critical**: The MHA core layer's `max_timestep` must be set to `MAX_SEQ_LEN`
(= `max_position_embeddings`). Setting it to `INIT_SEQ_LEN + NUM_TO_GENERATE`
causes an out-of-bounds access in the RoPE table during the first decode step.

### Step 3: Generate the differential reference fixture

#### 3-1. Write the generation script

Create
`test/unittest/models/causallm_reference/generators/generate_<model>_reference.py`.
Use the existing `generate_qwen3_reference.py` as a template.

Key points:
- `TINY_CONFIG` dimensions must exactly match C++ `makeTiny<Model>Config()`.
- **Reuse the production weight converter** (`res/<model>/weight_converter.py`).
  Do not write a separate conversion — reusing the converter validates it too.
  Import it via `REPO_ROOT / "Applications" / "CausalLM" / "res" / ...` (see the
  `CONVERTER_DIR` line in the qwen3 template).
- Use `input_ids = [1, 4, 2, 3]` and `n_gen = 4` as the standard test input.
- For models with `tie_word_embeddings=True`, do not save lm_head separately
  (`save_lm_head = not config.tie_word_embeddings`).
- For models with `query_pre_attn_scalar` (e.g. Gemma3), set it equal to `head_dim`
  so the HF attention scale matches the C++ MHA core (`1/sqrt(head_dim)`).

```bash
python3 test/unittest/models/causallm_reference/generators/generate_<model>_reference.py
```

Outputs:
- `config.json`, `generation_config.json`, `nntr_config.json`, `tokenizer.json`
- `nntr_<model>_tiny_fp32.bin`
- `input_ids.json`, `reference_logits.json`, `reference_tokens.json`
- `meta.json` — records tolerances and library versions

#### 3-2. Set tolerances in `meta.json`

`logits_atol_fp32` is determined by the HF ↔ nntrainer FP32 difference. For a
correct converter, `0.01` is sufficient.

`logits_atol_q40` accounts for Q4_0 quantization error. To calibrate it:
1. Start with a loose value (`5.0`) and verify the Q4_0 test passes.
2. Read the maximum deviation from the test output, then set the tolerance to
   that value multiplied by a safety factor (e.g. ×2).

```json
{
  "seed": 42,
  "n_gen": 4,
  "input_ids": [1, 4, 2, 3],
  "logits_atol_fp32": 0.01,
  "logits_atol_q40": 5.0,
  "prefix_match_min": 2
}
```

#### 3-3. Commit the fixture

```bash
git add test/unittest/models/causallm_reference/<model>_tiny/
git add test/unittest/models/causallm_reference/generators/generate_<model>_reference.py
```

Include the binary file (`.bin`). Tiny fixtures are approximately 100–200 KB,
which is negligible for the repository.

### Step 4: Write the differential reference test file

`unittest_causallm_<model>_reference.cpp` is intentionally minimal:

```cpp
// unittest_causallm_mymodel_reference.cpp

#include <causallm_test_utils.h>
#include <gtest/gtest.h>
#include <mymodel_causallm.h>
#include <memory>

namespace {

causallm_test::DifferentialModel myModelDescriptor() {
  return {
    "mymodel_tiny",
    [](causallm::json &cfg, causallm::json &gen, causallm::json &nntr) {
      return std::make_unique<
        causallm_test::CausalLMTestAdapter<causallm::MyModelCausalLM>>(
        cfg, gen, nntr);
    },
  };
}

TEST(MyModelDifferentialTest, FP32MatchesHFReference) {
  causallm_test::runFp32DifferentialChecks(myModelDescriptor());
}

TEST(MyModelDifferentialTest, Q40CloseToFP32Reference) {
  causallm_test::runQ40DifferentialChecks(myModelDescriptor());
}

} // namespace
```

Both tests skip automatically when the fixture is absent.

### Step 5: Register in `meson.build`

Add both source files to the `unittest_causallm_models` source list in
`Applications/CausalLM/meson.build`:

```meson
sources = [
  ...
  'unittest_causallm_mymodel.cpp',
  'unittest_causallm_mymodel_reference.cpp',
]
```

---

## Running the Tests

```bash
cd build
ninja unittest_causallm_models

# All tests (Q4_0 differential tests are SKIPPED without NNTR_QUANTIZE_BIN)
./Applications/CausalLM/unittest_causallm_models

# Include Q4_0 differential tests
NNTR_QUANTIZE_BIN=<path/to/nntr_quantize> \
  ./Applications/CausalLM/unittest_causallm_models

# One model only
./Applications/CausalLM/unittest_causallm_models --gtest_filter="*MyModel*"

# Differential tests only
./Applications/CausalLM/unittest_causallm_models --gtest_filter="*Differential*"
```

When `NNTR_QUANTIZE_BIN` is not set, `Q40CloseToFP32Reference` tests are reported
as SKIPPED, not FAILED.

To override the fixture root directory:
```bash
NNTRAINER_CAUSALLM_FIXTURE_DIR=/my/path ./unittest_causallm_models
```

---

## Checklist

- [ ] MHA core `max_timestep` is set to `MAX_SEQ_LEN`, not `INIT_SEQ_LEN + NUM_TO_GENERATE`
- [ ] Tiny config dimensions match between the C++ model and the Python generation script
- [ ] The production weight converter is reused (no separate conversion code)
- [ ] `lm_head` is not saved separately when `tie_word_embeddings = true`
- [ ] `query_pre_attn_scalar` (if applicable) is set equal to `head_dim`
- [ ] Weight converter saves tensors in DFS topological sort order
- [ ] Fixture is committed under `causallm_reference/<model>_tiny/`
- [ ] `logits_atol_fp32 ≤ 0.01` and the FP32 differential test passes
- [ ] Both source files are registered in `meson.build`
