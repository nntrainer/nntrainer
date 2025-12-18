# Test Module (`test/`)

## Responsibility

Provides automated tests and supporting assets for nntrainer:

- Unit tests for each core subsystem.
- Integration tests using real models and configurations.
- API-level tests for different frontends (Tizen CAPI, C++ API, JNI, nnstreamer).
- Test utilities and input generation tooling.

## Structure

Top-level layout under `test/`:

- `unittest/`
  - Main C++ unit tests, organized by module:
    - `compiler/` — tests for interpreters and realizers.
    - `datasets/` — tests for dataset and data producer behaviour.
    - `graph/` — tests for graph construction, execution order, and properties.
    - `layers/` — per-layer forward/backward tests and property checks.
    - `models/` — tests end-to-end models and training flows.
    - `optimizers/` — tests for optimizer correctness and edge cases.
    - `tensor/` — tests for tensor operations, pools, and CPU backends (including fp16).
    - Other directories mirroring modules as needed.
- `test_models/`
  - `.ini` model configuration files, binary snapshots, and TFLite models used by integration tests.
- `tizen_capi/`, `ccapi/`, `jni/`, `nnstreamer/`
  - Tests that exercise higher-level APIs and external integrations.
- `include/`
  - Test-only headers (`nntrainer_test_util.h`, timers, helpers).
- `input_gen/`
  - Scripts and small programs to generate inputs and model definitions for tests.
- Root files like:
  - `nntrainer_test_util.cpp` — shared test utilities.
  - `unittestcoverage.py` — coverage-related scripts.
  - `meson.build` — test build integration.

## Typical changes

- Adding new unit tests when core logic changes.
- Extending model coverage with new example configurations.
- Improving coverage and regression protection.

## Review focus

When changes touch `test/`:

- **Coverage vs changes**:
  - Verify that new or modified core logic has corresponding tests.
- **Signal quality**:
  - Tests should be deterministic and focused; avoid flaky or overly slow tests.
- **Maintainability**:
  - Keep test duplication low; reuse utilities when possible.

## Common pitfalls

- Tests that depend on environment-specific configuration or hardware.
- Missing negative tests for invalid configurations and error paths.
- Overly broad integration tests that are hard to debug when failing.
