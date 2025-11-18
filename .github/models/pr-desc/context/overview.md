# nntrainer Project Overview

This document gives a high-level map of the repository as context for an AI PR description generator.

The repository is split into two main trees:

- `nntrainer/` — core C++ library implementing the training/inference engine.
- `test/` — unit tests, integration tests, and test utilities.

## High-level layout

### `nntrainer/` (core library)

Core runtime and subsystems:

- **Root files**
  - `engine.*` — top-level training/inference engine orchestration.
  - `context.h` / `app_context.*` — global/application context, factory-style access to subsystems.
  - `mem_allocator.*` — memory allocation policies for tensors and buffers.
  - `cl_context.*`, `cl_buffer_manager.*` — OpenCL buffer/context helpers.
  - `nntrainer_error.h`, `nntrainer_log.h`, `nntrainer_logger.*` — error and logging utilities.
  - `meson.build` — build integration.

Subdirectories by responsibility (each has its own module document under `modules/`):

- `compiler/` — model configuration and graph compilation (INI/ONNX/TFLite interpreters, realization passes).
- `dataset/` — data loaders, pipelines, and producer interfaces.
- `graph/` — compute graph representation, execution order, and graph-level utilities.
- `layers/` — layer implementations and layer-related utilities.
- `models/` — model-level wrappers, configuration, and training/inference orchestration.
- `opencl/` — OpenCL device/program/queue management and kernel integration.
- `optimizers/` — optimizer implementations and learning-rate logic.
- `schema/` — flatbuffer/serialization schemas to persist and restore models.
- `tensor/` — tensor abstraction, tensor operations, pools, CPU/OpenCL backends.
- `utils/` — common helpers (properties, threading, INI wrapper, fp16, misc utilities).

### `test/` (tests and helpers)

- `unittest/` — main C++ unit tests grouped by subsystem:
  - `compiler/`, `datasets/`, `graph/`, `layers/`, `models/`, `optimizers/`, `tensor/`, etc.
- `test_models/` — model configuration files and assets used by integration tests.
- `tizen_capi/`, `ccapi/`, `jni/`, `nnstreamer/` — API-level tests and integration with external environments.
- `include/` — test-only headers and utilities.
- `input_gen/` — scripts/programs to generate test inputs and model definitions.

See `modules/Test.md` for a more detailed description of the test structure.

## How to use these documents in PR description generation

- Use `overview.md` to understand **where** a changed file sits in the overall architecture.
- Use the corresponding file inside `modules/` to understand **what that part of the system is responsible for**, which other subsystems it depends on, and what typical risks a change may introduce.
- When summarising a PR:
  - Group changes by these modules.
  - Highlight behaviour, API, and performance changes per module, not per file.
