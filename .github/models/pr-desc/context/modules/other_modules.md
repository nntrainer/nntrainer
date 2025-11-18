# Other Core Modules (root of `nntrainer/`)

## Responsibility

This module document covers core runtime files located directly under `nntrainer/` (not in a subdirectory):

- Application and global context.
- Engine orchestration.
- Memory allocation and OpenCL buffer helpers.
- Global logging and error helpers.

## Key components

Representative files:

- `engine.*` — orchestrates training and inference, coordinates models, datasets, and optimizers.
- `context.h` — provides access to global configuration and factories.
- `app_context.*` — higher-level context wrapper used by applications.
- `mem_allocator.*` — memory allocator abstraction for tensors and buffers.
- `cl_context.*`, `cl_buffer_manager.*` — OpenCL context and buffer helpers.
- `nntrainer_error.h`, `nntrainer_log.h`, `nntrainer_logger.*` — logging and error-reporting utilities.
- `meson.build` — build integration for the core library target.

## Dependencies and interactions

- Integrates all other modules (`models/`, `dataset/`, `optimizers/`, `tensor/`, etc.) into runnable workflows.
- Interaction point for external applications embedding nntrainer.
- Logging and error behaviour visible to all subsystems.

## Typical changes

- Adding new high-level engine features (e.g., new execution modes, profiling).
- Changing memory allocation strategies.
- Adjusting logging behaviour and error reporting.

## Review focus

For changes in these core files:

- **Global impact**:
  - Any change can affect most of the system; check for regressions and ABI breaks.
- **Resource management**:
  - Ensure contexts, allocators, and devices are created and destroyed correctly.
- **User-facing behaviour**:
  - Logging, error messages, and configuration defaults are part of the user experience.

## Common pitfalls

- Subtle behaviour changes that are not covered by tests in all modules.
- Initialization order issues when adding new global or static objects.
- Tight coupling to specific backends or build options.
