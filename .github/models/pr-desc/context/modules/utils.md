# Utils Module (`nntrainer/utils`)

## Responsibility

Provides shared utilities and infrastructure used across the project:

- Property and configuration helpers.
- Threading and thread pool abstractions.
- Auxiliary utilities for files, logging, math, and platform quirks.

## Key components

Representative files:

- Properties:
  - `base_properties.*`, other property helper files.
- Threading:
  - `bs_thread_pool.*`, `bs_thread_pool_manager.*`, `nntr_threads.*`.
- Numeric utilities:
  - `fp16.*` for half-precision conversions.
- Configuration and misc:
  - `ini_wrapper.*` for INI files.
  - `dynamic_library_loader.*`, `mman_windows.*`, and other platform helpers.

## Dependencies and interactions

- Used by most modules (compiler, models, tensor, etc.) where common functionality is needed.
- Some utilities are performance-critical (thread pool) and directly affect training speed.

## Typical changes

- Extending property system or configuration parsing.
- Improving thread pool behaviour or adding new synchronization primitives.
- Adding new helper utilities to reduce duplication elsewhere.

## Review focus

For changes in `nntrainer/utils/`:

- **Thread-safety and correctness** for concurrency utilities.
- **API stability** — utility types are widely used.
- **Performance impact** of changes to thread pools and fp16 handling.

## Common pitfalls

- Introducing deadlocks or contention in thread pools.
- Breaking non-Linux platforms when changing platform-specific code.
- Silent changes in default behaviour (e.g., thread count, logging verbosity).
