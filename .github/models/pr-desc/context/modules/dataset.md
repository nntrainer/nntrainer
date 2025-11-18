# Dataset Module (`nntrainer/dataset`)

## Responsibility

Provides data input pipelines for training and inference:

- Dataset configuration and lifecycle.
- Data producer interfaces and their implementations.
- Iteration utilities and pre-processing hooks.

## Key components

Typical files:

- `data_iteration.*` — iteration utilities for datasets.
- `dataset.*` — main dataset abstraction.
- `data_producer.*`, `data_producer_common_properties.*` — base producer interfaces and shared properties.
- Concrete producers, e.g.:
  - `func_data_producer.*`
  - `raw_file_data_producer.*`
- `meson.build` — build integration.

## Dependencies and interactions

- Used by `models/` and training loops inside `engine` to feed mini-batches.
- Uses `tensor/` for representing sampled data.
- May interact with `utils/` for configuration and thread management.

## Typical changes

- Adding a new data producer type (custom file formats, synthetic data, etc.).
- Extending dataset configuration (shuffling, augmentation, batching strategies).
- Performance tuning of data loading/iteration.

## Review focus

For changes in `nntrainer/dataset/`:

- **Correctness and determinism**:
  - Ensure label/feature pairing, shuffling, and epoch handling are consistent.
  - Deterministic behaviour when deterministic flags/seeds are used.
- **Resource management**:
  - No leaks or unbounded growth of buffers/threads.
  - Proper ownership of pointers and handles.
- **Throughput and latency**:
  - Critical for training performance; avoid unnecessary copies or sync points.
- **Error handling**:
  - Graceful handling of I/O errors and malformed inputs.

## Common pitfalls

- Off-by-one errors in epoch or batch counting.
- Incorrect handling of end-of-data conditions.
- Implicit assumptions about tensor shapes or layouts that do not generalize.
