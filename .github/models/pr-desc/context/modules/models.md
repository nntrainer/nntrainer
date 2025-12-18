# Models Module (`nntrainer/models`)

## Responsibility

Defines model-level abstractions and orchestrates training/inference:

- Model configuration structures.
- Composition of layers and graphs into models.
- Training loop policies and execution hooks.

## Key components

Typical files:

- `model.*`, `model_common_properties.*` — base model types and shared properties.
- Concrete model classes (e.g. `neuralnet.*` and variants).
- Training/inference utilities:
  - Dynamic training optimization (`dynamic_training_optimization.*`).
  - Callbacks and event handling.
- `meson.build` — build integration.

## Dependencies and interactions

- Use `compiler/` to compile configurations into graphs.
- Depend on `graph/`, `layers/`, and `tensor/` to execute models.
- Use `dataset/` for data feeding and `optimizers/` for parameter updates.
- Integrated with `engine.*` as part of the user-facing API.

## Typical changes

- Extending model configuration options.
- Modifying training loop behaviour (epochs, checkpoints, early stopping).
- Adding new model types or templates for specific tasks.

## Review focus

For changes in `nntrainer/models/`:

- **Training semantics**:
  - Verify that loss, metrics, and optimization behave as expected.
  - Ensure hooks (callbacks, events) fire at correct points.
- **Lifecycle and resource management**:
  - Proper ownership and cleanup of graphs, datasets, and tensors.
- **User-facing API**:
  - Changes should be clearly reflected in documentation and remain backward compatible when possible.

## Common pitfalls

- Partial updates that handle only a subset of training paths (e.g. ignoring inference-only flows).
- Poor error reporting when configuration is invalid or incomplete.
- Tight coupling to specific datasets or layers, reducing modularity.
