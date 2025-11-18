# Layers Module (`nntrainer/layers`)

## Responsibility

Implements individual neural network layers and layer-related utilities:

- Layer base classes and interfaces.
- Concrete layer implementations (convolution, dense, recurrent, normalization, etc.).
- Layer property definitions and validation.

## Key components

Typical files:

- Base layer definitions:
  - `layer.*`, `layer_context.*`, `layer_node.*`, etc.
- Core layers (names may vary in detail):
  - `fully_connected_layer.*`, `convolution_layer.*`, `pooling_layer.*`, `batch_normalization_layer.*`,
    `activation_layer.*`, `loss_layer.*`, etc.
- Specialized or experimental layers under subdirectories.
- Property and registration utilities for layers.

## Dependencies and interactions

- Use `tensor/` for data representation and operations.
- Integrated into graphs by `graph/` and `compiler/`.
- Interact with `optimizers/` through parameter tensors and gradient flows.
- Exposed to external configuration through `models/` and config parsers.

## Typical changes

- Adding a new layer type (forward/backward logic, configuration, and registration).
- Optimizing existing layers (better kernels, fused operations).
- Adjusting property validation or default values.

## Review focus

For changes in `nntrainer/layers/`:

- **Numerical correctness**:
  - Forward and backward implementations consistent with expected math.
  - Shape inference and tensor layouts correctly handled.
- **Performance**:
  - Hot path for training/inference; watch for unnecessary copies or allocations.
  - Vectorization and backend usage (CPU/OpenCL) where applicable.
- **Config and API**:
  - New properties should have clear names and validation rules.
  - Changes must be backward compatible or explicitly versioned.

## Common pitfalls

- Incomplete gradient definitions leading to silent training failures.
- Mismatched broadcasting or shape assumptions.
- Missing updates to registration tables or interpreters when adding a new layer.
