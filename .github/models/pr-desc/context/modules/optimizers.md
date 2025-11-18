# Optimizers Module (`nntrainer/optimizers`)

## Responsibility

Implements optimization algorithms for training:

- Optimizer base classes and configuration.
- Concrete optimizers (SGD, Adam, etc.).
- Learning rate scheduling hooks (when not delegated elsewhere).

## Key components

Typical files:

- `optimizer.*` and base classes.
- Concrete implementations:
  - `sgd.*`, `adam.*`, and others as available.
- Utility functions for weight decay, regularization, gradient clipping.
- `meson.build` — build integration.

## Dependencies and interactions

- Operate on parameter tensors defined by `layers/` and `models/`.
- Use properties and configuration helpers from `utils/`.
- Interact with training loops in `models/` and `engine/`.

## Typical changes

- Adding a new optimizer type.
- Adjusting default hyperparameters.
- Introducing advanced features (adaptive LR, warmup, etc.).

## Review focus

For changes in `nntrainer/optimizers/`:

- **Mathematical correctness** of update rules.
- **Stability**:
  - Behaviour with very small or large learning rates.
  - Handling of NaNs/Infs and numeric edge cases.
- **Config/API design**:
  - Clear mapping from config fields to algorithm behaviour.

## Common pitfalls

- Incorrect bias correction or moment updates (especially in Adam-like algorithms).
- Failing to support mixed precision or different tensor dtypes.
- Missing integration with existing tests covering convergence and regression.
