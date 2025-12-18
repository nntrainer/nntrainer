# Tensor CPU Backend (`nntrainer/tensor/cpu_backend`)

## Responsibility

Implements CPU-optimized math backends and kernels used by the tensor module:

- GEMM and other BLAS-like kernels for float/half/int quantized types.
- Architecture-specific optimizations (ARM NEON, x86, etc.).
- Abstraction of backend selection and dispatch.

## Key components

Representative files:

- `cpu_backend/arm/*` — ARM-specific backends:
  - `arm_compute_backend.*` — integration with ARM Compute Library where applicable.
  - `hgemm/*` — half-precision GEMM kernels and padding/packing utilities.
  - `kai/*` — quantized matmul implementations and related utilities.
- Backend interfaces and utilities described in `cpu_backend/README.md`.

## Dependencies and interactions

- Called by higher-level tensor operations in `nntrainer/tensor/`.
- Used heavily by `layers/` for fully connected, convolution, and other compute-heavy ops.
- Must remain compatible with tensor layouts and quantization schemes.

## Typical changes

- Adding new microkernels for specific shapes or data types.
- Tuning existing kernels for performance on new CPU targets.
- Refactoring backend interfaces for clarity or new features (e.g., quantization support).

## Review focus

For changes in CPU backend:

- **Numerical correctness**:
  - Compare against reference implementations for a wide range of sizes and values.
- **Performance**:
  - Kernels are hot paths; watch for unnecessary branches, loads, and stores.
- **Portability**:
  - Guard architecture-specific code with appropriate feature checks.
  - Confirm build still works on non-target architectures.

## Common pitfalls

- Alignment assumptions mismatched with actual allocation.
- Incorrect handling of edge tiles or remainder loops.
- Divergence between reference tensor semantics and backend implementation (e.g., layout or zero-padding assumptions).
