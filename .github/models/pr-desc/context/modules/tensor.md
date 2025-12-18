# Tensor Module (`nntrainer/tensor`)

## Responsibility

Provides the tensor abstraction and core tensor operations:

- Tensor shapes, strides, and memory layout.
- Tensor storage, pooling, and lifetime management.
- Device-specific operations (CPU and OpenCL).

## Key components

Representative files:

- Core tensor types:
  - `tensor.*`, `tensor_dim.*`, `var_grad.*`, etc.
- Tensor pools and planners:
  - `basic_planner.*`, `tensor_pool.*`, cache-related classes (`cache_*.*`).
- Data type and view helpers:
  - `char_tensor.*`, `manager.*`, etc.
- Device-specific operations:
  - `cl_operations/*` with OpenCL kernels in `cl_operations/cl_kernels/*.cl`.
  - `cpu_backend/*` for CPU math backends (see `Tensor_cpu_backend.md`).

## Dependencies and interactions

- Used directly by almost all other modules (`layers/`, `models/`, `graph/`, `dataset/`).
- Coordinates with `opencl/` and root-level OpenCL helpers for GPU execution.
- Uses `utils/` for properties, logging, and threading helpers.

## Typical changes

- Extending tensor APIs (new ops, views, or metadata).
- Improving memory reuse and planning.
- Integrating new device backends or kernels.

## Review focus

For changes in `nntrainer/tensor/`:

- **ABI and API stability**:
  - Tensor is a foundational type; interface changes have wide impact.
- **Correctness of broadcasting and indexing**:
  - Check all shape/stride calculations and boundary conditions.
- **Performance**:
  - In-place vs out-of-place operations.
  - Interaction with tensor pools and planners.

## Common pitfalls

- Hidden copies when constructing or slicing tensors.
- Concurrency issues when tensors are shared across threads.
- Divergent behaviour between CPU and OpenCL implementations of the same op.
