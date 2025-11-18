# OpenCL Module (`nntrainer/opencl`)

## Responsibility

Wraps OpenCL functionality and integrates GPU execution:

- Device enumeration and capabilities.
- Context and command queue management.
- Program and kernel compilation, loading, and invocation.
- Buffer management support for tensors.

## Key components

Typical files:

- `opencl_device_info.*` — device properties and capabilities.
- `opencl_context_manager.*` — creation and lifetime of OpenCL contexts.
- `opencl_command_queue_manager.*` — management of command queues.
- `opencl_program.*` — program build, caching, and kernel lookup.
- `opencl_kernel.*` — kernel invocation utilities.
- `opencl_loader.*` — dynamic OpenCL library loading.

## Dependencies and interactions

- Used indirectly by `tensor/` (via `cl_operations/`) and the core runtime.
- Works alongside `cl_context.*`, `cl_buffer_manager.*` in the root `nntrainer/` directory.
- Sensitive to build-time and runtime configuration (device selection, precision, etc.).

## Typical changes

- Adding support for new kernels or extending existing ones.
- Improving robustness of device/context selection and error handling.
- Performance tuning of kernel launches and buffer transfers.

## Review focus

For changes in `nntrainer/opencl/`:

- **Correctness across devices**:
  - Check for assumptions about specific vendors or device versions.
- **Error handling and fallbacks**:
  - Graceful handling when OpenCL is unavailable or specific capabilities are missing.
- **Performance**:
  - Ensure kernel launch parameters and data movement are efficient.

## Common pitfalls

- Hard-coding device indices or assuming a single GPU.
- Not propagating OpenCL errors, leading to silent failures.
- ABI or binary compatibility issues when caching programs/binaries.
