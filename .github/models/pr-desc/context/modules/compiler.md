# Compiler Module (`nntrainer/compiler`)

## Responsibility

Transforms high-level model descriptions (INI, ONNX, TFLite) into an executable computation graph. Handles:

- Parsing different configuration formats.
- Realization/rewriting passes that insert or rewrite layers and operations.
- Export of compiled graphs back to formats (e.g. TFLite).

## Key components

Typical files:

- `compiler.h`, `compiler_fwd.h` — main compiler interfaces and entry points.
- `interpreter.h` and concrete interpreters:
  - `ini_interpreter.*`
  - `onnx_interpreter.*`
- Realizer components:
  - `*_realizer.*` (e.g. `activation_realizer`, `bn_realizer`, `flatten_realizer`, `input_realizer`).
- `flatbuffer_opnode.*` — representation of operations for flatbuffer export.

## Dependencies and interactions

- Consumes model definitions from `models/` and configuration files.
- Produces graph structures that are consumed by `graph/` and `layers/`.
- Uses `schema/` for serialization formats and `utils/` for property handling/logging.
- Strongly tied to the semantics of `layers/` and `tensor/` (shapes, data formats).

## Typical changes

- Adding support for a new layer or operation to interpreters and realizers.
- Extending INI/ONNX parsing to support new attributes.
- Adjusting passes that insert auxiliary layers (e.g. activation, batch norm).

## Review focus

When files under `nntrainer/compiler/` change, focus on:

- **Semantic correctness**:
  - Are the inferred tensor shapes and data formats consistent with `layers/` and `tensor/`?
  - Are default properties and initializers aligned with existing behaviour?
- **Backward compatibility**:
  - INI or ONNX field changes should not silently break existing configs.
  - Versioned schema changes should be handled explicitly.
- **Graph transformations**:
  - Realizer passes must preserve numerical semantics and training behaviour.
  - Check that helper layers are inserted in the correct order and with correct connections.

## Common pitfalls

- Incomplete handling of corner cases (e.g. scalar/broadcast shapes, dynamic dimensions).
- Forgetting to register new operations in all relevant interpreters and tests.
- Introducing subtle changes in execution order that affect model numerics.
