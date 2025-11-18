# Schema Module (`nntrainer/schema`)

## Responsibility

Defines serialization formats for models and graphs, typically via FlatBuffers:

- Data structures for persisting model topology and parameters.
- Versioning and backward compatibility logic.
- Conversion helpers between in-memory objects and serialized forms.

## Key components

Typical files:

- Flatbuffer schema definitions and generated sources.
- Helpers to serialize and deserialize:
  - Network graphs.
  - Layers and their properties.
  - Optimizer and training configuration.

(Exact filenames depend on how FlatBuffers is organized in the project.)

## Dependencies and interactions

- Used by `compiler/`, `models/`, and `graph/` to save and load models.
- Must remain consistent with the semantics of `layers/`, `tensor/`, and `optimizers/`.
- Interacts with external tools that consume or produce the same schema.

## Typical changes

- Extending the schema to support new layers, properties, or training options.
- Introducing new schema versions and migration paths.
- Adding helper APIs for convenience loads/saves.

## Review focus

For changes in `nntrainer/schema/`:

- **Compatibility**:
  - Avoid breaking existing serialized models; provide migrations when fields change.
  - Add explicit versioning where necessary.
- **Completeness**:
  - New runtime features must be reflected in the schema when serialization is expected.
- **Security and robustness**:
  - Validate inputs to avoid crashes or undefined behaviour with malformed files.

## Common pitfalls

- Silent behaviour changes when adding optional fields without updating readers.
- Forgetting to update tests and tools that depend on the schema.
