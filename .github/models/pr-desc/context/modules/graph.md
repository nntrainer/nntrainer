# Graph Module (`nntrainer/graph`)

## Responsibility

Represents and manipulates the internal computation graph:

- Nodes, edges, and execution order.
- Graph-level analyses and utilities for training and inference.
- Integration with models and compiler output.

## Key components

Typical files:

- `graph_core.*`, `graph_util.*` — core data structures and helper functions.
- `network_graph.*` — high-level graph used by models.
- `graph_event.*`, `graph_manager.*` or similarly named files for managing graph lifecycle.
- `meson.build` — build integration.

(Exact filenames may evolve; the key concept is graph representation and utilities.)

## Dependencies and interactions

- Consumes compiled representations from `compiler/`.
- Drives which `layers/` are executed and in what order.
- Cooperates with `tensor/` tensor pools and memory planners.
- Used by `models/` to build and run networks.

## Typical changes

- Changing execution order logic, scheduling, or graph traversal.
- Adding new graph passes (e.g. optimization, validation).
- Enhancing graph serialization or visualization.

## Review focus

For changes in `nntrainer/graph/`:

- **Execution correctness**:
  - Topological order must respect data dependencies.
  - Backprop graph structure must match forward graph semantics.
- **Performance characteristics**:
  - Changes in traversal or scheduling may affect memory reuse and parallelism.
- **Integration with compiler and models**:
  - Ensure that new graph features have corresponding updates in `compiler/` and tests.

## Common pitfalls

- Cycles or invalid graphs slipping through validation.
- Failing to update gradient/optimizer related paths when graph changes.
- Inconsistent handling of training vs inference modes.
