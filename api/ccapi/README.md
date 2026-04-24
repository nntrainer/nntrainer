# `ml::train::Tensor` — Symbolic Graph API

This directory hosts the public C++ API of NNTrainer. The `Tensor` class
declared in [`include/tensor_api.h`](include/tensor_api.h) is a thin,
Pimpl-backed facade that lets callers build and execute models in terms
of *tensors that flow through layers* rather than in terms of layer
names and string-encoded `input_layers` properties.

## 1. Overview

### Motivation

The previous `ml::train::Tensor` was declared as a direct subclass of
the internal `nntrainer::Var_Grad`, which leaked gradient/optimizer
state into the public surface and offered no operations of its own.
Network authors therefore had to construct models layer-by-layer with
`addLayer()` calls and wire them together using string properties such
as `"input_layers=prev_name"`. Long models became error-prone and hard
to read, and the API could not describe features that were natural for
users (eager tensors, operator overloads, lazy chaining).

### Design goals

- **Separate the public `Tensor` from internal storage.** The public
  class hides an `Impl` struct and never exposes `nntrainer::Tensor`,
  `Var_Grad`, or `LayerNode` directly.
- **Two kinds of tensors, one class.** *Symbolic* tensors (constructed
  from a `TensorDim`) act as graph placeholders that are bound to
  internal storage at `compile()` time. *Eager* tensors (`fromData`,
  `zeros`, `ones`) carry live data immediately. Both expose the same
  method set; the internal state decides how each call is dispatched.
- **Graph is built by calling layers on tensors.** `LayerHandle` is a
  callable wrapper around `createLayer(...)`. `layer(x)` records an
  edge in the symbolic graph and returns the output tensor. The user
  never handles an input-layer string.
- **Allocation strategy follows construction.** `Tensor(dim)` maps to
  a `UNIQUE` tensor-pool slot; `Tensor::fromData(dim, ptr)` maps to
  `PLACEHOLDER` (external memory, never allocated by the pool). No
  extra registration API is needed.
- **Lazy chaining is a first-class concept.** In-place ops such as
  `add_i`, `subtract_i`, `multiply_i`, `pow_i`, `inv_sqrt_i` can be
  queued via `chain()` and flushed with `eval()`, matching how model
  authors express elementwise post-processing.

## 2. Tensor class basics

```cpp
namespace ml::train {
class Tensor {
public:
  Tensor();                                        // invalid / empty
  explicit Tensor(const TensorDim &dim,            // symbolic
                  const std::string &name = "");
  Tensor(const Tensor &rhs);                       // shallow (shares graph node)
  Tensor &operator=(const Tensor &rhs);
  Tensor(Tensor &&) noexcept;
  Tensor clone() const;                            // deep (eager tensors only)
  ~Tensor();

  bool isValid()        const;                     // construction succeeded
  bool isExternal()     const;                     // backed by user memory
  bool isMaterialized() const;                     // has data accessible now
  const TensorDim &shape()  const;
  const std::string &name() const;
  TensorDim::DataType dtype() const;
  // …
};
} // namespace ml::train
```

### Internal state (Pimpl)

`Tensor` owns a `std::unique_ptr<Impl>`. The `Impl` struct is defined in
`src/tensor_api_impl.h` (not installed) and records the subset of state
that distinguishes symbolic vs eager and bound vs unbound tensors:

| Field | Purpose |
|-------|---------|
| `dim`, `name` | Shape and identifier |
| `valid` | `true` once constructed with dim or data |
| `external` | `true` when memory is user-owned (`fromData`) |
| `eager_data` | `shared_ptr<nntrainer::Tensor>` for eager values |
| `external_ptr` | Raw pointer for external tensors |
| `bound_tensor` | Internal tensor populated by `compile()` + `initialize()` |
| `graph_edge` | `shared_ptr<SymbolicGraphNode>` — producing layer + inputs |
| `call_chain` | `std::vector<std::function<void(nntrainer::Tensor&)>>` for `chain()` |

Copy is intentionally shallow: copying a `Tensor` duplicates the Pimpl
struct but keeps the `graph_edge` `shared_ptr` so that downstream layers
still observe a single node in the symbolic graph. `clone()` performs a
deep data copy for eager tensors.

### Symbolic vs eager vs bound

| Source | `isValid` | `isExternal` | `isMaterialized` before `compile()` | After `compile() + initialize()` |
|--------|-----------|--------------|-------------------------------------|----------------------------------|
| `Tensor()` | ✗ | ✗ | ✗ | ✗ |
| `Tensor(dim, name)` | ✓ | ✗ | ✗ | ✓ (bound to pool slot) |
| `Tensor::fromData(dim, ptr)` | ✓ | ✓ | ✓ | ✓ (placeholder, pool not allocated) |
| `Tensor::zeros(dim)` / `ones(dim)` | ✓ | ✗ | ✓ | ✓ |

## 3. Creating tensors

```cpp
using namespace ml::train;

// Symbolic placeholder — no storage until compile/initialize
Tensor x({batch, 1, 28, 28}, "image");

// Eager tensors — data available immediately
Tensor w  = Tensor::zeros({1, 1, 784, 10}, "w");
Tensor b  = Tensor::ones({1, 1, 1, 10},    "b");

// External memory — caller owns the buffer, pool skips allocation
float kv_buf[seq_len * num_heads * head_dim] = {};
Tensor kv_cache =
    Tensor::fromData({1, num_heads, seq_len, head_dim}, kv_buf, "kv");
```

Eager and external tensors materialize immediately. Symbolic tensors
only materialize after the owning model calls `compile()` followed by
`initialize()`, at which point their `bound_tensor` pointer is set from
the tensor pool.

## 4. Graph construction

### `LayerHandle::operator()`

`createLayer(...)` now returns a `LayerHandle` (light wrapper around
`std::shared_ptr<Layer>`) that is callable. Calling it on a tensor
registers a new symbolic edge:

```cpp
LayerHandle fc1 = createLayer("fully_connected", {"unit=128", "name=fc1"});
LayerHandle fc2 = createLayer("fully_connected", {"unit=10",  "name=fc2"});

Tensor h = fc1(x);       // edge: fc1 consumes x, produces h
Tensor y = fc2(h);       // edge: fc2 consumes h, produces y
```

Multi-input layers accept a vector:

```cpp
LayerHandle concat = createLayer("concat", {"axis=3", "name=concat"});
Tensor merged = concat({a, b, c});
```

### Implicit operation layers

Arithmetic operators on symbolic tensors create addition / multiply
layers on the fly. They are registered with auto-generated names
(`add_0`, `mul_0`, …) so they appear in the graph exactly like any
other layer:

```cpp
Tensor r = a + b;            // internally calls createLayer("Addition", …)
Tensor s = r * scale;        // internally calls createLayer("Multiply", …)
Tensor t = r.reshape({1, 1, batch, -1});
```

Indexed outputs for multi-output layers are accessed via `output(i)`:

```cpp
LayerHandle split = createLayer("split", {"axis=3", "name=split"});
Tensor parts = split(h);
Tensor p0 = parts.output(0);
Tensor p1 = parts.output(1);
```

## 5. `Model::compile(Tensor, ...)`

Three overloads are provided on top of the pre-existing
`Model::compile(ExecutionMode)`:

```cpp
int Model::compile(Tensor &input,  Tensor &output,
                   ExecutionMode mode = ExecutionMode::INFERENCE);
int Model::compile(Tensor &input,  std::vector<Tensor> &outputs,
                   ExecutionMode mode = ExecutionMode::INFERENCE);
int Model::compile(std::vector<Tensor> &inputs,
                   std::vector<Tensor> &outputs,
                   ExecutionMode mode = ExecutionMode::INFERENCE);
```

### What compile does

1. **Backward DFS from outputs.** Starting at each output tensor, the
   builder walks the `graph_edge` chain toward the inputs and collects
   every distinct producing layer in topological order.
2. **Auto-insert `input` layers.** Leaf tensors constructed with
   `Tensor(dim, name)` become `createLayer("input", {"name=…",
   "input_shape=…"})` calls. If an additional leaf (non-`input` layer
   output) is observed during the walk, another `input` layer is
   synthesized so the graph remains well-formed.
3. **Populate `input_layers` from edges.** Each layer is added with
   `setProperty({"input_layers=…"})` derived from its `inputs` vector.
   No user-facing string wiring is required.
4. **Delegate to the classical `compile(ExecutionMode)`** on the
   internal model, then call `initialize(mode)` implicitly before
   returning so that `bound_tensor` becomes valid.

### Minimal example

```cpp
using namespace ml::train;

Tensor x({batch, 1, 1, 784}, "x");
auto h = createLayer("fully_connected",
                     {"unit=128", "activation=relu", "name=fc1"})(x);
auto y = createLayer("fully_connected",
                     {"unit=10", "activation=softmax", "name=fc2"})(h);

auto model = createModel(ModelType::NEURAL_NET, {"batch_size=1"});
model->compile(x, y);      // builds graph, adds layers, compiles, initializes
```

## 6. Lazy chaining

In-place math on a `Tensor` can be *deferred* so that a sequence of
elementwise updates is executed in one pass:

```cpp
auto t = Tensor::zeros({1, 1, 4, 4});
t.chain()
 .add_i(3.0f)            // t += 3
 .multiply_i(2.0f)       // t *= 2
 .pow_i(0.5f)            // t  = sqrt(t)
 .eval();                // flush the queued ops
```

The chain is stored as
`std::vector<std::function<void(nntrainer::Tensor&)>>` on the Pimpl
and runs over the internal tensor on `eval()`. Tensor-vs-tensor variants
capture the other operand's `Impl*`; the lambda resolves the source to
`bound_tensor` (if compiled) or `eager_data` (if eager) at execution
time, so the same chain works before and after `compile()`.

Supported queued operations: `add_i(float)`, `add_i(Tensor, alpha)`,
`subtract_i(float|Tensor)`, `multiply_i(float|Tensor)`,
`divide_i(float|Tensor)`, `pow_i(float)`, `inv_sqrt_i()`.

## 7. Migrating from the previous API

Old style — manual `addLayer` + string-based input wiring:

```cpp
model->addLayer(createLayer("input", {"name=x", "input_shape=1:1:784"}));
model->addLayer(createLayer("fully_connected",
                            {"name=fc1", "unit=128", "activation=relu",
                             "input_layers=x"}));
model->addLayer(createLayer("fully_connected",
                            {"name=fc2", "unit=10", "activation=softmax",
                             "input_layers=fc1"}));
model->compile();
```

New style — symbolic tensors:

```cpp
Tensor x({1, 1, 1, 784}, "x");
auto h = createLayer("fully_connected",
                     {"name=fc1", "unit=128", "activation=relu"})(x);
auto y = createLayer("fully_connected",
                     {"name=fc2", "unit=10", "activation=softmax"})(h);
model->compile(x, y);
```

### Mapping table

| Legacy                                | New                                             |
|---------------------------------------|-------------------------------------------------|
| `addLayer(createLayer("input", {...}))` | `Tensor input({...}, "name")`                 |
| `"input_layers=prev"` property        | `layer(prev)` / `layer({a, b, c})`              |
| manual `addLayer` walk + `compile()`  | `model->compile(input, output)`                 |
| user-created add/mul layers           | `a + b`, `a * b`                                |
| manual reshape layer                  | `t.reshape({...})`                              |
| pre-allocated external buffer         | `Tensor::fromData(dim, ptr, name)`              |

Existing `addLayer` + `compile()` paths remain fully supported — the
new overloads are additive and do not remove any public entry point.

## 8. End-to-end example

```cpp
#include <model.h>
#include <layer.h>
#include <tensor_api.h>

using namespace ml::train;

int main() {
  constexpr int batch = 32;

  // --- Graph ---
  Tensor x({batch, 1, 1, 784}, "x");

  auto h1 = createLayer(
      "fully_connected",
      {"unit=256", "activation=relu", "name=fc1"})(x);
  auto h2 = createLayer(
      "fully_connected",
      {"unit=128", "activation=relu", "name=fc2"})(h1);
  auto y  = createLayer(
      "fully_connected",
      {"unit=10",  "activation=softmax", "loss=cross", "name=fc3"})(h2);

  // --- Model ---
  auto model = createModel(ModelType::NEURAL_NET,
                           {"batch_size=" + std::to_string(batch),
                            "epochs=5"});
  model->setOptimizer(createOptimizer("adam", {"learning_rate=0.001"}));
  model->compile(x, y, ExecutionMode::TRAIN);

  model->setDataset(DatasetModeType::MODE_TRAIN,
                    createDataset(DatasetType::GENERATOR, train_cb));
  model->train();
  return 0;
}
```

## File layout

The implementation is split across four translation units so no single
file carries the whole surface:

| File | Contents |
|------|----------|
| `src/tensor_api.cpp`       | Construction, copy/move, accessors, state, data access, factories (`fromData`/`zeros`/`ones`), source/graph layer accessors, `getInternalPtr`/`wrapResult` helpers |
| `src/tensor_api_ops.cpp`   | Eager numeric ops (`add`/`subtract`/`multiply`/`divide`/`dot`/`transpose`/`pow`/`sum`/`average`/`l2norm`/`argmax`) and manipulation (`apply`/`apply_i`/`cat`/`getBatchSlice`/`getSharedDataTensor`) |
| `src/tensor_api_lazy.cpp`  | `chain()` / `add_i` / `subtract_i` / `multiply_i` / `divide_i` / `pow_i` / `inv_sqrt_i` / `eval()` |
| `src/tensor_api_graph.cpp` | `LayerHandle::operator()`, implicit op layers, graph-based `Model::compile` overloads |
| `src/tensor_api_impl.h`    | Private shared declarations: `struct Impl`, `struct SymbolicGraphNode`, `asInternal` inline helper |
