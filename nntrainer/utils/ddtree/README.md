# DDTree — Tree-based Speculative Decoding Core

A stateless, cache-agnostic C++ core implementing the **DDTree** tree-based
speculative-decoding algorithm. Given draft logits it builds a best-first
candidate token **tree**, compiles it into verify buffers and an additive
attention mask, follows the longest accepted root→leaf path after the target
model verifies the tree, and provides a reusable KV-tail compaction helper.

The core is **pure host logic** depending on the **C++ standard library only** —
it never holds a KV cache and has no `Tensor`/model dependency. The KV cache and
all model forwards live in the runtime that drives it (e.g. nntrainer's CausalLM
on CPU, or an NPU/QNN runtime on device).

---

## 1. Background

Autoregressive decoding emits one token per model forward. Speculative decoding
amortizes that cost: a cheap **draft** proposes several next tokens, and a single
**target verify** forward accepts the longest prefix the target agrees with.
DDTree proposes the candidates as a **tree** (not a single chain), so one verify
forward evaluates many alternative continuations at once, raising the accepted
tokens per step. Output is **lossless**: the target verify recomputes true
per-node logits, so draft quality affects only throughput, not correctness.

Two runtime values shape the tree:

- **`depthLimit`** — the draft horizon (`= block_size - 1`); the maximum tree
  depth. A block size of 16 gives a horizon of 15.
- **`budget`** — the number of nodes beyond the root. `budget = 31` yields a
  32-node tree.

Both are fields of `DDTreeConfig`; changing them needs no rebuild.

---

## 2. Architecture

The core decides *what to keep* and provides a *how-to-reorder* helper; the
runtime owns the cache and the forwards.

| Stage | Owner | Core function |
|-------|-------|---------------|
| draft forward → draft logits | caller | — |
| build candidate tree | **core** | `buildTree` |
| compile verify ids / positions / mask | **core** | `compile` |
| sliding-window mask variant | **core** | `makeSlidingMasks` (fp32) / `makeSlidingVisibility` (0/1) |
| target verify forward (appends the tree to KV) | caller | — |
| sample a posterior token per node | caller (`argmaxRow` / `sampleGreedy` helper) | `argmaxRow` |
| follow the accepted path | **core** | `followVerified` |
| keep indices | **core** | `followVerified().indices` |
| physically reorder the KV tail | runtime, via core helper | `compactTail` |
| KV cache storage | runtime (never the core) | — |

---

## 3. Module structure

```
nntrainer/utils/ddtree/
├── meson.build            # appends ddtree sources/headers to libnntrainer
├── ddtree_types.h         # DDTreeConfig, DDTreeStructure, CompiledTree, SlidingMasks, Accepted
├── ddtree.h / .cpp        # buildTree, compile, followVerified
├── ddtree_sliding.h/.cpp  # makeSlidingMasks (fp32 additive), makeSlidingVisibility (0/1)
├── ddtree_compact.h/.cpp  # compactTail (raw-pointer KV tail reorder)
└── ddtree_sampling.h/.cpp # argmaxRow / sampleGreedy (temperature-0 convenience)
```

Headers install to `include/nntrainer` (leaf-module convention), so downstream
consumers get them by linking `libnntrainer`.

---

## 4. API reference

Namespace `nntrainer::ddtree`.

```cpp
// Construct the best-first candidate tree from row-major [depthLimit, vocab]
// fp32 draft logits. Node count is up to 1 + min(budget, vocab).
DDTreeStructure buildTree(const float *draftLogits, int depthLimit, int vocab,
                          const DDTreeConfig &cfg);

// Flatten the tree into verify buffers (caller-owned) + an fp32 additive mask.
//  verifyInputIds[currentLength]    = [root, node tokens...]
//  verifyPositionIds[currentLength] = [start, start + depth...]  (siblings share a position)
//  attentionMask[currentLength, attnMaskRowStride]  (0 visible / maskFillValue hidden)
CompiledTree compile(int32_t rootTokenId, int start, int pastLength,
                     const DDTreeStructure &tree, const DDTreeConfig &cfg,
                     int32_t *verifyInputIds, int32_t *verifyPositionIds,
                     float *attentionMask, int attnMaskRowStride);

// Sliding-window variant of the fp32 additive mask (models that mix full and
// sliding attention). hasSlidingLayers==false -> sliding=nullptr; window<=0 -> sliding==full.
SlidingMasks makeSlidingMasks(float *attentionMask, const int32_t *verifyPositionIds,
                              int currentLength, int kvLength, int slidingWindow,
                              bool hasSlidingLayers, const DDTreeConfig &cfg,
                              float *slidingBuffer);

// Sliding-window visibility as a plain 0/1 bitmap (format-neutral). Equivalent to
// thresholding makeSlidingMasks's sliding output, with no fp32 round-trip and no
// maskFillValue dependence — convenient for integer/gating (e.g. QNN) consumers.
void makeSlidingVisibility(const uint8_t *treeVisibility, const int32_t *verifyPositionIds,
                           int currentLength, int kvLength, int slidingWindow,
                           uint8_t *outVisible);

// Walk the longest accepted root->leaf path given a per-node target argmax.
Accepted followVerified(
  const std::vector<std::unordered_map<int32_t, int32_t>> &childMaps,
  const int32_t *posterior);

// Reorder cache rows [pastLen, pastLen+tailLen) by keepIndices into
// [pastLen, pastLen+keepCount). Model-agnostic raw-pointer routine; call per layer.
void compactTail(void *cacheBase, int elemSizeBytes, int seqStrideElems, int rowElems,
                 int pastLen, int tailLen, const int32_t *keepIndices, int keepCount);

// Greedy sampling convenience (temperature 0). Ties resolve to the lowest index.
int32_t argmaxRow(const float *logits, int vocab);
void    sampleGreedy(const float *logits, int rows, int vocab, int32_t *out);
```

### Data structures (`ddtree_types.h`)

```cpp
struct DDTreeConfig {
  int   budget        = 31;  // node count beyond the root ("32 tree" == 31)
  int   depthLimit    = 0;   // draft horizon = block_size - 1
  float maskFillValue = 0;   // additive "-inf" (use a non-zero fp32/fp16 min)
};
struct DDTreeStructure {
  std::vector<int32_t> nodeTokenIds;  // [nodeCount]
  std::vector<int32_t> nodeDepths;    // [nodeCount], 1-based
  std::vector<int32_t> parents;       // [currentLength], parents[0] == -1
  std::vector<std::unordered_map<int32_t,int32_t>> childMaps; // [currentLength]
  std::vector<uint8_t> visibility;    // [currentLength*currentLength] row-major 0/1
  int nodeCount = 0, currentLength = 1; // currentLength == 1 + nodeCount
};
struct CompiledTree { int pastLength, currentLength; };
struct SlidingMasks { float *full, *sliding; bool hasSliding; };
struct Accepted { std::vector<int32_t> indices; int32_t nextToken; }; // indices[0]==0 (root)
```

---

## 5. Usage

```cpp
#include <ddtree.h>
#include <ddtree_sliding.h>
#include <ddtree_sampling.h>
#include <ddtree_compact.h>
using namespace nntrainer::ddtree;

DDTreeConfig cfg;
cfg.budget        = 31;                 // "32 tree"
cfg.depthLimit    = blockSize - 1;      // draft horizon
cfg.maskFillValue = -3.4028235e38f;     // finfo(float32).min (fp16 min for fp16)

// 1) build the candidate tree from draft logits [depthLimit, vocab].
DDTreeStructure tree = buildTree(draftLogits, cfg.depthLimit, vocab, cfg);

// 2) compile verify buffers + additive mask (caller owns the buffers).
const int cur = tree.currentLength, stride = past + cur;
std::vector<int32_t> ids(cur), pos(cur);
std::vector<float>   mask((size_t)cur * stride);
compile(rootTokenId, start, past, tree, cfg, ids.data(), pos.data(), mask.data(), stride);

// 3) (sliding models) full vs sliding masks. Use makeSlidingVisibility for a 0/1
//    consumer, or makeSlidingMasks for the fp32 additive variant.
std::vector<uint8_t> svis((size_t)cur * stride);
makeSlidingVisibility(tree.visibility.data(), pos.data(), cur, stride, slidingWindow, svis.data());

// 4) caller runs the target verify forward with ids/pos/mask -> verifyLogits,
//    then a posterior per node (temperature 0 == argmax):
std::vector<int32_t> posterior(cur);
sampleGreedy(verifyLogits, cur, vocab, posterior.data());

// 5) follow the accepted path; accepted tokens = ids[a.indices...], bonus = a.nextToken.
Accepted a = followVerified(tree.childMaps, posterior.data());

// 6) compact the KV tail to the accepted path (runtime owns the cache; call per layer).
//    compactTail(layerBase, elemSizeBytes, seqStride, rowElems, past, cur - 1,
//                a.indices.data() + 1, (int)a.indices.size() - 1);
```

---

## 6. Build & test

The core compiles into `libnntrainer` and is covered by `unittest_ddtree`
(21 tests across `DDTreeScaffold`, `DDTreeBuild`, `DDTreeCompile`,
`DDTreeSliding`, `DDTreeFollow`, `DDTreeCompact`, `DDTreeSampling`).

```bash
meson setup build -Denable-test=true
meson test -C build unittest_ddtree -v
```

To regenerate the golden vectors:

```bash
python3 test/unittest/ddtree_golden/gen_golden.py   # -> build_golden.{txt,json}
```
