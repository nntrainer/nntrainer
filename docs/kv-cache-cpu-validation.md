# CPU KV cache validation

This describes the supported CPU integration and reproducible commands. Local
investigation diaries and machine-specific artifacts are separate from these
instructions. Model weights, binaries, logits and snapshots are generated
artifacts and must not be committed.

## Supported integration

The managed graph adapter currently supports synchronous, batch-one, linear,
token-major CausalLM KV storage. Model declarations identify blocks that do not
produce KV (for example convolution blocks in a hybrid model). MHA validates
its actual Tensor slices against core write views before producing K/V, marks
the owner ready after both writes, and validates read views before attention.
Commit refuses missing producers. Graph exceptions abort the unpublished step.
This is KV publication, not atomic rollback of the complete model. Hybrid
convolution state and other mutable graph tensors are outside the lease;
after a mid-graph failure, recreate/reset the complete model before continuing
such a session. Changing only the KV position does not restore that state.

The bridge does not protect arbitrary escaped raw Tensor pointers, does not
provide asynchronous completion, and does not make non-KV encoder models use
KV leases. CPU CausalLM ring storage is explicitly rejected; lower-level ring
view tests are not evidence of CPU ring attention support.

Runtime selection is resolved against the registered engines and captured by
each owning object. With no GPU/CUDA context, an unset engine selects CPU.
New objects capture changed environment settings; existing layer execution
uses its saved settings.

| Option | Default | Purpose |
|---|---|---|
| `NNTR_ENGINE=cpu` | Registry-dependent | Explicit CPU execution |
| `NNTR_CPU_ATTENTION_FP32=1` | Off on Android | Preserve FP32 Q/K/V/O with half KV |
| `NNTR_CPU_KV_TILED=1` | Off | ARM bounded K/V packing experiment |
| `NNTR_CPU_KV_LEGACY=1` | Off | Compatibility host storage without core leases |
| `NNTR_KV_CPU_TRACE=1` | Off | Correctness audit only; disable for timing |

ARM packing defaults to the previous full-head implementation until an
alternative meets the performance criterion. Bounded packing reduces K/V
scratch space but may increase accumulated packing work. Necessary operation
scratch and hardware loads are distinct from persistent KV replication.

## Linux setup and checks

From the repository root, install the normal documented build dependencies,
Rust and Python dependencies for model conversion. Use a separate build tree:

```bash
tools/kv_cache/build_cpu.sh build-cpu
NNTR_ENGINE=cpu OPENBLAS_NUM_THREADS=1 NNTR_NUM_THREADS=4 \
  meson test -C build-cpu --print-errorlogs \
  unittest_kv_cache_core unittest_kv_cache_manager \
  unittest_kv_cache_manager_legacy unittest_causallm_runtime \
  unittest_nntrainer_graph unittest_causallm_models
python3 tools/kv_cache/prepare_qwen3_cpu.py \
  --build build-cpu --output build-cpu/models --download --quantize
```

The preparation tool pins the official Qwen3-0.6B revision and records source
and converted weight hashes. Linux packed quantized weights cannot be reused
as Android ARM weights: run the quantizer with `--isa ARM` for that target.
Weight quantization is separate from KV encoding.

## Performance measurements

Use the performance-specific observer; the correctness observer dumps logits
and is unsuitable for controlled timings.

```bash
python3 tools/kv_cache/freeze_cpu_runtime.py --build build-cpu \
  --output build-cpu/benchmark/runtime-candidate
python3 tools/kv_cache/build_qwen_cpu_probe.py --build build-cpu \
  --source tools/kv_cache/qwen_cpu_benchmark.cpp \
  --output build-cpu/benchmark/qwen_cpu_benchmark
python3 tools/kv_cache/benchmark_qwen3_cpu.py --target linux \
  --probe build-cpu/benchmark/qwen_cpu_benchmark \
  --runtime-dir build-cpu/benchmark/runtime-candidate \
  --model "$KV_MODEL_DIR" --output build-cpu/benchmark/results \
  --profile legacy-packed --label candidate \
  --context 1024 --context 4096 --decode-tokens 32 \
  --warmup 1 --repetitions 5 --threads 4
```

Set `KV_MODEL_DIR` to the prepared model directory containing its configuration
and weight files. The benchmark uses fixed token IDs to prevent early EOS;
it measures workload cost, not language quality. Load/packing, initialization,
warmup, first-token timing and decode timing are recorded separately.
The runtime freezer reads Meson targets instead of recursively searching the
build tree, which may contain older validation snapshots. It records source
paths/hashes and rejects missing targets, duplicate names and existing output.

On Android, use an NDK-built runner, matching runtime libraries, ARM-packed
weights and `--target android --serial "$ANDROID_SERIAL" --runtime-dir
"$ANDROID_RUNTIME_DIR"`. The serial is required. The harness writes only its
dedicated device subtree and records model/runtime hashes and thermal state.
The existing Android build script accepts `ANDROID_NDK` and explicit Meson
backend options; configure CPU-only for CPU comparisons.

Compare identical weights, token workload, capacities, thread settings and
arithmetic profiles. Preserve the baseline runtime. If class layout changed,
compile its observer against headers from that baseline revision using the
builder's `--header-tree` options. Do not mix current model headers with an
older shared library. Repeat paired measurements if thermal/frequency drift
or spread makes a 5% median regression decision inconclusive.

## Correctness after a change

Keep timing and correctness runs separate. The Qwen correctness harnesses
record token IDs, finite logits, owner addresses and same-device snapshot
continuations. Explicitly select tiled packing for a tiled trace; default
packed runs are functional compatibility checks and cannot claim zero
history copying. Keep the supported dtype/backend scope with every result.
Independent graph-mode, Gemma head-property and converter-order bug fixes
are maintained as separate review topics.
