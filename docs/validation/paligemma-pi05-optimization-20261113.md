# PaliGemma / π0.5 optimization — 2026-11-13

The optimized NNTrainer profiles are faster than the measured PyTorch eager baselines in all four model/device combinations. This result applies to the requests, precision and profiles below; it does not establish optimality for every workload or compare against `torch.compile` / TensorRT.

The earlier [validation](paligemma-pi05-20261113.md) and [implementation audit](paligemma-pi05-implementation-audit-20261113.md) remain historical records. The implementation changes below follow that audit. The companion JSON records source and binary hashes from the validated checkout before the support commits were created.

## Final measurements

Ubuntu 22.04, i9-14900, RTX 3050 OEM 8 GiB, CUDA 13.2. Each cell uses three fresh processes with six complete requests each. The warm statistic is the median of all 15 requests after each process’s first request. Framework order alternates between repetitions. CPU and GPU measurements run separately, with no concurrent compilation or model benchmarks. Relevant NNTrainer and Hugging Face checkpoint files are read into the OS file cache before each model group. All values below are measured, including initialization when stated.

| Model / engine | NNTrainer warm | PyTorch warm | PT / NNTR | NNTR peak RAM | PT peak RAM | NNTR peak VRAM | PT peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| paligemma / cpu | 2915.00 ms | 3270.94 ms | 1.122× | 11.34 GiB | 11.75 GiB | 0.00 GiB | 0.00 GiB |
| paligemma / cuda | 421.15 ms | 434.43 ms | 1.032× | 2.68 GiB | 12.29 GiB | 6.55 GiB | 6.39 GiB |
| pi05 / cpu | 2179.97 ms | 2234.75 ms | 1.025× | 13.55 GiB | 27.47 GiB | 0.00 GiB | 0.00 GiB |
| pi05 / cuda | 304.13 ms | 379.11 ms | 1.247× | 4.35 GiB | 27.74 GiB | 6.72 GiB | 7.02 GiB |

PyTorch environments: paligemma-cpu: Torch 2.13.0+cu132, Transformers 5.14.1; paligemma-cuda: Torch 2.13.0+cu132, Transformers 5.14.1; pi05-cpu: Torch 2.15.0+cpu, Transformers 4.53.2; pi05-cuda: Torch 2.13.0+cu132, Transformers 4.53.2.

RAM is process RSS including loading. VRAM is the maximum sampled whole-device used memory minus that process’s idle baseline, including managed memory; it is not the smaller NVML per-process dedicated counter. Sampling is approximately 100 ms and can miss shorter peaks. RAM and VRAM overlap for UVM and must not be added. π0.5 PyTorch’s approximately 27.7 GiB host peak comes from loading the FP32 checkpoint; it is not steady inference RSS. PaliGemma NNTrainer still uses slightly more VRAM than PyTorch.

| Model / engine | NNTR load | PT load | NNTR first request | PT first request | NNTR process warm medians | PT process warm medians |
|---|---:|---:|---:|---:|---|---|
| paligemma / cpu | 1.619 s | 1.226 s | 2.989 s | 3.386 s | 2.8966, 2.9968, 3.0046 s | 3.1756, 3.1816, 3.3393 s |
| paligemma / cuda | 2.585 s | 2.950 s | 0.556 s | 0.950 s | 0.4210, 0.4211, 0.4215 s | 0.4346, 0.4344, 0.4342 s |
| pi05 / cpu | 3.000 s | 21.284 s | 2.096 s | 2.288 s | 2.1548, 2.0972, 2.2790 s | 2.2372, 2.2476, 2.2163 s |
| pi05 / cuda | 2.817 s | 25.285 s | 0.437 s | 0.576 s | 0.3048, 0.3031, 0.3039 s | 0.3802, 0.3785, 0.3791 s |

Load and first-request columns are medians of three processes. The load phase includes weight placement and input-independent preparation. No observation is run during loading. First-request costs such as CUDA library/kernel initialization remain in the first-request measurement. Small speed margins, especially PaliGemma CUDA, should be rechecked on deployment hardware. The π0.5 CPU gain is only 2.5% in the pooled median: one fresh process had a 2.2790 s NNTrainer median versus 2.2163 s for PyTorch. This supports the aggregate result, not a claim that every run is faster.

## Comparable work and precision

- Batch size one. PaliGemma uses one 224×224 image, a 272-row prefix and the complete 11-token greedy continuation including EOS. π0.5 uses one image, a 261-row prefix, a 50×32 action chunk and ten Euler steps.
- CPU computes in FP32. CUDA uses FP32 vision, FP16 language weights/activations and FP32 expert/conditioning/exported prefix K/V. There is no quantization or reduced layer count. π0.5 RoPE frequencies originate from a true FP32 upstream model.
- Timed inference includes vision, language/backbone and complete generation or action sampling. Reference file reads, image preprocessing, tokenization and numerical comparisons are outside both timers. PyTorch preloads its image tensor on CUDA; NNTrainer’s host-input staging remains inside its inference boundary. PyTorch’s final CPU validation/export is outside its timer, while NNTrainer’s API exports are included.
- PyTorch uses `eval()`, `inference_mode()`, highest FP32 matmul precision and SDPA where available. Upstream π0.5 prefix/expert attention remains eager. This is the same public benchmark scope as the preceding report; compiled PyTorch is not measured.

## Changes that produced the result

1. Native `[output,input]` checkpoint layout for the backbone, vision and PaliGemma head avoids runtime transposition and a second full weight copy. The private CPU expert is transposed once during loading. CPU fuses Q/K/V and gate/up projections. The PaliGemma CPU profile reuses the tied FP32 embedding table as its head, removing a duplicate approximately 1.96 GiB allocation. Existing and quantized layouts keep their paths.
2. The optional MKL backend and shared Intel OpenMP worker pool let GEMMs, LayerNorm, bias and GeGLU reuse the same eight physical P cores. Small attention GEMMs avoid nested BLAS teams. CPU expert attention uses BLAS plus vector softmax. Its per-step workspaces are reused across layers; adaptive normalization and gated residuals share the row worker pool, without horizon-sized gate copies.
3. CUDA FP32 residual/bias operations remain on device. Vision/prefix/expert attention uses batched QK/PV GEMMs and device softmax, and prefix RoPE remains FP32 on device. Two softmax kernels needed an explicit barrier after reading the shared maximum before reusing its scratch for the sum; repeated full-output checks exposed this race.
4. FP16 PaliGemma stages its embedding input into managed activations before the first norm/residual. Decode graphs use a device position/key-count and the physical cache capacity, so replay includes newly appended keys and masks unused rows. Graph replay is enabled only through the certified FP16 embedding-input feed nodes.
5. Dense managed CUDA weights and private expert weights are placed during load. π0.5 input-independent time-conditioning projections are prepared once for the chosen step count and invalidated on weight reload or step-count change. Observations, vision, prefix inference and all expert steps still run on every request.

## Numerical and regression evidence

KV inputs are bound to graph inputs by name, replacing the earlier physical-slab/lexicographic workaround. Deferred CUDA logits are materialized before reference export; repeated requests cannot reuse an unwritten host row. Managed-memory copies complete before a CPU consumer can read the destination. This fixes an asynchronous FP32 connected-model race exposed by repeated tiny tests; graph-captured copies retain stream ordering without a host drain.

- Full PaliGemma vision outputs (524,288 values), last-row logits (257,216 values), and every generated token pass. CPU logits retain `atol=0.01, rtol=0.001`; CUDA FP16 logits retain `atol=0.1, rtol=0.001`. Vision retains `atol=0.03, rtol=0.002`; PIL preprocessing remains checked to `2e-7`.
- All π0.5 per-step velocities (16,000) and actions (1,600) pass `atol=0.005` and `0.01`, respectively. Non-finite values fail. Tolerances were not relaxed to obtain speed gains.
- Each model passes 50 CUDA requests with alternating images/prompts, and π0.5 alternating initial noise, on both synchronous and asynchronous profiles. PaliGemma also passes 50 asynchronous native-layout requests without MLP fusion. Each CPU profile passes ten alternating requests. The alternate PaliGemma request has a different prefix, a causal suffix and eight generated tokens, exercising graph/cache reset and token-budget changes.
- CPU model suite: 102 pass, one existing Gemma4 QINT4 skip, zero failures. MKL/shared-pool checks: three model differential tests, seven thread-manager tests and 17 compute-dispatch tests pass. CUDA VLA: four tests pass in ten consecutive repetitions, including fixed-capacity graph replay with a newly valid cache row and managed-copy followed immediately by a CPU consumer. The π0.5 tiny test also varies the number of Euler steps and reloads weights.
- All current files represented by the fixture tarball match its contents. `clang-format-14`, Python compilation and `git diff --check` pass.
- The unrelated Gemma4 CUDA FP32 failure documented in the earlier report remains an existing limitation; this is not a claim of a passing full CUDA model suite. Android/Windows/Tizen/Yocto builds and closed-loop robot behavior are not validated here.

## Reproduction

The measured packages and complete per-process commands/environments are stored under `.validation/paligemma-pi05-20261113/optimization/final/`. The companion [JSON report](paligemma-pi05-optimization-20261113.json) embeds all 24 result records, per-request timings, numerical output, selected configurations and source/binary SHA-256 hashes. Each `*-launcher.json` additionally preserves the CPU affinity and checkpoint cache-warming manifest. The JSON report also embeds the machine-local `run_matrix.py` source as `matrix_launcher`; model weights and local environments are not part of the repository.

Build with the optional native MKL provider (the default remains OpenBLAS):

```sh
meson setup build_cuda --reconfigure -Dblas-provider=mkl \
  -Dmkl-libdir="$PWD/.validation/paligemma-pi05-20261113/mkl-runtime/lib"
ninja -C build_cuda Applications/CausalLM/nntr_paligemma_e2e \
  Applications/CausalLM/nntr_pi05_e2e
```

CPU profile: `taskset -c 0,2,4,6,8,10,12,14`, eight operator/BLAS threads, `MKL_THREADING_LAYER=INTEL`, `MKL_DYNAMIC=FALSE`, `NNTR_CPU_SHARED_OMP=1`, `NNTR_CPU_AFFINITY=0`, `OMP_WAIT_POLICY=ACTIVE`, `KMP_AFFINITY=disabled`, `KMP_BLOCKTIME=20`, `GOMP_SPINCOUNT=100000`. PyTorch uses its GNU runtime with the same eight-core affinity, thread count and active wait policy. `NNTR_CPU_HUGE_PAGES=1` and `THP_MEM_ALLOC_ENABLE=1` provide allocation hints for the respective runtimes without changing system THP policy. The shared-pool optimization is optional on Linux x86 with Intel OpenMP; unsupported builds retain their normal worker pool.

CUDA profile: four host threads, `MKL_THREADING_LAYER=GNU`, `NNTR_CUDA_ASYNC=1`, `NNTR_CUDA_DECODE_GRAPH=1`, `NNTR_CUDA_PREFETCH_WEIGHTS=1`, `NNTR_CUBLAS_PEDANTIC=1`, and `PYTORCH_ALLOC_CONF=expandable_segments:True`. The VLA helper defaults to synchronous execution; the measured asynchronous profile is selected explicitly. Both settings passed the alternating stress checks. Other VLA defaults retain host-readable managed activations and enable residual/RoPE/device prefill paths.

The language and vision converters accept `--weight-transposed`; set `weight_transposed=true` in the corresponding configurations. CPU language profiles also set `fuse_mlp=true` (and default to `fuse_qkv=true`). For the tied PaliGemma CPU package, use `--cpu-external-head` and `cpu_external_head=true`. The final PaliGemma CUDA profile keeps `fuse_mlp=false`; π0.5 CUDA uses `fuse_mlp=true`. Export CPU FP32 weights separately from CUDA FP16 language weights. All selected configuration contents are in the JSON report.

On this machine, the complete matrix can be repeated sequentially with:

```sh
python3 .validation/paligemma-pi05-20261113/optimization/final/run_matrix.py cpu
python3 .validation/paligemma-pi05-20261113/optimization/final/run_matrix.py cuda
```

The CUDA monitor rejects competing compute processes both before and during a run. Interrupted measurements are excluded. The user-requested stop of competing Quick.KV CUDA benchmarks includes their `quickkv-lut-20261113.service` runner; it is left stopped, without restarting that workload.

CPU model implementation and execution infrastructure both required changes. These measurements justify retaining those changes; they do not prove that only kernel tuning remains. Further priorities are PaliGemma decode bandwidth and device-memory overhead, followed by profiling on the actual deployment hardware.
