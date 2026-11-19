#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a VLA differential driver and record Ubuntu CPU/CUDA time and memory.

Example:
  python tools/validate_vla_ports.py --out-dir /tmp/vla-validation \
    paligemma-cuda cuda build_cuda/Applications/CausalLM/nntr_paligemma_e2e \
    /path/to/text /path/to/vision /path/to/pytorch-reference

CUDA needs the toolkit's lib64 on LD_LIBRARY_PATH, including NVRTC builtins.
GPU device memory includes managed allocations; the per-process NVML figure
alone can omit managed memory. Both figures and the idle baseline are saved.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time


def gpu_query(fields, apps=False):
    option = "--query-compute-apps=" if apps else "--query-gpu="
    result = subprocess.run(
        ["nvidia-smi", option + fields, "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=3, check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return [row.split(",") for row in result.stdout.strip().splitlines() if row]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--operator-threads", type=int,
                        help="NNTrainer operator pool size; defaults to --threads")
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--cuda-async", choices=["0", "1"], default="0")
    parser.add_argument("name")
    parser.add_argument("engine", choices=["cpu", "cuda"])
    parser.add_argument("binary")
    parser.add_argument("arguments", nargs="+")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(NNTR_ENGINE=args.engine, NNTR_CUDA_DEV_ACT="0", NNTR_CUDA_ASYNC=args.cuda_async,
               OPENBLAS_NUM_THREADS=str(args.threads), OMP_NUM_THREADS=str(args.threads),
               NNTR_NUM_THREADS=str(args.operator_threads or args.threads))
    if args.engine == "cuda" and not args.allow_busy_gpu:
        existing = gpu_query("pid,process_name,used_memory", apps=True)
        if existing:
            raise RuntimeError(f"GPU is occupied; benchmark would be invalid: {existing}")
    gpu_baseline = int(gpu_query("memory.used")[0][0]) if args.engine == "cuda" else 0
    gpu_peak = gpu_baseline
    dedicated_peak = rss_peak = anon_peak = 0
    affinity_sample = {}
    huge_pages_peak = 0
    next_affinity_sample = 0
    command = [args.binary, *args.arguments]
    log_path = args.out_dir / (args.name + ".log")
    start = time.monotonic()
    with log_path.open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                try:
                    status = Path(f"/proc/{process.pid}/status").read_text()
                    for key in ["VmRSS", "VmHWM"]:
                        match = re.search(r"^" + key + r":\s+(\d+)", status, re.M)
                        if match:
                            rss_peak = max(rss_peak, int(match[1]))
                    match = re.search(r"^RssAnon:\s+(\d+)", status, re.M)
                    if match:
                        anon_peak = max(anon_peak, int(match[1]))
                    if time.monotonic() >= next_affinity_sample:
                        try:
                            rollup = Path(f"/proc/{process.pid}/smaps_rollup").read_text()
                            huge = re.search(r"^AnonHugePages:\s+(\d+)", rollup, re.M)
                            if huge:
                                huge_pages_peak = max(huge_pages_peak, int(huge[1]))
                        except (FileNotFoundError, PermissionError, ProcessLookupError):
                            pass
                        masks = {}
                        for task in Path(f"/proc/{process.pid}/task").glob("[0-9]*"):
                            try:
                                mask = ",".join(map(str, sorted(os.sched_getaffinity(int(task.name)))))
                                masks[mask] = masks.get(mask, 0)+1
                            except ProcessLookupError:
                                pass
                        if masks:
                            affinity_sample = masks
                        next_affinity_sample = time.monotonic()+1
                    if args.engine == "cuda":
                        gpu_peak = max(gpu_peak, int(gpu_query("memory.used")[0][0]))
                        for pid, used in gpu_query("pid,used_memory", apps=True):
                            if pid.strip() == str(process.pid):
                                dedicated_peak = max(dedicated_peak, int(used))
                            elif not args.allow_busy_gpu:
                                raise RuntimeError(
                                    f"Another CUDA process started during measurement: {pid.strip()}")
                except (FileNotFoundError, ProcessLookupError):
                    pass
                time.sleep(0.1)
        except BaseException:
            process.terminate()
            process.wait()
            raise
    result = dict(
        name=args.name, engine=args.engine, exit_code=process.returncode,
        wall_s=time.monotonic()-start, peak_rss_mib=rss_peak/1024,
        peak_anon_mib=anon_peak/1024, peak_gpu_process_dedicated_mib=dedicated_peak,
        gpu_idle_baseline_mib=gpu_baseline, peak_gpu_device_used_mib=gpu_peak,
        peak_gpu_device_delta_mib=max(0, gpu_peak-gpu_baseline), command=command,
        threads=args.threads,
        operator_threads=args.operator_threads or args.threads,
        last_thread_affinity_counts=affinity_sample,
        peak_anon_huge_pages_mib=huge_pages_peak/1024,
        runtime_environment={key: env[key] for key in [
            "NNTR_ENGINE", "NNTR_CUDA_DEV_ACT", "NNTR_CUDA_ASYNC",
            "NNTR_CUDA_KV_UVM", "NNTR_NUM_THREADS", "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "LD_LIBRARY_PATH", "PYTORCH_ALLOC_CONF",
            "NNTR_CPU_FC_TRANSPOSE", "NNTR_BENCH_REPEATS", "NNTR_CUDA_ELTWISE",
            "MKL_NUM_THREADS", "MKL_DYNAMIC", "MKL_THREADING_LAYER", "LD_PRELOAD",
            "NNTR_CPU_SPIN_COUNT", "NNTR_CUBLAS_PEDANTIC", "NNTR_MKL_VECTOR_THREADS",
            "NNTR_CPU_HUGE_PAGES", "THP_MEM_ALLOC_ENABLE", "NNTR_CPU_SHARED_OMP",
            "NNTR_CPU_AFFINITY", "OMP_WAIT_POLICY", "GOMP_SPINCOUNT",
            "NNTR_CUDA_ROPE", "NNTR_CUDA_VCOPY_PREFILL", "NNTR_CUDA_DECODE_GRAPH",
            "NNTR_CUDA_PREFETCH_WEIGHTS",
            "NNTR_VALIDATE_ALTERNATE_REFERENCE", "KMP_AFFINITY", "KMP_BLOCKTIME",
        ] if key in env},
    )
    text = log_path.read_text()
    result["requests"] = [
        {key: float(value) for key, value in re.findall(r"(\w+)=([\d.e+-]+)", row)}
        for row in re.findall(r"\[timing\](.*)", text)
    ]
    if result["requests"]:
        result.update(result["requests"][0])
    for label, prefix in [("prefill", "prefill"), ("generation", "decode")]:
        match = re.search(label+r": (\d+) tokens, (\d+) ms", text)
        if match:
            result[prefix+"_tokens"] = int(match[1])
            result[prefix+"_ms"] = int(match[2])
            result[prefix+"_tps"] = int(match[1])*1000/max(1,int(match[2]))
    result["numeric"] = re.findall(r"\[numeric\] (.*)", text)
    (args.out_dir / (args.name + ".json")).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
