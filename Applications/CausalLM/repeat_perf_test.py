##
# @file repeat_perf_test.py
# @brief Repeated performance test script for CausalLM on Android devices
# @author Eunju Yang <ej.yang@samsung.com>

import subprocess
import re
import time
import statistics
import sys

def get_thermal_temp():
    try:
        cmd = ["adb", "shell", "cat", "/sys/class/thermal/thermal_zone0/temp"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return float(result.stdout.strip()) / 1000.0
    except Exception as e:
        print(f"Error reading temp: {e}")
    return 0.0

def get_process_count():
    try:
        cmd = [
            "adb",
            "shell",
            "ps -ef | grep nntr_causallm | grep -v grep | wc -l",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception as e:
        print(f"Error counting processes: {e}")
    return 0

def set_cpu_governor(governor):
    print(f"Setting CPU governor to: {governor}")
    try:
        cmd = f"for path in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo {governor} > $path; done"
        subprocess.run(["adb", "shell", cmd], check=True)
        res = subprocess.run(["adb", "shell", "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"], capture_output=True, text=True)
        print(f"Verification (cpu0): {res.stdout.strip()}")
    except Exception as e:
        print(f"Failed to set governor: {e}")

def parse_output(output):
    """Parse prefill/generation tokens, ms, TPS from causallm output."""
    prefill_match = re.search(
        r"prefill:\s+(\d+)\s+tokens,\s+(\d+)\s+ms,\s+([\d\.]+)\s+TPS", output)
    gen_match = re.search(
        r"generation:\s+(\d+)\s+tokens,\s+(\d+)\s+ms,\s+([\d\.]+)\s+TPS", output)

    result = {}
    if prefill_match:
        result["prefill_tokens"] = int(prefill_match.group(1))
        result["prefill_ms"] = int(prefill_match.group(2))
        result["prefill_tps"] = float(prefill_match.group(3))
    if gen_match:
        result["gen_tokens"] = int(gen_match.group(1))
        result["gen_ms"] = int(gen_match.group(2))
        result["gen_tps"] = float(gen_match.group(3))

    return result

def run_causallm(model_path, omp_threads=None, taskset_mask=None, input_prompt=None):
    """Run causallm binary on device and return stdout."""
    export_cmd = f"export OMP_NUM_THREADS={omp_threads} && " if omp_threads else ""
    taskset_cmd = f"taskset {taskset_mask} " if taskset_mask else ""

    prompt_arg = f' "{input_prompt}"' if input_prompt is not None else ""

    cmd = [
        "adb", "shell",
        f"cd /data/local/tmp/Quick.AI && "
        f"{export_cmd}{taskset_cmd}./run_causallm.sh {model_path}{prompt_arg}"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode

def run_benchmark(run_id, model_path, omp_threads=None, taskset_mask=None):
    """Run two passes: normal prompt for prefill+e2e, 'test' prompt for generation."""
    start_temp = get_thermal_temp()
    start_procs = get_process_count()
    print(f"[{run_id}] Starting benchmark... (Temp: {start_temp:.1f}C, Procs: {start_procs})")

    # --- Pass 1: Normal prompt -> measure prefill TPS & e2e TPS ---
    print(f"  [{run_id}] Pass 1: Measuring prefill + e2e ...")
    out1, err1, rc1 = run_causallm(model_path, omp_threads, taskset_mask)
    p1 = parse_output(out1)

    prefill_tps = p1.get("prefill_tps", 0.0)
    prefill_tokens = p1.get("prefill_tokens", 0)
    prefill_ms = p1.get("prefill_ms", 0)
    gen_tokens_p1 = p1.get("gen_tokens", 0)
    gen_ms_p1 = p1.get("gen_ms", 0)

    total_tokens = prefill_tokens + gen_tokens_p1
    total_ms = prefill_ms + gen_ms_p1
    e2e_tps = (total_tokens / total_ms * 1000) if total_ms > 0 else 0.0

    # --- Pass 2: "test" prompt -> measure generation TPS ---
    print(f"  [{run_id}] Pass 2: Measuring generation (prompt='test') ...")
    out2, err2, rc2 = run_causallm(model_path, omp_threads, taskset_mask,
                                   input_prompt="test")
    p2 = parse_output(out2)

    generation_tps = p2.get("gen_tps", 0.0)
    generation_tokens = p2.get("gen_tokens", 0)

    end_temp = get_thermal_temp()
    end_procs = get_process_count()

    print(f"  [{run_id}] Done. Temp: {end_temp:.1f}C | "
          f"Prefill: {prefill_tps:.2f} | E2E: {e2e_tps:.2f} | "
          f"Gen: {generation_tps:.2f}")

    return {
        "prefill_tps": prefill_tps,
        "prefill_tokens": prefill_tokens,
        "prefill_ms": prefill_ms,
        "e2e_tps": e2e_tps,
        "e2e_tokens": total_tokens,
        "e2e_ms": total_ms,
        "generation_tps": generation_tps,
        "generation_tokens": generation_tokens,
        "start_temp": start_temp,
        "end_temp": end_temp,
        "start_procs": start_procs,
        "end_procs": end_procs,
        "error": (err1 if rc1 != 0 else "") + (err2 if rc2 != 0 else "")
    }

def fmt_stat(values, width=0):
    """Format as 'mean ± stddev (min / max)' like llama-bench, centered to width."""
    if not values:
        s = "N/A"
    else:
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        s = f"{mean:.2f} ± {std:.2f} ({min(values):.2f} / {max(values):.2f})"
    return s.center(width) if width else s

def main():
    model_path = "./models/qwen3-0.6b-q40-arm"
    omp_threads = 4
    taskset_mask = None
    governor = None
    num_runs = 10

    args = sys.argv[1:]
    if args and not args[0].startswith("-"):
        model_path = args.pop(0)

    for arg in args:
        if arg.startswith("--omp="):
            omp_threads = int(arg.split("=")[1])
        elif arg.startswith("--taskset="):
            taskset_mask = arg.split("=")[1]
        elif arg.startswith("--governor="):
            governor = arg.split("=")[1]
        elif arg.startswith("--runs="):
            num_runs = int(arg.split("=")[1])

    results = []

    print(f"Starting {num_runs} benchmark iterations for model: {model_path}")
    print(f"  Each iteration: Pass1 (normal prompt -> prefill+e2e), "
          f"Pass2 ('test' prompt -> generation)")
    if omp_threads:
        print(f"  OMP_NUM_THREADS={omp_threads}")
    if taskset_mask:
        print(f"  taskset mask={taskset_mask}")
    if governor:
        set_cpu_governor(governor)
    print("-" * 60)

    try:
        for i in range(num_runs):
            res = run_benchmark(i + 1, model_path, omp_threads, taskset_mask)
            results.append(res)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")

    if not results:
        print("No results collected.")
        return

    prefills = [r["prefill_tps"] for r in results if r["prefill_tps"] > 0]
    e2es = [r["e2e_tps"] for r in results if r["e2e_tps"] > 0]
    gens = [r["generation_tps"] for r in results if r["generation_tps"] > 0]

    # Token counts from the first valid run (constant across runs)
    first = results[0]
    pp_tok = first.get("prefill_tokens", 0)
    gen_tok = first.get("generation_tokens", 0)
    e2e_tok = first.get("e2e_tokens", 0)

    # --- llama-bench style summary table ---
    s_prefill = fmt_stat(prefills)
    s_gen = fmt_stat(gens)
    s_e2e = fmt_stat(e2es)

    h1 = f"Prefill ({pp_tok} tok, t/s)"
    h2 = f"Generation ({gen_tok} tok, t/s)"
    h3 = f"Prefill + Generation ({e2e_tok} tok, t/s)"
    h_row = "metric"
    row_label = "mean ± sd (min/max)"

    # Each column width = max(header, value) + 2 padding
    c0 = max(len(h_row), len(row_label)) + 2
    c1 = max(len(h1), len(s_prefill)) + 2
    c2 = max(len(h2), len(s_gen)) + 2
    c3 = max(len(h3), len(s_e2e)) + 2

    sep = f"+{'-' * c0}+{'-' * c1}+{'-' * c2}+{'-' * c3}+"

    print()
    print(f"model: {model_path}, runs: {len(results)}")
    print(sep)
    print(f"|{h_row:^{c0}}|{h1:^{c1}}|{h2:^{c2}}|{h3:^{c3}}|")
    print(sep)
    print(f"|{row_label:^{c0}}"
          f"|{fmt_stat(prefills, c1)}"
          f"|{fmt_stat(gens, c2)}"
          f"|{fmt_stat(e2es, c3)}|")
    print(sep)

    # --- Per-run detail ---
    print(f"\nPer-run detail:")
    print(f"{'run':>3}  {'startT':>6}  {'endT':>6}  {'Prefill':>10}  {'Generation':>10}  {'P+G':>10}")
    for i, r in enumerate(results):
        print(f"{i+1:3d}  {r['start_temp']:6.1f}  {r['end_temp']:6.1f}  "
              f"{r['prefill_tps']:10.2f}  {r['generation_tps']:10.2f}  "
              f"{r['e2e_tps']:10.2f}")

if __name__ == "__main__":
    """
    How to use:
      python3 Applications/CausalLM/repeat_perf_test.py {model_path} {options}
      python3 Applications/CausalLM/repeat_perf_test.py ./models/qwen3-0.6b --omp=4
      python3 Applications/CausalLM/repeat_perf_test.py ./models/qwen3-0.6b --runs=5

    Benchmark measures three metrics per iteration:
      - Prefill TPS:    from normal prompt run (prefill phase)
      - E2E TPS:        (prefill_tokens + gen_tokens) / (prefill_ms + gen_ms)
      - Generation TPS: from separate 'test' prompt run (generation phase)
    """
    main()
