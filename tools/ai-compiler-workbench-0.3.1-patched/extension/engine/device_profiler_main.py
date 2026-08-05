"""
Entry point the extension spawns for "Profile On-Device". Loads the
last completed pipeline run's state.json (from Run Pipeline) and hands
it to the Device Profiler Agent along with the user's configured
nntrainer install path.

Usage:
    python device_profiler_main.py <out_dir> <nntrainer_path>
"""
import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from agents.device_profiler import run as run_device_profiler
from agents.events import bus


def main():
    if len(sys.argv) < 3:
        bus.error("device_profiler", "usage: device_profiler_main.py <out_dir> <nntrainer_path>")
        sys.exit(1)

    out_dir, nntrainer_path = sys.argv[1], sys.argv[2]
    state_path = os.path.join(out_dir, "state.json")

    if not os.path.exists(state_path):
        bus.error(
            "device_profiler",
            "No completed pipeline run found in this workspace -- run 'Run Pipeline' first.",
        )
        sys.exit(1)

    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    if not state.get("cpp_path") or not os.path.exists(state.get("cpp_path", "")):
        bus.error(
            "device_profiler",
            "generated_model.cpp from the last run is missing -- run 'Run Pipeline' again.",
        )
        sys.exit(1)

    run_device_profiler(state, nntrainer_path)


if __name__ == "__main__":
    main()
