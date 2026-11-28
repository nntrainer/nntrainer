# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file orchestrator_main.py
# @brief Entry point the VS Code extension spawns as a child process for "Run Pipeline".
# @author prachi-t17 <prachi.t@samsung.com>

"""
Entry point the VS Code extension spawns as a child process for
"Run Pipeline". Streams one JSON event per line on stdout (see
agents/events.py) so the webview gets live updates; nothing is
buffered until the end.

Usage:
    python orchestrator_main.py <hf_model_name_or_path> --out DIR

The optional Anthropic API key comes only from $ANTHROPIC_API_KEY (a command-line
argument would be visible to other users in the process list).
"""
import argparse
import os
import sys

# Keep third-party progress bars (huggingface_hub / tqdm) off stdout: this
# process's stdout is the JSON event channel the extension parses line-by-line,
# and a progress bar interleaving mid-line would corrupt an event. Set before
# importing transformers/hf so it takes effect.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# A model given as a local folder never touches the network: set before
# transformers / huggingface_hub are imported (they read these at import).
LOCAL_MODEL = len(sys.argv) > 1 and os.path.isdir(sys.argv[1])
if LOCAL_MODEL:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

# These imports must come after the environment above is set.
# pylint: disable=wrong-import-position
from agents.events import bus
from agents.orchestrator import run_check_only, run_pipeline
# pylint: enable=wrong-import-position


def main():
    """Entry point the extension runs for Run Pipeline (and the check-only run)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="HF model name or local path")
    parser.add_argument("--out", default="./nntrainer_out")
    parser.add_argument(
        "--causallm-root",
        default=None,
        help=(
            "Path to a real CausalLM integration project (aiCompilerWorkbench.causallmProjectRoot)"
        ),
    )
    parser.add_argument(
        "--install-generated",
        action="store_true",
        help=(
            "Copy generated CausalLM .h/.cpp into --causallm-root"
            " (aiCompilerWorkbench.installGeneratedFiles)"
        ),
    )
    parser.add_argument("--generated-header-dir", default="include/generated")
    parser.add_argument("--generated-source-dir", default="src/generated")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="checklist + plan + generated code from config and tensor headers only "
        "(no download, model not loaded into memory)",
    )
    args = parser.parse_args()
    if LOCAL_MODEL:
        bus.log(f"Local model folder: {args.model} -- running offline, nothing is downloaded")

    # The optional API key: from the environment only (the extension passes it that way).
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    try:
        # pylint: disable-next=import-outside-toplevel,unused-import  # optional package
        from transformers import AutoModel  # noqa: F401
    except ImportError:
        bus.error(
            "environment",
            "transformers/torch not found in this Python environment. Run "
            "'AI Compiler Workbench: Set Up / Check Python Environment', or "
            "point aiCompilerWorkbench.pythonPath at an environment that has them.",
        )
        sys.exit(1)

    if args.check_only:
        run_check_only(args.model, args.out, api_key)
        return
    run_pipeline(
        args.model, args.out, api_key,
        causallm_project_root=args.causallm_root,
        install_generated_files=args.install_generated,
        generated_header_directory=args.generated_header_dir,
        generated_source_directory=args.generated_source_dir,
    )


if __name__ == "__main__":
    main()
