"""
Entry point the VS Code extension spawns as a child process for
"Run Pipeline". Streams one JSON event per line on stdout (see
agents/events.py) so the webview gets live updates; nothing is
buffered until the end.

Usage:
    python orchestrator_main.py <hf_model_name_or_path> --out DIR [--api-key KEY]
"""
import argparse
import os
import sys

# Keep third-party progress bars (huggingface_hub / tqdm) off stdout: this
# process's stdout is the JSON event channel the extension parses line-by-line,
# and a progress bar interleaving mid-line would corrupt an event. Set before
# importing transformers/hf so it takes effect. (M2)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from agents.events import bus
from agents.orchestrator import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="HF model name or local path")
    parser.add_argument("--out", default="./nntrainer_out")
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--causallm-root", default=None,
        help="Path to a real CausalLM integration project (aiCompilerWorkbench.causallmProjectRoot)",
    )
    parser.add_argument(
        "--install-generated", action="store_true",
        help="Copy generated CausalLM .h/.cpp into --causallm-root (aiCompilerWorkbench.installGeneratedFiles)",
    )
    parser.add_argument("--generated-header-dir", default="include/generated")
    parser.add_argument("--generated-source-dir", default="src/generated")
    args = parser.parse_args()

    # Prefer the API key from the environment (how the extension passes it now,
    # so it never appears in argv / process listings); fall back to the flag.
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")

    try:
        from transformers import AutoModel  # noqa: F401
    except ImportError:
        bus.error(
            "environment",
            "transformers/torch not found in this Python environment. Run "
            "'AI Compiler Workbench: Check / Install Python Dependencies', or "
            "point aiCompilerWorkbench.pythonPath at an environment that has them.",
        )
        sys.exit(1)

    run_pipeline(
        args.model, args.out, api_key,
        causallm_project_root=args.causallm_root,
        install_generated_files=args.install_generated,
        generated_header_directory=args.generated_header_dir,
        generated_source_directory=args.generated_source_dir,
    )


if __name__ == "__main__":
    main()
