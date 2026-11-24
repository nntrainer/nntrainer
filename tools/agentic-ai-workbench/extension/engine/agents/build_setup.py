# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file build_setup.py
# @brief Sets up the workbench's own nntrainer build that Verify uses.
# @author prachi-t17 <prachi.t@samsung.com>

"""
The workbench's own nntrainer build (tools/agentic-ai-workbench/nb), used
by verify to build and run generated models:

  python -m agents.build_setup status   [--host docker|native|auto]
  python -m agents.build_setup setup    [--host docker|native|auto]
  python -m agents.build_setup check    [--step setup|verify|run|all]
  python -m agents.build_setup cleanup  --list | --scope old|all
  python -m agents.build_setup selftest [--libs-only]

docker: an Ubuntu 24.04 container (knowledge/docker_host.py) building a
private copy of the checkout in a Docker volume. native: MSVC on Windows,
into tools/agentic-ai-workbench/nb. auto (default, or $WORKBENCH_BUILD_HOST):
Docker when it is running.

check: the system check (knowledge/preflight.py) -- is this computer ready
for the Docker build? It also runs by itself before a Docker setup; a
failure that setup needs stops it with the fix.

cleanup: what the workbench keeps in Docker (--list, event docker_usage),
and removing it (event docker_cleanup): scope old = old build / run images
only, the build stays; scope all = also the build container and volume.

selftest: the Docker self-test (knowledge/selftest.py) -- full build plus
upstream's tiny Qwen3 reference test, no Claude, no download (event selftest).

Prints JSON events like the pipeline. Setup installs only into the
workbench folder (meson + ninja in .buildenv); what would write elsewhere
(git submodules, Rust, Build Tools) is reported as missing with the command
to run, not done.
"""
import argparse
import sys

from knowledge import docker_host
from knowledge import preflight
from knowledge import selftest
from knowledge import verify

from .events import bus


def emit_preflight(result: dict) -> None:
    """Send a system check result to the webview."""
    bus.emit({"event": "preflight", **result})


def cleanup(scope: str, list_only: bool) -> int:
    """List (event docker_usage) or remove (event docker_cleanup) what the
    workbench keeps in Docker."""
    root = verify.resolve_root(None)
    running, detail = docker_host.daemon_status()
    if not running:
        payload = {"ok": False, "error": detail, "items": [], "total": 0, "old_total": 0}
        bus.emit({"event": "docker_usage" if list_only else "docker_cleanup", **payload})
        bus.log(detail, "warn")
        return 1
    if list_only:
        usage = docker_host.usage(root)
        bus.emit({"event": "docker_usage", **usage})
        if not usage["ok"]:
            bus.log("Could not read Docker's disk use: " + usage.get("error", ""), "warn")
            return 1
        bus.log(f"The workbench keeps {docker_host.fmt_size(usage['total'])} in Docker; "
                f"{docker_host.fmt_size(usage['old_total'])} of it is old images:")
        for i in usage["items"]:
            bus.log(
                f"  {i['kind'].replace('_', ' '):<11} {i['name']:<48} "
                f"{docker_host.fmt_size(i['size']):>9}"
                + ("  (old)" if i["old"] else "")
            )
        return 0
    res = docker_host.cleanup(root, scope, log=bus.log)
    bus.emit({"event": "docker_cleanup", **res})
    bus.log(
        f"Clean Up Docker ({scope}): removed {len(res['removed'])} item(s), freed about "
        f"{docker_host.fmt_size(res['freed'])}"
        + (f"; {len(res['failed'])} could not be removed" if res["failed"] else "")
        + ("." if scope == "old" else ". The next Docker step sets the build up again."),
        "info" if res["ok"] else "warn",
    )
    return 0 if res["ok"] else 1


def main(argv=None) -> int:
    """Command line of the Docker build actions: status, setup, check, cleanup, selftest."""
    parser = argparse.ArgumentParser(prog="python -m agents.build_setup")
    parser.add_argument("action", choices=["status", "setup", "check", "cleanup", "selftest"])
    parser.add_argument("--host", default=None, choices=["docker", "native", "auto"])
    parser.add_argument("--step", default="all", choices=["setup", "verify", "run", "all"])
    parser.add_argument("--scope", default="old", choices=["old", "all"])
    parser.add_argument(
        "--list", action="store_true", help="cleanup: only list what the workbench keeps in Docker"
    )
    parser.add_argument(
        "--libs-only", action="store_true", help="selftest: build only the libraries the test links"
    )
    args = parser.parse_args(argv)

    if args.action == "check":
        result = preflight.gate(args.step, bus.log, emit_preflight)
        return 0 if result["ok"] else 1

    if args.action == "cleanup":
        return cleanup(args.scope, args.list)

    if args.action == "selftest":
        result = selftest.run_selftest(log=bus.log, full_build=not args.libs_only)
        bus.emit({"event": "selftest", **result})
        emit_preflight(
            preflight.run_checks(step="all", network=False)
        )  # refresh the Docker setup view
        return 0 if result["ok"] else 1

    if args.action == "setup" and verify.select_executor(host=args.host).kind == "docker":
        result = preflight.gate("setup", bus.log, emit_preflight)
        if not result["ok"]:
            bus.emit(
                {
                    "event": "build_status",
                    "ok": False,
                    "status": "preflight_failed",
                    "host": "docker",
                    "why": result["summary"],
                    "missing": [],
                    "docker": docker_host.daemon_status()[0],
                }
            )
            return 1

    if args.action == "status":
        status = verify.build_status(host=args.host)
    else:
        status = verify.setup_build(log=bus.log, host=args.host)
    status["docker"] = docker_host.daemon_status()[0]
    bus.emit({"event": "build_status", **{k: v for k, v in status.items() if k != "libs"}})
    if status.get("ok") and not status.get("missing"):
        bus.log(
            f"nntrainer build ready ({status.get('host_label', '')}): {status.get('build_dir')}"
        )
        return 0
    for missing_name in status.get("missing") or []:
        bus.log(
            (
                ("Missing: " + missing_name)
                if isinstance(missing_name, str) and " " in missing_name
                else f"Not built yet: {missing_name}"
            ),
            "warn",
        )
    if status.get("why"):
        bus.log(status["why"], "warn")
    if status.get("log"):
        bus.log(status["log"][-3000:], "error")
    for changed_path in status.get("changed_outside") or []:
        bus.log(f"The nntrainer build changed a file in the checkout: {changed_path}", "warn")
    return 1 if args.action == "setup" else 0


if __name__ == "__main__":
    sys.exit(main())
