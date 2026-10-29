#!/usr/bin/env python3
"""Build a logits probe against a root or standalone CausalLM Meson CPU build.

Use --header-tree for an older complete project ABI, or --header-revision for
the legacy three-header override. Rebuild when the linked CausalLM ABI changes.
Existing libraries are linked without a Ninja rebuild, including Android builds.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path("build-kv-cpu"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--header-revision")
    parser.add_argument("--source", type=Path,
                        help="Alternate standalone runner source (default: logits probe)")
    parser.add_argument("--header-tree", type=Path,
                        help="Matching project source/header checkout for a preserved ABI")
    parser.add_argument("--header-tree-revision",
                        help="Provenance label for --header-tree, e.g. git archive revision")
    args = parser.parse_args()
    if args.header_revision and args.header_tree:
        parser.error("Choose --header-revision or --header-tree")
    root = Path(__file__).resolve().parents[2]
    source = (args.source or root / "tools/kv_cache/qwen_cpu_probe.cpp").resolve()
    build = args.build.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    commands = json.loads((build / "compile_commands.json").read_text())
    main_source = root / "Applications/CausalLM/main.cpp"
    entry = next(c for c in commands
                 if (Path(c["directory"]) / c["file"]).resolve() == main_source)
    compile_dir = Path(entry["directory"])
    argv = entry.get("arguments") or shlex.split(entry["command"])
    main_object = argv[argv.index("-o") + 1]
    # Meson object paths contain <target>.p in both build layouts. Derive the
    # target from that path rather than hardcoding Applications/CausalLM/.
    object_parts = Path(main_object).parts
    private_index = next(i for i, p in enumerate(object_parts) if p.endswith(".p"))
    target = str(Path(*object_parts[:private_index],
                      object_parts[private_index][:-2]))
    link = subprocess.check_output([
        "ninja", "-t", "commands", target], cwd=build, text=True).splitlines()[-1]
    link_args = shlex.split(link)
    if link_args.count(main_object) != 1:
        raise RuntimeError(f"Expected exactly one main object in {target}'s link command")
    machine_file = build / "meson-info/intro-machines.json"
    is_android = (json.loads(machine_file.read_text())["host"]["system"] == "android"
                  if machine_file.exists() else "android" in argv[0])
    compile_args = []
    i = 0
    while i < len(argv):
        if argv[i] in ("-o", "-MF", "-MQ", "-c"):
            i += 2
        elif argv[i] == "-MD":
            i += 1
        else:
            compile_args.append(argv[i])
            i += 1
    header_hashes = {}
    if args.header_tree:
        tree = args.header_tree.resolve()
        # json.hpp is downloaded/generated into the app directory and is not
        # present in git archives. Reuse that exact dependency without adding
        # current project headers back to the preserved ABI search path.
        generated = output.parent / "generated-dependencies"
        generated.mkdir(exist_ok=True)
        dependency = root / "Applications/CausalLM/json.hpp"
        if dependency.is_file():
            destination = generated / "json.hpp"
            if not destination.exists():
                destination.symlink_to(dependency)
            compile_args.insert(1, "-I" + str(generated))
            header_hashes["generated/json.hpp"] = hashlib.sha256(dependency.read_bytes()).hexdigest()

        def map_include(value):
            path = Path(value)
            absolute = (compile_dir / path).resolve() if not path.is_absolute() else path.resolve()
            try:
                relative = absolute.relative_to(root)
            except ValueError:
                return value
            if relative.parts[:2] == ("Applications", "CausalLM"):
                if any(part in ("third_party", "xgrammar", "lib") for part in relative.parts[2:]):
                    return value
            elif not relative.parts or relative.parts[0] not in ("nntrainer", "api"):
                return value
            mapped = tree / relative
            if not mapped.is_dir():
                raise RuntimeError(f"Matching header tree lacks {relative}")
            return str(mapped)

        for index, argument in enumerate(compile_args):
            if argument.startswith("-I") and len(argument) > 2:
                compile_args[index] = "-I" + map_include(argument[2:])
            elif index and compile_args[index - 1] in ("-I", "-isystem"):
                compile_args[index] = map_include(argument)
        for relative in ("Applications/CausalLM/models/transformer.h",
                         "Applications/CausalLM/models/causal_lm.h",
                         "Applications/CausalLM/kv_cache_manager.h",
                         "Applications/CausalLM/models/performance_metrics.h",
                         "Applications/CausalLM/factory.h"):
            path = tree / relative
            if path.exists():
                header_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.header_revision:
        include = output.parent / "baseline-headers"
        include.mkdir(exist_ok=True)
        for path in ("Applications/CausalLM/models/transformer.h",
                     "Applications/CausalLM/models/causal_lm.h",
                     "Applications/CausalLM/kv_cache_manager.h"):
            content = subprocess.check_output([
                "git", "show", f"{args.header_revision}:{path}"], cwd=root)
            (include / Path(path).name).write_bytes(content)
        compile_args.insert(1, "-I" + str(include))
    header_root = args.header_tree.resolve() if args.header_tree else root
    for relative in ("Applications/CausalLM/models/transformer.h",
                     "Applications/CausalLM/models/causal_lm.h",
                     "Applications/CausalLM/kv_cache_manager.h",
                     "Applications/CausalLM/models/performance_metrics.h",
                     "Applications/CausalLM/models/factory.h",
                     "nntrainer/kv_cache/kv_cache_manager.h",
                     "nntrainer/kv_cache/kv_cache_spec.h"):
        path = header_root / relative
        if args.header_revision and relative in (
                "Applications/CausalLM/models/transformer.h",
                "Applications/CausalLM/models/causal_lm.h",
                "Applications/CausalLM/kv_cache_manager.h"):
            path = include / Path(relative).name
        if path.exists():
            header_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    obj = output.with_suffix(".o")
    # The app uses -ffast-math on Android. The observer must still detect NaN/
    # infinity instead of having std::isfinite optimized to an unconditional true.
    compile_args += ["-fno-fast-math", "-c", str(source),
                     "-o", str(obj)]
    subprocess.run(compile_args, cwd=compile_dir, check=True)
    i = link_args.index("-o")
    link_args[i + 1] = str(output)
    link_args = [str(obj) if a == main_object else a
                 for a in link_args]
    if not is_android:
        # A relocated native probe needs the existing app/engine directories.
        # Android deployment provides LD_LIBRARY_PATH, never host absolute paths.
        link_args += ["-Wl,-rpath," + str(build),
                      "-Wl,-rpath," + str(build / "Applications/CausalLM"),
                      "-Wl,-rpath," + str(build / "nntrainer")]
    subprocess.run(link_args, cwd=build, check=True)
    metadata = {"header_revision": args.header_revision,
                "header_tree": str(args.header_tree.resolve()) if args.header_tree else None,
                "header_tree_revision": args.header_tree_revision,
                "header_sha256": header_hashes,
                "git_head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "source_tree_dirty": bool(subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=root)),
                "source_diff_sha256": hashlib.sha256(subprocess.check_output(
                    ["git", "diff"], cwd=root)).hexdigest(),
                "probe_source": str(source),
                "probe_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "target": target, "main_object": main_object,
                "android": is_android,
                "compile_directory": str(compile_dir),
                "compile_command": compile_args, "link_command": link_args}
    output.with_suffix(".abi.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
