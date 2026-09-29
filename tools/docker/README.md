# nntrainer Hexagon dev container

One `linux/amd64` image that builds nntrainer on x86 and cross-builds the
Hexagon backend (simulator tests, DSP skel, Android host harness). It is the
only build environment the agents in `.claude/agents/` use; the phone is
driven by a human (see `docs/plans/0000-agent-system-and-env.md`).

```
tools/docker/setup_wizard.sh          # one-time: runtime, image, SDK login, model files, smoke test
tools/docker/run.sh <command...>      # run anything inside the container, repo mounted at /work
tools/docker/run.sh                   # interactive shell
tools/docker/run.sh --build-image     # rebuild after editing the Dockerfile
```

Host layout the wrapper expects (all overridable by environment, see `run.sh`):

| Host path | In container | Contents |
|---|---|---|
| repo | `/work` | this checkout (build dirs are created inside it) |
| `~/Qualcomm/Hexagon_SDK/<ver>/` | `/opt/qcom/Hexagon_SDK/<ver>/` | Hexagon SDK installed by `qpm-cli` inside the container (licensed, never in the image) |
| `~/Qualcomm/hexkl_addon/` | `/opt/qcom/hexkl_addon/` | HexKL micro API (optional, HMX work) |
| `Applications/CausalLM/res/qwen3/qwen3-0.6b/` | `/model` (ro) | `hf/` (HuggingFace Qwen3-0.6B: safetensors + tokenizer, downloaded by the wizard) and `nntr_qwen3_0.6b_w8cx_DEFAULT.bin` (built from it by `tools/hexagon/make_w8cx_bin.py`) |

The entrypoint sources `setup_sdk_env.source` of the newest SDK version (or
`HEXAGON_SDK_VERSION`) so `HEXAGON_SDK_ROOT`, `DEFAULT_HEXAGON_TOOLS_ROOT`
and `DEFAULT_TOOLS_VARIANT` are set (the compiler itself is not on `PATH`;
the scripts call `$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-clang` by
full path and `run_sim_test.sh` picks the `run_main_on_hexagon` image of
`$DEFAULT_TOOLS_VARIANT`, e.g. `hexagon_toolv19_v75` for HEXAGON_Tools
19.0.04); the scripts under `tools/hexagon/` then work unchanged:

```
tools/docker/run.sh ./tools/hexagon/build_host_x86.sh
HEX_ARCH=v75 tools/docker/run.sh ./tools/hexagon/build_sim_test.sh
HEX_ARCH=v75 tools/docker/run.sh ./tools/hexagon/run_sim_test.sh profile acc
HEX_ARCH=v75 tools/docker/run.sh ./tools/hexagon/build_skel.sh
tools/docker/run.sh ./tools/hexagon/build_host_test.sh
```

Notes

* The simulator itself is an x86 binary running under Rosetta/QEMU on Apple
  silicon; expect `profile acc` to take several times longer than the "few
  minutes" recorded on a native workstation.
* `build_skel.sh` defaults to `HEX_ARCH=v79`; the shipping skel is v75
  (HEXAGON.md §7), so always pass `HEX_ARCH` explicitly.
* clang-format-14 lives in the image: `tools/docker/run.sh clang-format-14 -i <files>`.
