# Vendored NeuroPilot headers

`neuron/api/RuntimeAPI.h` and `neuron/api/Types.h` are copied verbatim from
the MediaTek NeuroPilot SDK:

- SDK: `neuropilot-sdk-premium-9.0.9-build20260629`
- Component: `neuron` 9.3.1
- Source path: `neuron_sdk/mt6991/include/neuron/api/`

These two headers are identical across every per-SoC tree shipped in the SDK
(`mt6881`, `mt6899`, `mt6989`, `mt6991`, `mt6993`, `usdk`, `dummy`) — verified
by a recursive diff — so a single copy covers all targets; there is no
per-SoC selection at build time, unlike Qualcomm QNN's HTP stub libraries.

Unlike the QNN backend, there is no configure-time vendor-tree generation
step here: the API surface is small (two self-contained headers, no
transitive vendor sources needed) and stable enough to check in directly.
Do not regenerate this directory from a build script.

To update: copy the same two files from a newer SDK's
`neuron_sdk/<any-soc>/include/neuron/api/` and diff against the previous
version before committing, since nntrainer/neuron/neuron_api.cpp's dlsym
table assumes these exact signatures.
