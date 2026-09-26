# tokenizers_c

A thin C ABI over the `tokenizers` Rust crate, vendored here with a
`Cargo.lock` so that every build resolves the same dependency versions.

## The `_win` suffix is historical

This crate is the source of truth for **both** consumers, not just Windows:

| consumer | script | artifact |
|---|---|---|
| Windows / MSVC | `../build_tokenizer_windows.ps1` | `tokenizers_c.lib` |
| Android / NDK  | `../build_tokenizer_android.sh`  | `libtokenizers_android_c.a` |

The Android script used to clone `mlc-ai/tokenizers-cpp` at build time; it now
builds this crate with `cargo build --locked`, which removes a network
dependency from a script that produces a linked artifact. The directory name
was not changed along with it because the build-directory layout it implies is
depended on independently by `../meson.build` and by the
`causallm-tokenizer-lib` option default in the top-level `meson_options.txt`.
Renaming it is a mechanical change across those files, deliberately kept out of
the change that repointed the Android script.
