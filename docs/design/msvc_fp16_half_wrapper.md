# fp16 on toolchains without `_Float16`: the uint16-backed `Half` wrapper

**Status:** implemented · **Scope:** `api/ccapi/include/half_fp16.h`, the
`fp16-impl` meson option, and the `_FP16` / `NNTR_GGML_FP16` macro selection.

## 1. Problem

`_FP16` is a preprocessor macro that resolves to a *language-level* half type:

```cpp
// api/ccapi/include/tensor_dim.h
#ifdef ENABLE_FP16
#ifdef USE__FP16
#define _FP16 __fp16      // ARM / Android
#else
#define _FP16 _Float16    // x86_64
#endif
#endif
```

Both `__fp16` and `_Float16` are GCC/Clang extensions. MSVC `cl.exe` implements
neither: `_Float16` is not a keyword there, so a declaration such as
`const _Float16 *p` does not even parse. `std::float16_t` (`<stdfloat>`, C++23)
is likewise not implemented, and `__STDCPP_FLOAT16_T__` is undefined. The
`__half` type from `<cuda_fp16.h>` is not a substitute either — its arithmetic
operators are device-only and unusable in host code without `nvcc`.

Consequently `-Denable-fp16=true` could not be built at all on MSVC, which in
turn blocks any Windows configuration that wants fp16 tensors.

What *is* available on MSVC are the F16C conversion intrinsics
(`_mm_cvtph_ps` / `_mm_cvtps_ph`, `<immintrin.h>`), supported since VS2012 and
emitted with no `/arch:` flag. The tree already relies on that same idiom in
`nntr_ggml_impl_fp16_fp32.cpp`.

## 2. Design

Add a self-contained leaf header defining `nntrainer::Half` — a
`uint16_t`-backed IEEE754 binary16 value that performs every operation in
`float` and rounds the result back — and select it through a **third branch** of
the `_FP16` macro, guarded by a new `USE_HALF_WRAPPER` define:

```cpp
#ifdef ENABLE_FP16
#ifdef USE__FP16
#define _FP16 __fp16
#elif defined(USE_HALF_WRAPPER)
#include "half_fp16.h"
#define _FP16 ::nntrainer::Half
#else
#define _FP16 _Float16
#endif
#endif
```

`nntr_ggml_impl.h` carries the identical three-way selection for
`NNTR_GGML_FP16` so that it stays self-contained when pulled in without
`tensor_dim.h`.

Two properties are load-bearing:

* **Compile-time only.** The backing type is fixed when the build is
  configured. There is no runtime switch, no environment variable, and no
  dispatch: `USE_HALF_WRAPPER` is either defined for the whole build or it is
  not. A runtime lever would be meaningless here — the two types cannot
  coexist in one translation unit, since they share one macro name.
* **Native builds are untouched.** GCC, Clang and every ARM build take exactly
  the branch they take today, because `USE_HALF_WRAPPER` is never defined for
  them. Nothing is removed; a branch is added.

### 2.1 Selection: capability probe, not compiler name

`meson_options.txt` gains a combo option:

```
option('fp16-impl', type: 'combo', choices: ['auto', 'native', 'wrapper'], value: 'auto')
```

* `auto` (default) — `meson.build` compiles a small `_Float16` *arithmetic*
  snippet with the actual compiler. If it compiles, native `_Float16` is used;
  otherwise the build falls back to the wrapper and adds
  `-DUSE_HALF_WRAPPER=1`. Testing arithmetic rather than mere declaration
  matters: a toolchain that parses the type but cannot operate on it must not
  be classified as native.
* `native` — force `_Float16`. Fails to build where the type does not exist,
  which is the point: it is a deliberate assertion.
* `wrapper` — force the wrapper even on GCC/Clang. This is what makes the MSVC
  path testable on Linux (see §4).

Selection is therefore a property of the compiler's capabilities, not of its
name or version, and does not depend on the host OS.

## 3. `Half` contract

### 3.1 Layout / ABI

```cpp
struct Half {
  uint16_t bits_;
  Half() = default;   // trivial, like _Float16: `new Half[n]` leaves it uninitialised
  ...
};
static_assert(sizeof(Half) == 2);
static_assert(std::is_trivially_copyable<Half>::value);
static_assert(std::is_standard_layout<Half>::value);
```

The static asserts are not decoration. Half-typed buffers are handed to
accelerator backends as raw bytes, sized with `sizeof(_FP16) * n` and
reinterpreted through `unsigned short *` / `uint16_t *`. Standard layout with
`uint16_t` as the first member makes those casts pointer-interconvertible and
therefore well-defined; 2-byte size keeps the byte arithmetic correct; trivial
copyability keeps `memcpy`, `memset` and `std::vector<_FP16>` valid. Bit-zero
is `+0.0`, so `memset(p, 0, ...)` retains its meaning.

Because the storage is bit-identical to native `_Float16`, model files,
checkpoints and device buffers written by one build are readable by the other.

### 3.2 Conversions

`float -> half` uses `_mm_cvtps_ph` under `_MSC_VER` (round-to-nearest-even via
the default MXCSR) and a software bit-twiddling path elsewhere, matching
`compute_fp32_to_fp16` / `compute_fp16_to_fp32`. `operator float()` is
implicit, mirroring `_Float16`'s promotion to `float`.

Note for the MSVC path: the rounding immediate of `_mm_cvtps_ph` must be in
`0..7` on cl.exe (`C4556`), so `_MM_FROUND_NO_EXC` (`0x08`) cannot be OR-ed in.
`_MM_FROUND_TO_NEAREST_INT` alone still yields round-to-nearest-even; only
exception masking is lost, and the produced bits are identical.

### 3.3 Operators

Every operator computes in `float` and rounds back — the same thing GCC does
for scalar `_Float16` arithmetic.

* Homogeneous `(Half, Half)` for `+ - * /` and the compound forms return
  `Half`. A single operation is bit-identical to native `_Float16`.
* Comparisons return `bool`.
* Increment / decrement are provided, since `_Float16` supports them.
* Mixed `(Half, T)` / `(T, Half)` operators are templated and SFINAE-
  constrained so they are exact matches on both arguments — this is what keeps
  `Half + float` unambiguous while leaving the converting constructor implicit
  (so copy-initialisation such as `_FP16 ret = 0;` still compiles). Integral
  `T` yields `Half`; floating `T` yields `common_type<float, T>`, matching
  `_Float16`'s promotion rules.

Streaming operators and `std::to_string` are deliberately not provided; no call
site needs them, and reaching for one is a compile error — the correct outcome
for a missing piece of a stand-in.

`std::numeric_limits<Half>` **is** provided, and is the exception to that rule
for one reason: it is the only member of the set whose absence does not fail to
compile. The primary template is defined for every type and value-initializes,
so an unspecialized `Half` answers `max() == 0.0`, `infinity() == 0.0` and
`is_specialized == false` — silently, from code that compiles clean. The
specialization is written as bit patterns through `Half::from_bits` so that
every member can be `constexpr` as the standard requires (the converting
constructor rounds through `float` with a `memcpy` and cannot be).

One caveat, measured rather than assumed. The native side has the same hole
today: libstdc++ specializes `numeric_limits` for `_Float16` only from C++23,
where `__STDCPP_FLOAT16_T__` is defined, and this project builds C++17 (C++20
on Windows). On GCC 13.3, `-std=c++17` gives `is_specialized == 0` and
`max() == 0`, while `-std=c++23` gives `1` and `65504`. So
`std::numeric_limits<_FP16>` is **not** usable on either backing type yet, and
a call site must not be added on the strength of this specialization alone;
what it buys is that the wrapper is no longer the half of the pair answering
zero. `unittest_half_fp16` asserts binary16's real constants directly, and
compares against `std::numeric_limits<_Float16>` only when that type reports
itself specialized — so the parity check starts working by itself the day the
standard level moves.

### 3.4 Where host half arithmetic actually lives

Accelerator staging code performs no host half arithmetic — it passes
pointers, computes byte sizes and casts to `unsigned short` for kernel
arguments — so it needs only §3.1 and §3.2. The full operator set of §3.3 is
required by the CPU tensor path (`half_tensor.cpp`, the fallback fp16 kernels,
the x86 fp16 backend), which is unconditionally in the build whenever
`enable-fp16` is on and therefore must compile.

## 4. Testing the MSVC path on Linux

`-Dfp16-impl=wrapper` is the harness. It builds the entire library and test
suite on GCC/Clang with `_FP16 == nntrainer::Half`, which exercises exactly the
code MSVC will compile, on a platform where the tests can actually be run.

It is wired into CI: `.github/workflows/ubuntu_clean_meson_build.yml` carries
`-Denable-fp16=true -Dfp16-impl=wrapper` as a third `meson_options` matrix
entry, so every PR builds and runs the suite under both backing types. That
leg is what catches a change using an `_FP16` operation `Half` does not
provide, or one leaning on native promotion where `Half` differs.
`unittest_half_fp16` cannot: it tests `Half` in isolation against
`_Float16`, not the tree's thousands of `_FP16` uses. Neither can the MSVC
job, which sets `enable-test = false` and is compile-and-link only.

The meaningful check beyond pass/fail is that the two implementations agree
numerically: run the fp16 tensor / cpu-backend / activation suites once under
`native` and once under `wrapper` and compare per test.

One expected difference, worth knowing before chasing it: for a *single*
operation the two are bit-identical, but in a multi-operation expression the
wrapper rounds every intermediate to binary16 while GCC/Clang evaluate native
`_Float16` with excess precision, keeping intermediates in `float`. Compiling
the native side with `-fexcess-precision=16` makes the two agree bit-for-bit;
without it, element-wise results can differ by one or two binary16 ULP. ARM
`__fp16` sits on the excess-precision side too, because its arithmetic promotes
to `float` by definition. Stored data is bit-identical either way, so model
files and device buffers remain interchangeable; only the last bit or two of a
chained host-side computation can move.

## 5. Related MSVC-only build gaps

Two fixes accompany the wrapper; both are no-ops for GCC/Clang codegen.

* `half_tensor.cpp` uses `std::accumulate` without including `<numeric>`.
  libstdc++ pulls it in transitively, MSVC's STL does not.
* The explicit `q8_K` specializations in `ggml_interface_fp16.cpp` carried
  top-level `__restrict` on their parameters while the primary template in
  `ggml_interface.h` does not. MSVC encodes top-level `__restrict` into the
  decorated name, so the caller's instantiation could not resolve against the
  definition (`LNK2019`); GCC ignores `restrict` when mangling, which hid the
  mismatch. The `restrict` locals inside the bodies are kept, so the aliasing
  hint — and GCC codegen — are unaffected.
