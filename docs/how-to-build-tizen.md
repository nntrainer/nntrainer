# How to build NNTrainer for Tizen

NNTrainer is packaged for Tizen with GBS. This page covers picking the right
target architecture, and how to build for a device whose CPU is newer than the
Tizen reference baseline.

## Prerequisites

Install GBS and point it at the Tizen snapshot repositories:

```bash
echo "deb [trusted=yes] http://download.tizen.org/tools/latest-release/Ubuntu_22.04/ /" \
  | sudo tee /etc/apt/sources.list.d/tizen.list
sudo apt-get update && sudo apt-get install -y gbs
cp .github/workflows/tizen.gbs.conf ~/.gbs.conf
```

## A plain build

```bash
gbs build -A armv7l  --define "unit_test 0"
gbs build -A aarch64 --define "unit_test 0"
gbs build -A x86_64  --define "unit_test 1"
```

`-A` must match the **RPM architecture of the target image**, not the CPU. A
64-bit core running a 32-bit Tizen rootfs is an `armv7l` target. Check on the
device with:

```bash
rpm --eval '%{_arch}'          # armv7l / aarch64 - this is the -A value
uname -m
grep -m1 'CPU part' /proc/cpuinfo   # e.g. 0xd0b = Cortex-A76
```

## Optional features

| Define | Effect |
| --- | --- |
| `--define "_with_fp16 1"` | Enable float16. Requires an ARMv8.2-A target; off by default because the aarch64 Tizen reference is ARMv8.0-A |
| `--define "_with_gpu 1"` | Enable the OpenCL backend |
| `--define "unit_test 1"` | Run the unit tests as part of the build |
| `--define "arm_tune <flags>"` | Raise the ARM baseline for a known device, see below |

## Building for a specific ARM core

The Tizen reference `Optflags` target the oldest core each architecture has to
support:

| Arch | Reference optflags |
| --- | --- |
| armv7l | `-march=armv7-a -mtune=cortex-a8 -mfpu=neon -mfloat-abi=softfp -mthumb` |
| aarch64 | `-march=armv8-a+fp+simd+crc+crypto -mtune=cortex-a57.cortex-a53` |

`arm_tune` is appended to `CFLAGS`/`CXXFLAGS` last, so it overrides both those
defaults and the `-march` that `meson.build` hardcodes for aarch64 (meson places
environment flags after `add_project_arguments()`).

### armv7l on an ARMv8.2-A core

This is the interesting case: a 32-bit Tizen rootfs on something like a
Cortex-A76. The core implements `FEAT_DotProd` at EL0 in AArch32, so the ggml
q4_0/q8_0 kernels can use `VSDOT` even though userspace is 32-bit. Without
`arm_tune` those kernels stay on their scalar path.

```bash
gbs build -A armv7l --include-all --skip-srcrpm \
  --define "_skip_debug_rpm 1" \
  --define "unit_test 0" \
  --define "arm_tune -march=armv8.2-a+dotprod -mtune=cortex-a76 -mfpu=neon-fp-armv8 -mfp16-format=ieee"
```

Each flag is load-bearing:

| Flag | Why |
| --- | --- |
| `-march=armv8.2-a` | gcc rejects `+dotprod` on plain `armv8-a` for A32 |
| `+dotprod` | defines `__ARM_FEATURE_DOTPROD`, which gates the kernels |
| `-mtune=cortex-a76` | scheduling only, no ISA effect |
| `-mfpu=neon-fp-armv8` | FMA plus fp16 <-> fp32 conversion |
| `-mfp16-format=ieee` | needed for `float16x4_t` and the block scale conversions |

Do **not** put `-mfloat-abi` in `arm_tune`: `softfp` is part of the armv7l ABI
and changing it breaks linking against the platform libraries. Do not add
`+i8mm` for Cortex-A76 either, since `FEAT_I8MM` is ARMv8.6-A and the core does
not implement it.

### Checking that the flags took effect

Tizen's global cflags include `-frecord-gcc-switches`, so the compile line is
recorded in the binary:

```bash
rpm2cpio ~/GBS-ROOT/local/repos/tizen/armv7l/RPMS/nntrainer-core-*.armv7l.rpm | cpio -idm
readelf -A usr/lib/debug/usr/lib/libnntrainer.so.debug | head -8
```

An `arm_tune`d armv7l build reports `Tag_CPU_arch: v8` and `Tag_FP_arch: FP for
ARMv8`; the reference build reports `v7` and `VFPv3`.

## Compatibility

An `arm_tune`d RPM no longer runs on cores below the baseline you pick. Build it
only for images pinned to known hardware, and keep the plain build for anything
that ships broadly.
