#!/usr/bin/env bash
# Regenerate the OpenCL kernel C++ source strings from the .cl files.
#
# meson's configure_file (cl_kernels/meson.build) emits one <kernel>.cpp
# per <kernel>.cl by wrapping each line in an R"(...)" "\n" literal. The
# ndk-build flow does not re-run meson, so editing a .cl file leaves the
# bundled .cpp stale and the GPU silently runs the old kernel source. This
# helper re-emits the .cpp strings so ndk-build picks up the latest .cl.
#
# Usage: tools/regen_cl_kernels.sh [builddir]
set -euo pipefail
BUILDDIR=${1:-builddir}
SRC=nntrainer/tensor/cl_operations/cl_kernels
DST="$BUILDDIR/nntrainer/tensor/cl_operations/cl_kernels"
if [ ! -d "$SRC" ]; then echo "src kernels dir not found: $SRC" >&2; exit 1; fi
if [ ! -d "$DST" ]; then echo "dst kernels dir not found: $DST (meson configured?)" >&2; exit 1; fi
python3 - "$SRC" "$DST" <<'PY'
import os, sys, pathlib
src_dir, dst_dir = sys.argv[1], sys.argv[2]
for cl in sorted(pathlib.Path(src_dir).glob('*.cl')):
    name = cl.stem
    # int4_gemm.cl gets a special variable name (see meson.build).
    var = 'gemm_int4_kernel' if name == 'int4_gemm' else name + '_kernel'
    lines = cl.read_text().split('\n')
    wrapped = '\n'.join(f'R"({l})" "\\n"' for l in lines)
    out = f'#include "{name}.h"\n\nnamespace nntrainer {{\nconst std::string {var} = {{ \n{wrapped}\n}};\n}} // namespace nntrainer\n'
    dst = pathlib.Path(dst_dir) / f'{name}.cpp'
    if not dst.exists():
        continue  # meson hasn't generated this yet; skip
    if dst.read_text() != out:
        dst.write_text(out)
        print(f'regen: {dst}')
PY
