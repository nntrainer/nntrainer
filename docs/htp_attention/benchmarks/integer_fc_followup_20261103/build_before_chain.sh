set -eu
source /tmp/nntr-integer-bench/env.sh
repo=/home/jungone-lee/workspace/nntrainer-pr4471-integer-fc
backend=$repo/nntrainer/tensor/htp_backend
variant=$1
out=/tmp/nntr-w4a16-opt-20261103/$variant
mkdir -p "$out"
if [ ! -f "$out/libnntr_hvx_skel.so" ]; then cp "$repo/test/htp/build/libnntr_hvx_skel.so" "$out/"; fi
flags=(-DHEXKL_BENCH_ROW_ONLY)
if [ "$variant" = scalar ]; then flags=(-DHEXKL_INTEGER_SCALAR); fi
"$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-clang" -mv81 -mhvx -mhvx-length=128B -mhmx -G0 -O3 -fPIC -shared -Wall -Werror "${flags[@]}" \
 -I "$repo/test/htp" -I "$repo/test/htp/generated" -I "$backend" -I "$backend/hmx" -I "$backend/hvx" \
 -I "$HEXKL_ROOT/include" -I "$HEXAGON_SDK_ROOT/rtos/qurt/computev81/include/qurt" \
 -I "$HEXAGON_SDK_ROOT/rtos/qurt/computev81/include/posix" -isystem "$HEXAGON_SDK_ROOT/incs" \
 -isystem "$HEXAGON_SDK_ROOT/incs/stddef" -isystem "$HEXAGON_SDK_ROOT/ipc/fastrpc/incs" \
 "$repo/test/htp/integer_fc_device_bench.c" "$out/libnntr_hvx_skel.so" -o "$out/libinteger_bench.so"
cp "$HEXAGON_SDK_ROOT/libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon" "$out/"
cp "$HEXAGON_SDK_ROOT/libs/run_main_on_hexagon/ship/hexagon_toolv19_v81/librun_main_on_hexagon_skel.so" "$out/"
