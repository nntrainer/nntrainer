#!/usr/bin/env bash

set -e

TARGET=$1
[ -z $1 ] && TARGET=$(pwd)
echo $TARGET


if [ ! -d $TARGET ]; then
    if [[ $1 == -D* ]] || [[ $1 == --arm-arch* ]]; then
	TARGET=$(pwd)
	echo $TARGET
    else
	echo $TARGET is not a directory. please put project root of nntrainer
	exit 1
    fi
fi

pushd $TARGET

filtered_args=()
# Preset/--arm-arch args go before the user's -D args so explicit -D wins
preset_args=()
arm_arch=""

for arg in "$@"; do
    if [[ $arg == -D* ]]; then
	filtered_args+=("$arg")
    fi
    # Handle --arm-arch=<version> argument
    if [[ $arg == --arm-arch=* ]]; then
        arm_arch="${arg#*=}"
    fi
done

# If --arm-arch specified, read configuration from JSON file
if [[ -z "$arm_arch" ]]; then
    arm_arch="armv8.2-a"
fi

if [[ -n "$arm_arch" ]]; then
    # Convert dots to dashes for filename (e.g., armv8.2-a -> armv8-2-a)
    arch_filename=$(echo "$arm_arch" | sed 's/\./-/g')
    json_file="${TARGET}/tools/cross/android_${arch_filename}.json"
    if [[ -f "$json_file" ]]; then
        echo "Using ARM architecture config from: $json_file"
        # Read values from JSON using Python (single invocation, portable, no jq dependency)
        eval "$(python3 -c "
import json, sys
try:
    data = json.load(open('$json_file'))
    print(f'enable_fp16={data.get(\"enable_fp16\", \"True\")}')
    # ISA extension toggles, e.g. enable_dotprod -> -Denable-dotprod=true
    exts = [f'-D{k.replace(\"_\", \"-\")}={str(v).lower()}' for k, v in data.items()
            if k.startswith('enable_') and k != 'enable_fp16']
    print(f'isa_args=({\" \".join(exts)})')
except Exception as e:
    print(f'echo \"Error reading JSON: {e}\" >&2', file=sys.stderr)
    sys.exit(1)
")"
        # Add ISA extension toggles to meson args
        preset_args+=("${isa_args[@]}")
        # Handle enable_fp16 based on JSON boolean
        if [[ "$enable_fp16" == "False" ]]; then
            preset_args+=("-Denable-fp16=false")
        fi
    else
        echo "Warning: JSON config file not found: $json_file (using meson defaults for ISA extensions)"
        echo "Available configurations:"
        ls -1 "${TARGET}/tools/cross/"*.json 2>/dev/null || echo "  No configurations found in tools/cross/"
    fi
    preset_args+=("-Darm-arch=${arm_arch}")
fi

# A builddir configured with a removed ARM option can't be reconfigured
if grep -qsE '^(arm-march|arm-arch = armv9\.2-a)' builddir/meson-private/cmd_line.txt; then
  echo "warning: $TARGET/builddir uses a removed ARM option, recreating it"
  rm -rf builddir
fi

if [ ! -f builddir/build.ninja ]; then
    #default value of openblas num threads is 1 for android
    #enable-tflite-interpreter=false is just temporally until ci system is stabel
    #enable-opencl=true will compile OpenCL related changes or remove this option to exclude OpenCL compilations.
  meson builddir -Dplatform=android -Dopenblas-num-threads=1 -Denable-tflite-interpreter=false -Denable-tflite-backbone=false -Denable-fp16=true -Dnntr-num-threads=4 -Dhgemm-experimental-kernel=false ${preset_args[@]} ${filtered_args[@]}
else
  echo "warning: $TARGET/builddir has already been taken, this script tries to reconfigure and try building"
  #default value of openblas num threads is 1 for android
  #enable-tflite-interpreter=false is just temporally until ci system is stabel
  #enable-opencl=true will compile OpenCL related changes or remove this option to exclude OpenCL compilations.
  # wipe first so options added since the builddir was made are known to configure
  meson setup --wipe builddir
  meson configure builddir -Dplatform=android -Dopenblas-num-threads=1 -Denable-tflite-interpreter=false -Denable-tflite-backbone=false -Denable-fp16=true -Dnntr-num-threads=4 -Dhgemm-experimental-kernel=false ${preset_args[@]} ${filtered_args[@]}
fi

pushd builddir
# `ninja install` runs meson's install target for the subprojects too, which
# tries to write /usr/local/lib/libgmock.a and fails with PermissionError on a
# tree that configured googletest (i.e. any tree without -Denable-test=false).
# Only nntrainer's own artefacts belong in android_build_result, so build and
# then install with --skip-subprojects.
ninja
meson install --no-rebuild --skip-subprojects

tar -czvf $TARGET/nntrainer_for_android.tar.gz --directory=android_build_result .

popd
popd

