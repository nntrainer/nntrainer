#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build only: actual app router helper/caller plus real native core and DSP."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def function(text, start):
    first = text.index(start)
    brace = text.index('{', first)
    depth, end = 1, brace + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[first:end] + '\n'


def extracts(source):
    first = source.index('    bool router_on_arm = false;')
    end = source.index('\n\n    {', first)
    return {
        'baseline.inc': function(source, 'static void normRows('),
        'helper.inc': 'static bool\n' + function(source, 'lfmRouterOnArm('),
        'accelerator.inc': 'static bool\n' + function(source, 'routerLogitsOnAccelerator('),
        'caller.inc': source[first:end] + '\n',
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('source', 'output', 'compiler', 'nntrainer-include', 'nntrainer-lib-dir'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    folder = a.source / 'Applications/CausalLM/models/lfm2_moe'
    source = (folder / 'lfm2_moe_layer.cpp').read_text()
    parts = extracts(source)
    for name, value in parts.items():
        (a.output / name).write_text(value)
    here = Path(__file__).parent
    for name in ('lfm2_router_arm_native_check.cpp', 'lfm2_router_arm_data.h'):
        shutil.copy2(here / name, a.output)
    common = [str(a.compiler.resolve()), '-std=c++17', '-O3', '-fPIC',
              '-frtti', '-fexceptions', '-fopenmp', '-march=armv8.2-a+fp16+dotprod+i8mm',
              '-DENABLE_FP16=1', '-DUSE__FP16=1', '-DENABLE_HEXKL=1',
              '-Wall', '-Wextra', '-ffunction-sections', '-fdata-sections',
              '-I' + str(a.nntrainer_include.resolve()), '-I' + str(folder.resolve())]
    kernel = common + ['-ffast-math', '-fno-fast-math', '-ffp-contract=off',
                       '-c', str((folder / 'lfm2_router_arm.cpp').resolve()),
                       '-o', str((a.output / 'router.o').resolve())]
    caller = common + ['-ffast-math', str((a.output / 'lfm2_router_arm_native_check.cpp').resolve()),
                       str((a.output / 'router.o').resolve()), '-Wl,--gc-sections',
                       '-Wl,--export-dynamic', '-Wl,-rpath,$ORIGIN',
                       '-L' + str(a.nntrainer_lib_dir.resolve()), '-lnntrainer', '-llog', '-ldl',
                       '-o', str((a.output / 'check').resolve())]
    sha = lambda b: hashlib.sha256(b).hexdigest()
    manifest = {'source_sha256': sha(source.encode()),
                'kernel_sha256': sha((folder / 'lfm2_router_arm.cpp').read_bytes()),
                'extracts': {k: sha(v.encode()) for k, v in parts.items()},
                'compile': [kernel, caller],
                'core_sha256': sha((a.nntrainer_lib_dir / 'libnntrainer.so').read_bytes()),
                'scope': 'Actual helper/caller/core and real DSP RPC interposition. Device run is external.'}
    (a.output / 'commands.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for cmd in (kernel, caller):
        subprocess.run(cmd, check=True)
    (a.output / 'binary.sha256').write_text(sha((a.output / 'check').read_bytes()) + '\n')
