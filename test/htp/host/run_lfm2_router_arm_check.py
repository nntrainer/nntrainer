#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Actual app/kernel plus real HVX C on lane emulation; no device execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from run_lfm2_router_arm_native import extracts

p = argparse.ArgumentParser()
for key in ('repo', 'source', 'output', 'simde-include'):
    p.add_argument('--' + key, type=Path, required=True)
p.add_argument('--cxx', default='g++')
p.add_argument('--cc', default='gcc')
p.add_argument('--sanitize', action='store_true')
p.add_argument('--build-only', action='store_true')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
folder = a.source / 'Applications/CausalLM/models/lfm2_moe'
source = (folder / 'lfm2_moe_layer.cpp').read_text()
for name, value in extracts(source).items():
    (a.output / name).write_text(value)
here = Path(__file__).parent
for name in ('lfm2_router_arm_check.cpp', 'lfm2_router_arm_data.h'):
    shutil.copy2(here / name, a.output)
# The public production kernel uses real arm_neon names; SIMDe supplies those
# names only in this host wrapper after all ordinary system headers are read.
(a.output / 'arm_neon.h').write_text('// SIMDe already included by host wrapper.\n')
(a.output / 'kernel.cpp').write_text(
    '#include <cstddef>\n#define SIMDE_ENABLE_NATIVE_ALIASES\n'
    '#include <simde/arm/neon.h>\n#define __aarch64__ 1\n#define ENABLE_HEXKL 1\n'
    '#include "' + str((folder / 'lfm2_router_arm.cpp').resolve()) + '"\n')
backend = a.repo / 'nntrainer/tensor/htp_backend'
emu = a.repo / 'test/htp/host/hvx_emu'
flags = ['-O2', '-g', '-ffp-contract=off', '-fno-fast-math',
         '-ffunction-sections', '-fdata-sections',
         '-Wall', '-Wextra', '-Wno-unused-parameter', '-Wno-unknown-pragmas',
         '-I' + str(backend.parent.resolve()), '-I' + str((backend / 'hvx').resolve()),
         '-I' + str(emu.resolve())]
if a.sanitize:
    flags += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
commands, objects = [], []
for stem in ('hvx_router_rows_f32', 'hvx_rmsnorm_rows_f32'):
    obj = a.output / (stem + '.o')
    commands.append([a.cc, '-std=c99'] + flags + ['-c', str((backend / 'hvx' / (stem + '.c')).resolve()), '-o', str(obj.resolve())])
    objects.append(str(obj.resolve()))
common = [a.cxx, '-std=c++17', '-pthread', '-mfma'] + flags + [
    '-I' + str(a.simde_include.resolve()), '-I' + str(a.output.resolve()),
    '-I' + str(folder.resolve())]
commands.append(common + ['-c', str((a.output / 'kernel.cpp').resolve()), '-o', str((a.output / 'kernel.o').resolve())])
commands.append(common + [str((a.output / 'lfm2_router_arm_check.cpp').resolve()),
                          str((a.output / 'kernel.o').resolve())] + objects + ['-Wl,--gc-sections', '-o', str((a.output / 'check').resolve())])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = {'commands': commands, 'source_sha256': sha(folder / 'lfm2_moe_layer.cpp'),
            'kernel_sha256': sha(folder / 'lfm2_router_arm.cpp'),
            'reference_sha256': sha(backend / 'hvx/hvx_router_rows_f32.c'),
            'scope': 'Actual app helper/caller and ARM kernel; Tensor/ops/scheduler stand-ins and real HVX C with IEEE lane emulation. Real special-value hardware behavior requires native fixture.'}
(a.output / 'commands.json').write_text(json.dumps(manifest, indent=2) + '\n')
for cmd in commands:
    subprocess.run(cmd, check=True)
if not a.build_only:
    for value in ('1', '0', 'unset', 'true', 'quantized'):
        env = os.environ.copy()
        env.update(ASAN_OPTIONS='detect_leaks=1:halt_on_error=1', UBSAN_OPTIONS='halt_on_error=1')
        env.pop('NNTR_HTP_ROUTER_U8I8', None)
        if value == 'unset':
            env.pop('NNTR_LFM2_ROUTER_ARM', None)
        else:
            env['NNTR_LFM2_ROUTER_ARM'] = '1' if value == 'quantized' else value
        if value == 'quantized':
            env['NNTR_HTP_ROUTER_U8I8'] = '0'  # Backend option deliberately uses presence.
        subprocess.run([str((a.output / 'check').resolve())], env=env, check=True)
