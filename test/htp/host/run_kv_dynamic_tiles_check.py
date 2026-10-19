#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare production dynamic KV tile scheduling with serial HVX quantization."""
import argparse
from pathlib import Path
import subprocess
from run_kv_fixed_tail_check import function

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
p.add_argument('--hexagon-tools', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--mutations', action='store_true')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
b = a.repo / 'nntrainer/tensor/htp_backend'
host = a.repo / 'test/htp/host'
native = a.hexagon_tools / 'Tools/libnative'
src = b / 'hmx/hexkl_kv_q.c'
s = src.read_text()
end = s.index('} dynamic_heads_ctx;') + len('} dynamic_heads_ctx;')
begin = s.rfind('typedef struct {', 0, end)
h = '''#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include "hexkl_kv_q.h"
#include "hvx_kv_quant.h"
#include "hvx_worker_pool.h"
static uint16_t g_wh_i8_pos[1024];
static int g_wh_i8_segments;
#include "hvx_tile_f16.h"
#define __hexagon__ 1
'''
h += function(s, 'wh_i8_segments_match') + '\n'
h += function(s, 'copy_interleaved_i8_tile') + '\n'
h += function(s, 'write_row_i8_tiles') + '\n'
h += function(s, 'append_dynamic_heads') + '\n' + s[begin:end] + '\n'
h += function(s, 'append_dynamic_tile64') + '\n'
h += function(s, 'dynamic_heads_worker') + '\n#undef __hexagon__\n'
(a.output / 'kv_dynamic_under_test.h').write_text(h)
flags = ['-std=gnu11', '-O1', '-g', '-pthread', '-fno-strict-aliasing',
         '-ffp-contract=off', '-include', 'malloc.h', '-include',
         str(host / 'attn_qprep_native_unaligned.h'), '-fsanitize=address,undefined',
         '-fno-sanitize-recover=all', '-fno-omit-frame-pointer']
inc = ['-isystem', str(native / 'include')]
inc += ['-I' + str(x) for x in (host / 'stub', b / 'hvx', b / 'hmx', a.output)]
objects = []
for name, source in (('metadata', src), ('quant', b / 'hvx/hvx_kv_quant.c'),
                     ('pool', b / 'hvx/hvx_worker_pool.c'),
                     ('fixture', host / 'kv_dynamic_tiles_check.c')):
    out = a.output / (name + '.o')
    subprocess.run(['gcc', *flags, *inc, '-c', str(source), '-o', str(out)], check=True)
    objects.append(str(out))
exe = a.output / 'check'
subprocess.run(['g++', '-pthread', '-fsanitize=address,undefined', *objects,
                str(native / 'lib/libnative.a'), '-lm', '-o', str(exe)], check=True)
subprocess.run([str(exe)], check=True)
if a.mutations:
    # Removing the masters' offset bias must be detected, proving the full-tile
    # implementation was exercised independently of its serial row reference.
    bad = h.replace('Q6_V_vxor_VV(out[half], bias)', 'out[half]', 1)
    if bad == h:
        raise RuntimeError('full-tile master-store mutation target changed')
    (a.output / 'kv_dynamic_under_test.h').write_text(bad)
    subprocess.run(['gcc', *flags, *inc, '-c',
                    str(host / 'kv_dynamic_tiles_check.c'), '-o',
                    str(a.output / 'fixture.o')], check=True)
    subprocess.run(['g++', '-pthread', '-fsanitize=address,undefined', *objects,
                    str(native / 'lib/libnative.a'), '-lm', '-o', str(exe)],
                   check=True)
    result = subprocess.run([str(exe)], capture_output=True, text=True)
    (a.output / 'missing_master_bias.log').write_text(result.stdout + result.stderr)
    (a.output / 'kv_dynamic_under_test.h').write_text(h)
    if result.returncode != -6 or 'Assertion' not in result.stderr:
        raise RuntimeError('full-tile master-store mutation was not detected')
    print('Dynamic KV missing-master-bias mutation: rejected as expected')
