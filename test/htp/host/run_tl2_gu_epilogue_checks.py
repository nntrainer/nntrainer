#!/usr/bin/env python3
"""Compare actual native-TL2 GU epilogues and requantization with SDK libnative.

Inputs are explicit repository/overlay paths; source files are never changed.
Static production helpers and their actual private types are extracted verbatim,
with symbol prefixes only. HVX dequant, quant, and LUT preparation are compiled
from production sources. The only scheduling stand-in is the existing pool-NULL
serial dispatch, so lane ownership is exercised explicitly by the fixture.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

BACK = Path('nntrainer/tensor/htp_backend')
MOE = BACK / 'hmx/hexkl_mm_u8i4_moe.c'
FUNCTIONS = ('moe_tl2_restore_magnitude', 'moe_tl2_gate_epilogue',
             'moe_tl2_gate_direct', 'moe_tl2_gate_epilogue_worker',
             'moe_m1_slice', 'moe_m1_requant_worker')


def function(text, name):
    match = re.search(r'^static (?:inline )?(?:void|int)\s+' + name + r'\(',
                      text, re.M)
    if not match:
        if name == 'moe_tl2_gate_direct':
            return ''
        raise ValueError('missing function ' + name)
    opening = text.index('{', match.end())
    # Strip comments/string literals before matching braces, preserving length.
    syntax = re.sub(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"',
                    lambda m: ' ' * len(m[0]), text, flags=re.S)
    depth = 1
    closing = opening + 1
    while depth:
        depth += (syntax[closing] == '{') - (syntax[closing] == '}')
        closing += 1
    return text[match.start():closing] + '\n'


def extract(text, prefix):
    begin = text.index('#define MOE_M1_MAX_ROWS')
    end = text.index('} moe_m1_ctx;', begin) + len('} moe_m1_ctx;')
    value = text[begin:end] + '\n' + '\n'.join(function(text, n) for n in FUNCTIONS)
    for name in (*FUNCTIONS, 'moe_m1_expert', 'moe_m1_ctx'):
        value = re.sub(r'\b' + name + r'\b', prefix + '_' + name, value)
    value = value.replace('hvx_dequant_swiglu_acc_tiles_to_f32(',
                          prefix + '_serial_swiglu(')
    if prefix == 'candidate':
        value = value.replace('hvx_dq_swiglu_worker(', 'counted_direct_swiglu(')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hexagon-tools', type=Path, required=True)
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--mutations', action='store_true')
    args = parser.parse_args()
    repo = args.repo.resolve()
    source = (args.source or repo).resolve()
    ref, out = args.reference.resolve(), args.output.resolve()
    native = args.hexagon_tools.resolve() / 'Tools/libnative'
    out.mkdir(parents=True, exist_ok=True)
    if not (native / 'lib/libnative.a').is_file():
        parser.error('--hexagon-tools must contain Tools/libnative/lib/libnative.a')
    if args.mutations and not args.sanitize:
        parser.error('--mutations requires --sanitize')
    original = (source / MOE).read_text()
    (out / 'reference-unit.inc').write_text(extract((ref / MOE).read_text(), 'reference'))
    (out / 'candidate-unit.inc').write_text(extract(original, 'candidate'))
    (out / 'native-types.h').write_text(
        '#include <hexagon_types.h>\n#include <hvx_hexagon_protos.h>\n'
        'typedef HVX_Vector nntr_unaligned_hvx __attribute__((aligned(1)));\n'
        '#define HVX_UVector nntr_unaligned_hvx\n#define __hexagon__ 1\n')
    cc, cxx = os.environ.get('CC', 'gcc'), os.environ.get('CXX', 'g++')
    flags = ['-std=gnu11', '-O1' if args.sanitize else '-O3', '-g', '-Wall',
             '-Wextra', '-Wno-unused-parameter', '-ffp-contract=off',
             '-fno-strict-aliasing', '-ffunction-sections', '-fdata-sections',
             '-include', 'malloc.h', '-include', str(out / 'native-types.h')]
    if args.sanitize:
        flags += ['-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                  '-fno-omit-frame-pointer', '-fno-pie']
    for path in (out, native / 'include', repo / 'test/htp/host/stub',
                 repo / 'nntrainer/tensor', repo / BACK,
                 repo / BACK / 'hmx', repo / BACK / 'hvx'):
        flags += ['-I', str(path)]
    commands = []
    inputs = [ref / MOE, source / MOE, Path(__file__).resolve()]
    def run(command, label, expected_failure=False):
        commands.append(command)
        (out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        log = out / (label + '.log')
        with log.open('w') as stream:
            proc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if expected_failure:
            if proc.returncode == 0 or 'Assertion' not in log.read_text():
                raise SystemExit(f'{label}: expected assertion rejection; see {log}')
        elif proc.returncode:
            raise SystemExit(f'{label} failed ({proc.returncode}); see {log}')
        return proc.returncode
    objects = []
    for name in ('hvx_dequant_i32', 'hvx_quant_u8', 'hvx_gemv_u8tl2',
                 'hvx_rmsnorm_rows_f32'):
        src = source / BACK / 'hvx' / (name + '.c')
        if not src.exists():
            src = repo / BACK / 'hvx' / (name + '.c')
        obj = out / (name + '.o')
        inputs.append(src)
        run([cc, *flags, '-c', str(src), '-o', str(obj)], name)
        objects.append(str(obj))
    check = source / 'test/htp/host/tl2_gu_epilogue_check.c'
    if not check.exists():
        check = Path(__file__).with_name('tl2_gu_epilogue_check.c')
    inputs.append(check)
    executable = out / 'check'
    def build(label):
        obj = out / 'check.o'
        run([cc, *flags, '-c', str(check), '-o', str(obj)], label + '-compile')
        link = [cxx]
        if args.sanitize:
            link += ['-fsanitize=address,undefined', '-no-pie']
        run([*link, str(obj), *objects, str(native / 'lib/libnative.a'),
             '-Wl,--gc-sections', '-lm', '-o', str(executable)], label + '-link')
    build('check')
    run([str(executable)], 'run')
    mutations = {}
    if args.mutations:
        # Each mutant changes the actual extracted candidate. The control stays
        # untouched, and the valid-candidate output is restored before exit.
        tests = {
            'wrong-up-plane': ('c->inter_ntiles,\n', 'c->inter_ntiles - 1u,\n'),
            'all-lanes-write': ('hvx_dq_swiglu_worker(threads, id, &job);',
                                'hvx_dq_swiglu_worker(1u, 0u, &job);'),
            'accept-split': ('c->tl2_splits != 1u ||', '0 ||'),
            'accept-mag7': ('htp_tl2_palette_magnitude(e->g->pal) != 1u ||',
                            '0 ||'),
        }
        for label, (old, new) in tests.items():
            if old not in original:
                raise SystemExit('mutant anchor missing: ' + label)
            changed = original.replace(old, new, 1)
            (out / 'candidate-unit.inc').write_text(extract(changed, 'candidate'))
            build(label)
            mutations[label] = run([str(executable)], label, True)
        (out / 'candidate-unit.inc').write_text(extract(original, 'candidate'))
        build('restored')
        run([str(executable)], 'restored-run')
    summary = {
        'passed': True, 'sanitized': args.sanitize,
        'result': (out / 'run.log').read_text().strip(), 'mutations': mutations,
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'scope': 'Actual extracted M1 GU/requant helpers and production HVX math; explicit lanes, pool NULL only. No DMA, RPC or silicon timing claim.'}
    (out / 'validation.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
