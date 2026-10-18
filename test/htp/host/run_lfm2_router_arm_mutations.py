#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reject selected ownership/math/caller mutations using the actual helpers.

Run run_lfm2_router_arm_check.py --sanitize first. This runner reuses its two
unchanged HVX reference objects, but recompiles each mutated ARM/app helper.
No production file is modified.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--checked-output', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
base = a.checked_output.resolve()
a.output.mkdir(parents=True, exist_ok=True)
commands = json.loads((base / 'commands.json').read_text())['commands']
kernel_command, link_command = commands[-2:]
wrapper = (base / 'kernel.cpp').read_text()
kernel_path = Path(wrapper.rsplit('#include "', 1)[1].split('"', 1)[0])
kernel = kernel_path.read_text()
mutations = [
    ('input_subnormal', 'kernel', 'if (hasSubnormal(x, 4 * hidden))',
     'if (false && hasSubnormal(x, 4 * hidden))', 'primitive special qualification', '1'),
    ('weight_subnormal', 'kernel', 'return !hasSubnormal(w, 2048 * 64);',
     'return true;', 'primitive special qualification', '1'),
    ('nonfinite_output', 'kernel', 'return vmaxvq_u32(nonfinite) == 0;',
     'return true;', 'primitive special qualification', '1'),
    ('input_alias', 'helper.inc',
     '!disjoint(input, static_cast<size_t>(count) * 2048 * sizeof(float)) ||',
     'false ||', 'metadata guard accepted', '1'),
    ('weight_context', 'helper.inc',
     '(row_context && weight_context && row_context != weight_context)',
     '(false)', 'metadata guard accepted', '1'),
    ('last_task', 'helper.inc', '0, count / 4,', '0, count / 4 - 1,',
     'task coverage', '1'),
    ('failure_propagation', 'helper.inc',
     'finite.store(false, std::memory_order_relaxed);',
     'finite.store(true, std::memory_order_relaxed);',
     'special caller used ARM', '1'),
    ('fallback_overwrite', 'caller.inc', 'router_on_arm ||', 'true ||',
     'actual caller RPC count', '0'),
    ('gelu_guard', 'caller.inc', '!router_norm && !gelu_act &&',
     '!router_norm &&', 'unsupported caller policy', '1'),
    ('opt_in', 'helper.inc', 'if (!enabled)', 'if (false && !enabled)',
     'actual caller qualification', '0'),
]
results = []
for name, target, old, new, reason, flag in mutations:
    out = (a.output / name).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for src in base.iterdir():
        if src.is_file() and src.suffix in ('.inc', '.h', '.cpp'):
            shutil.copy2(src, out / src.name)
    source = kernel if target == 'kernel' else (out / target).read_text()
    if source.count(old) != 1:
        raise RuntimeError(f'{name}: expected unique mutation anchor')
    source = source.replace(old, new)
    if target == 'kernel':
        (out / 'production-kernel.cpp').write_text(source)
        (out / 'kernel.cpp').write_text(wrapper.replace(str(kernel_path), str(out / 'production-kernel.cpp')))
    else:
        (out / target).write_text(source)
    # Only generated fixture/kernel paths move. The reference objects remain
    # the already checked actual C translation units in the original directory.
    kc = [v.replace(str(base), str(out)) for v in kernel_command]
    lc = [v if v.endswith(('hvx_router_rows_f32.o', 'hvx_rmsnorm_rows_f32.o'))
          else v.replace(str(base), str(out)) for v in link_command]
    (out / 'commands.json').write_text(json.dumps([kc, lc], indent=2) + '\n')
    with (out / 'build.log').open('w') as log:
        subprocess.run(kc, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(lc, stdout=log, stderr=subprocess.STDOUT, check=True)
    env = os.environ.copy()
    env.pop('NNTR_HTP_ROUTER_U8I8', None)
    env.update(NNTR_LFM2_ROUTER_ARM=flag,
               ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',
               UBSAN_OPTIONS='halt_on_error=1')
    run = subprocess.run([str(out / 'check')], env=env, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / 'run.log').write_text(run.stdout)
    if run.returncode == 0 or reason not in run.stdout:
        raise RuntimeError(f'{name}: did not reject by intended assertion: {run.stdout}')
    results.append({'name': name, 'returncode': run.returncode,
                    'expected_assertion': reason, 'pass': True})
    print(name + ' rejected: ' + reason, flush=True)
(a.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
