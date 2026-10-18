#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract actual app/staging helpers; SIMDe supplies host NEON semantics."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--baseline',type=Path,required=True)
p.add_argument('--simde-include',type=Path)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--cxx',default='g++')
p.add_argument('--sanitize',action='store_true')
p.add_argument('--native-arm',action='store_true')
p.add_argument('--build-only',action='store_true')
a=p.parse_args()
if not a.native_arm and not a.simde_include:
 p.error('--simde-include is required for the host SIMD fixture')
a.output.mkdir(parents=True,exist_ok=True)
rel='Applications/CausalLM/layers/conv_block_layer.cpp'
s=(a.source/rel).read_text();b=(a.baseline/rel).read_text()
# The existing full-row helper must remain byte-for-byte the baseline body.
def full_helper(text):
 begin=text.index('static bool lfmConvPostGatePrefill(')
 end=text.index('\n}\n', begin)+3
 return text[begin:end]
if full_helper(s)!=full_helper(b):
 raise RuntimeError('Existing full-row helper differs from the baseline')
start=s.index('// The app enables fast-math,')
end=s.index('ConvBlockLayer::ConvBlockLayer() :')
bs=b.index('// The app enables fast-math,');be=b.index('ConvBlockLayer::ConvBlockLayer() :')
chunks={'candidate_helpers.inc':s[start:end], 'baseline_helpers.inc':b[bs:be]}
core=(a.repo/'nntrainer/tensor/htp_backend/htp_compute_ops.cpp').read_text()
cs=core.index('  struct Held {');ce=core.index('  /** @brief The attention\'s u16 context',cs)
chunks['staging.inc']=core[cs:ce]
hs=core.index('  const void *host_view(');he=core.index('  bool gemm_qs4cx_res_add_fp32(',hs)
chunks['views.inc']=core[hs:he]
# Actual app early route. Type/dtype/from/owns-state guards are not recoded.
rs=s.index('  const bool row_resident = rows == 1 && htpDecodeRowResident(from);')
re=s.index('  // Match invokeFc\'s K=2048 row cap:',rs)
chunks['app_route.inc']=s[rs:re]
for name,data in chunks.items():(a.output/name).write_text(data)
shutil.copy2(Path(__file__).with_name('lfm2_conv_chunk_check.cpp'),a.output)
cmd=[a.cxx,'-std=c++17','-O2','-g','-ffast-math','-pthread','-Wall','-Wextra',
 '-I'+str((a.repo/'nntrainer/tensor/htp_backend').resolve()),
 str((a.output/'lfm2_conv_chunk_check.cpp').resolve()),'-o',str((a.output/'check').resolve())]
if a.simde_include:
 cmd[1:1]=['-I'+str(a.simde_include.resolve())]
if a.native_arm:
 cmd[1:1]=['-march=armv8.2-a+fp16+dotprod+i8mm','-static-libstdc++']
else:
 cmd[1:1]=['-mfma']
if a.sanitize:cmd[1:1]=['-fsanitize=address,undefined','-fno-omit-frame-pointer']
manifest={'source_sha256':hashlib.sha256(s.encode()).hexdigest(),'baseline_sha256':hashlib.sha256(b.encode()).hexdigest(),
 'core_sha256':hashlib.sha256(core.encode()).hexdigest(),'extracts':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in chunks.items()},
 'compile':cmd,'run':[str((a.output/'check').resolve())],
 'arithmetic':'native ARM NEON' if a.native_arm else 'SIMDe0.8.2 on x86 FMA',
 'standins':['Actual app arithmetic; simulated NEON results require separate native ARM validation.',
 'HTP allocation uses aligned host heap with actual fit/stage/input/alias helpers; fake RPC writes deterministic projection bytes.',
 'Synchronous ThreadManager substitute can use1or4pthread row lanes; no DSP/FastRPC timing claim.']}
(a.output/'commands.json').write_text(json.dumps(manifest,indent=2)+'\n')
subprocess.run(cmd,check=True)
if not a.build_only:subprocess.run(manifest['run'],check=True)
