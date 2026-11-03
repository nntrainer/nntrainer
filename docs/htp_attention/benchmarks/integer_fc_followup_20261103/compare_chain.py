import csv, pathlib, re, shutil, statistics, subprocess

root = pathlib.Path('/tmp/nntr-w4a16-opt-20261103')
out = root / 'chain-comparison'
out.mkdir(exist_ok=True)
adb = ['adb', '-s', 'R5KL30G6MCX']
remote = '/data/local/tmp/nntr-w4a16-opt-chain'
for variant, name in [('before', 'before_chain'), ('after', 'packed-unroll')]:
    subprocess.run(adb + ['push', str(root/name) + '/.', remote+'/'+variant+'/'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
for rnd in range(3):
    for shape in range(6):
        for variant in (['after', 'before'] if rnd == 1 else ['before', 'after']):
            name = f'{variant}-round{rnd}-shape{shape}'
            directory = remote+'/'+variant
            command = (f'cd {directory} && ADSP_LIBRARY_PATH="{directory};/vendor/lib/rfsa/adsp;/dsp" '
                       f'LD_LIBRARY_PATH=/vendor/lib64:{directory} ./run_main_on_hexagon 3 '
                       f'libinteger_bench.so {directory}/{name}.txt chain {shape}')
            with (out/(name+'.runner')).open('w') as f:
                subprocess.run(adb+['shell', command], stdout=f, stderr=subprocess.STDOUT, check=True)
            subprocess.run(adb+['pull',directory+'/'+name+'.txt',str(out/(name+'.txt'))],
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
            txt=(out/(name+'.txt')).read_text()
            assert 'BENCH OK' in txt and '\nFAIL' not in txt, name
            print(name+' OK',flush=True)
rows=[]; components=[]; hashes={}
for p in sorted(out.glob('*-round*-shape*.txt')):
    variant=p.name.split('-')[0]
    for line in p.read_text().splitlines():
        d=dict(re.findall(r'(\w+)=([\w.]+)',line))
        if line.startswith('CHAIN '): rows.append(dict(variant=variant,run=p.stem,**d))
        if line.startswith('COMPONENT '): components.append(dict(variant=variant,run=p.stem,**d))
        if line.startswith('CHAIN_CHECK '):
            hashes.setdefault(tuple(d[k] for k in ('M','K','N')),[]).append(d['hash'])
assert len(hashes)==6 and all(len(x)==6 and len(set(x))==1 for x in hashes.values()),hashes
for name,ds in [('samples',rows),('components',components)]:
    with (out/(name+'.csv')).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(ds[0]));w.writeheader();w.writerows(ds)
with (out/'medians.csv').open('w') as f:
    w=csv.writer(f); w.writerow(['variant','M','K','N','samples','row_us','packed_us'])
    for key in sorted({tuple(d[k] for k in ('variant','M','K','N')) for d in rows}):
        ds=[d for d in rows if tuple(d[k] for k in ('variant','M','K','N'))==key]
        assert len(ds)==18
        values=[round(statistics.median(float(d[k]) for d in ds),3) for k in ['row_us','packed_us']]
        w.writerow([*key,len(ds),*values]); print(key,values,flush=True)
(out/'validation.txt').write_text('36/36 processes passed. All six shapes have matching complete consumer INT32 hashes before/after. After runs also compare full producer and consumer INT32 buffers, all UINT16 codes and padding, common/per-channel requantization.\n')
