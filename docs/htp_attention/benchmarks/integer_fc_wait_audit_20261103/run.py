from pathlib import Path
import subprocess, re, csv, statistics

root = Path('/tmp/nntr-w4a16-waits-20261103')
remote = '/data/local/tmp/nntr-w4a16-waits-20261103'
adb = ['adb', '-s', 'R5KL30G6MCX']
for rnd in range(3):
    for shape in (1, 4, 0, 3, 2, 5):
        name = f'round{rnd}-shape{shape}'
        cmd = (f'cd {remote} && ADSP_LIBRARY_PATH="{remote};/vendor/lib/rfsa/adsp;/dsp" '
               f'LD_LIBRARY_PATH=/vendor/lib64:{remote} ./run_main_on_hexagon 3 '
               f'libinteger_bench.so {remote}/{name}.txt {shape}')
        with (root / (name + '.runner')).open('w') as f:
            subprocess.run(adb + ['shell', cmd], stdout=f, stderr=subprocess.STDOUT, check=True)
        subprocess.run(adb + ['pull', f'{remote}/{name}.txt', str(root / (name + '.txt'))],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        txt = (root / (name + '.txt')).read_text()
        assert 'BENCH OK' in txt and '\nFAIL' not in txt, name
        print(name, 'OK', flush=True)

rows = []
for p in sorted(root.glob('round*-shape*.txt')):
    for line in p.read_text().splitlines():
        if line.startswith('TRACE '):
            rows.append(dict(run=p.stem, **dict(re.findall(r'(\w+)=([\w.]+)', line))))
with (root / 'trace_samples.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
medians = []
for key in sorted({(r['M'], r['K'], r['N'], r['mode']) for r in rows}):
    group = [r for r in rows if (r['M'], r['K'], r['N'], r['mode']) == key]
    d = dict(zip(('M','K','N','mode'),key)); d['samples'] = len(group)
    for k in rows[0]:
        if k not in ('run','M','K','N','mode','pass'):
            d[k] = round(statistics.median(float(r[k]) for r in group),3)
    medians.append(d)
with (root / 'trace_medians.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=list(medians[0])); w.writeheader(); w.writerows(medians)
print('Wrote trace_samples.csv and trace_medians.csv', flush=True)
