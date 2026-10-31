"""Recompute every committed CI report from its raw CSV and compare ratio / CI bounds with the stored JSON."""
import json, sys, pathlib
sys.path.insert(0, '.')
from research.phase5.analyze_ci import COMPARISONS, analyze, load_raw
from research.phase5.benchmark_inprocess import comparisons_for
worst, checked, mismatched = 0.0, 0, []
for js in sorted(pathlib.Path('research/phase5/ci_runs').rglob('CI_REPORT*.json')):
    raw = js.parent / 'benchmark_raw.csv'
    if not raw.exists():
        continue
    stored = json.loads(js.read_text())
    stored = stored['results'] if isinstance(stored, dict) else stored
    data = load_raw(raw)
    pairs = sorted({(r['baseline'], r['candidate']) for r in stored})
    fresh = {(r['batch'], r['prompt_tokens'], r['generated_tokens'], r['baseline'], r['candidate']): r
             for r in analyze(data, 10000, 0, pairs)}
    for r in stored:
        key = (r['batch'], r['prompt_tokens'], r['generated_tokens'], r['baseline'], r['candidate'])
        f = fresh.get(key)
        checked += 1
        if f is None:
            mismatched.append((js.name, key, 'missing')); continue
        d = max(abs(f[k] - r[k]) for k in ('ratio', 'ci_low', 'ci_high'))
        worst = max(worst, d)
        if d > 1e-9 or f['verdict'] != r['verdict']:
            mismatched.append((str(js.relative_to('research/phase5/ci_runs')), key, round(d, 6), r['verdict'], f['verdict']))
print(f'rows checked: {checked}; max |difference| {worst:.2e}; mismatches: {len(mismatched)}')
for m in mismatched[:10]: print('  ', m)
