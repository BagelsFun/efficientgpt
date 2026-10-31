"""Re-grade saved responses under grading v1 (no preamble) and v1.1 (stdlib preamble). No new generations."""
import json, sys, pathlib
sys.path.insert(0, '.')
from assistant.bench import quality_v1
from assistant.bench.grading import grade
items = {i['id']: i for i in quality_v1.items()}
out = {}
for f in sorted(pathlib.Path(sys.argv[1]).glob('*.responses.jsonl')):
    rows = [json.loads(l) for l in f.open(encoding='utf-8')]
    res = {}
    for ver, pre in (('v1', False), ('v1.1', True)):
        passed = {r['id']: grade(items[r['id']], r['response'], preamble=pre)[0] for r in rows}
        res[ver] = {f'{sp}/{cat}': f"{sum(passed[i] for i in passed if (sp == 'all' or items[i]['split'] == sp) and (cat == 'all' or items[i]['category'] == cat))}/"
                                   f"{sum(1 for i in passed if (sp == 'all' or items[i]['split'] == sp) and (cat == 'all' or items[i]['category'] == cat))}"
                    for sp in ('dev', 'test', 'all') for cat in ('code', 'math', 'mcq', 'all')}
        res[ver + '_changed_items'] = sorted(i for i in passed if passed[i] != next(r['passed'] for r in rows if r['id'] == i)) if ver == 'v1.1' else None
    out[f.stem.replace('.responses', '')] = res
    print(f.stem, '| v1:', res['v1']['all/all'], 'test', res['v1']['test/all'], '| v1.1:', res['v1.1']['all/all'], 'test', res['v1.1']['test/all'], '| changed by v1.1:', res['v1.1_changed_items'])
json.dump(out, open(pathlib.Path(sys.argv[1]) / 'regrade.json', 'w'), indent=2)
