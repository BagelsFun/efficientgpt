"""In-domain held-out eval: execution-graded pass@1 on the distillation validation split (never trained on).

    python -m assistant.bench.distill_eval --backend hf:Qwen/Qwen3-1.7B --quant nf4 [--adapter <dir>] \
        --data results/assistant/data/distill_v2_full.jsonl --names <adapter>/val_names.json --out <file>.json

The trainer writes `val_names.json` (the shuffled validation rows it excluded from training). Each prompt is the
body-free signature prompt; the response passes if the function passes the teacher's verified tests in the sandboxed
grader. Greedy, thinking off. Complements assistant-quality-v1 (16 code items is too few to detect small gains). The
split comes from the same teacher distribution as the training data, so it measures in-domain gain, not general skill.
"""

import argparse
import json
from pathlib import Path

from .grading import grade_code


def load_split(data, names):
    wanted = set(json.loads(names.read_text(encoding='utf-8')))
    rows = [json.loads(l) for l in data.read_text(encoding='utf-8').splitlines() if l.strip()]
    split = [r for r in rows if r['name'] in wanted]
    if len(split) != len(wanted):
        raise SystemExit(f'{len(wanted) - len(split)} validation names not found in {data}')
    return split


def mcnemar(base, cand):
    """Paired comparison on the same items: (wins, losses, exact two-sided p) for cand vs base ({name: passed})."""
    from math import comb
    names = sorted(set(base) & set(cand))
    wins = sum(cand[n] and not base[n] for n in names)
    losses = sum(base[n] and not cand[n] for n in names)
    k, n = min(wins, losses), wins + losses
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return wins, losses, p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--backend', required=True)
    ap.add_argument('--quant', default='nf4')
    ap.add_argument('--revision', default=None)
    ap.add_argument('--adapter', default=None)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--names', type=Path, required=True)
    ap.add_argument('--max-tokens', type=int, default=768)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    from ..cli import make_backend
    split = load_split(args.data, args.names)
    backend = make_backend(args.backend, args.quant, args.revision, adapter=args.adapter)
    records = []
    for r in split:
        reply = backend.chat([r['messages'][0]], max_tokens=args.max_tokens)
        ok, why = grade_code(reply.text, r['name'], r['tests'], timeout=20)
        records.append({'name': r['name'], 'topic': r.get('topic'), 'passed': ok, 'reason': why[:200], 'response': reply.text})
    passed = sum(x['passed'] for x in records)
    out = {'backend': args.backend, 'quant': args.quant, 'adapter': args.adapter, 'data': str(args.data),
           'n': len(records), 'passed': passed, 'pass_rate': passed / max(1, len(records)), 'records': records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('backend', 'adapter', 'n', 'passed', 'pass_rate')}))


if __name__ == '__main__':
    main()
