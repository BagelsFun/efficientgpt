"""Re-grade saved agent_e2e results with the current classifier (no model calls), then apply manual labels.

    python -m assistant.bench.regrade_agent runs/<stamp>/agent_v12/*.json [--labels runs/<stamp>/agent_v12/manual_labels.json]

Writes <name>.regraded.json next to each input (the original file is kept; earlier label in `verdict_prev`).
- Crashed turns are 'error'.
- Saved answers from runs before 2026-10-31 15:00 were truncated to 600 characters; if the run-time verdict was
  'correct' but the truncated text no longer matches, the row goes to 'review' instead of silently flipping.
- `--labels` maps "<file stem>|<config>|<row index>" -> {"verdict": ..., "note": ...}: a human decision recorded with
  its reason, applied last. Rows still in 'review' after labels are reported as unresolved.
"""

import argparse
import json
from pathlib import Path

from .agent_e2e import QUESTIONS, VERDICTS, classify


def regrade(path, labels):
    accept = dict(QUESTIONS)
    data = json.loads(path.read_text(encoding='utf-8'))
    out = {}
    for cfg, c in data['configs'].items():
        for i, r in enumerate(c['rows']):
            prev = r.get('verdict_v1', r.get('verdict'))
            r['verdict_prev'] = prev
            r.pop('verdict_v1', None)
            if r.get('error'):
                v = 'error'
            else:
                v = classify(r['answer'], accept[r['question']])
                if prev == 'correct' and v != 'correct' and len(r['answer']) >= 600:
                    v = 'review'  # run-time match may have been beyond the saved 600 characters
            key = f'{path.stem}|{cfg}|{i}'
            if key in labels:
                r['manual'] = labels[key]
                v = labels[key]['verdict']
            r['verdict'] = v
            r['passed'] = v == 'correct'
        counts = {v: sum(r['verdict'] == v for r in c['rows']) for v in VERDICTS}
        for k in ('hallucinated',):
            c.pop(k, None)
        c.update(counts, score=f"{counts['correct']}/{len(c['rows'])}")
        out[cfg] = counts
    data['regraded'] = 'classifier v3 (anchored numbers, review/error classes) + manual labels'
    path.with_name(path.stem + '.regraded.json').write_text(json.dumps(data, indent=1), encoding='utf-8')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+', type=Path)
    ap.add_argument('--labels', type=Path, default=None)
    args = ap.parse_args()
    labels = json.loads(args.labels.read_text(encoding='utf-8')) if args.labels else {}
    for path in args.files:
        if path.name.endswith('.regraded.json') or path.name == 'manual_labels.json':
            continue
        for cfg, counts in regrade(path, labels).items():
            print(f'{path.stem:28s} {cfg:10s} {counts}')


if __name__ == '__main__':
    main()
