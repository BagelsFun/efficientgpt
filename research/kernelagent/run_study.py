"""Full KernelAgent-Ada study: every task x strategy x seed at an equal budget, then one held-out run per task winner.

    python -m research.kernelagent.run_study --out results/kernelagent/study_<date> [--wait-pid <pid>] [--busy-pct 15]

Waits (optionally) for another process to exit and for GPU utilisation to fall below --busy-pct, then runs the search
CLI for each combination in a fresh process (resumable: an existing output file is skipped). Evaluations are shared
through one history file with --reuse-history, so identical candidates are timed once per study. The held-out split
is evaluated only for the single best dev candidate per task (across all strategies and seeds), after the search.
Writes study.json (summaries) next to the per-run files; the report is written by hand from that evidence.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .evaluate import gpu_utilization
from .tasks import TASKS

STRATEGIES = ('single_shot', 'random', 'local', 'local_meta')


def pid_alive(pid):
    out = subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/NH'], capture_output=True, text=True).stdout
    return str(pid) in out


def wait_idle(busy_pct, max_wait_s, log):
    t0 = time.time()
    while True:
        u = gpu_utilization()
        if u['util_pct'] != u['util_pct']:  # NaN: nvidia-smi failed, contention cannot be judged
            log(f'GPU utilisation unknown: {u}; stopping')
            raise SystemExit(3)
        if u['util_pct'] <= busy_pct:
            return u
        if time.time() - t0 > max_wait_s:
            log(f'GPU still busy after {max_wait_s}s: {u}; stopping')
            raise SystemExit(2)
        time.sleep(60)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tasks', nargs='*', default=list(TASKS))
    ap.add_argument('--seeds', nargs='*', type=int, default=[0, 1])
    ap.add_argument('--budget', type=int, default=12)
    ap.add_argument('--trials', type=int, default=8)
    ap.add_argument('--busy-pct', type=float, default=15.0)
    ap.add_argument('--max-wait-s', type=int, default=4 * 3600)
    ap.add_argument('--wait-pid', type=int, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    logf = args.out / 'study.log'

    def log(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        print(line, flush=True)
        with logf.open('a', encoding='utf-8') as f:
            f.write(line + '\n')

    if args.wait_pid:
        log(f'waiting for pid {args.wait_pid}')
        while pid_alive(args.wait_pid):
            time.sleep(120)
    history = args.out / 'history.jsonl'
    summaries = []
    for task in args.tasks:
        complete = True
        for strategy in STRATEGIES:
            for seed in (args.seeds if strategy in ('random', 'local') else args.seeds[:1]):  # deterministic ones once
                out = args.out / f'search_{task}_{strategy}_s{seed}.json'
                if not out.exists():
                    log(f'idle check before {out.name}: {wait_idle(args.busy_pct, args.max_wait_s, log)}')
                    cmd = [sys.executable, '-m', 'research.kernelagent.search', '--task', task, '--strategy', strategy,
                           '--budget', str(args.budget), '--seed', str(seed), '--trials', str(args.trials),
                           '--history', str(history), '--reuse-history', '--busy-pct', str(args.busy_pct), '--out', str(out)]
                    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
                    log(f'{out.name}: rc={p.returncode} {p.stdout.strip()[-300:]} {p.stderr.strip()[-300:] if p.returncode else ""}')
                    if p.returncode:
                        complete = False
                        continue
                summaries.append(json.loads(out.read_text(encoding='utf-8')))
        best = min((s for s in summaries if s['task'] == task and s.get('best')), key=lambda s: s['best']['score_ms'],
                   default=None)
        final = args.out / f'heldout_{task}.json'
        if not complete:  # never pick a held-out winner from partial searches (resume re-runs the missing ones first)
            log(f'{task}: searches incomplete; held-out evaluation deferred')
        elif best and not final.exists():
            from . import templates
            from .evaluate import evaluate
            wait_idle(args.busy_pct, args.max_wait_s, log)
            b = best['best']
            res = evaluate(task, templates.render(task, b['family'], b['params']), 'heldout', timing=True,
                           trials=args.trials) | {'family': b['family'], 'params': b['params'], 'chosen_by': best['strategy']}
            final.write_text(json.dumps(res, indent=1, default=str), encoding='utf-8')
            log(f'{final.name}: {res["status"]}')
    (args.out / 'study.json').write_text(json.dumps(
        [{k: s.get(k) for k in ('task', 'strategy', 'seed', 'evaluated', 'statuses', 'best')} for s in summaries],
        indent=1, default=str), encoding='utf-8')
    log('done')


if __name__ == '__main__':
    main()
