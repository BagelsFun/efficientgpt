"""Optimization controller: search strategies over template families at an equal evaluation budget.

    python -m research.kernelagent.search --task temperature_softmax --strategy local --budget 12 --seed 0 \
        --history results/kernelagent/history.jsonl --out results/kernelagent/search_<task>_<strategy>_s0.json

Strategies (all draw from the same candidate space and pay one budget unit per evaluated candidate):
  single_shot : the default params of every family (budget = number of families; the no-search baseline)
  random      : uniform without replacement over all (family, params) points
  local       : start from each family's default, then repeatedly evaluate the untried neighbour (one param moved one
                step) of the current best; restarts from the next family when a neighbourhood is exhausted
  local_meta  : `local`, but neighbours are ordered by profiler metadata: if the best uses spills, try fewer warps /
                smaller blocks first; if it reaches < 30% of the roofline, try more parallelism first
Score = median candidate ms on the task's dev shapes (geometric mean over shapes). Incorrect, crashing or screened
candidates cost budget and are recorded but never become the incumbent. Identical sources already in the history are
reused, not re-run. The held-out split is never touched here; `--final` evaluates only the chosen winner on it.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

from . import templates


def points(task):
    return [(f, p) for f in templates.FAMILIES[task] for p in templates.space_points(task, f)]


def key(fam, params):
    return fam + json.dumps(params, sort_keys=True)


def neighbours(task, fam, params):
    space = templates.FAMILIES[task][fam]['space']
    out = []
    for k, values in sorted(space.items()):
        i = values.index(params[k]) if params[k] in values else None
        if i is None:
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(values):
                q = dict(params, **{k: values[j]})
                if templates.valid(task, fam, q):
                    out.append((k, j - i, q))
    return out


def score(result):
    """Geometric-mean candidate ms over dev shapes; inf unless correct with timings."""
    if result.get('status') != 'correct' or not result.get('performance'):
        return math.inf
    ms = [row['median_ms'].get('candidate') for row in result['performance'].values()]
    if not ms or any(m is None or m <= 0 for m in ms):
        return math.inf
    return math.exp(sum(math.log(m) for m in ms) / len(ms))


def meta_order(result, cand):
    """Sort key for local_meta: lower = try first. cand = (param, direction, params)."""
    k, direction, _ = cand
    meta = (result or {}).get('launch_metadata') or [{}]
    spills = any((m or {}).get('n_spills') for m in meta)
    perf = (result or {}).get('performance') or {}
    roof = [row.get('roofline_fraction') for row in perf.values() if row.get('roofline_fraction') is not None]
    low_roof = bool(roof) and max(roof) < 0.3
    shrink = k in ('WARPS', 'BLOCK', 'BLOCK_N', 'ROWS') and direction < 0
    grow = k in ('WARPS', 'SPLITS', 'SPLIT', 'NSPLIT') and direction > 0
    if spills:
        return 0 if shrink else 1
    if low_roof:
        return 0 if grow else 1
    return 1


class Search:
    def __init__(self, task, evaluate_fn, budget, seed=0, cache=None):
        self.task, self.eval, self.budget, self.rng = task, evaluate_fn, budget, random.Random(seed)
        self.cache = cache if cache is not None else {}
        self.trace, self.best = [], None

    def run_one(self, fam, params):
        k = key(fam, params)
        if k in {t['key'] for t in self.trace}:
            return None  # never pay twice for the same point
        cached = k in self.cache
        res = self.cache[k] if cached else self.eval(self.task, fam, params)
        self.cache[k] = res
        s = score(res)
        self.trace.append({'key': k, 'family': fam, 'params': params, 'status': res.get('status'), 'score_ms': s,
                           'from_history': cached})
        if s < math.inf and (self.best is None or s < self.best[0]):
            self.best = (s, fam, params, res)
        return res

    def left(self):
        return self.budget - len(self.trace)

    def single_shot(self):
        for fam, spec in templates.FAMILIES[self.task].items():
            if self.left() <= 0:
                break
            self.run_one(fam, spec['default'])

    def random(self):
        pts = points(self.task)
        self.rng.shuffle(pts)
        for fam, p in pts:
            if self.left() <= 0:
                break
            self.run_one(fam, p)

    def local(self, use_meta=False):
        tried = set()
        fams = list(templates.FAMILIES[self.task])
        for fam in fams:  # seed with every family's default (as single_shot)
            if self.left() <= 0:
                return
            self.run_one(fam, templates.FAMILIES[self.task][fam]['default'])
            tried.add(key(fam, templates.FAMILIES[self.task][fam]['default']))
        frontier = sorted((t for t in self.trace if t['score_ms'] < math.inf), key=lambda t: t['score_ms'])
        starts = [(t['family'], t['params']) for t in frontier]
        while self.left() > 0 and starts:
            fam, params = starts.pop(0)
            current = (score(self.cache[key(fam, params)]), params)
            improved = True
            while improved and self.left() > 0:
                improved = False
                cands = [c for c in neighbours(self.task, fam, current[1]) if key(fam, c[2]) not in tried]
                if use_meta:
                    res = self.cache.get(key(fam, current[1]))
                    cands.sort(key=lambda c: meta_order(res, c))
                else:
                    self.rng.shuffle(cands)
                for _, _, q in cands:
                    if self.left() <= 0:
                        break
                    tried.add(key(fam, q))
                    s = score(self.run_one(fam, q) or self.cache[key(fam, q)])
                    if s < current[0]:
                        current, improved = (s, q), True
                        break  # first improvement: move and re-expand
        rest = [(f, p) for f, p in points(self.task) if key(f, p) not in tried]
        self.rng.shuffle(rest)  # neighbourhoods exhausted: spend the remaining budget like `random` (equal budget)
        for f, p in rest:
            if self.left() <= 0:
                break
            self.run_one(f, p)

    def run(self, strategy):
        {'single_shot': self.single_shot, 'random': self.random, 'local': self.local,
         'local_meta': lambda: self.local(use_meta=True)}[strategy]()
        return self.summary(strategy)

    def summary(self, strategy):
        statuses = {}
        for t in self.trace:
            statuses[t['status']] = statuses.get(t['status'], 0) + 1
        out = {'task': self.task, 'strategy': strategy, 'budget': self.budget, 'evaluated': len(self.trace),
               'unused_budget': self.left(),
               'statuses': statuses, 'trace': self.trace}
        if self.best:
            out['best'] = {'score_ms': self.best[0], 'family': self.best[1], 'params': self.best[2]}
        return out


def real_evaluator(history, trials, env, contention_ok, busy_pct=10.0):
    from .evaluate import evaluate, record

    def fn(task, fam, params):
        res = evaluate(task, templates.render(task, fam, params), 'dev', timing=True, trials=trials, busy_pct=busy_pct)
        res |= {'family': fam, 'params': params}
        record(history, res, {'env': env, 'time': time.strftime('%Y-%m-%dT%H:%M:%S')})
        if res.get('contended') and not contention_ok:
            raise SystemExit(f'GPU busy before timing ({res.get("gpu_before")}); refusing to search on contended timings')
        return res
    return fn


REUSABLE = {'correct', 'incorrect', 'rejected_screen', 'compile_error', 'import_error'}  # not crash/timeout (transient)


def load_cache(history, task, trials):
    """Earlier evaluations reusable for this search: same task, dev split, evaluator protocol, trial count and exact
    source (sha256 of the current rendering), deterministic outcome, and uncontended timings."""
    from .evaluate import PROTOCOL, candidate_id
    cache = {}
    if history.exists():
        for line in history.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            if r.get('task') != task or not r.get('family') or r.get('split') != 'dev' or r.get('protocol') != PROTOCOL:
                continue
            if r.get('status') not in REUSABLE or r['family'] not in templates.FAMILIES[task]:
                continue
            if r['id'] != candidate_id(task, templates.render(task, r['family'], r['params'])):
                continue  # template changed since the row was recorded
            if r['status'] == 'correct' and (not r.get('performance') or r.get('contended') or r.get('trials') != trials):
                continue
            cache[key(r['family'], r['params'])] = r
    return cache


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', required=True)
    ap.add_argument('--strategy', required=True, choices=['single_shot', 'random', 'local', 'local_meta'])
    ap.add_argument('--budget', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--trials', type=int, default=8)
    ap.add_argument('--history', type=Path, default=Path('results/kernelagent/history.jsonl'))
    ap.add_argument('--reuse-history', action='store_true', help='reuse identical earlier evaluations (saves GPU time)')
    ap.add_argument('--busy-pct', type=float, default=10.0, help='GPU utilisation above which timings count as contended')
    ap.add_argument('--allow-contended', action='store_true', help='debug only: results are not reportable')
    ap.add_argument('--final', action='store_true', help='evaluate the winner once on the held-out shapes')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    from .evaluate import environment, evaluate
    env = environment()
    cache = load_cache(args.history, args.task, args.trials) if args.reuse_history else {}
    s = Search(args.task, real_evaluator(args.history, args.trials, env, args.allow_contended, args.busy_pct), args.budget, args.seed, cache)
    out = s.run(args.strategy) | {'env': env, 'seed': args.seed}
    if args.final and s.best:
        _, fam, params, _ = s.best
        out['heldout'] = evaluate(args.task, templates.render(args.task, fam, params), 'heldout', timing=True,
                                  trials=args.trials)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('task', 'strategy', 'evaluated', 'statuses', 'best') if k in out}, default=str))


if __name__ == '__main__':
    main()
