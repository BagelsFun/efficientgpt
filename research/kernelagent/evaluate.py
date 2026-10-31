"""Isolated candidate evaluation: policy screen -> compile -> correctness -> CUDA-graph timing.

    python -m research.kernelagent.evaluate --task argmax --family row_whole [--params '{"WARPS": 4}']
        [--split dev|heldout] [--no-timing] [--history results/kernelagent/history.jsonl]
    python -m research.kernelagent.evaluate --smoke [--no-timing]   # every default template on every dev shape

Each candidate runs in a fresh subprocess (`--worker`, spec on stdin, JSON on stdout) with a wall-clock timeout and a
GPU memory cap, so a hang, crash or illegal access cannot take down the controller. Correctness is judged against an
fp64 truth with dtype-aware rules per task kind:
  float  : max |cand - truth| <= TOL_FACTOR * max |eager_fp32 - truth| + floor, and finite wherever truth is finite;
  index  : exact equality with the reference (argmax of identical fp32 values; first index wins ties);
  sample : the chosen token has p > 0 and its fp64 CDF interval contains u * total within SAMPLE_TOL * total
           (exact agreement with eager is reported, not required: fp32 cumsum order may move a boundary).
Timing: each callable is captured in its own CUDA graph; trials interleave candidate / eager / Inductor / best existing
in a rotating order; per-call times feed research.phase5.analyze_ci.ratio_ci (bootstrap over trials). Timings taken
while another process loads the GPU are marked `contended` and must not be reported as results.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROTOCOL = 'kernelagent-ada/eval-v2'  # v2: INNER calls per graph (v1 timed one graph launch per call)
INNER = 20
TOL_FACTOR = 4.0
SAMPLE_TOL = 1e-5
MEM_FRACTION = 0.45
DEFAULT_TIMEOUT = 300


def candidate_id(task, source):
    return hashlib.sha256(f'{task}\n{source}'.encode()).hexdigest()[:16]


def gpu_utilization():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=10).stdout.split(',')
        return {'util_pct': float(out[0]), 'mem_used_mib': float(out[1])}
    except Exception:
        return {'util_pct': math.nan, 'mem_used_mib': math.nan}


def git_revision():
    try:
        rev = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--', 'research/kernelagent'],
                                    capture_output=True, text=True, timeout=10).stdout.strip())
        return rev + ('+dirty' if dirty else '')
    except Exception:
        return 'unknown'


# ------------------------------------------------------------------ correctness (worker side)

def check_output(task, inp, out):
    """Return (ok, detail) for one input under the task's numerical contract."""
    import torch
    ref = task['reference']
    kind = task['kind']
    if not isinstance(out, torch.Tensor):
        return False, {'error': f'run() returned {type(out).__name__}'}
    eager = ref(inp)
    if out.shape != eager.shape:
        return False, {'error': f'shape {tuple(out.shape)} != {tuple(eager.shape)}'}
    if kind == 'index':
        mism = int((out.long() != eager.long()).sum())
        return mism == 0, {'mismatches': mism}
    if kind == 'sample':
        probs = inp['probs'].double()
        cdf = probs.cumsum(-1)
        total = cdf[:, -1:]
        target = inp['u'].double().unsqueeze(-1) * total
        t = out.long().clamp(0, probs.size(-1) - 1)
        in_range = bool(((out.long() >= 0) & (out.long() < probs.size(-1))).all())
        hi = cdf.gather(-1, t)
        lo = hi - probs.gather(-1, t)
        tol = SAMPLE_TOL * total
        ok_rows = (probs.gather(-1, t) > 0) & (target >= lo - tol) & (target <= hi + tol)
        exact = float((out.long() == eager.long()).float().mean())
        return in_range and bool(ok_rows.all()), {'bad_rows': int((~ok_rows).sum()), 'exact_vs_eager': exact}
    truth = ref({k: (v.double() if isinstance(v, torch.Tensor) and v.is_floating_point() else v) for k, v in inp.items()},
                dtype=torch.float64)
    finite = torch.isfinite(truth)
    if not bool(torch.isfinite(out.double())[finite].all()):
        return False, {'error': 'non-finite output where truth is finite'}
    err_c = float((out.double() - truth)[finite].abs().max()) if finite.any() else 0.0
    err_e = float((eager.double() - truth)[finite].abs().max()) if finite.any() else 0.0
    bound = TOL_FACTOR * err_e + task['floor']
    return err_c <= bound, {'err_candidate': err_c, 'err_eager_fp32': err_e, 'bound': bound}


def correctness(task, run, shapes, seeds=(0, 1)):
    rows = []
    for shape in shapes:
        cases = [(s, None) for s in seeds] + [(seeds[0], e) for e in task['edges']]
        for seed, edge in cases:
            inp = task['inputs'](shape, seed, edge)
            given = {k: (v.clone() if hasattr(v, 'clone') else v) for k, v in inp.items()}
            try:
                out = run(given)
                changed = [k for k, v in inp.items() if hasattr(v, 'clone') and not bool((given[k] == v).all())]
                if changed:  # judge against the pristine inputs; a candidate may not write to them
                    ok, detail = False, {'error': f'inputs modified: {changed}'}
                else:
                    ok, detail = check_output(task, inp, out)
            except Exception as e:  # compile errors surface here on first launch
                ok, detail = False, {'error': f'{type(e).__name__}: {str(e)[:300]}'}
            rows.append({'shape': list(shape), 'seed': seed, 'edge': edge, 'ok': ok, **detail})
    return rows


# ------------------------------------------------------------------ timing (worker side)

def _graphed(fn, inp, warmup=3, inner=INNER):
    """One graph holding `inner` back-to-back calls: per-call time excludes the graph-launch overhead (about 7 us on
    this Windows/WDDM machine), which in eval-v1 put a floor under every kernel and pushed all ratios toward 1.
    Inputs are reused across calls, so data that fits in the 24 MB L2 stays cache-resident (warm-L2 timing)."""
    import torch
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn(inp)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn(inp)
    return g


def timing(callables, inp, trials, reps, calls_per_rep):
    """{name: {trial: [per-call ms, ...]}} with graphs replayed in a rotating interleaved order."""
    import torch
    graphs, failed = {}, {}
    for name, fn in callables.items():
        try:
            graphs[name] = _graphed(fn, inp)
        except Exception as e:
            failed[name] = f'{type(e).__name__}: {str(e)[:200]}'
    names = list(graphs)
    out = {n: {} for n in names}
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for t in range(trials):
        order = names[t % len(names):] + names[:t % len(names)] if names else []
        for n in order:
            g = graphs[n]
            g.replay()  # warm the instruction/L2 state for this graph
            times = []
            for _ in range(reps):
                start.record()
                for _ in range(calls_per_rep):
                    g.replay()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end) / (calls_per_rep * INNER))
            out[n][t] = times
    return out, failed


def launch_metadata(module):
    meta = []
    for h in getattr(module, 'LAUNCHES', [])[:4]:
        m = {}
        for key in ('n_regs', 'n_spills'):
            m[key] = getattr(h, key, None)
        md = getattr(h, 'metadata', None)
        m['shared_bytes'] = getattr(md, 'shared', None) if md is not None else None
        m['num_warps'] = getattr(md, 'num_warps', None) if md is not None else None
        meta.append(m)
    return meta


def worker(spec):
    """Runs inside the isolated subprocess."""
    import importlib.util

    import torch
    from . import tasks as T
    from .screen import screen
    from ..phase5.analyze_ci import classify, ratio_ci

    task = T.task(spec['task'])
    res = {'protocol': PROTOCOL, 'task': spec['task'], 'split': spec['split']}
    problems = screen(spec['source'])
    if problems:
        return res | {'status': 'rejected_screen', 'problems': problems}
    torch.cuda.set_per_process_memory_fraction(MEM_FRACTION)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / f"cand_{spec['id']}.py"
        path.write_text(spec['source'], encoding='utf-8')
        t0 = time.perf_counter()
        try:
            mod_spec = importlib.util.spec_from_file_location(f"cand_{spec['id']}", path)
            module = importlib.util.module_from_spec(mod_spec)
            mod_spec.loader.exec_module(module)
        except Exception as e:
            return res | {'status': 'import_error', 'error': f'{type(e).__name__}: {str(e)[:300]}'}
        shapes = [tuple(s) for s in task[spec['split']]]
        first = time.perf_counter()
        rows = correctness(task, module.run, shapes)
        res['first_pass_s'] = round(time.perf_counter() - first, 3)  # includes Triton JIT compilation
        res['import_s'] = round(first - t0, 3)
    res['correctness'] = rows
    res['launch_metadata'] = launch_metadata(module)
    if not all(r['ok'] for r in rows):
        compile_err = any('error' in r and ('Compil' in r['error'] or 'triton' in r['error'].lower()) for r in rows)
        return res | {'status': 'compile_error' if compile_err else 'incorrect'}
    res['status'] = 'correct'
    if not spec.get('timing'):
        return res
    res['gpu_before'] = spec.get('gpu_before')
    util = spec.get('gpu_before', {}).get('util_pct', math.nan)
    res['contended'] = not (util <= spec.get('busy_pct', 10))  # unknown (NaN) utilisation counts as contended
    perf = {}
    for shape in shapes:
        inp = task['inputs'](shape, 0, None)
        callables = {'candidate': module.run, 'eager': task['reference']}
        if spec.get('inductor', True):
            callables['inductor'] = torch.compile(task['reference'], dynamic=False)
        if task.get('best_existing'):
            callables['best_existing'] = task['best_existing']()
        times, failed = timing(callables, inp, spec['trials'], spec['reps'], spec['calls'])
        row = {'failed': failed, 'median_ms': {n: statistics.median(x for v in t.values() for x in v)
                                               for n, t in times.items()}}
        if 'candidate' in times:
            nbytes = task['bytes'](inp)
            row['roofline_fraction'] = T.roofline_fraction(nbytes, row['median_ms']['candidate'] / 1e3)
            for base in [n for n in times if n != 'candidate']:
                r = ratio_ci(times[base], times['candidate'])  # >1 means candidate faster than base
                row[f'speedup_vs_{base}'] = {k: r[k] for k in ('ratio', 'ci_low', 'ci_high')} | {'class': classify(r)}
        perf['x'.join(map(str, shape))] = row
    res['performance'] = perf
    return res


# ------------------------------------------------------------------ controller side

def evaluate(task, source, split='dev', timing=True, timeout=DEFAULT_TIMEOUT, trials=8, reps=5, calls=10,
             inductor=True, busy_pct=10.0):
    spec = {'task': task, 'source': source, 'split': split, 'timing': timing, 'trials': trials, 'reps': reps,
            'calls': calls, 'inductor': inductor, 'busy_pct': busy_pct, 'id': candidate_id(task, source)}
    if timing:
        spec['gpu_before'] = gpu_utilization()
    t0 = time.perf_counter()
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    try:
        p = subprocess.run([sys.executable, '-m', 'research.kernelagent.evaluate', '--worker'], input=json.dumps(spec),
                           capture_output=True, text=True, timeout=timeout, env=env, encoding='utf-8')
    except subprocess.TimeoutExpired:
        return {'protocol': PROTOCOL, 'task': task, 'split': split, 'id': spec['id'], 'status': 'timeout',
                'wall_s': round(time.perf_counter() - t0, 1)}
    lines = [l for l in p.stdout.splitlines() if l.startswith('{"protocol"')]
    if p.returncode != 0 or not lines:
        res = {'protocol': PROTOCOL, 'task': task, 'split': split, 'status': 'crash', 'returncode': p.returncode,
               'stderr_tail': p.stderr[-800:]}
    else:
        res = json.loads(lines[-1])
    res['id'] = spec['id']
    res['trials'] = trials if timing else None
    res['wall_s'] = round(time.perf_counter() - t0, 1)
    return res


def environment():
    import torch
    import triton
    return {'torch': torch.__version__, 'triton': triton.__version__, 'gpu': torch.cuda.get_device_name(0),
            'python': platform.python_version(), 'git': git_revision()}


def record(history, result, extra):
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open('a', encoding='utf-8') as f:
        f.write(json.dumps(result | extra) + '\n')


def seen_ids(history):
    """Candidate ids already evaluated (any status) so controllers never re-run identical source."""
    if not history.exists():
        return set()
    return {json.loads(l)['id'] for l in history.read_text(encoding='utf-8').splitlines() if l.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--task')
    ap.add_argument('--family')
    ap.add_argument('--params', default=None)
    ap.add_argument('--split', default='dev', choices=['dev', 'heldout'])
    ap.add_argument('--no-timing', action='store_true')
    ap.add_argument('--trials', type=int, default=8)
    ap.add_argument('--history', type=Path, default=Path('results/kernelagent/history.jsonl'))
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    if args.worker:
        res = worker(json.loads(sys.stdin.read()))
        print(json.dumps(res, default=str), flush=True)
        return
    from . import templates
    env = environment()
    jobs = []
    if args.smoke:
        jobs = [(t, f, spec['default']) for t, fams in templates.FAMILIES.items() for f, spec in fams.items()]
    else:
        fam = templates.FAMILIES[args.task][args.family]
        jobs = [(args.task, args.family, json.loads(args.params) if args.params else fam['default'])]
    results = []
    for t, f, params in jobs:
        src = templates.render(t, f, params)
        res = evaluate(t, src, args.split, timing=not args.no_timing, trials=args.trials)
        res |= {'family': f, 'params': params}
        record(args.history, res, {'env': env, 'time': time.strftime('%Y-%m-%dT%H:%M:%S')})
        results.append(res)
        n_ok = sum(r['ok'] for r in res.get('correctness', []))
        print(f"{t:20s} {f:18s} {res['status']:16s} {n_ok}/{len(res.get('correctness', []))} cases  {res['wall_s']}s",
              flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({'env': env, 'results': results}, indent=1, default=str), encoding='utf-8')


if __name__ == '__main__':
    main()
