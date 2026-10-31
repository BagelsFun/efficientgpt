"""In-process interleaved comparison of Phase 5 modes (for differences below ~5%).

    python -m research.phase5.benchmark_inprocess --modes adaptive_graph compiled_adaptive_graph \
        --output research/phase5/ci_runs/<stamp>_inprocess/benchmark_raw.csv

Separate-process runs carry per-process offsets of about +-3% even for
identical work (EXPERIMENT_LOG AA-01). Here every child process builds ALL
requested modes, validates them on every workload (exact or near-tie rule, as
in benchmark_review), then times them interleaved request by request, rotating
the mode order each request so each mode visits each position equally often.
Shared process state therefore cancels within each comparison. Clusters for
the bootstrap are (process, block). Peak memory is NOT reported: all modes
are resident together. Output: raw CSV (analyze_ci schema), CI_REPORT.md/.json.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from ..phase4.benchmark_decode import load_model, make_prompts
from .analyze_ci import COMPARISONS, analyze, markdown
from .benchmark_requests import workloads
from .benchmark_review import APPROX_MODES, EXTRA_MODES, MODES, ModeRunner

MARKER = 'RESULT_JSON:'


def rotation(modes, step):
    """Mode order for one interleaved request; position counts balance every len(modes) steps."""
    k = step % len(modes)
    return modes[k:] + modes[:k]


def comparisons_for(modes):
    """Known contract comparisons among the modes, then modes[0] against each other mode."""
    pairs = [(a, b) for a, b in COMPARISONS if a in modes and b in modes]
    pairs += [(modes[0], m) for m in modes[1:] if (modes[0], m) not in pairs and (m, modes[0]) not in pairs]
    return pairs


def run_child(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA hardware is required for the benchmark')
    model = load_model(args.checkpoint)
    tok = Tokenizer.from_file(str(args.tokenizer))
    cases = workloads(args.quick)
    inputs = {case: make_prompts(tok, model.config.vocab_size, case[0], case[1], args.variants)
              for case in cases}
    old = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    try:
        with torch.inference_mode():
            runners = {mode: ModeRunner(model, mode) for mode in args.modes}
            near_ties = {mode: 0 for mode in args.modes}
            for case in cases:
                for prompt in inputs[case]:
                    for mode, runner in runners.items():
                        near_ties[mode] += runner.check(prompt, case[2])
            for case in cases:
                for runner in runners.values():
                    for _ in range(2):
                        runner.get(inputs[case][0], case[2]).generate(inputs[case][0], case[2])
            torch.cuda.synchronize()
            raw = []
            for batch, p, n in cases:
                case = (batch, p, n)
                decs = {mode: r.get(inputs[case][0], n) for mode, r in runners.items()}
                for block in range(args.blocks):
                    for i in range(args.requests):
                        ix = i % args.variants
                        prompt = inputs[case][ix]
                        for mode in rotation(list(args.modes), i + block):
                            torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            decs[mode].generate(prompt, n)
                            torch.cuda.synchronize()
                            raw.append({'block': block, 'mode': mode, 'batch': batch, 'prompt_tokens': p,
                                        'generated_tokens': n, 'trial': i, 'variant': ix,
                                        'ms': round(1000 * (time.perf_counter() - t0), 5)})
    finally:
        torch.backends.cuda.matmul.fp32_precision = old
    return {'raw': raw, 'near_ties': near_ties,
            'compile_ms': {m: round(r.compile_ms, 1) for m, r in runners.items()}}


def run_all(args):
    rows, meta = [], []
    for proc in range(args.processes):
        # Rotate the mode list per process too, so construction order varies.
        modes = rotation(list(args.modes), proc)
        print(f'Process {proc + 1}/{args.processes}: modes built in order {modes}', flush=True)
        cmd = [sys.executable, '-m', 'research.phase5.benchmark_inprocess', '--child',
               '--modes', *modes, '--blocks', str(args.blocks), '--requests', str(args.requests),
               '--variants', str(args.variants), '--checkpoint', str(args.checkpoint),
               '--tokenizer', str(args.tokenizer), '--output', str(args.output)]
        if args.quick:
            cmd.append('--quick')
        proc_out = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if proc_out.returncode or MARKER not in proc_out.stdout:
            print(proc_out.stdout[-4000:], flush=True)
            print(proc_out.stderr[-4000:], file=sys.stderr, flush=True)
            raise RuntimeError(f'Process {proc + 1} failed; no comparison is valid')
        result = json.loads(proc_out.stdout.rsplit(MARKER, 1)[1])
        for r in result['raw']:
            rows.append({'round': proc * args.blocks + r['block'] + 1, 'process': proc + 1, **r})
        meta.append({'process': proc + 1, 'build_order': modes,
                     'near_ties': result['near_ties'], 'compile_ms': result['compile_ms']})
        print(f"  correctness=PASS near_ties (approx modes: rows differing from fp32)={result['near_ties']}", flush=True)
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    data = {}
    for r in rows:
        case = (r['batch'], r['prompt_tokens'], r['generated_tokens'])
        data.setdefault(case, {}).setdefault(r['mode'], {}).setdefault(r['round'], []).append(r['ms'])
    results = analyze(data, args.boots, 0, comparisons_for(list(args.modes)))
    report = out.with_name('CI_REPORT.md')
    text = markdown(results, out, args.boots).replace(
        '# Phase 5 bootstrap latency ratios',
        '# Phase 5 in-process interleaved latency ratios\n\n'
        f'{args.processes} processes x {args.blocks} blocks x {args.requests} requests per mode and workload; '
        'clusters = (process, block). Peak memory not comparable (all modes resident together).')
    report.write_text(text, encoding='utf-8')
    report.with_suffix('.json').write_text(json.dumps({'processes': meta, 'results': results}, indent=2),
                                           encoding='utf-8')
    for r in results:
        print(f"{r['batch']:2} {r['prompt_tokens']:3} {r['generated_tokens']:2} {r['baseline']:>23} -> "
              f"{r['candidate']:<23} {r['ratio']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] {r['verdict']}")
    print(f'Wrote {out}, {report} and {report.with_suffix(".json")}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--modes', nargs='+', choices=MODES + EXTRA_MODES + APPROX_MODES, required=True)
    ap.add_argument('--processes', type=int, default=3)
    ap.add_argument('--blocks', type=int, default=6)
    ap.add_argument('--requests', type=int, default=12)
    ap.add_argument('--variants', type=int, default=3)
    ap.add_argument('--boots', type=int, default=10000)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--checkpoint', type=Path, default=Path('out-tinystories-v2/baseline-2500.pt'))
    ap.add_argument('--tokenizer', type=Path, default=Path('data/tinystories_v2/tokenizer.json'))
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if len(set(args.modes)) != len(args.modes) or len(args.modes) < 2:
        ap.error('Need at least two distinct modes')
    if min(args.processes, args.blocks, args.requests, args.variants) < 1 or args.blocks * args.processes < 4:
        ap.error('Need positive sizes and at least 4 (process, block) clusters')
    if args.child:
        print(MARKER + json.dumps(run_child(args)), flush=True)
    else:
        run_all(args)


if __name__ == '__main__':
    main()
