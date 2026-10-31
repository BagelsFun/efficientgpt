"""Bootstrap confidence intervals for Phase 5 latency ratios from raw timings.

    python -m research.phase5.analyze_ci --raw <benchmark>_raw.csv --output <report>.md

Ratio = baseline median ms / candidate median ms (>1 means the candidate is
faster). Two-level bootstrap paired on process rounds: resample rounds with
replacement (same rounds for both modes), then resample requests within each
chosen round. Between-process drift is therefore inside the interval.

Classification (benchmark contract): WIN if ratio >= 1.05 and CI low > 1;
SLOWER if CI high < 1; otherwise INCONCLUSIVE. Prompt=192 rows compare full
vs adaptive graphs at the same 256-token capacity, i.e. identical work, and
serve as an A/A control whose CI must contain 1.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

COMPARISONS = (  # (baseline, candidate)
    ('full_graph', 'adaptive_graph'),
    ('prefix_sdpa_eager', 'full_graph'),
    ('prefix_sdpa_eager', 'adaptive_graph'),
    ('compiled_static', 'full_graph'),
    ('compiled_static', 'adaptive_graph'),
    ('prefix_sdpa_eager', 'compiled_static'),
    ('full_graph', 'compiled_full_graph'),
    ('adaptive_graph', 'compiled_adaptive_graph'),
    ('full_graph', 'compiled_adaptive_graph'),
    ('compiled_full_graph', 'compiled_adaptive_graph'),
    ('full_graph', 'triton_full_graph'),
    ('adaptive_graph', 'triton_adaptive_graph'),
    ('compiled_adaptive_graph', 'triton_adaptive_graph'),
    ('full_graph', 'triton_adaptive_graph'),
    ('triton_full_graph', 'triton_adaptive_graph'),
    ('triton_full_graph', 'compiled_triton_full_graph'),
    ('triton_adaptive_graph', 'compiled_triton_adaptive_graph'),
    ('adaptive_graph', 'compiled_triton_full_graph'),
    ('compiled_triton_full_graph', 'compiled_triton_adaptive_graph'),
    ('compiled_triton_full_graph', 'fp16_compiled_triton_full_graph'),
    ('compiled_triton_full_graph', 'bf16_compiled_triton_full_graph'),
    ('fp16_compiled_triton_full_graph', 'bf16_compiled_triton_full_graph'),
    ('fp16_triton_full_graph', 'fp16_compiled_triton_full_graph'),
    ('adaptive_graph', 'fp16_compiled_triton_full_graph'),
)
AA_PROMPT = 192
# Same 256-token capacity at prompt=192, so identical work: CI must contain 1.
AA_PAIRS = {('full_graph', 'adaptive_graph'), ('compiled_full_graph', 'compiled_adaptive_graph'),
            ('triton_full_graph', 'triton_adaptive_graph'),
            ('compiled_triton_full_graph', 'compiled_triton_adaptive_graph')}


def load_raw(path):
    """Return {(batch, prompt, gen): {mode: {round: [ms, ...]}}}."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with Path(path).open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            case = (int(r['batch']), int(r['prompt_tokens']), int(r['generated_tokens']))
            data[case][r['mode']][int(r['round'])].append(float(r['ms']))
    return data


def _stack(by_round, rounds):
    sizes = {len(by_round[r]) for r in rounds}
    if len(sizes) != 1:
        raise ValueError(f'Unequal requests per round: {sorted(sizes)}')
    return np.array([by_round[r] for r in rounds])


def ratio_ci(base, cand, boots=5000, seed=0, level=0.95):
    """Point ratio and percentile CI of median(base)/median(cand); inputs {round: [ms]}."""
    rounds = sorted(set(base) & set(cand))
    if not rounds:
        raise ValueError('No shared rounds')
    a, b = _stack(base, rounds), _stack(cand, rounds)
    rng = np.random.default_rng(seed)
    n_rounds = len(rounds)
    pick = rng.integers(0, n_rounds, (boots, n_rounds))[..., None]
    sa = a[pick, rng.integers(0, a.shape[1], (boots, n_rounds, a.shape[1]))]
    sb = b[pick, rng.integers(0, b.shape[1], (boots, n_rounds, b.shape[1]))]
    ratios = np.median(sa.reshape(boots, -1), 1) / np.median(sb.reshape(boots, -1), 1)
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(ratios, [tail, 100 - tail])
    per_round = np.median(a, 1) / np.median(b, 1)
    return {'ratio': float(np.median(a) / np.median(b)), 'ci_low': float(lo), 'ci_high': float(hi),
            'rounds': n_rounds, 'requests_per_round': int(a.shape[1]),
            'per_round_min': float(per_round.min()), 'per_round_max': float(per_round.max()),
            'baseline_median_ms': float(np.median(a)), 'candidate_median_ms': float(np.median(b))}


def classify(result, threshold=1.05):
    if result['ratio'] >= threshold and result['ci_low'] > 1:
        return 'WIN'
    if result['ci_high'] < 1:
        return 'SLOWER'
    return 'INCONCLUSIVE'


def analyze(data, boots=5000, seed=0, comparisons=COMPARISONS):
    out = []
    for case in sorted(data):
        for base, cand in comparisons:
            if base in data[case] and cand in data[case]:
                r = ratio_ci(data[case][base], data[case][cand], boots, seed)
                aa = case[1] == AA_PROMPT and (base, cand) in AA_PAIRS
                r.update(batch=case[0], prompt_tokens=case[1], generated_tokens=case[2],
                         baseline=base, candidate=cand, aa_control=aa,
                         verdict=('AA_PASS' if r['ci_low'] <= 1 <= r['ci_high'] else 'AA_FAIL')
                         if aa else classify(r))
                out.append(r)
    return out


def markdown(results, raw_path, boots):
    lines = [f'# Phase 5 bootstrap latency ratios', '',
             f'Source: `{raw_path}`; {boots} two-level bootstrap resamples (rounds, then requests); 95% percentile CI.',
             'Ratio = baseline median / candidate median (>1: candidate faster). '
             'WIN needs ratio >= 1.05 and CI low > 1. Prompt=192 full-vs-adaptive is the A/A control (identical capacity).', '',
             '| B / prompt / gen | baseline -> candidate | base ms | cand ms | ratio | 95% CI | per-round range | rounds x n | verdict |',
             '|---|---|---:|---:|---:|---|---|---|---|']
    for r in results:
        lines.append(f"| {r['batch']} / {r['prompt_tokens']} / {r['generated_tokens']} | {r['baseline']} -> {r['candidate']} "
                     f"| {r['baseline_median_ms']:.3f} | {r['candidate_median_ms']:.3f} | {r['ratio']:.3f} "
                     f"| [{r['ci_low']:.3f}, {r['ci_high']:.3f}] | {r['per_round_min']:.3f}-{r['per_round_max']:.3f} "
                     f"| {r['rounds']} x {r['requests_per_round']} | {r['verdict']} |")
    counts = defaultdict(int)
    for r in results:
        counts[(r['baseline'], r['candidate'], r['verdict'])] += 1
    lines += ['', '## Verdict counts (full denominator)', '']
    lines += [f'- {b} -> {c}: {v} x{n}' for (b, c, v), n in sorted(counts.items())]
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--boots', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    results = analyze(load_raw(args.raw), args.boots, args.seed)
    args.output.write_text(markdown(results, args.raw, args.boots), encoding='utf-8')
    args.output.with_suffix('.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    for r in results:
        print(f"{r['batch']:2} {r['prompt_tokens']:3} {r['generated_tokens']:2} {r['baseline']:>17} -> {r['candidate']:<14} "
              f"{r['ratio']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] {r['verdict']}")
    print(f'Wrote {args.output} and {args.output.with_suffix(".json")}')


if __name__ == '__main__':
    main()
