"""A/B: HF generate with the default dynamic KV cache vs transformers' static cache (compiled decode), one process.

    python -m assistant.bench.static_cache_ab --out results/assistant/bench/<run>/static_cache_ab.json

Measures first-call cost, steady TTFT / decode tok/s (unique-prefix prompts, interleaved A/B), and whether greedy outputs
on the 52 quality prompts are identical; quality under static cache is graded with grading v1.1.
"""

import argparse
import json
import statistics
import time
import uuid
from pathlib import Path

from . import quality_v1
from .grading import grade


def _gpu_util():
    import subprocess
    try:
        return float(subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=10).stdout.split()[0])
    except Exception:
        return float('nan')


def paired_ratio_ci(a, b, boots=10000, seed=0):
    """median(a)/median(b) with a percentile bootstrap over paired repetitions (a[i], b[i] interleaved in one rep)."""
    import random
    rng = random.Random(seed)
    n = len(a)
    ratios = []
    for _ in range(boots):
        idx = [rng.randrange(n) for _ in range(n)]
        ratios.append(statistics.median(a[i] for i in idx) / statistics.median(b[i] for i in idx))
    ratios.sort()
    return {'ratio': statistics.median(a) / statistics.median(b), 'ci_low': ratios[int(0.025 * boots)],
            'ci_high': ratios[int(0.975 * boots) - 1], 'reps': n}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='Qwen/Qwen3-1.7B')
    ap.add_argument('--revision', default='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e')
    ap.add_argument('--quant', default='bf16')
    ap.add_argument('--repeats', type=int, default=10, help='>= 10: a bootstrap of medians over fewer reps is too coarse')
    ap.add_argument('--skip-equivalence', action='store_true', help='speed only (equivalence measured in ASSIST-03)')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    import torch
    from ..backends import HFBackend
    b = HFBackend(args.model, revision=args.revision, quant=args.quant)
    prompt = 'Explain in detail how a GPU executes a matrix multiplication, step by step.'
    msg = lambda: [{'role': 'user', 'content': f'[request {uuid.uuid4().hex}]\n{prompt}'}]
    out = {'model': args.model, 'revision': args.revision, 'quant': args.quant, 'first_call_s': {}, 'steady': {},
           'gpu_util_before_pct': _gpu_util()}
    for mode in (False, True):
        b.static_cache = mode
        t = time.perf_counter()
        b.chat(msg(), max_tokens=64)
        out['first_call_s']['static' if mode else 'dynamic'] = round(time.perf_counter() - t, 2)
        b.chat(msg(), max_tokens=64)  # second call: any recompilation for the new length shows up here, not in steady state
    stats = {'dynamic': [], 'static': []}
    for i in range(args.repeats):
        for mode in ((False, True) if i % 2 == 0 else (True, False)):
            b.static_cache = mode
            stats['static' if mode else 'dynamic'].append(b.chat(msg(), max_tokens=256).stats)
    for k, v in stats.items():
        out['steady'][k] = {'decode_tok_s': [round(s.decode_tok_s, 2) for s in v], 'decode_tok_s_median': statistics.median(s.decode_tok_s for s in v),
                            'ttft_s_median': statistics.median(s.ttft_s for s in v), 'completion_tokens': [s.completion_tokens for s in v]}
        print(k, json.dumps(out['steady'][k]), flush=True)
    out['decode_speedup_static'] = paired_ratio_ci([s.decode_tok_s for s in stats['static']],
                                                   [s.decode_tok_s for s in stats['dynamic']])
    out['ttft_ratio_static'] = paired_ratio_ci([s.ttft_s for s in stats['static']], [s.ttft_s for s in stats['dynamic']])
    out['gpu_util_after_pct'] = _gpu_util()
    print('decode speedup (static/dynamic)', out['decode_speedup_static'], flush=True)
    if args.skip_equivalence:
        out['torch_peak_allocated_mib'] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2), encoding='utf-8')
        return
    items = quality_v1.items()
    limits = {'code': 768, 'math': 512, 'mcq': 48}
    same, passed, t0 = 0, {'dynamic': 0, 'static': 0}, time.perf_counter()
    diffs = []
    for item in items:
        texts = {}
        for mode in (False, True):
            b.static_cache = mode
            texts['static' if mode else 'dynamic'] = b.chat([{'role': 'user', 'content': item['prompt']}], max_tokens=limits[item['category']]).text
        same += texts['dynamic'] == texts['static']
        if texts['dynamic'] != texts['static']:
            diffs.append(item['id'])
        for k, t in texts.items():
            passed[k] += grade(item, t)[0]
    out['equivalence'] = {'identical_responses': f'{same}/{len(items)}', 'differing_items': diffs,
                          'quality_v1.1': {k: f'{v}/{len(items)}' for k, v in passed.items()},
                          'seconds': round(time.perf_counter() - t0, 1)}
    out['torch_peak_allocated_mib'] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    print(json.dumps(out['equivalence']), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
