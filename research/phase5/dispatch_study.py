"""Dispatch grid: single_pass vs split-K decode attention over (batch, capacity) for the TinyStories heads (6 x 64).

    python -m research.phase5.dispatch_study --out runs/<stamp>/dispatch_grid.json

Uses the KernelAgent-Ada eval-v2 timing (20 calls per CUDA graph, interleaved trials, bootstrap CI). Positions are drawn
uniformly in [0, cap) per row (the average over a request's lifetime). Selection data for the shape rule only: the rule
is then judged by the end-to-end benchmark, not by this grid.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from ..kernelagent.evaluate import gpu_utilization, timing
from ..kernelagent.tasks import HEAD_DIM, HEADS
from ..phase5.analyze_ci import classify, ratio_ci
from .decode_attention import decode_attention, decode_attention_single_pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--batches', nargs='*', type=int, default=[1, 2, 4, 8, 16, 32])
    ap.add_argument('--caps', nargs='*', type=int, default=[32, 64, 128, 256])
    ap.add_argument('--trials', type=int, default=8)
    ap.add_argument('--position-mode', choices=['shared', 'per_row'], default='shared',
                    help='shared (default, as the graph decoders run): one scalar position for all rows, swept over '
                         'cap/8, 3cap/8, 5cap/8, 7cap/8; per_row: independent random positions (KA-04 grid; a kernel then '
                         'waits for its longest row, a different condition; independent review KA-06)')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    for b in args.batches:
        for cap in args.caps:
            g = torch.Generator(device='cuda').manual_seed(b * 1000 + cap)
            base = {'q': torch.randn(b, HEADS, 1, HEAD_DIM, generator=g, device='cuda'),
                    'k': torch.randn(b, HEADS, cap, HEAD_DIM, generator=g, device='cuda'),
                    'v': torch.randn(b, HEADS, cap, HEAD_DIM, generator=g, device='cuda')}
            if args.position_mode == 'per_row':
                positions = [torch.randint(0, cap, (b,), generator=g, device='cuda')]
            else:
                positions = [torch.tensor([max(0, (2 * j + 1) * cap // 8)], device='cuda') for j in range(4)]
            for pos in positions:
                rows.append(measure(b, cap, dict(base, position=pos), args.trials))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    for r in rows:  # lifetime-average cost per shape: sum of medians over the swept positions
        s = summary.setdefault(f"{r['batch']}x{r['cap']}", {'split_us': 0.0, 'single_us': 0.0, 'classes': []})
        s['split_us'] += r['split_us']
        s['single_us'] += r['single_us']
        s['classes'].append(r['class'])
    for k, s in summary.items():
        s['speedup_single'] = s['split_us'] / s['single_us']
        print(f"{k:7s} lifetime speedup_single x{s['speedup_single']:.2f} per-position {s['classes']}", flush=True)
    args.out.write_text(json.dumps({'env': {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)},
                                    'position_mode': args.position_mode, 'rows': rows, 'summary': summary},
                                   indent=1), encoding='utf-8')


def measure(b, cap, inp, trials):
    fns = {'split_k': lambda i: decode_attention(i['q'], i['k'], i['v'], i['position']),
           'single_pass': lambda i: decode_attention_single_pass(i['q'], i['k'], i['v'], i['position'])}
    err = float((fns['split_k'](inp) - fns['single_pass'](inp)).abs().max())
    time.sleep(2)  # nvidia-smi averages over its sample window: let our own previous shape drain first
    u = gpu_utilization()
    times, failed = timing(fns, inp, trials, 5, 10)
    r = ratio_ci(times['split_k'], times['single_pass'])  # > 1: single_pass faster
    row = {'batch': b, 'cap': cap, 'programs': b * HEADS, 'position': inp['position'].tolist(), 'gpu_before': u,
           'max_abs_diff': err, 'split_us': r['baseline_median_ms'] * 1e3, 'single_us': r['candidate_median_ms'] * 1e3,
           'speedup_single': r['ratio'], 'ci': [r['ci_low'], r['ci_high']], 'class': classify(r)}
    pos = row['position'] if len(row['position']) == 1 else 'per-row'
    print(f"B={b:2} cap={cap:3} pos={pos} split {row['split_us']:.2f}us single {row['single_us']:.2f}us "
          f"x{r['ratio']:.2f} [{r['ci_low']:.2f},{r['ci_high']:.2f}] {row['class']} util {u['util_pct']}", flush=True)
    return row


if __name__ == '__main__':
    main()
