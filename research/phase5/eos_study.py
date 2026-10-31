"""EOS behavior of the trained model on real story openings, and mask overhead.

    python -m research.phase5.eos_study --output research/phase5/ci_runs/<stamp>_eos/eos_study.json

Prompts are the first p tokens of validation stories (data/<set>/val.bin,
stories split at EOS = 0, only stories longer than p). Greedy decoding with
the exact adaptive graph pool + EOSDecoder, budget = block_size - p. Reports
per (batch, p): share of rows that emit EOS within budget, length quantiles,
and the UPPER BOUND on what early exit (plan option B) could save: decode
steps after the slowest row in each batch finished, ignoring prefill and the
cost of checking. Mask overhead is timed in-process, interleaved.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from ..phase4.benchmark_decode import load_model
from .adaptive_pool import DecoderPool
from .eos import EOSDecoder, generate_reference_fed_pad


def story_prompts(path, eos_id, p, count):
    data = np.fromfile(path, dtype=np.uint16)
    ends = np.flatnonzero(data == eos_id)
    starts = np.concatenate([[0], ends[:-1] + 1])
    picked = [data[s:s + p] for s, e in zip(starts, ends) if e - s > p][:count]
    if len(picked) < count:
        raise ValueError(f'Only {len(picked)} stories longer than {p} tokens')
    return torch.tensor(np.stack(picked).astype(np.int64))


def quantiles(values):
    q = np.percentile(values, [10, 50, 90])
    return [float(x) for x in q]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', type=Path, default=Path('out-tinystories-v2/baseline-2500.pt'))
    ap.add_argument('--val', type=Path, default=Path('data/tinystories_v2/val.bin'))
    ap.add_argument('--eos-id', type=int, default=0)
    ap.add_argument('--stories', type=int, default=128)
    ap.add_argument('--overhead-reps', type=int, default=40)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    model = load_model(args.checkpoint)
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    limit = model.config.block_size
    pool = DecoderPool(model, 'adaptive')
    out = {'checkpoint': str(args.checkpoint), 'eos_id': args.eos_id, 'stories': args.stories, 'configs': []}
    with torch.inference_mode():
        for p in (8, 32, 64):
            prompts = story_prompts(args.val, args.eos_id, p, args.stories).cuda()
            n = limit - p
            for batch in (1, 16):
                lengths, savings, emitted, checked = [], [], 0, 0
                for start in range(0, args.stories, batch):
                    x = prompts[start:start + batch]
                    dec = EOSDecoder(pool.get(x, n)[0], args.eos_id)
                    tokens, lens = dec.generate(x, n)
                    if checked < 2:  # exactness spot check on real EOS
                        ref, ref_len = generate_reference_fed_pad(model, x, n, args.eos_id)
                        if not (torch.equal(ref, tokens) and torch.equal(ref_len, lens)):
                            raise AssertionError(f'EOS mismatch at p={p}, batch={batch}, start={start}')
                        checked += 1
                    emitted += int((tokens[:, p:] == args.eos_id).any(1).sum())
                    lens = lens.tolist()
                    lengths += lens
                    savings.append(1 - max(lens) / n)
                cfg = {'batch': batch, 'prompt_tokens': p, 'budget': n, 'rows': len(lengths),
                       'rows_emitting_eos': emitted, 'eos_share': round(emitted / len(lengths), 4),
                       'length_p10_p50_p90': quantiles(lengths),
                       'early_exit_upper_bound_mean': round(statistics.mean(savings), 4),
                       'early_exit_upper_bound_p10_p50_p90': quantiles(savings)}
                out['configs'].append(cfg)
        # Mask overhead: same decoder, plain vs EOS-wrapped, interleaved (ABBA).
        x = story_prompts(args.val, args.eos_id, 8, 16).cuda()
        n = limit - 8
        inner = pool.get(x, n)[0]
        wrapped = EOSDecoder(inner, args.eos_id)
        times = {'plain': [], 'eos': []}
        for i in range(args.overhead_reps):
            for name in (('plain', 'eos') if i % 2 == 0 else ('eos', 'plain')):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                (inner if name == 'plain' else wrapped).generate(x, n)
                torch.cuda.synchronize()
                times[name].append(1000 * (time.perf_counter() - t0))
        plain, eos = statistics.median(times['plain']), statistics.median(times['eos'])
        out['mask_overhead'] = {'batch': 16, 'prompt_tokens': 8, 'budget': n, 'reps': args.overhead_reps,
                                'plain_median_ms': round(plain, 3), 'eos_median_ms': round(eos, 3),
                                'overhead_ms': round(eos - plain, 3), 'overhead_pct': round(100 * (eos / plain - 1), 3)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding='utf-8')
    for c in out['configs']:
        print(c)
    print(out['mask_overhead'])


if __name__ == '__main__':
    main()
