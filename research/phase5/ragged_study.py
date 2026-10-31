"""Ragged batching on real story prompts: cost and agreement with per-row decoding.

    python -m research.phase5.ragged_study --output research/phase5/ci_runs/<stamp>_ragged/run1.json

Each batch: 16 validation stories, prompt lengths drawn uniformly from
[--min-len, --max-len] (seeded), budget --count. Modes, timed interleaved in
one process (adaptive pools; block = batch):
  ragged     one RaggedGraphDecoder call for the whole batch
  per_row    16 sequential batch-1 calls of the exact uniform graph pool
             (the only exact-semantics option without ragged support)
  uniform    one uniform batch-16 call with every row at Pmax tokens
             (cost reference only: different prompts, different output)
Agreement: ragged rows vs per_row (== batch-1 generate_reference, bit-exact)
under the near-tie rule; exact-row rate and near-tie count are reported.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..phase4.benchmark_decode import load_model
from .adaptive_pool import DecoderPool
from .analyze_ci import ratio_ci
from .benchmark_inprocess import rotation
from .benchmark_review import near_tie_match
from .ragged import RaggedEngine, RaggedGraphDecoder


def story_starts(path, eos_id, min_tokens, count):
    data = np.fromfile(path, dtype=np.uint16)
    ends = np.flatnonzero(data == eos_id)
    starts = np.concatenate([[0], ends[:-1] + 1])
    picked = [data[s:e] for s, e in zip(starts, ends) if e - s > min_tokens][:count]
    if len(picked) < count:
        raise ValueError('Not enough long stories')
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', type=Path, default=Path('out-tinystories-v2/baseline-2500.pt'))
    ap.add_argument('--val', type=Path, default=Path('data/tinystories_v2/val.bin'))
    ap.add_argument('--batches', type=int, default=8)
    ap.add_argument('--requests', type=int, default=6)
    ap.add_argument('--min-len', type=int, default=8)
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--count', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    model = load_model(args.checkpoint)
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    rng = np.random.default_rng(args.seed)
    stories = story_starts(args.val, 0, args.max_len, 16 * args.batches)
    ragged_pool = DecoderPool(model, 'adaptive', factory=RaggedGraphDecoder)
    pool = DecoderPool(model, 'adaptive')
    n = args.count
    batches = []
    for i in range(args.batches):
        lengths = rng.integers(args.min_len, args.max_len + 1, 16)
        pmax = int(lengths.max())
        x = torch.tensor(np.stack([s[:pmax] for s in stories[16 * i:16 * i + 16]]).astype(np.int64)).cuda()
        batches.append((x, torch.tensor(lengths, device='cuda')))
    out = {'checkpoint': str(args.checkpoint), 'seed': args.seed, 'count': n,
           'length_range': [args.min_len, args.max_len], 'batches': []}
    rows = exact = ties = 0
    times = {'ragged': {}, 'per_row': {}, 'uniform': {}}
    with torch.inference_mode():
        for bi, (x, lengths) in enumerate(batches):
            dec = ragged_pool.get(x, n)[0]
            got = dec.generate(x, n, lengths)
            if not torch.equal(got, RaggedEngine(model, 16, dec.capacity).generate_eager(x, n, lengths)):
                raise AssertionError(f'ragged graph != eager engine, batch {bi}')
            singles = [(x[b:b + 1, :int(lengths[b])], pool.get(x[b:b + 1, :int(lengths[b])], n)[0])
                       for b in range(16)]
            per_row = torch.cat([d.generate(p, n)[:, p.size(1):] for p, d in singles])
            for b, (p, _) in enumerate(singles):
                ok, k = near_tie_match(model, torch.cat([p, per_row[b:b + 1]], 1), torch.cat([p, got[b:b + 1]], 1))
                if not ok:
                    raise AssertionError(f'ragged row outside near-tie rule: batch {bi} row {b}')
                ties += k
                exact += int(torch.equal(per_row[b], got[b]))
                rows += 1
            udec = pool.get(x, n)[0]
            run = {'ragged': lambda: dec.generate(x, n, lengths),
                   'per_row': lambda: [d.generate(p, n) for p, d in singles],
                   'uniform': lambda: udec.generate(x, n)}
            for fn in run.values():
                fn()
            for name in run:
                times[name][bi] = []
            for i in range(args.requests):
                for name in rotation(list(run), i + bi):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    run[name]()
                    torch.cuda.synchronize()
                    times[name][bi].append(1000 * (time.perf_counter() - t0))
            out['batches'].append({'lengths': lengths.tolist(), 'pmax': int(lengths.max()), 'capacity': dec.capacity})
    for base, cand in (('per_row', 'ragged'), ('uniform', 'ragged')):
        r = ratio_ci(times[base], times[cand], boots=10000)
        out[f'{base}->{cand}'] = {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}
        print(f"{base} -> {cand}: {r['baseline_median_ms']:.2f} ms / {r['candidate_median_ms']:.2f} ms = "
              f"{r['ratio']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]", flush=True)
    med = {k: float(np.median([t for v in d.values() for t in v])) for k, d in times.items()}
    out['tokens_per_s'] = {k: round(16 * n / (ms / 1000), 1) for k, ms in med.items()}
    out['agreement'] = {'rows': rows, 'exact_rows': exact, 'near_tie_rows': ties,
                        'exact_share': round(exact / rows, 4)}
    print('tokens/s (generated, median request):', out['tokens_per_s'])
    print('agreement vs per-row decoding:', out['agreement'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
