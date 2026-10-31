"""Sampling cost vs greedy (in-process, interleaved) and EOS behavior under sampling.

    python -m research.phase5.sampling_study --output research/phase5/ci_runs/<stamp>_sampling/run1.json

Prompts: first 32 tokens of validation stories (as eos_study). Exact adaptive
graph pools: greedy vs SampledGraphDecoder for each config. Each timed sampled
request includes drawing its uniforms. Ratios are greedy/sampled medians
(< 1: sampling is slower) with a block bootstrap (analyze_ci.ratio_ci).
EOS stats use EOSDecoder (eos 0) over all stories with budget 256 - 32.
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
from .eos import EOSDecoder
from .eos_study import quantiles, story_prompts
from .sampling import SamplingConfig, draw_uniforms, generate_reference_sampled, sampled_factory

CONFIGS = {'T0.8_k50': SamplingConfig(0.8, top_k=50), 'T1.0_p0.9': SamplingConfig(1.0, top_p=0.9),
           'T0.8_k50_p0.95': SamplingConfig(0.8, top_k=50, top_p=0.95)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', type=Path, default=Path('out-tinystories-v2/baseline-2500.pt'))
    ap.add_argument('--val', type=Path, default=Path('data/tinystories_v2/val.bin'))
    ap.add_argument('--blocks', type=int, default=8)
    ap.add_argument('--requests', type=int, default=10)
    ap.add_argument('--stories', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    model = load_model(args.checkpoint)
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    gen = torch.Generator(device='cuda').manual_seed(args.seed)
    prompts = story_prompts(args.val, 0, 32, args.stories).cuda()
    pools = {'greedy': DecoderPool(model, 'adaptive')}
    pools.update({name: DecoderPool(model, 'adaptive', factory=sampled_factory(cfg)) for name, cfg in CONFIGS.items()})
    out = {'checkpoint': str(args.checkpoint), 'seed': args.seed, 'timing': [], 'eos': {}}
    with torch.inference_mode():
        for batch, n in ((1, 128), (16, 128), (16, 224)):
            x = prompts[:batch]
            decs = {name: pool.get(x, n)[0] for name, pool in pools.items()}
            for name, cfg in CONFIGS.items():  # exactness spot check on real prompts
                u = draw_uniforms(batch, n, gen)
                if not torch.equal(generate_reference_sampled(model, x, n, cfg, u), decs[name].generate(x, n, u)):
                    raise AssertionError(f'sampled mismatch {name} B={batch} n={n}')
            run = {name: (lambda d=d: d.generate(x, n)) for name, d in decs.items() if name == 'greedy'}
            run.update({name: (lambda d=decs[name]: d.generate(x, n, draw_uniforms(batch, n, gen))) for name in CONFIGS})
            for fn in run.values():
                for _ in range(3):
                    fn()
            times = {name: {} for name in run}
            for block in range(args.blocks):
                for name in run:
                    times[name][block] = []
                for i in range(args.requests):
                    for name in rotation(list(run), i + block):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        run[name]()
                        torch.cuda.synchronize()
                        times[name][block].append(1000 * (time.perf_counter() - t0))
            for name in CONFIGS:
                r = ratio_ci(times['greedy'], times[name], boots=10000)
                row = {'batch': batch, 'prompt_tokens': 32, 'budget': n, 'config': name,
                       'greedy_ms': round(r['baseline_median_ms'], 3), 'sampled_ms': round(r['candidate_median_ms'], 3),
                       'greedy_over_sampled': round(r['ratio'], 4),
                       'ci': [round(r['ci_low'], 4), round(r['ci_high'], 4)],
                       'overhead_pct': round(100 * (r['candidate_median_ms'] / r['baseline_median_ms'] - 1), 2)}
                out['timing'].append(row)
                print(row, flush=True)
        n = 256 - 32
        for name in ['greedy'] + list(CONFIGS):
            lengths, emitted = [], 0
            for start in range(0, args.stories, 16):
                x = prompts[start:start + 16]
                dec = EOSDecoder(pools[name].get(x, n)[0], 0)
                tokens, lens = dec.generate(x, n) if name == 'greedy' else dec.generate(x, n, draw_uniforms(16, n, gen))
                emitted += int((tokens[:, 32:] == 0).any(1).sum())
                lengths += lens.tolist()
            out['eos'][name] = {'rows': len(lengths), 'eos_share': round(emitted / len(lengths), 4),
                                'length_p10_p50_p90': quantiles(np.array(lengths))}
            print(name, out['eos'][name], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
