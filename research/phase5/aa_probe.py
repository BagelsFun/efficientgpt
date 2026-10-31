"""In-process A/A probe: full vs adaptive pools at prompt=192 (same 256 capacity).

    python -m research.phase5.aa_probe --order full_first --output <file>.json

Builds eager-kernel and Inductor-compiled full/adaptive pools in ONE process,
validates them, then times their 256-capacity graphs interleaved request by
request (ABBA order per pair). Blocks of requests play the role of rounds in
analyze_ci.ratio_ci, so the CI covers block-to-block drift. --order decides
which pool type is created first, to separate pool type from allocation order.
Buffer addresses are recorded for the layout hypothesis.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from ..phase4.benchmark_decode import load_model, make_prompts
from ..phase4.decoder import generate_reference
from .adaptive_pool import DecoderPool
from .analyze_ci import classify, ratio_ci
from .benchmark_review import near_tie_match
from .compiled_baseline import CompiledGraphDecoder

CASES = ((1, 192, 32), (16, 192, 32))
# Requests from every bucket first, so the adaptive pool holds the same graphs as in the benchmark.
WARM = ((1, 8, 16), (1, 64, 64), (1, 128, 32), (16, 8, 16), (16, 64, 64), (16, 128, 32))
PAIRS = (('full_graph', 'adaptive_graph'), ('compiled_full_graph', 'compiled_adaptive_graph'))


def build(model, order):
    kinds = ('full', 'adaptive') if order == 'full_first' else ('adaptive', 'full')
    pools = {}
    for prefix, factory in (('', None), ('compiled_', CompiledGraphDecoder)):
        for kind in kinds:
            pools[f'{prefix}{kind}_graph'] = DecoderPool(model, kind, factory=factory)
    return pools


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--order', choices=('full_first', 'adaptive_first'), required=True)
    ap.add_argument('--blocks', type=int, default=10)
    ap.add_argument('--requests', type=int, default=30)
    ap.add_argument('--checkpoint', type=Path, default=Path('out-tinystories-v2/baseline-2500.pt'))
    ap.add_argument('--tokenizer', type=Path, default=Path('data/tinystories_v2/tokenizer.json'))
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    model = load_model(args.checkpoint)
    tok = Tokenizer.from_file(str(args.tokenizer))
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    pools = build(model, args.order)
    out = {'order': args.order, 'blocks': args.blocks, 'requests': args.requests, 'cases': {}}
    with torch.inference_mode():
        for batch, p, n in WARM + CASES:
            prompt = make_prompts(tok, model.config.vocab_size, batch, p, 1)[0]
            ref = generate_reference(model, prompt, n)
            for name, pool in pools.items():
                got = pool.get(prompt, n)[0].generate(prompt, n)
                ok = torch.equal(ref, got) if not name.startswith('compiled_') else near_tie_match(model, ref, got)[0]
                if not ok:
                    raise AssertionError(f'correctness: {name} {(batch, p, n)}')
        for case in CASES:
            batch, p, n = case
            prompts = make_prompts(tok, model.config.vocab_size, batch, p, 3)
            decs = {name: pool.get(prompts[0], n)[0] for name, pool in pools.items()}
            for d in decs.values():
                for _ in range(3):
                    d.generate(prompts[0], n)
            times = {name: {} for name in decs}
            for block in range(args.blocks):
                for name in decs:
                    times[name][block] = []
                for i in range(args.requests):
                    prompt = prompts[i % len(prompts)]
                    for a, b in PAIRS:
                        for name in ((a, b) if (i + block) % 2 == 0 else (b, a)):
                            torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            decs[name].generate(prompt, n)
                            torch.cuda.synchronize()
                            times[name][block].append(1000 * (time.perf_counter() - t0))
            entry = {'addresses': {name: {'key0_ptr_mod_2MiB': d.engine.keys[0].data_ptr() % (2 << 20),
                                          'capacity': int(d.engine.keys[0].size(2))}
                                   for name, d in decs.items()}}
            for a, b in PAIRS:
                r = ratio_ci(times[a], times[b], boots=10000)
                r['verdict'] = 'AA_PASS' if r['ci_low'] <= 1 <= r['ci_high'] else 'AA_FAIL'
                r['classify'] = classify(r)
                entry[f'{a}->{b}'] = r
                print(f"{args.order} B={batch} {p}+{n} {a} -> {b}: {r['ratio']:.4f} "
                      f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}] {r['verdict']} "
                      f"(medians {r['baseline_median_ms']:.3f} / {r['candidate_median_ms']:.3f} ms)", flush=True)
            out['cases'][f'{batch}_{p}_{n}'] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
