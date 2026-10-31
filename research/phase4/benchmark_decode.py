"""End-to-end greedy-decoding experiment. Run as a module from nanoGPT root.

Example:
python -m research.phase4.benchmark_decode --quick
python -m research.phase4.benchmark_decode

Only use checkpoints you created/trust: torch.load(weights_only=False) is unsafe
with untrusted files. No model weights or private prompts are written to CSV.
"""

import argparse
import csv
import math
import platform
import statistics
import time
from pathlib import Path

import torch
from model import GPT, GPTConfig
from tokenizers import Tokenizer
from .decoder import GraphDecoder, StaticKVDecoder, generate_reference


def percentile(samples, fraction):
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def make_prompts(tokenizer, vocab, batch, length, count=4):
    base = tokenizer.encode('Once upon a time, a little fox').ids
    if not base:
        raise ValueError('Tokenizer generated an empty input')
    prompts = []
    for request in range(count):
        rows = []
        for i in range(batch):
            words = (base * math.ceil(length / len(base)))[:length]
            words[-1] = (words[-1] + request + i) % vocab
            rows.append(words)
        prompts.append(torch.tensor(rows, dtype=torch.long, device='cuda'))
    return prompts


def load_model(path):
    if not path.exists():
        raise FileNotFoundError(f'Trusted local checkpoint not found: {path}')
    # This is safe ONLY for checkpoints you created and trust.
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = GPT(GPTConfig(**ckpt['model_args']))
    weights = {k.removeprefix('_orig_mod.'): v for k,v in ckpt['model'].items()}
    model.load_state_dict(weights)
    return model.eval().cuda()


def bench_workload(model, tokenizer, batch, length, generated, trials, requests):
    prompts = make_prompts(tokenizer, model.config.vocab_size, batch, length, requests)
    eager_static = StaticKVDecoder(model, batch)
    print(f'\nCAPTURING GRAPH: batch={batch}, prompt={length}, generated={generated}', flush=True)
    graph = GraphDecoder(model, prompts[0])

    runners = {
        'dynamic_kv_eager': lambda p: generate_reference(model, p, generated),
        'static_kv_eager': lambda p: eager_static.generate_eager(p, generated),
        'static_kv_graph': lambda p: graph.generate(p, generated),
    }
    print('VERIFYING outputs across changing inputs...', flush=True)
    for p in prompts:
        reference = runners['dynamic_kv_eager'](p)
        for name in ('static_kv_eager','static_kv_graph'):
            candidate = runners[name](p)
            torch.cuda.synchronize()
            if not torch.equal(reference, candidate):
                differences = (reference != candidate).sum().item()
                raise AssertionError(f'{name} produced {differences} differing tokens; no speedup claim is valid')
    print('GREEDY TOKEN CORRECTNESS: PASS', flush=True)

    for runner in runners.values():
        runner(prompts[0])
    torch.cuda.synchronize()

    latencies = {name: [] for name in runners}
    # Alternating method order controls some temporal bias.
    modes = list(runners)
    for trial in range(trials):
        order = modes[trial % len(modes):] + modes[:trial % len(modes)]
        if trial % 2:
            order.reverse()
        for name in order:
            for p in prompts:
                torch.cuda.synchronize()
                start = time.perf_counter()
                runners[name](p)
                torch.cuda.synchronize()
                latencies[name].append((time.perf_counter() - start) * 1000)

    results = []
    baseline = statistics.median(latencies['dynamic_kv_eager'])
    for name in modes:
        v = latencies[name]
        med = statistics.median(v)
        row = {
            'batch': batch, 'prompt_tokens': length, 'generated_tokens': generated,
            'method': name, 'num_requests': len(v),
            'p50_ms': round(med, 4),
            'p95_ms': round(percentile(v, .95), 4),
            'min_ms': round(min(v), 4), 'max_ms': round(max(v), 4),
            'generated_tokens_per_sec': round(batch * generated / (med / 1000), 2),
            'speedup_vs_dynamic_kv': round(baseline / med, 4),
            'graph_capture_ms': round(graph.capture_ms, 4) if name == 'static_kv_graph' else 0,
            'output_tokens_equal': True,
        }
        results.append(row)
        print(f"{name}: p50={med:.3f}ms  p95={row['p95_ms']:.3f}ms "
              f"speedup={row['speedup_vs_dynamic_kv']:.3f}x", flush=True)
    print(f'Graph capture cost (separate): {graph.capture_ms:.3f}ms', flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('out-tinystories-v2/baseline-2500.pt'))
    parser.add_argument('--tokenizer', type=Path, default=Path('data/tinystories_v2/tokenizer.json'))
    parser.add_argument('--output', type=Path, default=Path('research/phase4/decode_results.csv'))
    parser.add_argument('--trials', type=int, default=4)
    parser.add_argument('--requests', type=int, default=4)
    parser.add_argument('--quick', action='store_true', help='One batch-1 workload, 16 generated tokens')
    args = parser.parse_args()
    if args.trials < 2 or args.requests < 2:
        parser.error('Use at least 2 trials and 2 varying requests')
    if not torch.cuda.is_available():
        parser.error('CUDA GPU required for the graph comparison')
    if not args.tokenizer.exists():
        parser.error(f'Tokenizer not found: {args.tokenizer}')

    print('Local research benchmark — not production evidence', flush=True)
    print('PyTorch:',torch.__version__,'Python:',platform.python_version(), flush=True)
    print('GPU:',torch.cuda.get_device_name(0),flush=True)
    print('Only read and load checkpoints you created and trust.',flush=True)
    model = load_model(args.checkpoint)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    precision = torch.backends.cuda.matmul
    previous = precision.fp32_precision
    precision.fp32_precision = 'tf32'
    try:
        tests = [(1, 8, 16)] if args.quick else [(1, 8, 64), (16, 128, 64)]
        all_rows = []
        with torch.inference_mode():
            for batch, length, tokens in tests:
                if length + tokens > model.config.block_size:
                    raise ValueError('Requested workload exceeds model context window')
                all_rows.extend(bench_workload(
                    model, tokenizer, batch, length, tokens, args.trials, args.requests,
                ))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', newline='', encoding='utf-8') as fp:
            writer = csv.DictWriter(fp, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        print('\nPHASE 4 BENCHMARK COMPLETE',flush=True)
        print('CSV:', args.output.resolve(),flush=True)
        print('Caution: one GPU, synthetic prompts; full-load and deployment costs excluded.',flush=True)
    finally:
        precision.fp32_precision = previous


if __name__ == '__main__':
    main()
