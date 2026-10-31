"""Correctness + rough speed of compiled-step CUDA graphs vs hand-captured eager graphs."""
import sys, time, statistics, torch
sys.path.insert(0, '.')
from pathlib import Path
from research.phase4.benchmark_decode import load_model
from research.phase4.decoder import generate_reference
from research.phase5.adaptive_pool import DecoderPool
from research.phase5.benchmark_review import near_tie_match
from research.phase5.compiled_baseline import CompiledGraphDecoder
torch.backends.cuda.matmul.fp32_precision = 'tf32'
model = load_model(Path('out-tinystories-v2/baseline-2500.pt'))
V = model.config.vocab_size
make = lambda m, prompt, cap: CompiledGraphDecoder(m, prompt, cap)
g = torch.Generator().manual_seed(0)
def timeit(dec, x, n):
    ts = []
    for _ in range(15):
        torch.cuda.synchronize(); t = time.perf_counter(); dec.generate(x, n); torch.cuda.synchronize()
        ts.append(1000 * (time.perf_counter() - t))
    return statistics.median(ts)
for kind in ('full', 'adaptive'):
    cpool, epool = DecoderPool(model, kind, factory=make), DecoderPool(model, kind)
    t0 = time.perf_counter(); bad = ties = seqs = 0
    with torch.inference_mode():
        for b, p, n in [(1, 70, 10), (1, 8, 16), (1, 1, 79), (1, 100, 28), (1, 191, 1), (1, 3, 253),
                        (16, 70, 10), (16, 5, 20), (16, 64, 64), (16, 128, 64), (16, 192, 32)]:
            for _ in range(2):
                x = torch.randint(0, V, (b, p), generator=g).cuda()
                dec, cap = cpool.get(x, n)
                ok, k = near_tie_match(model, generate_reference(model, x, n), dec.generate(x, n))
                bad += not ok; ties += k; seqs += b
        init = time.perf_counter() - t0
        for b, p, n in [(1, 64, 64), (16, 64, 64), (16, 128, 64)]:
            x = torch.randint(0, V, (b, p), generator=g).cuda()
            c, _ = cpool.get(x, n); e, _ = epool.get(x, n)
            print(f'{kind} B={b} {p}+{n}: compiled-graph {timeit(c, x, n):.2f}ms  eager-graph {timeit(e, x, n):.2f}ms')
    print(f'{kind}: graphs={cpool.graph_count} sequences={seqs} failures={bad} near_ties={ties} init={init:.1f}s')
