"""Does torch.compile(fullgraph) accept the step with user Triton kernels? Correctness + rough per-step timing + kernel profile."""
import sys, time, statistics, torch
sys.path.insert(0, '.')
from pathlib import Path
from torch.profiler import profile, ProfilerActivity
from research.phase4.benchmark_decode import load_model
from research.phase4.decoder import generate_reference
from research.phase5.adaptive_pool import DecoderPool
from research.phase5.benchmark_review import near_tie_match
from research.phase5.decode_attention import CompiledTritonGraphDecoder, TritonGraphDecoder
torch.backends.cuda.matmul.fp32_precision = 'tf32'
model = load_model(Path('out-tinystories-v2/baseline-2500.pt'))
g = torch.Generator().manual_seed(0)
for B in (1, 16):
    x = torch.randint(0, 4096, (B, 64), generator=g).cuda()
    t0 = time.perf_counter(); c = DecoderPool(model, 'full', factory=CompiledTritonGraphDecoder).get(x, 128)[0]; init = time.perf_counter() - t0
    t = DecoderPool(model, 'full', factory=TritonGraphDecoder).get(x, 128)[0]
    bad = 0
    with torch.inference_mode():
        for p, n in [(64, 128), (8, 200), (191, 1), (1, 255)]:
            y = torch.randint(0, 4096, (B, p), generator=g).cuda()
            bad += not near_tie_match(model, generate_reference(model, y, n), c.generate(y, n))[0]
        res = {}
        for name, d in (('triton', t), ('compiled_triton', c)):
            d.engine.reset(x); ts = []
            for _ in range(7):
                torch.cuda.synchronize(); s = time.perf_counter()
                for _ in range(100): d.graph.replay()
                torch.cuda.synchronize(); ts.append((time.perf_counter() - s) / 100 * 1000); d.engine.reset(x)
            res[name] = statistics.median(ts)
        # kernel profile of one compiled step (eager call of the compiled fn, not the graph)
        e = c.engine; e.reset(x); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(20): e.step()
            torch.cuda.synchronize()
        ev = [v for v in prof.key_averages() if v.device_type.name == 'CUDA']
        tot = sum(v.self_device_time_total for v in ev) / 20 / 1000
    print(f'B={B} failures={bad} init={init:.1f}s replay ms/step: triton={res["triton"]:.4f} compiled_triton={res["compiled_triton"]:.4f} '
          f'ratio={res["triton"]/res["compiled_triton"]:.3f}; compiled step GPU {tot:.3f} ms, kernels/step {sum(v.count for v in ev)/20:.0f}')
    for v in sorted(ev, key=lambda v: -v.self_device_time_total)[:6]:
        print(f'    {v.self_device_time_total/20/1000:.4f} ms x{v.count//20} {v.key[:80]}')
