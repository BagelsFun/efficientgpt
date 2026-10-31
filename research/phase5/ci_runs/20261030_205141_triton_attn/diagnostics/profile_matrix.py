"""Per-step GPU time by kernel category: eager-kernel step vs Inductor step, B x capacity; plus replay time and bandwidth probe."""
import sys, json, torch, statistics, time
sys.path.insert(0, '.')
from pathlib import Path
from torch.profiler import profile, ProfilerActivity
from research.phase4.benchmark_decode import load_model
from research.phase4.compact_cache import CompactStaticKVDecoder
from research.phase4.decoder import StaticKVDecoder
from research.phase5.compiled_baseline import step_index_copy
torch.backends.cuda.matmul.fp32_precision = 'tf32'
model = load_model(Path('out-tinystories-v2/baseline-2500.pt'))
c = model.config
print('config', {k: getattr(c, k) for k in ('n_layer', 'n_head', 'n_embd', 'block_size', 'vocab_size')})
def category(name):
    n = name.lower()
    if 'fmha' in n or 'attention' in n or 'flash' in n: return 'attention'
    if 'gemm' in n or 'gemv' in n or 'splitk' in n or 'sm80_xmma' in n or 'cutlass' in n: return 'gemm'
    if 'layer_norm' in n: return 'layernorm'
    return 'other'
comp = torch.compile(step_index_copy, fullgraph=True, dynamic=False)
out = []
with torch.inference_mode():
    for B in (1, 16):
        for cap in (80, 128, 192, 256):
            x = torch.randint(0, c.vocab_size, (B, 8)).cuda()
            for name, fn in (('eager', lambda e: StaticKVDecoder.step.__wrapped__(e)), ('compiled', lambda e: comp(e))):
                e = CompactStaticKVDecoder(model, B, cap); e.reset(x)
                for _ in range(5): fn(e)
                e.reset(x); torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for _ in range(20): fn(e)
                    torch.cuda.synchronize()
                cats = {}
                for ev in prof.key_averages():
                    if ev.device_type.name == 'CUDA':
                        cats[category(ev.key)] = cats.get(category(ev.key), 0) + ev.self_device_time_total / 20 / 1000
                total = sum(cats.values())
                row = {'B': B, 'capacity': cap, 'step': name, 'gpu_ms': round(total, 4), **{k: round(v, 4) for k, v in sorted(cats.items())},
                       'attention_share': round(cats.get('attention', 0) / total, 3)}
                out.append(row); print(row, flush=True)
    # replay ground truth for the eager-kernel graph at each capacity (ms per step)
    from research.phase4.compact_cache import CompactGraphDecoder
    for B in (1, 16):
        for cap in (80, 128, 192, 256):
            x = torch.randint(0, c.vocab_size, (B, 8)).cuda()
            d = CompactGraphDecoder(model, x, cap - 8)
            d.engine.reset(x); ts = []
            for _ in range(5):
                torch.cuda.synchronize(); t = time.perf_counter()
                for _ in range(50): d.graph.replay()
                torch.cuda.synchronize(); ts.append((time.perf_counter() - t) / 50 * 1000)
                d.engine.reset(x)
            print({'B': B, 'capacity': cap, 'graph_replay_ms_per_step': round(statistics.median(ts), 4)}, flush=True)
    # achievable DRAM bandwidth (cache-cold, 512 MiB): copy and read-reduce
    a = torch.empty(128 * 2**20, device='cuda'); b = torch.empty_like(a); a.normal_()
    for label, op, bytes_moved in (('copy', lambda: b.copy_(a), 2 * a.numel() * 4), ('read_sum', lambda: a.sum(), a.numel() * 4)):
        for _ in range(3): op()
        torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(20): op()
        e.record(); torch.cuda.synchronize()
        print({'bandwidth_probe': label, 'GB_per_s': round(bytes_moved * 20 / (s.elapsed_time(e) / 1000) / 1e9, 1)}, flush=True)
