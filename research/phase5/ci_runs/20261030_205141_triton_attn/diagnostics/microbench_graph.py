"""Attention-only GPU time per layer call, measured from CUDA Graph replays (no host launch overhead).
hot: graph of N attention calls. cold: graph of N (L2 flush + attention) minus graph of N flushes."""
import sys, statistics, torch, torch.nn.functional as F
sys.path.insert(0, '.')
from research.phase5.decode_attention import decode_attention
torch.backends.cuda.matmul.fp32_precision = 'tf32'
N = 20
flush = torch.empty(64 * 2**20 // 4, device='cuda')
def graph_us(body, reps=15):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): body()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): body()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts) * 1000 / N
flush_us = graph_us(lambda: [flush.fill_(1.0) for _ in range(N)])
print({'flush_us': round(flush_us, 1)})
for B in (1, 16):
    for cap in (128, 256):
        q = torch.randn(B, 6, 1, 64, device='cuda'); k = torch.randn(B, 6, cap, 64, device='cuda'); v = torch.randn_like(k)
        for L in sorted({16, 64, cap // 2, cap}):
            pos = torch.tensor([L - 1], device='cuda')
            allowed = torch.arange(cap, device='cuda').view(1, 1, 1, cap) <= pos.view(1, 1, 1, 1)
            sd = lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
            ours = lambda: decode_attention(q, k, v, pos)
            row = {'B': B, 'cap': cap, 'L': L}
            for name, fn in (('sdpa', sd), ('triton', ours)):
                row[f'{name}_hot_us'] = round(graph_us(lambda: [fn() for _ in range(N)]), 1)
                row[f'{name}_cold_us'] = round(graph_us(lambda: [(flush.fill_(1.0), fn()) for _ in range(N)]) - flush_us, 1)
            row['speedup_hot'] = round(row['sdpa_hot_us'] / row['triton_hot_us'], 2)
            row['speedup_cold'] = round(row['sdpa_cold_us'] / row['triton_cold_us'], 2)
            row['triton_cold_useful_GBps'] = round(2 * B * 6 * L * 64 * 4 / (row['triton_cold_us'] * 1e-6) / 1e9, 1)
            print(row, flush=True)
