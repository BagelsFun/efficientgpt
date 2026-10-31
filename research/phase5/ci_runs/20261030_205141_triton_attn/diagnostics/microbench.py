"""Attention-only latency per layer call: masked SDPA (full cache) vs decode_attention, L2 flushed (cold) and hot."""
import sys, json, statistics, torch, torch.nn.functional as F
sys.path.insert(0, '.')
from research.phase5.decode_attention import decode_attention
torch.backends.cuda.matmul.fp32_precision = 'tf32'
flush = torch.empty(64 * 2**20 // 4, device='cuda')  # 64 MiB > 24 MiB L2
def timed(fn, cold, reps=60):
    ts = []
    for _ in range(reps):
        if cold: flush.fill_(1.0)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts) * 1000  # us
CONFIGS = {1: [(16, 16, 2), (32, 16, 2), (8, 16, 2), (32, 32, 4)], 16: [(64, 32, 4), (32, 32, 4), (128, 32, 4), (64, 64, 4), (128, 64, 8)]}
rows = []
for B in (1, 16):
    for cap in (128, 256):
        q = torch.randn(B, 6, 1, 64, device='cuda'); k = torch.randn(B, 6, cap, 64, device='cuda'); v = torch.randn_like(k)
        for L in sorted({16, 64, cap // 2, cap}):
            pos = torch.tensor([L - 1], device='cuda')
            allowed = torch.arange(cap, device='cuda').view(1, 1, 1, cap) <= pos.view(1, 1, 1, 1)
            sd = lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
            best = None
            for cfg in CONFIGS[B]:
                fn = lambda cfg=cfg: decode_attention(q, k, v, pos, cfg)
                fn(); t = timed(fn, True)
                if best is None or t < best[0]: best = (t, cfg)
            for cold in (True, False):
                t_sdpa = timed(sd, cold); t_ours = timed(lambda: decode_attention(q, k, v, pos, best[1]), cold)
                t_def = timed(lambda: decode_attention(q, k, v, pos), cold)
                useful = 2 * B * 6 * L * 64 * 4
                row = dict(B=B, cap=cap, L=L, cache='cold' if cold else 'hot', sdpa_us=round(t_sdpa, 1), default_cfg_us=round(t_def, 1),
                           best_cfg=best[1], best_us=round(t_ours, 1), speedup_default=round(t_sdpa / t_def, 2),
                           useful_GBps_default=round(useful / (t_def * 1e-6) / 1e9, 1))
                rows.append(row); print(row, flush=True)
