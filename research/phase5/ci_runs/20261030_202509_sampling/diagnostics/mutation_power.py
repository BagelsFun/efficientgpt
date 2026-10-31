"""Power check for TestDistribution: p-values for the real sampler and for wrong transforms."""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, '.')
import research.phase5.sampling as S
from research.phase5.test_sampling import TestDistribution, CONFIGS, chi_square_p
from research.phase5.sampling import SamplingConfig
original = S.filtered_probs
def ignore_temperature(logits, cfg): return original(logits, SamplingConfig(1.0, cfg.top_k, cfg.top_p))
def top_k_plus_one(logits, cfg): return original(logits, SamplingConfig(cfg.temperature, None if cfg.top_k is None else cfg.top_k + 1, cfg.top_p))
def top_p_inclusive(logits, cfg):  # keeps one token too many: drops only where cumsum (incl. self) > p ... shifted by one
    if cfg.top_p is None: return original(logits, cfg)
    probs = original(logits, SamplingConfig(cfg.temperature, cfg.top_k, None))
    ordered, order = torch.sort(probs, dim=-1, descending=True, stable=True)
    drop_sorted = (ordered.cumsum(-1) - ordered) >= cfg.top_p
    drop_sorted = torch.cat([torch.zeros_like(drop_sorted[:, :1]), drop_sorted[:, :-1]], 1)
    drop = torch.zeros_like(drop_sorted).scatter(-1, order, drop_sorted)
    probs = probs.masked_fill(drop, 0.0); return probs / probs.sum(-1, keepdim=True)
t = TestDistribution(); TestDistribution.setUpClass()
x = torch.tensor([t.PROMPT]).repeat(t.N, 1)
for name, fn in [('real', original), ('ignore_temperature', ignore_temperature), ('top_k+1', top_k_plus_one), ('top_p_one_extra', top_p_inclusive)]:
    S.filtered_probs = fn
    out = []
    for i, cfg in enumerate(CONFIGS):
        u = torch.rand((t.N, 2), generator=torch.Generator().manual_seed(100 + i))
        obs = t.counts(S.generate_reference_sampled(t.model, x, 2, cfg, u))
        exact = t.exact(cfg)
        zero_hits = int(obs[exact == 0].sum())
        p, stat, df = chi_square_p(obs, exact * t.N)
        out.append(f"T={cfg.temperature},k={cfg.top_k},p={cfg.top_p}: p={p:.2e} zero-cell hits={zero_hits}")
    print(name.ljust(20), ' | '.join(out))
S.filtered_probs = original
