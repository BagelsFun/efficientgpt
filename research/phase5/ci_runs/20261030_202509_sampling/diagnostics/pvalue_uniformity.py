"""Real sampler: chi-square p-values over 20 seeds per config should look U(0,1); also the same for GPT.generate."""
import sys, torch
sys.path.insert(0, '.')
from research.phase5.test_sampling import TestDistribution, CONFIGS, chi_square_p
from research.phase5.sampling import generate_reference_sampled
t = TestDistribution(); TestDistribution.setUpClass()
x = torch.tensor([t.PROMPT]).repeat(t.N, 1)
for cfg in CONFIGS:
    exact = t.exact(cfg); ps = []
    for seed in range(20):
        u = torch.rand((t.N, 2), generator=torch.Generator().manual_seed(1000 + seed))
        ps.append(chi_square_p(t.counts(generate_reference_sampled(t.model, x, 2, cfg, u)), exact * t.N)[0])
    ps.sort(); below = sum(p < 0.05 for p in ps)
    print(f'ours T={cfg.temperature},k={cfg.top_k},p={cfg.top_p}: median p={ps[10]:.3f}, min={ps[0]:.2e}, {below}/20 below 0.05')
for cfg in [c for c in CONFIGS if c.top_p is None]:
    exact = t.exact(cfg); ps = []
    for seed in range(20):
        torch.manual_seed(2000 + seed)
        with torch.inference_mode():
            tok = t.model.generate(x, 2, temperature=cfg.temperature, top_k=cfg.top_k)
        ps.append(chi_square_p(t.counts(tok), exact * t.N)[0])
    ps.sort(); print(f'GPT.generate T={cfg.temperature},k={cfg.top_k}: median p={ps[10]:.3f}, min={ps[0]:.2e}, {sum(p < 0.05 for p in ps)}/20 below 0.05')
