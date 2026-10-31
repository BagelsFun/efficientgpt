"""CPU tests for Phase 5 sampling: transforms, inverse CDF, exactness given u, distributions."""
import unittest
import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
from ..phase4.decoder import generate_reference, reference_step
from .sampling import (SampledEngine, SamplingConfig, filtered_probs, generate_reference_sampled,
                       sample_from)

CONFIGS = [SamplingConfig(1.0), SamplingConfig(0.7, top_k=3), SamplingConfig(1.3, top_p=0.8),
           SamplingConfig(0.9, top_k=5, top_p=0.9)]


def tiny_model(vocab, block, scale=1.0, seed=242):
    torch.manual_seed(seed)
    model = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=16, vocab_size=vocab,
                          block_size=block, bias=True, dropout=0)).eval()
    with torch.no_grad():
        model.lm_head.weight.mul_(scale)  # tied with wte: sharpens the next-token distribution
    return model


class TestTransforms(unittest.TestCase):
    def test_top_p_keeps_smallest_prefix(self):
        logits = torch.log(torch.tensor([[0.15, 0.5, 0.05, 0.3]]))
        for p, kept in [(0.5, {1}), (0.6, {1, 3}), (0.95, {0, 1, 3}), (0.99, {0, 1, 2, 3}), (1e-9, {1})]:
            probs = filtered_probs(logits, SamplingConfig(1.0, top_p=p))
            self.assertEqual(set(torch.nonzero(probs[0]).flatten().tolist()), kept, p)
            self.assertAlmostEqual(float(probs.sum()), 1.0, places=6)

    def test_top_k_matches_gpt_generate_masking(self):
        g = torch.Generator().manual_seed(0)
        logits = torch.randn(5, 50, generator=g)
        logits[0, :4] = 3.0  # ties at the k-th value are kept, as in GPT.generate
        for t, k in [(1.0, 1), (0.8, 4), (1.5, 10), (1.0, 50), (1.0, 80)]:
            ref = logits / t
            v, _ = torch.topk(ref, min(k, ref.size(-1)))
            ref = ref.masked_fill(ref < v[:, [-1]], float('-inf'))
            self.assertTrue(torch.equal(F.softmax(ref, dim=-1), filtered_probs(logits, SamplingConfig(t, top_k=k))))

    def test_temperature_only_is_softmax(self):
        logits = torch.randn(3, 20, generator=torch.Generator().manual_seed(1))
        self.assertTrue(torch.allclose(filtered_probs(logits, SamplingConfig(0.5)), F.softmax(logits / 0.5, -1)))

    def test_invalid_configs(self):
        for kw in [dict(temperature=0), dict(top_k=0), dict(top_p=0), dict(top_p=1.5)]:
            with self.assertRaises(ValueError):
                SamplingConfig(**kw)


class TestInverseCDF(unittest.TestCase):
    def test_mapping_skips_zero_mass(self):
        probs = torch.tensor([[0.25, 0.0, 0.75, 0.0]]).repeat(5, 1)
        u = torch.tensor([0.0, 0.2499, 0.25, 0.9999, 1 - 2 ** -24])
        self.assertEqual(sample_from(probs, u).flatten().tolist(), [0, 0, 2, 2, 2])

    def test_never_picks_zero_probability(self):
        g = torch.Generator().manual_seed(2)
        probs = F.softmax(torch.randn(64, 30, generator=g), -1)
        probs[:, ::3] = 0
        probs = probs / probs.sum(-1, keepdim=True)
        for _ in range(20):
            tokens = sample_from(probs, torch.rand(64, generator=g))
            self.assertTrue(bool((probs.gather(1, tokens) > 0).all()))


class TestExactGivenUniforms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tiny_model(64, 48, scale=3.0)

    def test_engine_matches_reference(self):
        g = torch.Generator().manual_seed(4)
        for cfg in CONFIGS:
            for batch, p, n in [(1, 1, 20), (3, 7, 15), (4, 30, 17)]:
                engine = SampledEngine(self.model, batch, p + n, cfg)
                for _ in range(2):  # reuse the engine with new prompts and draws
                    x = torch.randint(0, 64, (batch, p), generator=g)
                    u = torch.rand((batch, n), generator=g)
                    expected = generate_reference_sampled(self.model, x, n, cfg, u)
                    self.assertTrue(torch.equal(expected, engine.generate_eager(x, n, u)), (cfg, batch, p, n))

    def test_greedy_limits(self):
        g = torch.Generator().manual_seed(5)
        x = torch.randint(0, 64, (3, 5), generator=g)
        greedy = generate_reference(self.model, x, 12)
        for cfg in (SamplingConfig(1.0, top_k=1), SamplingConfig(2.0, top_p=1e-9)):
            for u in (torch.zeros(3, 12), torch.rand((3, 12), generator=g), torch.full((3, 12), 1 - 2 ** -24)):
                self.assertTrue(torch.equal(greedy, generate_reference_sampled(self.model, x, 12, cfg, u)), cfg)


def chi_square_p(observed, expected):
    """Pearson chi-square p-value; cells with expected < 5 are merged into one bin."""
    small = expected < 5
    obs = torch.cat([observed[~small], observed[small].sum().view(1)])
    exp = torch.cat([expected[~small], expected[small].sum().view(1)])
    keep = exp > 0
    obs, exp = obs[keep], exp[keep]
    stat = float(((obs - exp) ** 2 / exp).sum())
    df = obs.numel() - 1
    return float(torch.special.gammaincc(torch.tensor(df / 2, dtype=torch.float64),
                                         torch.tensor(stat / 2, dtype=torch.float64))), stat, df


class TestDistribution(unittest.TestCase):
    """Enumerate all 2-token continuations (V=8) exactly; sample 100k sequences; chi-square."""
    V, N, PROMPT = 8, 100_000, [3, 1]

    @classmethod
    def setUpClass(cls):
        cls.model = tiny_model(cls.V, 8, scale=4.0)

    def exact(self, cfg):
        x = torch.tensor([self.PROMPT])
        with torch.inference_mode():
            first = filtered_probs(reference_step(self.model, x)[0][:, -1, :], cfg)[0]
            nxt = torch.cat([x.repeat(self.V, 1), torch.arange(self.V).view(-1, 1)], 1)
            second = filtered_probs(reference_step(self.model, nxt)[0][:, -1, :], cfg)
        return (first.view(-1, 1) * second).flatten().double()

    def counts(self, tokens):
        cells = tokens[:, -2] * self.V + tokens[:, -1]
        return torch.bincount(cells, minlength=self.V * self.V).double()

    def test_sampler_matches_exact_distribution(self):
        x = torch.tensor([self.PROMPT]).repeat(self.N, 1)
        for i, cfg in enumerate(CONFIGS):
            probs = self.exact(cfg)
            u = torch.rand((self.N, 2), generator=torch.Generator().manual_seed(100 + i))
            observed = self.counts(generate_reference_sampled(self.model, x, 2, cfg, u))
            self.assertEqual(float(observed[probs == 0].sum()), 0.0, cfg)
            p, stat, df = chi_square_p(observed, probs * self.N)
            self.assertGreaterEqual(df, 5, (cfg, 'distribution too concentrated for a meaningful test'))
            self.assertGreater(p, 1e-3, (cfg, stat, df))

    def test_gpt_generate_matches_same_distribution(self):
        """nanoGPT's multinomial sampler agrees with our transforms (temperature, top-k only)."""
        x = torch.tensor([self.PROMPT]).repeat(self.N, 1)
        for i, cfg in enumerate(c for c in CONFIGS if c.top_p is None):
            torch.manual_seed(200 + i)
            with torch.inference_mode():
                tokens = self.model.generate(x, 2, temperature=cfg.temperature, top_k=cfg.top_k)
            p, stat, df = chi_square_p(self.counts(tokens), self.exact(cfg) * self.N)
            self.assertGreater(p, 1e-3, (cfg, stat, df))


if __name__ == '__main__':
    unittest.main()
