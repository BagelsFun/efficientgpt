"""GPU tests for the Phase 5 combinations: Triton attention x ragged, x sampling; KV state and EOS across lengths.

Contracts (docs/PHASE5_CONTRACT.md):
  * Triton ragged: every row within the near-tie rule vs the exact ragged engine and vs batch-1 decoding;
    pad values never influence output (bit-identical).
  * Triton sampling: NOT judged by token agreement. (1) chi-square vs exactly enumerated probabilities on a
    tiny vocabulary through the real graph path; (2) teacher-forced total-variation distance to the exact
    sampled step no larger than the TF32-vs-IEEE noise floor between two exact implementations.
  * KV state: a request decoded after a longer one equals the same request on a fresh decoder, bit for bit.
"""
import unittest
from pathlib import Path
import torch
from model import GPT, GPTConfig
from ..phase4.decoder import generate_reference, reference_step
from .adaptive_pool import DecoderPool

CHECKPOINT = Path('out-tinystories-v2/baseline-2500.pt')
try:
    import triton  # noqa: F401
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False
GPU = unittest.skipUnless(torch.cuda.is_available() and HAVE_TRITON, 'CUDA and Triton required')


def tiny(vocab=128, block=64, seed=242, scale=1.0):
    torch.manual_seed(seed)
    model = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=128, vocab_size=vocab,  # head dim 64 for the kernel
                          block_size=block, bias=True, dropout=0)).eval().cuda()
    with torch.no_grad():
        model.lm_head.weight.mul_(scale)
    return model


def triton_ragged_factory():
    from .decode_attention import decode_attention
    from .ragged import RaggedGraphDecoder
    return lambda model, prompt, capacity: RaggedGraphDecoder(model, prompt, capacity, decode_attention)


@GPU
class TestTritonRagged(unittest.TestCase):
    def run_mixes(self, model, mixes, caps):
        from .ragged import RaggedEngine, generate_reference_ragged
        from .test_ragged import ragged_match
        g = torch.Generator().manual_seed(41)
        pool = DecoderPool(model, 'adaptive', capacities=caps, factory=triton_ragged_factory())
        for lengths, n in mixes:
            lengths = torch.tensor(lengths, device='cuda')
            x = torch.randint(0, model.config.vocab_size, (len(lengths), int(lengths.max())), generator=g).cuda()
            with torch.inference_mode():
                dec = pool.get(x, n)[0]
                got = dec.generate(x, n, lengths)
                eager = RaggedEngine(model, len(lengths), dec.capacity).generate_eager(x, n, lengths)
                for ref in (eager, generate_reference_ragged(model, x, lengths, n)):
                    ok, _, _ = ragged_match(model, x, lengths, ref, got)
                    self.assertTrue(ok, (lengths.tolist(), n))
                other = x.clone()  # pad values must never matter
                for b, L in enumerate(lengths.tolist()):
                    other[b, L:] = (other[b, L:] + 7) % model.config.vocab_size
                self.assertTrue(torch.equal(got, dec.generate(other, n, lengths)))

    def test_tiny_model(self):
        self.run_mixes(tiny(), [([14, 3, 9, 1], 2), ([8, 8, 8, 8], 20), ([2, 30, 17, 5], 33), (list(range(1, 17)), 40)],
                       (16, 32, 48, 64))

    @unittest.skipUnless(CHECKPOINT.is_file(), 'trusted local checkpoint not present')
    def test_trained_checkpoint(self):
        from ..phase4.benchmark_decode import load_model
        self.run_mixes(load_model(CHECKPOINT), [([8, 64, 32, 17, 1, 120, 64, 5, 90, 33, 2, 64, 48, 11, 7, 100], 128),
                                                ([191], 1), ([8, 200], 40)], (80, 128, 192, 256))


def chi_square_p(observed, expected):
    from .test_sampling import chi_square_p as cs
    return cs(observed, expected)


@GPU
class TestTritonSampling(unittest.TestCase):
    """Distribution checks for the Triton-attention sampled graph (token agreement is not used as proof)."""

    V, N, PROMPT = 8, 100_000, [3, 1]

    def test_chi_square_vs_enumerated_distribution(self):
        from .decode_attention import decode_attention
        from .sampling import SamplingConfig, filtered_probs, sampled_factory
        # scale 1.0 keeps >= 9 cells with >= 5 expected counts in every config (sharper models collapse
        # some configs to one cell, which makes chi-square vacuous; df is asserted below)
        model = tiny(vocab=self.V, block=8, scale=1.0)
        x1 = torch.tensor([self.PROMPT], device='cuda')
        for i, cfg in enumerate([SamplingConfig(1.0), SamplingConfig(0.7, top_k=3), SamplingConfig(1.3, top_p=0.8),
                                 SamplingConfig(0.9, top_k=5, top_p=0.9)]):
            with torch.inference_mode():  # exact enumeration of all V*V two-token continuations (fp32 reference)
                first = filtered_probs(reference_step(model, x1)[0][:, -1, :], cfg)[0]
                nxt = torch.cat([x1.repeat(self.V, 1), torch.arange(self.V, device='cuda').view(-1, 1)], 1)
                exact = (first.view(-1, 1) * filtered_probs(reference_step(model, nxt)[0][:, -1, :], cfg)).flatten().double()
                pool = DecoderPool(model, 'full', factory=sampled_factory(cfg, decode_attention))
                gen = torch.Generator(device='cuda').manual_seed(300 + i)
                counts = torch.zeros(self.V * self.V, dtype=torch.float64, device='cuda')
                chunk = 20_000
                x = x1.repeat(chunk, 1)
                dec = pool.get(x, 2)[0]
                for _ in range(self.N // chunk):
                    tok = dec.generate(x, 2, torch.rand((chunk, 2), generator=gen, device='cuda'))
                    counts += torch.bincount(tok[:, -2] * self.V + tok[:, -1], minlength=self.V * self.V).double()
            self.assertEqual(float(counts[exact == 0].sum()), 0.0, cfg)
            p, stat, df = chi_square_p(counts.cpu(), (exact * self.N).cpu())
            self.assertGreaterEqual(df, 5, (cfg, 'distribution too concentrated for a meaningful test'))
            self.assertGreater(p, 1e-3, (cfg, stat, df))

    @unittest.skipUnless(CHECKPOINT.is_file(), 'trusted local checkpoint not present')
    def test_tv_within_precision_noise_floor(self):
        """Teacher-forced: TV(exact, triton) median and p99 <= those of TV(exact TF32, exact IEEE) on the same states."""
        from ..phase4.benchmark_decode import load_model
        from .compiled_baseline import decode_logits
        from .decode_attention import decode_attention
        from .sampling import SampledEngine, SamplingConfig, filtered_probs, sample_from
        model = load_model(CHECKPOINT)
        P = torch.backends.cuda.matmul
        old = P.fp32_precision
        x = torch.randint(0, 4096, (16, 32), generator=torch.Generator().manual_seed(5)).cuda()
        try:
            for cfg in (SamplingConfig(1.0), SamplingConfig(1.0, top_p=0.9)):
                tv = {}
                for label in ('triton', 'ieee'):
                    gen = torch.Generator(device='cuda').manual_seed(9)
                    with torch.inference_mode():
                        P.fp32_precision = 'tf32'
                        a = SampledEngine(model, 16, 128, cfg)
                        b = SampledEngine(model, 16, 128, cfg, decode_attention if label == 'triton' else None)
                        a.reset(x)
                        b.reset(x)
                        b.token.copy_(a.token)
                        vals = []
                        for _ in range(80):
                            P.fp32_precision = 'tf32'
                            pa = filtered_probs(decode_logits(a)[:, -1], cfg)
                            P.fp32_precision = 'tf32' if label == 'triton' else 'ieee'
                            pb = filtered_probs(decode_logits(b, attention=b.attention)[:, -1], cfg)
                            vals.append(0.5 * (pa.double() - pb.double()).abs().sum(-1))
                            tok = sample_from(pa, torch.rand(16, generator=gen, device='cuda'))
                            for e in (a, b):
                                e.history.index_copy_(1, e.position, e.token)
                                e.token.copy_(tok)
                                e.position.add_(1)
                    tv[label] = torch.cat(vals)
                self.assertLessEqual(float(tv['triton'].median()), float(tv['ieee'].median()), cfg)
                self.assertLessEqual(float(tv['triton'].quantile(0.99)), float(tv['ieee'].quantile(0.99)), cfg)
        finally:
            P.fp32_precision = old


@GPU
class TestStateAcrossLengths(unittest.TestCase):
    """A request after a longer one must equal the same request on a fresh decoder (no stale KV or position state)."""

    def test_fresh_vs_reused(self):
        from .decode_attention import CompiledTritonGraphDecoder, TritonGraphDecoder
        from .sampling import SamplingConfig, draw_uniforms, sampled_factory
        from .decode_attention import decode_attention
        model = tiny()
        g = torch.Generator().manual_seed(51)
        long_x = torch.randint(0, 128, (4, 40), generator=g).cuda()
        short_x = torch.randint(0, 128, (4, 3), generator=g).cuda()
        u_long = draw_uniforms(4, 23, torch.Generator(device='cuda').manual_seed(1))
        u_short = draw_uniforms(4, 30, torch.Generator(device='cuda').manual_seed(2))
        cfg = SamplingConfig(0.9, top_k=20, top_p=0.95)
        uniform = {
            'exact_graph': (None, False),
            'triton_graph': (TritonGraphDecoder, False),
            'compiled_triton_graph': (CompiledTritonGraphDecoder, False),
            'sampled_exact': (sampled_factory(cfg), True),
            'sampled_triton': (sampled_factory(cfg, decode_attention), True),
        }
        with torch.inference_mode():
            for name, (factory, sampled) in uniform.items():
                fresh = DecoderPool(model, 'full', factory=factory).get(short_x, 30)[0]
                reused = DecoderPool(model, 'full', factory=factory).get(long_x, 23)[0]
                args_long, args_short = ((u_long,), (u_short,)) if sampled else ((), ())
                reused.generate(long_x, 23, *args_long)
                self.assertTrue(torch.equal(fresh.generate(short_x, 30, *args_short),
                                            reused.generate(short_x, 30, *args_short)), name)
            for label, factory in (('ragged_exact', None), ('ragged_triton', triton_ragged_factory())):
                from .ragged import RaggedGraphDecoder
                factory = factory or RaggedGraphDecoder
                lengths_long = torch.tensor([40, 31, 2, 40], device='cuda')
                lengths_short = torch.tensor([1, 3, 2, 3], device='cuda')
                fresh = factory(model, short_x, 64)
                reused = factory(model, long_x, 64)
                reused.generate(long_x, 23, lengths_long)
                self.assertTrue(torch.equal(fresh.generate(short_x, 30, lengths_short),
                                            reused.generate(short_x, 30, lengths_short)), label)

    @unittest.skipUnless(CHECKPOINT.is_file() and Path('data/tinystories_v2/val.bin').is_file(),
                         'trusted local checkpoint or TinyStories validation split not present')
    def test_ragged_eos_lengths_match_per_row(self):
        """Ragged + EOS: wherever a ragged row equals per-row decoding, EOS lengths equal the feed-pad reference."""
        import numpy as np
        from ..phase4.benchmark_decode import load_model
        from .eos import generate_reference_fed_pad, mask_after_eos
        from .ragged import RaggedGraphDecoder
        from .ragged_study import story_starts
        model = load_model(CHECKPOINT)
        stories = story_starts('data/tinystories_v2/val.bin', 0, 64, 16)
        lengths = torch.tensor([8, 64, 32, 17, 12, 50, 64, 5, 40, 33, 20, 64, 48, 11, 7, 60], device='cuda')
        x = torch.tensor(np.stack([s[:64] for s in stories]).astype(np.int64)).cuda()
        n = 192
        compared = stops = 0
        with torch.inference_mode():
            gen = DecoderPool(model, 'full', factory=RaggedGraphDecoder).get(x, n)[0].generate(x, n, lengths)
            masked, lens = mask_after_eos(gen, 0, 0)
            for b, L in enumerate(lengths.tolist()):
                ref_tokens, ref_len = generate_reference_fed_pad(model, x[b:b + 1, :L], n, 0)
                if torch.equal(generate_reference(model, x[b:b + 1, :L], n)[0, L:], gen[b]):
                    compared += 1
                    stops += int(ref_len[0] < n)
                    self.assertEqual(int(lens[b]), int(ref_len[0]), b)
                    self.assertTrue(torch.equal(masked[b], ref_tokens[0, L:]), b)
        self.assertGreaterEqual(compared, 12)
        self.assertGreater(stops, 0)


if __name__ == '__main__':
    unittest.main()
