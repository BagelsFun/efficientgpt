"""CUDA Graph pool exactness under adversarial reuse; skipped without CUDA.

Unlike the benchmark's validation prompts (one sentence repeated, last token
varied), these use uniformly random tokens and deliberately reuse one graph
long-prompt-then-short, fill buckets exactly, and request a single token.
"""
import unittest
from pathlib import Path
import torch
from model import GPT, GPTConfig
from ..phase4.decoder import generate_reference
from .adaptive_pool import DecoderPool

CHECKPOINT = Path('out-tinystories-v2/baseline-2500.pt')
POLICIES = ('tf32', 'ieee')


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for graph capture')
class TestGraphPoolExact(unittest.TestCase):
    def setUp(self):
        self.old_policy = torch.backends.cuda.matmul.fp32_precision

    def tearDown(self):
        torch.backends.cuda.matmul.fp32_precision = self.old_policy

    def check(self, model, mode, sequence, capacities, seeds=(101, 202)):
        for policy in POLICIES:
            torch.backends.cuda.matmul.fp32_precision = policy
            for seed in seeds:
                g = torch.Generator().manual_seed(seed)
                pool = DecoderPool(model, mode, capacities=capacities)
                for batch, p, n in sequence:
                    for _ in range(2):
                        x = torch.randint(0, model.config.vocab_size, (batch, p), generator=g).cuda()
                        dec, _ = pool.get(x, n)
                        with torch.inference_mode():
                            got = dec.generate(x, n)
                            expected = generate_reference(model, x, n)
                        self.assertTrue(torch.equal(expected, got), (mode, policy, seed, batch, p, n))

    def test_tiny_model(self):
        torch.manual_seed(242)
        model = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=32, vocab_size=128,
                              block_size=64, bias=True, dropout=0)).eval().cuda()
        # Long-then-short in each bucket, exact bucket edges, n=1, batch 4.
        sequence = [(1, 14, 2), (1, 3, 5), (1, 1, 15), (1, 15, 1), (1, 20, 12),
                    (1, 2, 30), (1, 63, 1), (1, 1, 63), (4, 14, 2), (4, 5, 27)]
        for mode in ('full', 'adaptive'):
            self.check(model, mode, sequence, capacities=(16, 32, 48, 64))

    @unittest.skipUnless(CHECKPOINT.is_file(), 'trusted local checkpoint not present')
    def test_trained_checkpoint(self):
        from ..phase4.benchmark_decode import load_model
        model = load_model(CHECKPOINT)
        sequence = [(1, 70, 10), (1, 8, 16), (1, 1, 79), (1, 79, 1), (1, 100, 28),
                    (1, 9, 100), (1, 191, 1), (1, 3, 253), (1, 255, 1),
                    (16, 70, 10), (16, 5, 20), (16, 33, 95)]
        for mode in ('full', 'adaptive'):
            self.check(model, mode, sequence, capacities=(80, 128, 192, 256), seeds=(101,))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for graph capture')
class TestGraphPoolEOS(unittest.TestCase):
    """EOS-wrapped graph pools == feed-pad reference, exactly, including early/simultaneous stops."""

    def check(self, model, sequence, capacities, eos_ids):
        from .eos import EOSDecoder, generate_reference_fed_pad
        g = torch.Generator().manual_seed(11)
        stops = 0
        for mode in ('full', 'adaptive'):
            pool = DecoderPool(model, mode, capacities=capacities)
            for batch, p, n in sequence:
                x = torch.randint(0, model.config.vocab_size, (batch, p), generator=g).cuda()
                with torch.inference_mode():
                    plain = generate_reference(model, x, n)
                    # Real EOS plus the most frequent generated token, so stops actually occur.
                    for eos in eos_ids + [int(plain[:, p:].flatten().mode().values)]:
                        expected, exp_len = generate_reference_fed_pad(model, x, n, eos)
                        got, got_len = EOSDecoder(pool.get(x, n)[0], eos).generate(x, n)
                        self.assertTrue(torch.equal(expected, got), (mode, batch, p, n, eos))
                        self.assertTrue(torch.equal(exp_len, got_len), (mode, batch, p, n, eos))
                        stops += int((got_len < n).sum())
        self.assertGreater(stops, 0)

    def test_tiny_model(self):
        torch.manual_seed(242)
        model = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=32, vocab_size=16,
                              block_size=64, bias=True, dropout=0)).eval().cuda()
        self.check(model, [(1, 14, 2), (4, 3, 29), (4, 1, 63), (1, 20, 12), (16, 5, 40)],
                   (16, 32, 48, 64), [0])

    @unittest.skipUnless(CHECKPOINT.is_file(), 'trusted local checkpoint not present')
    def test_trained_checkpoint(self):
        from ..phase4.benchmark_decode import load_model
        self.check(load_model(CHECKPOINT), [(1, 8, 200), (16, 8, 240), (16, 64, 128), (1, 191, 1)],
                   (80, 128, 192, 256), [0])


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for graph capture')
class TestSampledGraphPool(unittest.TestCase):
    """Sampled full/adaptive graph pools == sampled reference token for token given the same uniforms."""

    def check(self, model, sequence, capacities, cfgs):
        from .eos import EOSDecoder, mask_after_eos
        from .sampling import draw_uniforms, generate_reference_sampled, sampled_factory
        gen = torch.Generator(device='cuda').manual_seed(21)
        g = torch.Generator().manual_seed(22)
        for cfg in cfgs:
            for mode in ('full', 'adaptive'):
                pool = DecoderPool(model, mode, capacities=capacities, factory=sampled_factory(cfg))
                for batch, p, n in sequence:
                    x = torch.randint(0, model.config.vocab_size, (batch, p), generator=g).cuda()
                    u = draw_uniforms(batch, n, gen)
                    with torch.inference_mode():
                        expected = generate_reference_sampled(model, x, n, cfg, u)
                        dec = pool.get(x, n)[0]
                        self.assertTrue(torch.equal(expected, dec.generate(x, n, u)), (cfg, mode, batch, p, n))
                        got, lengths = EOSDecoder(dec, 0).generate(x, n, u)
                        exp, exp_len = mask_after_eos(expected, p, 0)
                        self.assertTrue(torch.equal(exp, got) and torch.equal(exp_len, lengths), (cfg, mode))

    def test_tiny_model(self):
        from .sampling import SamplingConfig
        torch.manual_seed(242)
        model = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=32, vocab_size=64,
                              block_size=64, bias=True, dropout=0)).eval().cuda()
        cfgs = [SamplingConfig(1.0), SamplingConfig(0.7, top_k=5), SamplingConfig(1.3, top_p=0.8),
                SamplingConfig(0.9, top_k=10, top_p=0.9)]
        self.check(model, [(1, 14, 2), (1, 3, 5), (4, 1, 63), (1, 63, 1), (16, 5, 40)], (16, 32, 48, 64), cfgs)

    @unittest.skipUnless(CHECKPOINT.is_file(), 'trusted local checkpoint not present')
    def test_trained_checkpoint(self):
        from ..phase4.benchmark_decode import load_model
        from .sampling import SamplingConfig, draw_uniforms, sampled_factory
        model = load_model(CHECKPOINT)
        cfgs = [SamplingConfig(0.8, top_k=50), SamplingConfig(1.0, top_p=0.9)]
        self.check(model, [(1, 8, 200), (16, 8, 120), (16, 64, 64), (1, 191, 1)], (80, 128, 192, 256), cfgs)
        # Different draws must give different text (guards against u being ignored).
        x = torch.randint(0, model.config.vocab_size, (16, 8), generator=torch.Generator().manual_seed(5)).cuda()
        gen = torch.Generator(device='cuda').manual_seed(6)
        dec = DecoderPool(model, 'adaptive', factory=sampled_factory(cfgs[1])).get(x, 64)[0]
        with torch.inference_mode():
            a = dec.generate(x, 64, draw_uniforms(16, 64, gen))
            b = dec.generate(x, 64, draw_uniforms(16, 64, gen))
        self.assertFalse(torch.equal(a, b))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for graph capture')
class TestRaggedGraphPool(unittest.TestCase):
    """Ragged graph pools: == eager engine exactly, == uniform pools when lengths match, per-row near-tie."""

    def check(self, model, batches, capacities):
        from .ragged import RaggedEngine, RaggedGraphDecoder, generate_reference_ragged
        from .test_ragged import ragged_match
        g = torch.Generator().manual_seed(31)
        rows = exact = 0
        for mode in ('full', 'adaptive'):
            pool = DecoderPool(model, mode, capacities=capacities, factory=RaggedGraphDecoder)
            uniform = DecoderPool(model, mode, capacities=capacities)
            for lengths, n in batches:  # several length mixes reuse the same graphs
                lengths = torch.tensor(lengths, device='cuda')
                pmax = int(lengths.max())
                x = torch.randint(0, model.config.vocab_size, (len(lengths), pmax), generator=g).cuda()
                with torch.inference_mode():
                    dec = pool.get(x, n)[0]
                    got = dec.generate(x, n, lengths)
                    eager = RaggedEngine(model, len(lengths), dec.capacity).generate_eager(x, n, lengths)
                    self.assertTrue(torch.equal(eager, got), (mode, lengths.tolist(), n))
                    ok, _, k = ragged_match(model, x, lengths, generate_reference_ragged(model, x, lengths, n), got)
                    self.assertTrue(ok, (mode, lengths.tolist(), n))
                    rows += len(lengths)
                    exact += k
                    if bool((lengths == pmax).all()):
                        self.assertTrue(torch.equal(uniform.get(x, n)[0].generate(x, n)[:, pmax:], got))
        return rows, exact

    def test_tiny_model(self):
        torch.manual_seed(242)
        model = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=32, vocab_size=128,
                              block_size=64, bias=True, dropout=0)).eval().cuda()
        rows, exact = self.check(model, [([14, 3, 9, 1], 2), ([3, 1, 2, 3], 5), ([8, 8, 8, 8], 20), ([1, 15, 7, 15], 1),
                                         ([2, 30, 17, 5], 33), ([1] * 16, 63), (list(range(1, 17)), 40)],
                                 (16, 32, 48, 64))
        self.assertGreater(exact, rows // 2)

    @unittest.skipUnless(CHECKPOINT.is_file(), 'trusted local checkpoint not present')
    def test_trained_checkpoint(self):
        from ..phase4.benchmark_decode import load_model
        lengths16 = [8, 64, 32, 17, 1, 120, 64, 5, 90, 33, 2, 64, 48, 11, 7, 100]
        self.check(load_model(CHECKPOINT), [(lengths16, 128), ([64] * 16, 64), ([191], 1), ([8, 200], 40),
                                            (lengths16[::-1], 20)], (80, 128, 192, 256))


try:
    import triton  # noqa: F401
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


@unittest.skipUnless(torch.cuda.is_available() and HAVE_TRITON, 'CUDA and Triton required')
class TestDecodeAttention(unittest.TestCase):
    """Triton decode attention vs an fp64 reference: error no worse than masked SDPA's (or 1e-6)."""

    def test_against_fp64(self):
        import torch.nn.functional as F
        from .decode_attention import decode_attention
        g = torch.Generator(device='cuda').manual_seed(0)
        old = torch.backends.cuda.matmul.fp32_precision
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
        try:
            for batch, cap in [(1, 80), (1, 256), (16, 128), (16, 256), (3, 37), (2, 1 + 1)]:
                for scale in (1.0, 8.0):
                    q = torch.randn(batch, 6, 1, 64, generator=g, device='cuda') * scale
                    k = torch.randn(batch, 6, cap, 64, generator=g, device='cuda') * scale
                    v = torch.randn(batch, 6, cap, 64, generator=g, device='cuda')
                    for pos in (torch.tensor([0]), torch.tensor([cap - 1]), torch.tensor([cap // 2]),
                                torch.randint(0, cap, (batch,), generator=torch.Generator().manual_seed(batch))):
                        pos = pos.cuda()
                        allowed = torch.arange(cap, device='cuda').view(1, 1, 1, cap) <= pos.view(-1, 1, 1, 1)
                        ref = F.scaled_dot_product_attention(q.double(), k.double(), v.double(), attn_mask=allowed)
                        sdpa_err = float((F.scaled_dot_product_attention(q, k, v, attn_mask=allowed).double() - ref).abs().max())
                        err = float((decode_attention(q, k, v, pos).double() - ref).abs().max())
                        self.assertLessEqual(err, max(2 * sdpa_err, 1e-6), (batch, cap, scale, pos.tolist()))
        finally:
            torch.backends.cuda.matmul.fp32_precision = old

    def test_low_precision_inputs(self):
        """bf16/fp16 q/k/v: fp32 math inside; error vs fp64 no worse than 2x same-dtype masked SDPA (or 1e-3)."""
        import torch.nn.functional as F
        from .decode_attention import decode_attention
        g = torch.Generator(device='cuda').manual_seed(1)
        for dtype in (torch.bfloat16, torch.float16):
            for batch, cap in [(1, 256), (16, 128), (3, 37)]:
                q = torch.randn(batch, 6, 1, 64, generator=g, device='cuda').to(dtype)
                k = torch.randn(batch, 6, cap, 64, generator=g, device='cuda').to(dtype)
                v = torch.randn(batch, 6, cap, 64, generator=g, device='cuda').to(dtype)
                pos = torch.randint(0, cap, (batch,), generator=torch.Generator().manual_seed(cap)).cuda()
                allowed = torch.arange(cap, device='cuda').view(1, 1, 1, cap) <= pos.view(-1, 1, 1, 1)
                ref = F.scaled_dot_product_attention(q.double(), k.double(), v.double(), attn_mask=allowed)
                out = decode_attention(q, k, v, pos)
                self.assertEqual(out.dtype, dtype)
                sdpa_err = float((F.scaled_dot_product_attention(q, k, v, attn_mask=allowed).double() - ref).abs().max())
                self.assertLessEqual(float((out.double() - ref).abs().max()), max(2 * sdpa_err, 1e-3), (dtype, batch, cap))
        with self.assertRaises(ValueError):
            q = torch.randn(1, 6, 1, 64, device='cuda')
            decode_attention(q, q.half().expand(1, 6, 8, 64).contiguous(), q.expand(1, 6, 8, 64).contiguous(),
                             torch.tensor([3], device='cuda'))

    def test_never_reads_past_position(self):
        """NaN-poisoned slots after the position must not affect the output."""
        from .decode_attention import decode_attention
        q = torch.randn(4, 6, 1, 64, device='cuda')
        k = torch.randn(4, 6, 96, 64, device='cuda')
        v = torch.randn_like(k)
        pos = torch.tensor([0, 17, 63, 94], device='cuda')
        clean = decode_attention(q, k, v, pos)
        for b, p in enumerate(pos.tolist()):
            k[b, :, p + 1:] = float('nan')
            v[b, :, p + 1:] = float('nan')
        self.assertTrue(torch.equal(clean, decode_attention(q, k, v, pos)))

    def test_compiled_triton_pool_within_near_tie_rule(self):
        """Inductor-compiled step with the Triton attention inside, captured in the CUDA graph."""
        from .benchmark_review import near_tie_match
        from .decode_attention import CompiledTritonGraphDecoder
        torch.manual_seed(242)
        model = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=128, vocab_size=128,
                              block_size=64, bias=True, dropout=0)).eval().cuda()
        g = torch.Generator().manual_seed(10)
        for mode in ('full', 'adaptive'):
            pool = DecoderPool(model, mode, capacities=(16, 32, 48, 64), factory=CompiledTritonGraphDecoder)
            for batch, p, n in [(1, 14, 2), (1, 3, 5), (1, 63, 1), (4, 5, 27), (16, 1, 63)]:
                x = torch.randint(0, 128, (batch, p), generator=g).cuda()
                with torch.inference_mode():
                    ok, _ = near_tie_match(model, generate_reference(model, x, n), pool.get(x, n)[0].generate(x, n))
                self.assertTrue(ok, (mode, batch, p, n))

    def test_triton_graph_pools_within_near_tie_rule(self):
        from .benchmark_review import near_tie_match
        from .decode_attention import TritonGraphDecoder
        torch.manual_seed(242)
        tiny = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=128, vocab_size=128,
                             block_size=64, bias=True, dropout=0)).eval().cuda()
        g = torch.Generator().manual_seed(9)
        cases = [(tiny, (16, 32, 48, 64), [(1, 14, 2), (1, 3, 5), (1, 15, 1), (1, 20, 12), (1, 2, 30),
                                             (1, 63, 1), (4, 5, 27), (16, 1, 63)])]
        if CHECKPOINT.is_file():
            from ..phase4.benchmark_decode import load_model
            cases.append((load_model(CHECKPOINT), (80, 128, 192, 256),
                          [(1, 70, 10), (1, 8, 200), (16, 8, 120), (16, 64, 64), (1, 191, 1), (16, 128, 128)]))
        for model, caps, sequence in cases:
            for mode in ('full', 'adaptive'):
                pool = DecoderPool(model, mode, capacities=caps, factory=TritonGraphDecoder)
                for batch, p, n in sequence:
                    x = torch.randint(0, model.config.vocab_size, (batch, p), generator=g).cuda()
                    with torch.inference_mode():
                        ok, _ = near_tie_match(model, generate_reference(model, x, n), pool.get(x, n)[0].generate(x, n))
                    self.assertTrue(ok, (mode, batch, p, n))


@unittest.skipUnless(torch.cuda.is_available() and HAVE_TRITON, 'CUDA and Triton required')
class TestSinglePassDispatch(unittest.TestCase):
    """KernelAgent-Ada's single_pass decode attention and the shape dispatch: same contract as split-K."""

    def test_against_fp64_and_low_precision(self):
        import torch.nn.functional as F
        # Bound: 2x same-dtype masked SDPA (the split-K test's rule) OR 1.25x the accepted split-K kernel on the same
        # input. The SDPA-only bound is seed-fragile: at (fp32, B=1, cap=256, x8, pos 37) SDPA is unusually accurate and
        # split-K itself errs 1.68e-5 > 9.6e-6, identically to single_pass (EXPERIMENT_LOG KA-04).
        from .decode_attention import decode_attention, decode_attention_single_pass as attn
        g = torch.Generator(device='cuda').manual_seed(3)
        old = torch.backends.cuda.matmul.fp32_precision
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
        try:
            for dtype, floor in ((torch.float32, 1e-6), (torch.bfloat16, 1e-3), (torch.float16, 1e-3)):
                for batch, cap in [(1, 80), (1, 256), (16, 128), (16, 256), (3, 37), (2, 2), (8, 65)]:
                    for scale in (1.0, 8.0):
                        q = (torch.randn(batch, 6, 1, 64, generator=g, device='cuda') * scale).to(dtype)
                        k = (torch.randn(batch, 6, cap, 64, generator=g, device='cuda') * scale).to(dtype)
                        v = torch.randn(batch, 6, cap, 64, generator=g, device='cuda').to(dtype)
                        for pos in (torch.tensor([0]), torch.tensor([cap - 1]), torch.tensor([63]).clamp(max=cap - 1),
                                    torch.randint(0, cap, (batch,), generator=torch.Generator().manual_seed(batch))):
                            pos = pos.cuda()
                            allowed = torch.arange(cap, device='cuda').view(1, 1, 1, cap) <= pos.view(-1, 1, 1, 1)
                            ref = F.scaled_dot_product_attention(q.double(), k.double(), v.double(), attn_mask=allowed)
                            sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
                            out = attn(q, k, v, pos)
                            self.assertEqual(out.dtype, dtype)
                            err = float((out.double() - ref).abs().max())
                            split_err = float((decode_attention(q, k, v, pos).double() - ref).abs().max())
                            bound = max(2 * float((sdpa.double() - ref).abs().max()), 1.25 * split_err, floor)
                            self.assertLessEqual(err, bound, (dtype, batch, cap, scale, pos.tolist()))
        finally:
            torch.backends.cuda.matmul.fp32_precision = old

    def test_bound_is_seed_robust(self):
        """Audit (KA-05): the KA-04 bound must not depend on one lucky seed. Ratios single/split > 1.25 occur only at
        absolute errors < 1e-6 (fp32 rounding level), which the floor covers; worst cases of both kernels match."""
        import torch.nn.functional as F
        from .decode_attention import decode_attention, decode_attention_single_pass as attn
        old = torch.backends.cuda.matmul.fp32_precision
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
        try:
            worst = {'split': 0.0, 'single': 0.0}
            for seed in (101, 202, 303):
                g = torch.Generator(device='cuda').manual_seed(seed)
                for batch, cap in [(1, 256), (4, 256), (16, 128)]:
                    for scale in (1.0, 8.0):
                        for _ in range(4):
                            q = torch.randn(batch, 6, 1, 64, generator=g, device='cuda') * scale
                            k = torch.randn(batch, 6, cap, 64, generator=g, device='cuda') * scale
                            v = torch.randn(batch, 6, cap, 64, generator=g, device='cuda')
                            pos = torch.randint(0, cap, (batch,), generator=g, device='cuda')
                            allowed = torch.arange(cap, device='cuda').view(1, 1, 1, cap) <= pos.view(-1, 1, 1, 1)
                            ref = F.scaled_dot_product_attention(q.double(), k.double(), v.double(), attn_mask=allowed)
                            err = lambda o: float((o.double() - ref).abs().max())
                            e_sdpa = err(F.scaled_dot_product_attention(q, k, v, attn_mask=allowed))
                            e_split, e_single = err(decode_attention(q, k, v, pos)), err(attn(q, k, v, pos))
                            self.assertLessEqual(e_single, max(2 * e_sdpa, 1.25 * e_split, 1e-6), (seed, batch, cap, scale))
                            # independent of split-K (a bug shared by both kernels must still fail; KA-06 review)
                            self.assertLessEqual(e_single, max(4 * e_sdpa, 1e-6), (seed, batch, cap, scale))
                            worst['split'], worst['single'] = max(worst['split'], e_split), max(worst['single'], e_single)
            self.assertLessEqual(worst['single'], 1.1 * worst['split'] + 1e-7, worst)
        finally:
            torch.backends.cuda.matmul.fp32_precision = old

    def test_model_logits_vs_fp64(self):
        """Teacher-forced decoding of the real checkpoint: logits within the SDPA path's own fp64 error class and no
        argmax change (KA-05 audit measured 17,408 row-steps: max |dlogit| 1.3e-5 single, 1.7e-5 SDPA, 0 disagreements)."""
        if not CHECKPOINT.is_file():
            self.skipTest('checkpoint not present')
        from .attention_error_audit import model_part
        old = torch.backends.cuda.matmul.fp32_precision
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
        try:
            res = model_part(CHECKPOINT, sequences=2, length=128, seed=5)['impls']
        finally:
            torch.backends.cuda.matmul.fp32_precision = old
        for name in ('split', 'single'):
            self.assertEqual(res[name]['argmax_disagree'], 0, name)
            self.assertLessEqual(res[name]['max_abs_dlogit'], max(2 * res['sdpa']['max_abs_dlogit'], 1e-5), (name, res))

    def test_never_reads_past_position_and_strided_inputs(self):
        from .decode_attention import decode_attention, decode_attention_single_pass as attn
        q = torch.randn(4, 6, 1, 64, device='cuda')
        k = torch.randn(4, 6, 96, 64, device='cuda')
        v = torch.randn_like(k)
        pos = torch.tensor([0, 17, 63, 94], device='cuda')
        clean = attn(q, k, v, pos)
        for b, p in enumerate(pos.tolist()):
            k[b, :, p + 1:] = float('nan')
            v[b, :, p + 1:] = float('nan')
        self.assertTrue(torch.equal(clean, attn(q, k, v, pos)))
        # non-contiguous K/V (head-major view of a [B, cap, H, D] buffer), as a compact cache may hand over
        kt = torch.randn(4, 96, 6, 64, device='cuda').transpose(1, 2)
        vt = torch.randn(4, 96, 6, 64, device='cuda').transpose(1, 2)
        self.assertLess(float((attn(q, kt, vt, pos) - decode_attention(q, kt, vt, pos)).abs().max()), 1e-5)

    def test_dispatch_rule(self):
        from .decode_attention import use_single_pass
        self.assertTrue(use_single_pass(8, 6, 256))      # 48 programs: single_pass won every capacity in the grid
        self.assertTrue(use_single_pass(1, 6, 64))       # short cache
        self.assertFalse(use_single_pass(1, 6, 256))     # few programs, long cache: split-K (single_pass 0.73x)
        self.assertFalse(use_single_pass(4, 6, 256))     # the held-out shape where single_pass lost (0.83x)

    def test_dispatch_pools_within_near_tie_rule(self):
        from .benchmark_review import near_tie_match
        from .decode_attention import DispatchTritonGraphDecoder
        torch.manual_seed(242)
        from .decode_attention import use_single_pass
        tiny = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=128, vocab_size=128,
                             block_size=128, bias=True, dropout=0)).eval().cuda()
        g = torch.Generator().manual_seed(11)
        # caps above 64 with few programs exercise the split-K branch without the checkpoint (KA-06 review)
        tiny_seq = [(1, 14, 2), (1, 3, 5), (1, 63, 1), (1, 90, 20), (2, 100, 27), (4, 5, 27), (16, 1, 63), (32, 30, 30)]
        self.assertFalse(use_single_pass(1, 2, 128))
        self.assertTrue(use_single_pass(32, 2, 64))
        cases = [(tiny, (16, 32, 64, 96, 128), tiny_seq)]
        if CHECKPOINT.is_file():
            from ..phase4.benchmark_decode import load_model
            cases.append((load_model(CHECKPOINT), (80, 128, 192, 256),
                          [(1, 70, 10), (1, 8, 200), (16, 8, 120), (16, 64, 64), (1, 191, 1), (16, 128, 128), (8, 192, 64)]))
        for model, caps, sequence in cases:
            for mode in ('full', 'adaptive'):
                pool = DecoderPool(model, mode, capacities=caps, factory=DispatchTritonGraphDecoder)
                for batch, p, n in sequence:
                    x = torch.randint(0, model.config.vocab_size, (batch, p), generator=g).cuda()
                    with torch.inference_mode():
                        ok, _ = near_tie_match(model, generate_reference(model, x, n), pool.get(x, n)[0].generate(x, n))
                    self.assertTrue(ok, (mode, batch, p, n))


@unittest.skipUnless(torch.cuda.is_available() and HAVE_TRITON, 'CUDA and Triton required')
class TestCompiledBaseline(unittest.TestCase):
    def test_tiny_model_within_near_tie_rule(self):
        from .benchmark_review import near_tie_match
        from .compiled_baseline import CompiledStaticDecoder
        torch.manual_seed(242)
        model = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=32, vocab_size=128,
                              block_size=64, bias=True, dropout=0)).eval().cuda()
        g = torch.Generator().manual_seed(7)
        dec = CompiledStaticDecoder(model, torch.randint(0, 128, (4, 8), generator=g).cuda())
        with torch.inference_mode():
            for p, n in [(8, 16), (1, 63), (63, 1), (30, 20)]:
                x = torch.randint(0, 128, (4, p), generator=g).cuda()
                ok, _ = near_tie_match(model, generate_reference(model, x, n), dec.generate(x, n))
                self.assertTrue(ok, (p, n))

    def test_compiled_graph_pool_reuse(self):
        from .benchmark_review import near_tie_match
        from .compiled_baseline import CompiledGraphDecoder
        torch.manual_seed(242)
        model = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=32, vocab_size=128,
                              block_size=64, bias=True, dropout=0)).eval().cuda()
        g = torch.Generator().manual_seed(9)
        sequence = [(1, 14, 2), (1, 3, 5), (1, 15, 1), (1, 20, 12), (1, 2, 30), (1, 63, 1), (4, 5, 27)]
        for mode in ('full', 'adaptive'):
            pool = DecoderPool(model, mode, capacities=(16, 32, 48, 64), factory=CompiledGraphDecoder)
            for batch, p, n in sequence:
                x = torch.randint(0, 128, (batch, p), generator=g).cuda()
                dec, _ = pool.get(x, n)
                with torch.inference_mode():
                    ok, _ = near_tie_match(model, generate_reference(model, x, n), dec.generate(x, n))
                self.assertTrue(ok, (mode, batch, p, n))


if __name__ == '__main__':
    unittest.main()
