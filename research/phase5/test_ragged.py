"""CPU tests for ragged batches: validation, prefill, pad invariance, uniform equivalence, per-row reference."""
import unittest
import torch
from model import GPT, GPTConfig
from ..phase4.compact_cache import CompactStaticKVDecoder
from ..phase4.decoder import reference_step
from .benchmark_review import near_tie_match
from .ragged import RaggedEngine, generate_reference_ragged, ragged_prefill


def ragged_match(model, prompts, lengths, ref_gen, out_gen):
    """Per row: (all rows within the near-tie rule, near-tie count, bit-exact rows)."""
    ties = exact = 0
    for b, n in enumerate(lengths.tolist()):
        ref = torch.cat([prompts[b:b + 1, :n], ref_gen[b:b + 1]], 1)
        out = torch.cat([prompts[b:b + 1, :n], out_gen[b:b + 1]], 1)
        ok, k = near_tie_match(model, ref, out)
        if not ok:
            return False, ties, exact
        ties += k
        exact += int(torch.equal(ref, out))
    return True, ties, exact


def ragged_batch(g, vocab, lengths):
    lengths = torch.tensor(lengths)
    prompts = torch.randint(0, vocab, (len(lengths), int(lengths.max())), generator=g)
    return prompts, lengths


class TestRaggedCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(242)
        cls.model = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=32, vocab_size=64,
                                  block_size=48, bias=True, dropout=0)).eval()
        cls.g = torch.Generator().manual_seed(8)

    def test_validation(self):
        engine = RaggedEngine(self.model, 2, 20)
        x = torch.zeros((2, 5), dtype=torch.long)
        for lengths in (torch.tensor([0, 5]), torch.tensor([6, 1]), torch.tensor([1]), torch.tensor([2.0, 3.0])):
            with self.assertRaises((ValueError, RuntimeError)):
                engine.generate_eager(x, 3, lengths)
        with self.assertRaises(ValueError):
            engine.generate_eager(x, 16, torch.tensor([5, 5]))  # 5 + 16 > capacity 20

    def test_prefill_matches_batch_one(self):
        prompts, lengths = ragged_batch(self.g, 64, [1, 9, 4, 13])
        logits, cache = ragged_prefill(self.model, prompts, lengths)
        for b, n in enumerate(lengths.tolist()):
            ref_logits, ref_cache = reference_step(self.model, prompts[b:b + 1, :n])
            self.assertTrue(torch.allclose(logits[b], ref_logits[0], atol=1e-5, rtol=1e-5), b)
            for (k, v), (rk, rv) in zip(cache, ref_cache):
                self.assertTrue(torch.allclose(k[b, :, :n], rk[0], atol=1e-5, rtol=1e-5))
                self.assertTrue(torch.allclose(v[b, :, :n], rv[0], atol=1e-5, rtol=1e-5))

    def test_pad_values_never_read(self):
        """Tokens AND the K/V of every valid slot must be bit-identical for any pad values."""
        prompts, lengths = ragged_batch(torch.Generator().manual_seed(9), 64, [3, 11, 7, 1, 11])
        engine = RaggedEngine(self.model, 5, 40)

        def run(x):
            tokens = engine.generate_eager(x, 20, lengths)
            # slots written so far: prompt + the first 19 generated tokens of each row
            state = [torch.cat([t[b, :, :n + 19] for t in engine.keys + engine.values])
                     for b, n in enumerate(lengths.tolist())]
            return tokens, state

        base_tokens, base_state = run(prompts)
        for fill in (0, 63, 17):
            other = prompts.clone()
            for b, n in enumerate(lengths.tolist()):
                other[b, n:] = fill
            tokens, state = run(other)
            self.assertTrue(torch.equal(base_tokens, tokens), fill)
            for b, (s0, s1) in enumerate(zip(base_state, state)):
                self.assertTrue(torch.equal(s0, s1), (fill, b))

    def test_uniform_lengths_equal_existing_decoder(self):
        for batch, p, n in [(1, 5, 10), (3, 8, 20), (4, 30, 17)]:
            x = torch.randint(0, 64, (batch, p), generator=self.g)
            lengths = torch.full((batch,), p)
            expected = CompactStaticKVDecoder(self.model, batch, p + n).generate_eager(x, n)[:, p:]
            got = RaggedEngine(self.model, batch, p + n).generate_eager(x, n, lengths)
            self.assertTrue(torch.equal(expected, got), (batch, p, n))

    def test_rows_match_batch_one_reference(self):
        exact_rows = rows = 0
        engine = RaggedEngine(self.model, 6, 47)
        for lengths in ([1, 2, 3, 4, 5, 6], [20, 1, 20, 7, 13, 2], [9, 9, 9, 9, 9, 1]):
            prompts, lengths = ragged_batch(self.g, 64, lengths)
            n = 47 - int(lengths.max())
            ok, _, exact = ragged_match(self.model, prompts, lengths,
                                        generate_reference_ragged(self.model, prompts, lengths, n),
                                        engine.generate_eager(prompts, n, lengths))
            self.assertTrue(ok, lengths.tolist())
            exact_rows += exact
            rows += len(lengths)
        self.assertGreater(exact_rows, rows // 2)


if __name__ == '__main__':
    unittest.main()
