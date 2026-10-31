"""CPU tests for the near-tie-tolerant greedy check used by compiled_static."""
import unittest
import torch
from model import GPT, GPTConfig
from ..phase4.decoder import generate_reference
from .benchmark_review import near_tie_match


class TestNearTieMatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(242)
        cls.model = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=32, vocab_size=128,
                                  block_size=48, bias=True, dropout=0)).eval()
        cls.ref = generate_reference(cls.model, torch.randint(0, 128, (3, 5)), 10)

    def flipped(self, row, pos):
        out = self.ref.clone()
        out[row, pos:] = (out[row, pos:] + 1) % 128
        return out

    def test_identical_passes_without_ties(self):
        self.assertEqual(near_tie_match(self.model, self.ref, self.ref.clone()), (True, 0))

    def test_divergence_at_clear_margin_fails(self):
        self.assertEqual(near_tie_match(self.model, self.ref, self.flipped(1, 8), margin=0.0), (False, 0))

    def test_divergence_below_margin_counts_as_tie(self):
        out = self.flipped(0, 6)
        out[2, 9] = (out[2, 9] + 3) % 128
        self.assertEqual(near_tie_match(self.model, self.ref, out, margin=1e9), (True, 2))

    def test_stops_at_first_bad_row(self):
        out = self.flipped(0, 6)
        out[2, 9] = (out[2, 9] + 3) % 128
        ok, ties = near_tie_match(self.model, self.ref, out, margin=0.0)
        self.assertFalse(ok)
        self.assertEqual(ties, 0)


if __name__ == '__main__':
    unittest.main()
