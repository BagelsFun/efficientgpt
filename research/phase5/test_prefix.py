"""Tests for a more credible PyTorch SDPA eager decoding baseline."""
import unittest
import torch
from model import GPT, GPTConfig
from ..phase4.decoder import generate_reference
from .prefix_eager import PrefixSDPADecoder


class TestPrefixSDPA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(242)
        cls.model = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=32,
                                  vocab_size=128, block_size=48,
                                  bias=True, dropout=0)).eval()

    def test_many_prompt_shapes_and_reuse(self):
        for batch, p, n in [(1, 1, 1), (1, 5, 8), (2, 7, 6), (3, 23, 12), (1, 40, 8)]:
            dec = PrefixSDPADecoder(self.model, batch)
            for seed in [17, 28, 39]:
                torch.manual_seed(seed)
                x = torch.randint(0, self.model.config.vocab_size, (batch, p))
                expected = generate_reference(self.model, x, n)
                actual = dec.generate(x, n)
                self.assertTrue(torch.equal(expected, actual), (batch, p, n, seed))

    def test_context_limit(self):
        dec = PrefixSDPADecoder(self.model, 1)
        with self.assertRaises(ValueError):
            dec.generate(torch.zeros((1, 46), dtype=torch.long), 3)

    def test_batch_mismatch(self):
        dec = PrefixSDPADecoder(self.model, 1)
        with self.assertRaises(ValueError):
            dec.generate(torch.zeros((2, 4), dtype=torch.long), 2)

    def test_empty_prompt(self):
        dec = PrefixSDPADecoder(self.model, 1)
        with self.assertRaises(ValueError):
            dec.generate(torch.zeros((1, 0), dtype=torch.long), 2)


if __name__ == '__main__':
    unittest.main()
