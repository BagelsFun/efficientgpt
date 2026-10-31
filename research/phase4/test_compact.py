"""CPU correctness tests for smaller KV-cache capacity.

Run from nanoGPT root:
    python -m unittest research.phase4.test_compact -v
"""
import unittest

import torch
from model import GPT, GPTConfig
from .compact_cache import CompactStaticKVDecoder
from .decoder import generate_reference


class TestCompactDecoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(121)
        cls.model = GPT(GPTConfig(block_size=32, vocab_size=128,
                                  n_layer=2, n_head=2, n_embd=32,
                                  dropout=0.0, bias=True)).eval()

    def test_same_greedy_outputs_for_multiple_prompts(self):
        for batch, length, count in [(1, 3, 8), (2, 7, 6), (3, 13, 4), (1, 29, 3)]:
            decoder = CompactStaticKVDecoder(self.model, batch, length + count)
            for seed in (1, 3, 5):
                torch.manual_seed(seed)
                prompt = torch.randint(0, 128, (batch, length))
                expected = generate_reference(self.model, prompt, count)
                actual = decoder.generate_eager(prompt, count)
                self.assertTrue(torch.equal(expected, actual), (batch, length, count, seed))

    def test_one_token(self):
        prompt = torch.randint(0, 128, (1, 5))
        decoder = CompactStaticKVDecoder(self.model, 1, 6)
        self.assertTrue(torch.equal(generate_reference(self.model, prompt, 1),
                                    decoder.generate_eager(prompt, 1)))

    def test_reject_over_capacity(self):
        decoder = CompactStaticKVDecoder(self.model, 1, 8)
        prompt = torch.randint(0, 128, (1, 5))
        with self.assertRaises(ValueError):
            decoder.generate_eager(prompt, 4)

    def test_reject_invalid_capacity(self):
        with self.assertRaises(ValueError):
            CompactStaticKVDecoder(self.model, 1, 33)


if __name__ == "__main__":
    unittest.main()
