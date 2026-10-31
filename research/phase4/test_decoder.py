"""CPU correctness checks; run from nanoGPT folder using -m unittest."""
import unittest
import torch
from model import GPT, GPTConfig
from research.phase4.decoder import generate_reference, StaticKVDecoder


class TestDecoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(10)
        cls.model = GPT(GPTConfig(
            block_size=32, vocab_size=128, n_layer=2,
            n_head=2, n_embd=32, dropout=0.0, bias=True,
        )).eval()

    def test_greedy_equal_for_multiple_inputs(self):
        for batch, length, count in [(1, 3, 8), (2, 7, 6), (3, 13, 4)]:
            dec = StaticKVDecoder(self.model, batch)
            for seed in (1, 3, 5):
                torch.manual_seed(seed)
                ids = torch.randint(0, 128, (batch, length))
                expected = generate_reference(self.model, ids, count)
                obtained = dec.generate_eager(ids, count)
                self.assertTrue(torch.equal(expected, obtained), (batch,length,count,seed))

    def test_one_token_and_last_valid_position(self):
        dec = StaticKVDecoder(self.model, 1)
        for length, count in ((5, 1), (29, 3)):
            tokens = torch.randint(0, 128, (1, length))
            self.assertTrue(torch.equal(
                generate_reference(self.model, tokens, count),
                dec.generate_eager(tokens, count),
            ))

    def test_batch_mismatch_rejected(self):
        dec = StaticKVDecoder(self.model, 2)
        with self.assertRaises(ValueError):
            dec.generate_eager(torch.randint(0, 128, (1, 5)), 3)

    def test_context_rejection(self):
        ids = torch.randint(0, 128, (1, 31))
        dec = StaticKVDecoder(self.model, 1)
        with self.assertRaises(ValueError):
            dec.generate_eager(ids, 2)


if __name__ == '__main__':
    unittest.main()
