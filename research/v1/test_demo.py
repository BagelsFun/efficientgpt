"""Checkpoint-free CPU correctness tests for the V1 generation interface."""
import unittest
import torch
from model import GPT, GPTConfig
from ..phase4.decoder import generate_reference
from .demo import generate

class TestV1Demo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(550)
        cls.model = GPT(GPTConfig(block_size=48, vocab_size=128,
                 n_layer=2, n_head=2, n_embd=32, dropout=0.0, bias=True)).eval()

    def test_prefix_outputs_match_reference(self):
        for batch, prompt_len, count in [(1,3,1),(1,7,9),(2,12,7)]:
            prompt = torch.randint(0,128,(batch,prompt_len),dtype=torch.long)
            expected = generate_reference(self.model,prompt,count)
            actual = generate(self.model,prompt,count,"prefix")
            self.assertTrue(torch.equal(expected,actual))

    def test_reject_invalid_budget(self):
        with self.assertRaises(ValueError):
            generate(self.model, torch.randint(0,128,(1,3)), 0, "prefix")

    def test_reject_cpu_graph(self):
        with self.assertRaises(ValueError):
            generate(self.model, torch.randint(0,128,(1,3)), 3, "compact-graph")

if __name__ == "__main__":
    unittest.main()
