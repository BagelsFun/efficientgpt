"""GPU-free policy and reuse tests; no checkpoints needed."""
import unittest
from types import SimpleNamespace
import torch
from .adaptive_pool import DecoderPool, choose_capacity


class FakeDecoder:
    capture_ms = 2.0
    def __init__(self, model, prompt, cap):
        self.cap = cap
        self.batch = prompt.shape[0]


class TestPool(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNamespace(config=SimpleNamespace(block_size=256))
        self.make = lambda model, prompt, cap: FakeDecoder(model, prompt, cap)

    def test_bucket_choice(self):
        self.assertEqual(choose_capacity(8, 16, (80,128,192,256), 256), 80)
        self.assertEqual(choose_capacity(64, 64, (80,128,192,256), 256), 128)
        self.assertEqual(choose_capacity(128, 64, (80,128,192,256), 256), 192)
        self.assertEqual(choose_capacity(192, 32, (80,128,192,256), 256), 256)

    def test_reuse_with_varying_prompt_lengths(self):
        pool = DecoderPool(self.model, "adaptive", factory=self.make)
        a, cap_a = pool.get(torch.zeros((1,8),dtype=torch.long),16)
        b, cap_b = pool.get(torch.zeros((1,64),dtype=torch.long),16)
        self.assertIs(a,b)
        self.assertEqual((cap_a,cap_b,pool.graph_count), (80,80,1))

    def test_batch_and_bucket_are_separate(self):
        pool = DecoderPool(self.model, "adaptive", factory=self.make)
        a, _ = pool.get(torch.zeros((1,8),dtype=torch.long),16)
        b, _ = pool.get(torch.zeros((16,8),dtype=torch.long),16)
        c, _ = pool.get(torch.zeros((1,128),dtype=torch.long),32)
        self.assertEqual(pool.graph_count,3)
        self.assertIsNot(a,b)
        self.assertIsNot(a,c)
        self.assertEqual(pool.total_capture_ms,6)

    def test_full_mode_one_graph_per_batch(self):
        pool = DecoderPool(self.model, "full", factory=self.make)
        a, _ = pool.get(torch.zeros((1,8),dtype=torch.long),16)
        b, _ = pool.get(torch.zeros((1,192),dtype=torch.long),32)
        self.assertIs(a,b)
        self.assertEqual(pool.graph_count,1)

    def test_rejects_invalid_budget(self):
        pool = DecoderPool(self.model, "adaptive", factory=self.make)
        with self.assertRaises(ValueError):
            pool.get(torch.zeros((1,240),dtype=torch.long),32)
        with self.assertRaises(ValueError):
            pool.get(torch.zeros((1,8),dtype=torch.long),0)
        self.assertEqual(pool.graph_count,0)


if __name__ == "__main__":
    unittest.main()
