"""CPU test: the index_copy_ step used under Inductor is bit-identical in eager."""
import unittest
import torch
from model import GPT, GPTConfig
from ..phase4.compact_cache import CompactStaticKVDecoder
from ..phase4.decoder import StaticKVDecoder
from .compiled_baseline import step_index_copy


class TestStepIndexCopy(unittest.TestCase):
    def test_matches_scatter_step_bitwise(self):
        torch.manual_seed(242)
        model = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=32, vocab_size=128,
                              block_size=48, bias=True, dropout=0)).eval()
        for make in (lambda: StaticKVDecoder(model, 3), lambda: CompactStaticKVDecoder(model, 3, 30)):
            a, b = make(), make()
            x = torch.randint(0, 128, (3, 7))
            with torch.inference_mode():
                a.reset(x); b.reset(x)
                for _ in range(20):
                    StaticKVDecoder.step(a)
                    step_index_copy(b)
            for left, right in zip(a.keys + a.values + [a.history, a.token, a.position],
                                   b.keys + b.values + [b.history, b.token, b.position]):
                self.assertTrue(torch.equal(left, right))


if __name__ == '__main__':
    unittest.main()
