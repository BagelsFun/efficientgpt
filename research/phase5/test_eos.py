"""CPU tests for EOS stopping: mask semantics and equivalence with feed-pad decoding."""
import unittest
import torch
from model import GPT, GPTConfig
from ..phase4.decoder import StaticKVDecoder, generate_reference
from .eos import EOSDecoder, generate_reference_fed_pad, mask_after_eos
from .prefix_eager import PrefixSDPADecoder


class _Eager:
    """Expose StaticKVDecoder.generate_eager as generate()."""
    def __init__(self, decoder):
        self.decoder = decoder

    def generate(self, prompt, count):
        return self.decoder.generate_eager(prompt, count)


def frequent_token(tokens, prompt_len):
    """Most common generated token: used as a stand-in EOS so stops actually happen."""
    return int(tokens[:, prompt_len:].flatten().mode().values)


class TestMaskAfterEOS(unittest.TestCase):
    def test_semantics(self):
        t = torch.tensor([[9, 9, 5, 7, 5, 6],    # EOS(5) at gen index 0, later EOS masked too
                          [9, 9, 1, 2, 3, 4],    # no EOS
                          [9, 9, 1, 2, 3, 5],    # EOS on the last step
                          [5, 9, 1, 5, 5, 2]])   # EOS in prompt ignored; first gen EOS at index 1
        out, lengths = mask_after_eos(t, 2, eos_id=5)
        self.assertEqual(out.tolist(), [[9, 9, 5, 5, 5, 5], [9, 9, 1, 2, 3, 4],
                                        [9, 9, 1, 2, 3, 5], [5, 9, 1, 5, 5, 5]])
        self.assertEqual(lengths.tolist(), [1, 4, 4, 2])
        self.assertTrue(torch.equal(t[:, :2], out[:, :2]))

    def test_separate_pad_id(self):
        out, lengths = mask_after_eos(torch.tensor([[1, 5, 3, 5]]), 1, eos_id=5, pad_id=-1)
        self.assertEqual((out.tolist(), lengths.tolist()), ([[1, 5, -1, -1]], [1]))

    def test_input_not_mutated(self):
        t = torch.tensor([[1, 5, 3]])
        mask_after_eos(t, 1, eos_id=5)
        self.assertEqual(t.tolist(), [[1, 5, 3]])


class TestEOSDecoding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(242)
        cls.model = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=32, vocab_size=16,
                                  block_size=48, bias=True, dropout=0)).eval()

    def cases(self):
        g = torch.Generator().manual_seed(3)
        for batch, p, n in [(1, 1, 20), (3, 7, 15), (4, 13, 30), (2, 40, 8)]:
            yield torch.randint(0, 16, (batch, p), generator=g), n

    def test_post_mask_equals_feed_pad_reference(self):
        stops = 0
        for x, n in self.cases():
            plain = generate_reference(self.model, x, n)
            eos = frequent_token(plain, x.size(1))
            for pad in (None, 15):
                expected, exp_len = generate_reference_fed_pad(self.model, x, n, eos, pad)
                got, got_len = mask_after_eos(plain, x.size(1), eos, pad)
                self.assertTrue(torch.equal(expected, got), (x.shape, n, eos, pad))
                self.assertTrue(torch.equal(exp_len, got_len))
            stops += int((got_len < n).sum())
        self.assertGreater(stops, 0, 'no row stopped early; test would be vacuous')

    def test_wrapper_on_eager_decoders(self):
        for x, n in self.cases():
            eos = frequent_token(generate_reference(self.model, x, n), x.size(1))
            expected = generate_reference_fed_pad(self.model, x, n, eos)
            for inner in (_Eager(StaticKVDecoder(self.model, x.size(0))), PrefixSDPADecoder(self.model, x.size(0))):
                got = EOSDecoder(inner, eos).generate(x, n)
                self.assertTrue(torch.equal(expected[0], got[0]))
                self.assertTrue(torch.equal(expected[1], got[1]))


if __name__ == '__main__':
    unittest.main()
