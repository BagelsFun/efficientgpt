"""GPU tests for the V3 graph decoder on a tiny random Qwen3 (fast, no download)."""
import unittest

import torch

HAVE_CUDA = torch.cuda.is_available()


def tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=32, max_position_embeddings=256, tie_word_embeddings=True,
                      initializer_range=0.2)  # default 0.02 gives one repeated token: equality would be trivial
    return Qwen3ForCausalLM(cfg).cuda().to(torch.bfloat16).eval()


def hf_greedy(model, x, n):
    out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=n, min_new_tokens=n,
                         do_sample=False, cache_implementation='static', pad_token_id=0)
    return out[0, x.shape[1]:].tolist()


@unittest.skipUnless(HAVE_CUDA, 'needs CUDA')
class TestGraphGreedyDecoder(unittest.TestCase):
    def test_matches_hf_generate_bf16_and_int8(self):
        from .fast_decode import GraphGreedyDecoder
        from .int8_linear import quantize_model
        for quant in ('bf16', 'int8'):
            model = tiny_qwen3()
            if quant == 'int8':
                quantize_model(model)
            dec = GraphGreedyDecoder(model, max_cache_len=128)
            g = torch.Generator().manual_seed(1)
            for p in (1, 5, 37):
                x = torch.randint(0, 512, (1, p), generator=g).cuda()
                got, st = dec.generate(x, 40)
                self.assertGreater(len(set(got)), 8, 'degenerate model output would make equality trivial')
                self.assertEqual(got, hf_greedy(model, x, 40), (quant, p))
                self.assertEqual(st['tokens'], 40)

    def test_eos_stops_exactly_and_never_leaks_lookahead_tokens(self):
        from .fast_decode import GraphGreedyDecoder
        model = tiny_qwen3()
        dec = GraphGreedyDecoder(model, max_cache_len=128, lookahead=8)
        x = torch.randint(0, 512, (1, 7), generator=torch.Generator().manual_seed(2)).cuda()
        full, _ = dec.generate(x, 60)
        # EOS = the latest token that first appears at its position (so stopping there is unambiguous)
        k = max(i for i in range(len(full)) if full[i] not in full[:i])
        self.assertGreater(k, 10, 'EOS must fall well past the first lookahead window')
        streamed = []
        got, st = dec.generate(x, 60, eos_ids={full[k]}, on_token=streamed.append)
        self.assertEqual(got, full[:k + 1])        # stops at (and includes) the EOS token
        self.assertEqual(streamed, got)            # tokens computed past EOS by run-ahead are never emitted
        again, _ = dec.generate(x, 60)             # cache reset between requests: same result as the first run
        self.assertEqual(again, full)

    def test_guards(self):
        from .fast_decode import GraphGreedyDecoder
        dec = GraphGreedyDecoder(tiny_qwen3(), max_cache_len=32)
        with self.assertRaises(ValueError):
            dec.generate(torch.zeros((1, 20), dtype=torch.long, device='cuda'), 20)
        with self.assertRaises(ValueError):  # batch > 1 would silently decode only row 0 (V3 audit)
            dec.generate(torch.zeros((2, 4), dtype=torch.long, device='cuda'), 4)
        with self.assertRaises(ValueError):
            dec.generate(torch.zeros((1, 4), dtype=torch.long, device='cuda'), 0)
        out, st = dec.generate(torch.zeros((1, 4), dtype=torch.long, device='cuda'), 3)
        self.assertEqual((len(out), st['tokens']), (3, 3))
        self.assertIn('capture_s', st)


if __name__ == '__main__':
    unittest.main()
