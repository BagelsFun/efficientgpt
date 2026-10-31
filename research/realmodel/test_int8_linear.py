"""GPU tests for the experimental weight-only int8 linear (R1; not part of the V1 claims)."""
import unittest

import torch

HAVE_CUDA = torch.cuda.is_available()


@unittest.skipUnless(HAVE_CUDA, 'needs CUDA')
class TestInt8Linear(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _check(self, m, k, n, config=None):
        from . import int8_linear as mod
        q = torch.randint(-127, 128, (n, k), device='cuda', dtype=torch.int8)
        s = torch.rand(n, device='cuda') * 0.01 + 1e-4
        x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
        if config:
            mod.CONFIGS[(k, n)] = config
        try:
            y = mod.int8_linear(x, q, s)
        finally:
            mod.CONFIGS.pop((k, n), None)
        ref = x.double() @ (q.double() * s.double()[:, None]).t()
        naive = (x @ (q.to(torch.bfloat16) * s.to(torch.bfloat16)[:, None]).t()).double()
        self.assertEqual((y.shape, y.dtype), ((m, n), torch.bfloat16))
        err, naive_err = float((y.double() - ref).abs().max()), float((naive - ref).abs().max())
        self.assertLessEqual(err, max(2 * naive_err, 1e-3), (m, k, n, config))  # no worse than bf16 dequant + cuBLAS

    def test_decode_prefill_and_splitk(self):
        for m, k, n in [(1, 2048, 6144), (4, 6144, 2048), (16, 2048, 1024), (3, 100, 70), (17, 2048, 512), (64, 512, 256)]:
            self._check(m, k, n)
        self._check(2, 2048, 1024, ('dot16_splitk', 64, 128, 4, 4))
        self._check(1, 6144, 2048, ('dot16_splitk', 32, 256, 8, 4))

    def test_more_than_16_rows_refused_by_kernel(self):  # V1 audit #7: no silently uninitialised rows
        from .int8_linear import int8_matmul
        q = torch.zeros(64, 32, device='cuda', dtype=torch.int8)
        with self.assertRaises(ValueError):
            int8_matmul(torch.zeros(17, 32, device='cuda', dtype=torch.bfloat16), q, torch.ones(64, device='cuda'))

    def test_quantize_roundtrip_and_model_surgery(self):
        from .int8_linear import Int8Linear, quantize_model, quantize_weight
        big = torch.randn(20000, 64, device='cuda')  # > one 8192-row chunk: must equal the whole-matrix formula
        qb, sb = quantize_weight(big)
        s_ref = (big.abs().amax(1) / 127.0).clamp_min(1e-12)
        self.assertTrue(torch.equal(sb, s_ref))
        self.assertTrue(torch.equal(qb, torch.round(big / s_ref[:, None]).clamp(-127, 127).to(torch.int8)))
        w = torch.randn(300, 128, device='cuda')
        q, s = quantize_weight(w)
        self.assertEqual((q.dtype, int(q.abs().max())), (torch.int8, 127))
        self.assertLessEqual(float((q.float() * s[:, None] - w).abs().max()), float(s.max()) / 2 + 1e-6)  # half a step
        net = torch.nn.Sequential(torch.nn.Linear(128, 64, bias=True), torch.nn.GELU(), torch.nn.Linear(64, 32)).cuda()
        net = net.to(torch.bfloat16)
        x = torch.randn(5, 128, device='cuda', dtype=torch.bfloat16)
        ref = net(x).float()
        self.assertEqual(quantize_model(net, include_lm_head=False), 2)
        self.assertIsInstance(net[0], Int8Linear)
        self.assertLess(float((net(x).float() - ref).abs().max()), 0.05 * float(ref.abs().max()) + 1e-2)

    def test_compile_and_cuda_graph(self):
        from .int8_linear import int8_linear
        q = torch.randint(-127, 128, (512, 256), device='cuda', dtype=torch.int8)
        s = torch.rand(512, device='cuda') * 0.01
        x = torch.randn(2, 256, device='cuda', dtype=torch.bfloat16)
        eager = int8_linear(x, q, s)
        self.assertTrue(torch.equal(torch.compile(lambda t: int8_linear(t, q, s), fullgraph=True)(x), eager))
        static_x = x.clone()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            int8_linear(static_x, q, s)  # warm-up / JIT outside capture
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = int8_linear(static_x, q, s)
        static_x.copy_(x * 2)
        g.replay()
        self.assertTrue(torch.equal(out, int8_linear(x * 2, q, s)))


if __name__ == '__main__':
    unittest.main()
