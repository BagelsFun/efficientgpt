"""Power check: an off-by-one Triton attention (drops the current key) must fail the distribution tests."""
import sys, unittest, torch
sys.path.insert(0, '.')
import research.phase5.decode_attention as DA
import research.phase5.test_gpu_combinations as T
real = DA.decode_attention
def off_by_one(q, k, v, position, config=None):
    return real(q, k, v, (position - 1).clamp(min=0), config)
for label, fn in (('real', real), ('off_by_one', off_by_one)):
    DA.decode_attention = fn
    r = unittest.TextTestRunner(stream=open('nul', 'w')).run(unittest.defaultTestLoader.loadTestsFromTestCase(T.TestTritonSampling))
    print(label, 'failures:', [t.id().split('.')[-1] for t, _ in r.failures + r.errors])
DA.decode_attention = real
