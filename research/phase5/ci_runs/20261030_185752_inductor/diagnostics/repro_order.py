import sys, torch
sys.path.insert(0, '.')
from pathlib import Path
from research.phase4.benchmark_decode import load_model
from research.phase4.decoder import generate_reference
from research.phase5.compiled_baseline import CompiledStaticDecoder
torch.backends.cuda.matmul.fp32_precision = 'tf32'
model = load_model(Path('out-tinystories-v2/baseline-2500.pt'))
g = torch.Generator().manual_seed(0)
decs = {}
for b in [int(v) for v in sys.argv[1].split(',')]:
    decs[b] = CompiledStaticDecoder(model, torch.randint(0, 50, (b, 64), generator=g).cuda())
with torch.inference_mode():
    for b, dec in decs.items():
        bad = []
        for p, n in [(64, 64), (8, 16), (191, 1), (1, 255), (128, 32)]:
            y = torch.randint(0, model.config.vocab_size, (b, p), generator=g).cuda()
            got, ref = dec.generate(y, n), generate_reference(model, y, n)
            if not torch.equal(got, ref):
                bad.append((p, n, int((got != ref).any(1).sum()), int((got != ref).sum())))
        print(f'order={sys.argv[1]} B={b}: failing (p, n, rows, tokens) = {bad}')
print('recompiles:', torch._dynamo.utils.counters['stats'].get('unique_graphs'))
