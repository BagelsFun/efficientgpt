import sys, time, statistics, torch
sys.path.insert(0, '.')
from pathlib import Path
from research.phase4.benchmark_decode import load_model
from research.phase4.decoder import generate_reference
from research.phase5.adaptive_pool import DecoderPool
from research.phase5.compiled_baseline import CompiledStaticDecoder
torch.backends.cuda.matmul.fp32_precision = 'tf32'
model = load_model(Path('out-tinystories-v2/baseline-2500.pt'))
g = torch.Generator().manual_seed(0)
for mode in sys.argv[1:]:
    for batch in (1, 16):
        x = torch.randint(0, model.config.vocab_size, (batch, 64), generator=g).cuda()
        try:
            dec = CompiledStaticDecoder(model, x, mode=mode)
        except Exception as e:
            print(f'{mode} B={batch} COMPILE FAIL {type(e).__name__}: {str(e)[:300]}'); continue
        ok = True
        with torch.inference_mode():
            for p, n in [(64, 64), (8, 16), (191, 1), (1, 255), (128, 32)]:
                y = torch.randint(0, model.config.vocab_size, (batch, p), generator=g).cuda()
                ok &= torch.equal(dec.generate(y, n), generate_reference(model, y, n))
            ts = []
            for _ in range(15):
                torch.cuda.synchronize(); t = time.perf_counter(); dec.generate(x, 64); torch.cuda.synchronize()
                ts.append(1000 * (time.perf_counter() - t))
            full, _ = DecoderPool(model, 'full').get(x, 64)
            fs = []
            for _ in range(15):
                torch.cuda.synchronize(); t = time.perf_counter(); full.generate(x, 64); torch.cuda.synchronize()
                fs.append(1000 * (time.perf_counter() - t))
        print(f'{mode} B={batch} exact={ok} compile+warm={dec.compile_ms:.0f}ms p50(64+64)={statistics.median(ts):.2f}ms full_graph={statistics.median(fs):.2f}ms')
