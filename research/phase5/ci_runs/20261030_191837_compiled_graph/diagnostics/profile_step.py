"""Per-kernel CUDA time of one decode step: eager StaticKVDecoder.step vs Inductor-compiled."""
import sys, torch
sys.path.insert(0, '.')
from pathlib import Path
from torch.profiler import profile, ProfilerActivity
from research.phase4.benchmark_decode import load_model
from research.phase4.decoder import StaticKVDecoder
torch.backends.cuda.matmul.fp32_precision = 'tf32'
model = load_model(Path('out-tinystories-v2/baseline-2500.pt'))
for B in (1, 16):
    x = torch.randint(0, model.config.vocab_size, (B, 64)).cuda()
    eng = StaticKVDecoder(model, B)
    comp = torch.compile(StaticKVDecoder.step.__wrapped__, fullgraph=True, dynamic=False)
    with torch.inference_mode():
        for name, fn in (('eager', lambda: StaticKVDecoder.step.__wrapped__(eng)), ('compiled', lambda: comp(eng))):
            eng.reset(x)
            for _ in range(5): fn()
            eng.reset(x); torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(20): fn()
                torch.cuda.synchronize()
            ev = [e for e in prof.key_averages() if e.device_type.name == 'CUDA']
            total = sum(e.self_device_time_total for e in ev) / 20
            print(f'\nB={B} {name}: GPU kernel time per step {total/1000:.3f} ms, kernels/step {sum(e.count for e in ev)/20:.0f}')
            for e in sorted(ev, key=lambda e: -e.self_device_time_total)[:6]:
                print(f'  {e.self_device_time_total/20/1000:7.3f} ms  x{e.count//20:<3} {e.key[:90]}')
