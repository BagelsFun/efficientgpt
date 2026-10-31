import sys, torch
sys.path.insert(0, '.')
from pathlib import Path
from torch.profiler import profile, ProfilerActivity
from research.phase4.benchmark_decode import load_model
from research.phase4.compact_cache import CompactStaticKVDecoder
from research.phase5.decode_attention import triton_attention_step
torch.backends.cuda.matmul.fp32_precision = 'tf32'
model = load_model(Path('out-tinystories-v2/baseline-2500.pt'))
comp = torch.compile(triton_attention_step, fullgraph=True, dynamic=False)
x = torch.randint(0, 4096, (16, 64)).cuda()
with torch.inference_mode():
    for name, fn in (('eager_triton', triton_attention_step), ('compiled_triton', comp)):
        e = CompactStaticKVDecoder(model, 16, 256); e.reset(x)
        for _ in range(3): fn(e)
        e.reset(x); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(10): fn(e)
            torch.cuda.synchronize()
        ev = sorted([v for v in prof.key_averages() if v.device_type.name == 'CUDA'], key=lambda v: -v.count)
        print(f'== {name}: kernels/step {sum(v.count for v in ev)/10:.0f}, GPU {sum(v.self_device_time_total for v in ev)/10/1000:.3f} ms')
        for v in ev: print(f'   x{v.count//10:<3} {v.self_device_time_total/10/1000:.4f} ms  {v.key[:70]}')
