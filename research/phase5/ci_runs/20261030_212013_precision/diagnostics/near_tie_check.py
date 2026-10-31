"""Do the low-precision modes satisfy the fp32 near-tie rule? Benchmark validation prompts + random prompts."""
import sys, copy, torch
sys.path.insert(0, '.')
from pathlib import Path
from tokenizers import Tokenizer
from research.phase4.benchmark_decode import load_model, make_prompts
from research.phase4.decoder import generate_reference, reference_step
from research.phase5.adaptive_pool import DecoderPool
from research.phase5.benchmark_requests import workloads
from research.phase5.decode_attention import CompiledTritonGraphDecoder
torch.backends.cuda.matmul.fp32_precision = 'tf32'
m32 = load_model(Path('out-tinystories-v2/baseline-2500.pt')); tok = Tokenizer.from_file('data/tinystories_v2/tokenizer.json')
g = torch.Generator().manual_seed(3)
sets = {'benchmark_prompts': [(c, make_prompts(tok, 4096, c[0], c[1], 3)) for c in workloads(False)],
        'random_prompts': [((b, p, n), [torch.randint(0, 4096, (b, p), generator=g).cuda() for _ in range(3)])
                           for b, p, n in [(16, 8, 120), (16, 64, 64), (16, 128, 128), (1, 8, 248)]]}
for dtype in (torch.float16, torch.bfloat16):
    pool = DecoderPool(copy.deepcopy(m32).to(dtype), 'full', factory=CompiledTritonGraphDecoder)
    for name, cases in sets.items():
        rows = diverged = within = 0; margins = []
        with torch.inference_mode():
            for (b, p, n), prompts in cases:
                for x in prompts:
                    ref, out = generate_reference(m32, x, n), pool.get(x, n)[0].generate(x, n)
                    for r in range(b):
                        rows += 1
                        d = (ref[r] != out[r]).nonzero()
                        if len(d):
                            diverged += 1; i = int(d[0])
                            top = reference_step(m32, ref[r:r + 1, :i])[0][0, -1].topk(2).values
                            mg = float(top[0] - top[1]); margins.append(mg); within += mg < 1e-2
        print(f'{str(dtype).split(".")[-1]} {name}: rows={rows} diverged={diverged} within_near_tie_rule={within} '
              f'max_margin_at_divergence={max(margins) if margins else 0:.2e}')
