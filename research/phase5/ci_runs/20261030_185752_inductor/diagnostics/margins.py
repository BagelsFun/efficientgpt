import sys, torch
sys.path.insert(0, '.')
from pathlib import Path
from research.phase4.benchmark_decode import load_model
from research.phase4.decoder import generate_reference, reference_step
from research.phase5.compiled_baseline import CompiledStaticDecoder
torch.backends.cuda.matmul.fp32_precision = 'tf32'
model = load_model(Path('out-tinystories-v2/baseline-2500.pt'))
V = model.config.vocab_size
g = torch.Generator().manual_seed(0)
dec = CompiledStaticDecoder(model, torch.randint(0, 50, (16, 64), generator=g).cuda())
fwd = torch.compile(lambda ids: model(ids)[0])
margins, all_margins, errs = [], [], []
with torch.inference_mode():
    for trial in range(6):
        for p, n in [(64, 64), (1, 255), (128, 32)]:
            y = torch.randint(0, V, (16, p), generator=g).cuda()
            got, ref = dec.generate(y, n), generate_reference(model, y, n)
            for r in range(16):
                d = (got[r] != ref[r]).nonzero()
                if len(d):
                    i = int(d[0]); lg, _ = reference_step(model, ref[r:r+1, :i])
                    t = lg[0, -1].topk(2).values; margins.append(float(t[0] - t[1]))
            # typical compiled-vs-eager logit error and margin distribution on the same sequences
            lc, le = fwd(ref[:, :min(ref.size(1), 256)]), model(ref[:, :min(ref.size(1), 256)])[0]
            errs.append(float((lc - le).abs().max()))
            all_margins += le[:, -1].topk(2).values.diff(dim=-1).abs().squeeze(-1).tolist()
seqs = 6 * 3 * 16
print(f'divergent sequences: {len(margins)}/{seqs}')
print('ref top1-top2 margin at first divergence:', ['%.1e' % m for m in sorted(margins)])
print('compiled-vs-eager max |logit diff| per batch: min %.1e max %.1e' % (min(errs), max(errs)))
am = sorted(all_margins); print('typical last-position margins: p1 %.1e median %.1e' % (am[len(am)//100], am[len(am)//2]))
