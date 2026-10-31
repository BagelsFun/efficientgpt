"""(A) Triton ragged agreement; (B) teacher-forced TV distance of Triton vs exact sampled-step distributions."""
import sys, json, torch
sys.path.insert(0, '.')
from pathlib import Path
from research.phase4.benchmark_decode import load_model
from research.phase5.adaptive_pool import DecoderPool
from research.phase5.benchmark_review import near_tie_match
from research.phase5.compiled_baseline import decode_logits
from research.phase5.decode_attention import decode_attention
from research.phase5.eos_study import story_prompts
from research.phase5.ragged import RaggedEngine, RaggedGraphDecoder, generate_reference_ragged
from research.phase5.ragged_study import story_starts
from research.phase5.sampling import SampledEngine, SamplingConfig, filtered_probs, sample_from
torch.backends.cuda.matmul.fp32_precision = 'tf32'
m = load_model(Path('out-tinystories-v2/baseline-2500.pt')); out = {}
g = torch.Generator().manual_seed(0)
# (A) ragged with Triton attention
stories = story_starts('data/tinystories_v2/val.bin', 0, 128, 64)
tri = DecoderPool(m, 'adaptive', factory=lambda mm, p, c: RaggedGraphDecoder(mm, p, c, decode_attention))
rows = exact_vs_engine = exact_vs_b1 = 0; ties_engine = ties_b1 = 0; fail = 0
with torch.inference_mode():
    for i in range(4):
        lengths = torch.randint(8, 129, (16,), generator=g).cuda(); pmax = int(lengths.max())
        import numpy as np
        x = torch.tensor(np.stack([s[:pmax] for s in stories[16*i:16*i+16]]).astype(np.int64)).cuda()
        n = 128; dec = tri.get(x, n)[0]
        got = dec.generate(x, n, lengths)
        eng = RaggedEngine(m, 16, dec.capacity).generate_eager(x, n, lengths)
        b1 = generate_reference_ragged(m, x, lengths, n)
        for b, L in enumerate(lengths.tolist()):
            p = x[b:b+1, :L]; rows += 1
            for ref, key in ((eng, 'e'), (b1, 'b')):
                ok, k = near_tie_match(m, torch.cat([p, ref[b:b+1]], 1), torch.cat([p, got[b:b+1]], 1))
                fail += not ok
                if key == 'e': exact_vs_engine += int(torch.equal(ref[b], got[b])); ties_engine += k
                else: exact_vs_b1 += int(torch.equal(ref[b], got[b])); ties_b1 += k
out['ragged_triton'] = dict(rows=rows, failures_outside_near_tie=fail, identical_to_exact_ragged_engine=exact_vs_engine,
                            near_ties_vs_engine=ties_engine, identical_to_batch1=exact_vs_b1, near_ties_vs_batch1=ties_b1)
print(out['ragged_triton'], flush=True)
# (B) teacher-forced distribution distance, sampled step: exact SDPA vs Triton attention, identical state each step
prompts = story_prompts('data/tinystories_v2/val.bin', 0, 32, 64).cuda()
cfgs = {'T1.0': SamplingConfig(1.0), 'T0.8_k50': SamplingConfig(0.8, top_k=50), 'T1.0_p0.9': SamplingConfig(1.0, top_p=0.9),
        'T0.8_k50_p0.95': SamplingConfig(0.8, top_k=50, top_p=0.95)}
gen = torch.Generator(device='cuda').manual_seed(1)
with torch.inference_mode():
    for name, cfg in cfgs.items():
        tvs, set_changes, steps = [], 0, 0
        for s in range(0, 64, 16):
            x = prompts[s:s+16]
            a = SampledEngine(m, 16, 32 + 160, cfg); bt = SampledEngine(m, 16, 32 + 160, cfg, decode_attention)
            a.reset(x); bt.reset(x); bt.token.copy_(a.token)
            for t in range(159):
                la, lb = decode_logits(a), decode_logits(bt, attention=decode_attention)
                pa, pb = filtered_probs(la[:, -1], cfg), filtered_probs(lb[:, -1], cfg)
                tvs.append((0.5 * (pa.double() - pb.double()).abs().sum(-1)).cpu())
                set_changes += int(((pa > 0) != (pb > 0)).any(-1).sum()); steps += 16
                tok = sample_from(pa, torch.rand(16, generator=gen, device='cuda'))
                for e in (a, bt):
                    e.history.index_copy_(1, e.position, e.token); e.token.copy_(tok); e.position.add_(1)
        tv = torch.cat(tvs)
        out[f'tv_{name}'] = dict(steps=steps, tv_median=float(tv.median()), tv_p99=float(tv.quantile(0.99)), tv_max=float(tv.max()),
                                 kept_set_changes=set_changes)
        print(name, out[f'tv_{name}'], flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=2)
