"""Noise-floor control: TV distance of exact-SDPA sampled-step distributions, TF32 vs IEEE matmul, same states (teacher-forced)."""
import sys, json, torch
sys.path.insert(0, '.')
from pathlib import Path
from research.phase4.benchmark_decode import load_model
from research.phase5.compiled_baseline import decode_logits
from research.phase5.decode_attention import decode_attention
from research.phase5.eos_study import story_prompts
from research.phase5.sampling import SampledEngine, SamplingConfig, filtered_probs, sample_from
m = load_model(Path('out-tinystories-v2/baseline-2500.pt'))
prompts = story_prompts('data/tinystories_v2/val.bin', 0, 32, 64).cuda()
cfgs = {'T1.0': SamplingConfig(1.0), 'T0.8_k50': SamplingConfig(0.8, top_k=50), 'T1.0_p0.9': SamplingConfig(1.0, top_p=0.9),
        'T0.8_k50_p0.95': SamplingConfig(0.8, top_k=50, top_p=0.95)}
P = torch.backends.cuda.matmul
out = {}
with torch.inference_mode():
    for name, cfg in cfgs.items():
        res = {}
        for label, policy_b, attn_b in (('tf32_vs_ieee_exact', 'ieee', None), ('exact_vs_triton_tf32', 'ieee', decode_attention)):
            gen = torch.Generator(device='cuda').manual_seed(1); tvs, changes, steps = [], 0, 0
            for s in range(0, 64, 16):
                x = prompts[s:s+16]
                P.fp32_precision = 'tf32'
                a = SampledEngine(m, 16, 192, cfg); b = SampledEngine(m, 16, 192, cfg, attn_b); a.reset(x)
                P.fp32_precision = policy_b if attn_b is None else 'tf32'
                b.reset(x); b.token.copy_(a.token)
                for t in range(159):
                    P.fp32_precision = 'tf32'; la = decode_logits(a)
                    P.fp32_precision = policy_b if attn_b is None else 'tf32'; lb = decode_logits(b, attention=attn_b)
                    pa, pb = filtered_probs(la[:, -1], cfg), filtered_probs(lb[:, -1], cfg)
                    tvs.append((0.5 * (pa.double() - pb.double()).abs().sum(-1)).cpu())
                    changes += int(((pa > 0) != (pb > 0)).any(-1).sum()); steps += 16
                    tok = sample_from(pa, torch.rand(16, generator=gen, device='cuda'))
                    for e in (a, b):
                        e.history.index_copy_(1, e.position, e.token); e.token.copy_(tok); e.position.add_(1)
            tv = torch.cat(tvs)
            res[label] = dict(steps=steps, tv_median=float(tv.median()), tv_p99=float(tv.quantile(0.99)), tv_max=float(tv.max()), kept_set_changes=changes)
        out[name] = res; print(name, json.dumps(res), flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=2)
