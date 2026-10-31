"""Quality of low-precision weights (bf16 / fp16) against the fp32 model. APPROXIMATE modes.

    python -m research.phase5.precision_study --output research/phase5/ci_runs/<stamp>_precision/quality.json

Weights (and therefore the KV cache) are stored in the low dtype; GEMMs
accumulate in fp32 (cuBLAS), the Triton attention computes in fp32.
A. Teacher-forced on validation windows of 256 tokens (full-sequence forward):
   mean NLL / perplexity per dtype, paired bootstrap CI of the NLL change vs
   fp32, next-token argmax agreement with fp32, mean KL(fp32 || low).
B. Free-running greedy decoding (decode path, CUDA graphs) on real story
   prompts (32 tokens, budget 224) vs the bit-exact fp32 adaptive graph:
   identical-sequence rate, first-divergence position, whether the first
   divergence is a near-tie (fp32 margin < 1e-2), mean NLL of the generated
   continuation scored by the fp32 model (lower = more fp32-plausible), and
   EOS behavior. fp32 compiled_triton is the control (near-ties only).
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..phase4.benchmark_decode import load_model
from .adaptive_pool import DecoderPool
from .benchmark_review import NEAR_TIE_LOGIT_MARGIN
from .decode_attention import CompiledTritonGraphDecoder
from .eos import mask_after_eos
from .eos_study import story_prompts
from ..phase4.decoder import reference_step

DTYPES = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}


def paired_ci(diff, boots=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (boots, len(diff)))
    means = diff[idx].mean(1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


@torch.inference_mode()
def teacher_forced(models, val_path, windows, batch=16):
    data = np.fromfile(val_path, dtype=np.uint16).astype(np.int64)
    n = min(windows, (len(data) - 1) // 257)
    x_all = torch.tensor(data[:n * 257].reshape(n, 257))
    nll = {k: [] for k in models}
    agree = {k: [] for k in models}
    kl = {k: [] for k in models}
    for start in range(0, n, batch):
        x = x_all[start:start + batch].cuda()
        inp, tgt = x[:, :-1].contiguous(), x[:, 1:].contiguous()
        logp = {}
        for name, m in models.items():
            logits, _ = m(inp, tgt)  # targets given -> logits for every position
            logp[name] = F.log_softmax(logits.float(), -1)
        ref_arg = logp['fp32'].argmax(-1)
        for name in models:
            tok_nll = -logp[name].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            nll[name] += tok_nll.mean(1).tolist()
            agree[name] += (logp[name].argmax(-1) == ref_arg).float().mean(1).tolist()
            kl[name] += (logp['fp32'].exp() * (logp['fp32'] - logp[name])).sum(-1).mean(1).tolist()
    out = {'windows': n, 'tokens': n * 256}
    base = np.array(nll['fp32'])
    for name in models:
        arr = np.array(nll[name])
        d, lo, hi = paired_ci(arr - base)
        out[name] = {'mean_nll': float(arr.mean()), 'perplexity': float(np.exp(arr.mean())),
                     'delta_nll_vs_fp32': d, 'delta_nll_ci': [lo, hi],
                     'delta_nll_relative_pct': 100 * d / float(base.mean()),
                     'argmax_agreement': float(np.mean(agree[name])), 'mean_kl_from_fp32': float(np.mean(kl[name]))}
    return out


@torch.inference_mode()
def continuation_nll(model32, prompts, gen):
    """Mean NLL of generated tokens under the fp32 model, per row."""
    seq = torch.cat([prompts, gen], 1)
    logits, _ = model32(seq[:, :-1].contiguous(), seq[:, 1:].contiguous())
    logp = F.log_softmax(logits.float(), -1)[:, prompts.size(1) - 1:]
    return (-logp.gather(-1, gen.unsqueeze(-1)).squeeze(-1)).mean(1)


@torch.inference_mode()
def free_running(model32, candidates, val_path, stories, p=32, n=224):
    prompts = story_prompts(val_path, 0, p, stories).cuda()
    ref_pool = DecoderPool(model32, 'adaptive')
    results = {}
    ref_rows, ref_nll = [], []
    for s in range(0, stories, 16):
        x = prompts[s:s + 16]
        ref = ref_pool.get(x, n)[0].generate(x, n)
        ref_rows.append(ref)
        ref_nll.append(continuation_nll(model32, x, ref[:, p:]))
    ref_all, ref_nll = torch.cat(ref_rows), torch.cat(ref_nll).cpu().numpy()
    results['fp32_exact_reference'] = {'mean_continuation_nll': float(ref_nll.mean()), **eos_stats(ref_all, p)}
    for name, (model, factory) in candidates.items():
        pool = DecoderPool(model, 'full', factory=factory)
        outs = torch.cat([pool.get(prompts[s:s + 16], n)[0].generate(prompts[s:s + 16], n)
                          for s in range(0, stories, 16)])
        first, near = [], 0
        for r in range(stories):
            diff = (outs[r, p:] != ref_all[r, p:]).nonzero()
            if len(diff):
                i = int(diff[0])
                first.append(i)
                logits, _ = reference_step(model32, ref_all[r:r + 1, :p + i])
                top = logits[0, -1].float().topk(2).values
                near += int(float(top[0] - top[1]) < NEAR_TIE_LOGIT_MARGIN)
        cand_nll = torch.cat([continuation_nll(model32, prompts[s:s + 16], outs[s:s + 16, p:])
                              for s in range(0, stories, 16)]).cpu().numpy()
        d, lo, hi = paired_ci(cand_nll - ref_nll)
        results[name] = {
            'identical_sequences': stories - len(first), 'diverged': len(first),
            'first_divergence_p10_p50_p90': [float(x) for x in np.percentile(first, [10, 50, 90])] if first else None,
            'first_divergence_is_near_tie': near,
            'mean_continuation_nll': float(cand_nll.mean()), 'delta_nll_vs_fp32_greedy': d, 'delta_ci': [lo, hi],
            **eos_stats(outs, p)}
    return results


def eos_stats(tokens, p):
    _, lengths = mask_after_eos(tokens, p, 0)
    emitted = (tokens[:, p:] == 0).any(1)
    return {'eos_share': float(emitted.float().mean()), 'median_length': float(lengths.float().median())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', type=Path, default=Path('out-tinystories-v2/baseline-2500.pt'))
    ap.add_argument('--val', type=Path, default=Path('data/tinystories_v2/val.bin'))
    ap.add_argument('--windows', type=int, default=1024)
    ap.add_argument('--stories', type=int, default=128)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    model32 = load_model(args.checkpoint)
    models = {name: model32 if dtype == torch.float32 else copy.deepcopy(model32).to(dtype)
              for name, dtype in DTYPES.items()}
    out = {'checkpoint': str(args.checkpoint), 'matmul_fp32_precision': 'tf32',
           'teacher_forced': teacher_forced(models, args.val, args.windows)}
    print(json.dumps(out['teacher_forced'], indent=1), flush=True)
    candidates = {f'{name}_compiled_triton_full_graph': (models[name], CompiledTritonGraphDecoder) for name in DTYPES}
    out['free_running'] = free_running(model32, candidates, args.val, args.stories)
    print(json.dumps(out['free_running'], indent=1), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
