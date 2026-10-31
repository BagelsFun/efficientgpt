"""Independent audit of the single-pass decode-attention numerical contract (KA-04 tolerance change).

    python -m research.phase5.attention_error_audit --out runs/<stamp>/attention_error_audit.json

Part 1 (kernel): over many random cases (dtype x shape x logit scale x positions), the max-abs error vs an fp64 SDPA
reference for masked SDPA (same dtype), split-K and single_pass; distributions, the single/split error ratio, and how often
each kernel exceeds the per-case "2x SDPA" rule. Part 2 (model): teacher-forced decoding of the real checkpoint with each
attention implementation, comparing per-step logits against the same model in fp64 (max |dlogit|, argmax agreement,
top-1 margin at disagreements). Correctness-only: safe to run while other GPU work is active.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from .decode_attention import decode_attention, decode_attention_single_pass


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def kernel_part(cases_per_config, seed):
    g = torch.Generator(device='cuda').manual_seed(seed)
    out = {}
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        errs = {'sdpa': [], 'split': [], 'single': []}
        ratio, over = [], {'split': 0, 'single': 0}
        n = 0
        for batch, cap in [(1, 80), (1, 256), (2, 128), (4, 256), (8, 65), (16, 128), (16, 256), (32, 64)]:
            for scale in (1.0, 4.0, 8.0):
                for _ in range(cases_per_config):
                    q = (torch.randn(batch, 6, 1, 64, generator=g, device='cuda') * scale).to(dtype)
                    k = (torch.randn(batch, 6, cap, 64, generator=g, device='cuda') * scale).to(dtype)
                    v = torch.randn(batch, 6, cap, 64, generator=g, device='cuda').to(dtype)
                    pos = torch.randint(0, cap, (batch,), generator=g, device='cuda')
                    allowed = torch.arange(cap, device='cuda').view(1, 1, 1, cap) <= pos.view(-1, 1, 1, 1)
                    ref = F.scaled_dot_product_attention(q.double(), k.double(), v.double(), attn_mask=allowed)
                    e = {'sdpa': F.scaled_dot_product_attention(q, k, v, attn_mask=allowed),
                         'split': decode_attention(q, k, v, pos), 'single': decode_attention_single_pass(q, k, v, pos)}
                    e = {name: float((o.double() - ref).abs().max()) for name, o in e.items()}
                    for name, val in e.items():
                        errs[name].append(val)
                    ratio.append(e['single'] / max(e['split'], 1e-30))
                    floor = 1e-6 if dtype == torch.float32 else 1e-3
                    for name in over:
                        over[name] += e[name] > max(2 * e['sdpa'], floor)
                    n += 1
        out[str(dtype).removeprefix('torch.')] = {
            'cases': n,
            'error': {name: {'median': statistics.median(v), 'p99': pct(v, 0.99), 'max': max(v)} for name, v in errs.items()},
            'single_over_split': {'median': statistics.median(ratio), 'p99': pct(ratio, 0.99), 'max': max(ratio)},
            'cases_over_2x_sdpa_rule': over}
    return out


@torch.inference_mode()
def model_part(checkpoint, sequences, length, seed):
    import copy

    from ..phase4.benchmark_decode import load_model
    from ..phase4.compact_cache import CompactStaticKVDecoder
    from .compiled_baseline import decode_logits
    model = load_model(checkpoint)
    ref_model = copy.deepcopy(model).double()
    g = torch.Generator().manual_seed(seed)
    impls = {'sdpa': None, 'split': decode_attention, 'single': decode_attention_single_pass}
    stats = {name: {'max_abs_dlogit': 0.0, 'argmax_disagree': 0, 'margins_at_disagree': []} for name in impls}
    steps = 0
    for s in range(sequences):
        batch = (1, 16)[s % 2]
        toks = torch.randint(0, model.config.vocab_size, (batch, length), generator=g).cuda()
        engines = {name: CompactStaticKVDecoder(model, batch, length) for name in impls}
        ref = CompactStaticKVDecoder(ref_model, batch, length)
        for t in range(length):
            for eng in [ref, *engines.values()]:
                eng.token.copy_(toks[:, t:t + 1])
                eng.position.fill_(t)
            r = decode_logits(ref)[:, -1].double()
            top2 = r.topk(2, -1).values
            for name, attn in impls.items():
                lo = decode_logits(engines[name], attention=attn)[:, -1].double()
                st = stats[name]
                st['max_abs_dlogit'] = max(st['max_abs_dlogit'], float((lo - r).abs().max()))
                bad = lo.argmax(-1) != r.argmax(-1)
                st['argmax_disagree'] += int(bad.sum())
                st['margins_at_disagree'] += (top2[:, 0] - top2[:, 1])[bad].tolist()
            steps += batch
    for st in stats.values():
        m = st.pop('margins_at_disagree')
        st['max_fp64_margin_at_disagree'] = max(m) if m else None
    return {'row_steps': steps, 'length': length, 'sequences': sequences, 'impls': stats}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cases', type=int, default=10, help='random cases per (shape, scale) config and dtype')
    ap.add_argument('--sequences', type=int, default=8)
    ap.add_argument('--length', type=int, default=256)
    ap.add_argument('--checkpoint', type=Path, default=Path('out-tinystories-v2/baseline-2500.pt'))
    ap.add_argument('--seed', type=int, default=20261031)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    old = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
    try:
        res = {'kernel': kernel_part(args.cases, args.seed),
               'model': model_part(args.checkpoint, args.sequences, args.length, args.seed)}
    finally:
        torch.backends.cuda.matmul.fp32_precision = old
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
