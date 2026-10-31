"""Why is int8 time-to-first-token worse? Kernel profile of one prefill (the benchmark's ~61-token chat prompt).

    python -m research.realmodel.profile_prefill --out runs/<stamp>/profile_prefill.json

For bf16 and int8 (weights quantised exactly as in the benchmark), warms up, then profiles one eager prefill forward
(`logits_to_keep=1`, as the graph decoder does) with torch.profiler (CUDA activity) and reports: wall time of the prefill,
summed GPU kernel time, and kernel time grouped into dequantisation copies (int8 -> bf16 casts), GEMMs, and everything else.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from .bench_int8 import MODEL, PROMPT, REVISION


def group(name):
    n = name.lower()
    if 'gemm' in n or 'gemv' in n or 'cutlass' in n or 'sm80_xmma' in n or 'ampere' in n or 'sm90' in n:
        return 'gemm'
    if 'copy' in n or 'elementwise' in n and 'convert' in n:
        return 'copy/cast (incl. int8->bf16 dequantisation)'
    if 'fmha' in n or 'attention' in n or 'flash' in n:
        return 'attention'
    return 'other'


def profile(quant):
    import torch
    from torch.profiler import ProfilerActivity, profile as prof

    from assistant.backends import HFBackend
    b = HFBackend(MODEL, revision=REVISION, quant=quant)
    ids = b.tokenizer.apply_chat_template([{'role': 'user', 'content': PROMPT}], add_generation_prompt=True,
                                          enable_thinking=False, return_tensors='pt', return_dict=True)['input_ids'].cuda()
    with torch.inference_mode():
        for _ in range(3):
            b.model(input_ids=ids, logits_to_keep=1)
        torch.cuda.synchronize()
        walls = []
        for _ in range(10):  # unprofiled wall time
            t = time.perf_counter()
            b.model(input_ids=ids, logits_to_keep=1)
            torch.cuda.synchronize()
            walls.append(1e3 * (time.perf_counter() - t))
        with prof(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            b.model(input_ids=ids, logits_to_keep=1)
            torch.cuda.synchronize()
    groups = defaultdict(float)
    kernels = defaultdict(lambda: [0.0, 0])
    for e in p.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            groups[group(e.name)] += e.device_time_total / 1e3
            kernels[e.name][0] += e.device_time_total / 1e3
            kernels[e.name][1] += 1
    top = sorted(kernels.items(), key=lambda kv: -kv[1][0])[:8]
    res = {'quant': quant, 'prompt_tokens': ids.shape[1], 'prefill_wall_ms_median': sorted(walls)[len(walls) // 2],
           'gpu_kernel_ms': sum(groups.values()), 'groups_ms': dict(groups),
           'top_kernels': [{'kernel': k[:100], 'ms': round(v[0], 3), 'calls': v[1]} for k, v in top]}
    del b
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    out = {q: profile(q) for q in ('bf16', 'int8')}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1), encoding='utf-8')
    for q, r in out.items():
        print(f"{q}: prompt {r['prompt_tokens']} tokens, prefill wall {r['prefill_wall_ms_median']:.1f} ms, "
              f"GPU kernels {r['gpu_kernel_ms']:.1f} ms -> " + ', '.join(f'{k} {v:.1f}' for k, v in r['groups_ms'].items()))
        for t in r['top_kernels'][:5]:
            print(f"    {t['ms']:7.2f} ms x{t['calls']:4d}  {t['kernel']}")


if __name__ == '__main__':
    main()
