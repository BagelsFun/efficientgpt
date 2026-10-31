"""Where does a Qwen3-1.7B decode token go? Kernel-level profile of the HF static-cache decode (bf16 or int8).

    python -m research.realmodel.profile_decode --quant int8 --out runs/<stamp>/profile_int8.json

After warm-up (compile + CUDA-graph capture), one greedy generation of --tokens tokens runs under torch.profiler (CUDA
activity, which records kernels replayed inside CUDA graphs). Reported per decoded token: wall time, summed GPU kernel time,
and their gap (host/launch/sync overhead: the GPU idles while Python, the streamer or HF generate run), plus the kernel
breakdown by name with call counts. The prefill kernels of the generation are included; with --tokens 129 they are a small
share and are listed separately when their names differ.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from .bench_int8 import MODEL, PROMPT, REVISION


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quant', default='int8')
    ap.add_argument('--tokens', type=int, default=129)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    import torch
    from torch.profiler import ProfilerActivity, profile

    from assistant.backends import HFBackend
    b = HFBackend(MODEL, revision=REVISION, quant=args.quant, static_cache=True)
    msg = [{'role': 'user', 'content': PROMPT}]
    for n in (32, args.tokens):  # compile / capture for the timed length
        b.chat(msg, max_tokens=n)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        reply = b.chat(msg, max_tokens=args.tokens)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    n = reply.stats.completion_tokens
    kern = defaultdict(lambda: [0.0, 0])
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            kern[e.name][0] += e.device_time_total / 1e3  # ms
            kern[e.name][1] += 1
    total_gpu = sum(v[0] for v in kern.values())
    rows = sorted(({'kernel': k[:120], 'ms_total': round(v[0], 3), 'calls': v[1], 'ms_per_token': round(v[0] / n, 4),
                    'share': round(v[0] / total_gpu, 4)} for k, v in kern.items()), key=lambda r: -r['ms_total'])
    res = {'quant': args.quant, 'tokens': n, 'wall_ms_per_token': 1e3 * wall / n,
           'gpu_kernel_ms_per_token': total_gpu / n, 'gap_ms_per_token': 1e3 * wall / n - total_gpu / n,
           'decode_tok_s_streamer': reply.stats.decode_tok_s, 'kernels': rows[:40]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps({k: res[k] for k in res if k != 'kernels'}, indent=1))
    for r in rows[:15]:
        print(f"{r['share']:6.1%} {r['ms_per_token']:8.4f} ms/tok  x{r['calls']:5d}  {r['kernel'][:90]}")


if __name__ == '__main__':
    main()
