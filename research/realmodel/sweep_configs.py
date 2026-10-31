"""End-to-end sweep of int8 kernel launch configs through the graph decoder (V3 optimisation evidence).

    python -m research.realmodel.sweep_configs --out runs/<stamp>/config_sweep.json

KernelAgent-Ada picked one global config on warm-L2 microbenchmarks; this measures what matters, end-to-end decode, for a
few candidate configs (global and per-shape). For each: a fresh GraphGreedyDecoder is captured with the config, then
(a) pure GPU time per decode step = CUDA-event time of 200 back-to-back graph replays / 200 (no host in the loop), and
(b) streamed decode tok/s of a 256-token generation (3 reps). Same process, one model; configs are applied at capture time.
"""

import argparse
import json
import statistics
from pathlib import Path

CANDIDATES = {
    'shipped_splitk8_bn64_bk64': {'*': ('dot16_splitk', 64, 64, 8, 4)},
    'dot16_bn64_bk128': {'*': ('dot16', 64, 128, 0, 4)},
    'dot16_bn32_bk256': {'*': ('dot16', 32, 256, 0, 4)},
    'dot16_bn32_bk128': {'*': ('dot16', 32, 128, 0, 4)},
    'dot16_bn64_bk256_w8': {'*': ('dot16', 64, 256, 0, 8)},
    'splitk2_bn64_bk128': {'*': ('dot16_splitk', 64, 128, 2, 4)},
    # per-shape: split only where N gives few output blocks (k/v: N=1024; down/q/o: N=2048), plain dot16 elsewhere
    'mixed_split_small_n': {'*': ('dot16', 64, 128, 0, 4), (2048, 1024): ('dot16_splitk', 64, 128, 4, 4),
                            (6144, 2048): ('dot16_splitk', 64, 128, 4, 4), (2048, 2048): ('dot16_splitk', 64, 128, 2, 4)},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--configs', nargs='*', default=list(CANDIDATES))
    args = ap.parse_args()
    import torch

    from assistant.backends import HFBackend

    from . import int8_linear as il
    from .bench_int8 import MODEL, PROMPT, REVISION
    from .fast_decode import GraphGreedyDecoder
    b = HFBackend(MODEL, revision=REVISION, quant='int8', static_cache=False)
    tok = b.tokenizer
    x = tok.apply_chat_template([{'role': 'user', 'content': PROMPT}], add_generation_prompt=True, enable_thinking=False,
                                return_tensors='pt', return_dict=True)['input_ids'].cuda()
    results = {}
    for name in args.configs:
        cfg = CANDIDATES[name]
        il.DEFAULT = cfg['*']
        il.CONFIGS.clear()
        il.CONFIGS.update({k: v for k, v in cfg.items() if k != '*'})
        torch._dynamo.reset()
        dec = GraphGreedyDecoder(b.model, max_cache_len=512)
        dec.generate(x, 16)  # capture
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        dec.generate(x, 2)  # leaves the cache at position p + 1; 200 replays stay below max_cache_len
        with torch.inference_mode():
            start.record()
            for _ in range(200):
                dec.graph.replay()
            end.record()
            end.synchronize()
        gpu_ms = start.elapsed_time(end) / 200
        rates = []
        for _ in range(3):
            out, st = dec.generate(x, 256)
            rates.append((st['tokens'] - 1) / (st['total_s'] - st['ttft_s']))
        results[name] = {'gpu_ms_per_step': gpu_ms, 'decode_tok_s': rates, 'decode_tok_s_median': statistics.median(rates),
                         'first_tokens': out[:16]}
        print(f'{name:28s} gpu {gpu_ms:.3f} ms/step ({1e3 / gpu_ms:.1f} steps/s)  decode {statistics.median(rates):.1f} tok/s',
              flush=True)
        del dec
        torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
