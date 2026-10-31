"""Where does HF generate()'s per-token host overhead come from? (evidence for an upstream issue; V3 found ~2 ms/token)

    python -m research.realmodel.streamer_overhead --out runs/<stamp>/streamer_overhead.json

Qwen3-1.7B bf16, static cache (torch.compile), greedy, 256 new tokens forced (min_new_tokens), one process, interleaved
reps of: (a) no streamer, (b) a no-op BaseStreamer (isolates the streamer hook's own cost), (c) TextStreamer to a null sink,
and (d, e) the same with transformers' `DeferredStopCheck` force-enabled on CUDA (upstream enables it for MPS only, "we should
enable it for cuda if we observe perf gains"); outputs are checked to be token-identical to the default path. Reported: total generation time per
token (prefill included, identical across variants) and the EfficientGPT graph decoder on the same model and prompt for
reference (bf16, no int8, so the comparison isolates the decode loop).
"""

import argparse
import inspect
import io
import json
import statistics
import time
from pathlib import Path

from .bench_int8 import MODEL, PROMPT, REVISION


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reps', type=int, default=6)
    ap.add_argument('--tokens', type=int, default=256)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    import contextlib

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
    from transformers.generation.streamers import BaseStreamer

    from .fast_decode import GraphGreedyDecoder
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16).cuda().eval()
    x = tok.apply_chat_template([{'role': 'user', 'content': PROMPT}], add_generation_prompt=True, enable_thinking=False,
                                return_tensors='pt', return_dict=True)['input_ids'].cuda()

    class NoOp(BaseStreamer):
        def put(self, value):
            pass

        def end(self):
            pass

    sink = io.StringIO()
    variants = {'no_streamer': lambda: None, 'noop_streamer': NoOp,
                'text_streamer': lambda: TextStreamer(tok, skip_prompt=True),
                # transformers' DeferredStopCheck (stop flag read one step late; enabled upstream for MPS only) forced on CUDA
                'deferred_no_streamer': lambda: None,
                'deferred_text_streamer': lambda: TextStreamer(tok, skip_prompt=True)}
    outputs = {}
    from transformers.generation import utils as gen_utils
    original = inspect.getattr_static(gen_utils.DeferredStopCheck, 'is_supported')  # keeps the staticmethod wrapper

    def run(make, name=None):
        deferred = name is not None and name.startswith('deferred')
        gen_utils.DeferredStopCheck.is_supported = (staticmethod(lambda device, cache, cache_is_returned, is_assistant: not is_assistant)
                                                    if deferred else original)
        streamer = make()
        torch.cuda.synchronize()
        t = time.perf_counter()
        with contextlib.redirect_stdout(sink):
            out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=args.tokens,
                                 min_new_tokens=args.tokens, do_sample=False, cache_implementation='static',
                                 pad_token_id=tok.eos_token_id, streamer=streamer)
        torch.cuda.synchronize()
        assert out.shape[1] - x.shape[1] == args.tokens
        outputs.setdefault(name, set()).add(tuple(out[0, x.shape[1]:].tolist()))
        return 1e3 * (time.perf_counter() - t) / args.tokens

    times = {k: [] for k in variants}
    try:
        for name, make in variants.items():  # compile / warm-up each path
            run(make, name)
            run(make, name)
        for i in range(args.reps):
            for k in (list(variants) if i % 2 == 0 else list(variants)[::-1]):
                times[k].append(run(variants[k], k))
    finally:
        gen_utils.DeferredStopCheck.is_supported = original
    with torch.inference_mode():
        dec = GraphGreedyDecoder(model, max_cache_len=512)
        dec.generate(x, 16)
        graph = []
        for _ in range(args.reps):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out, st = dec.generate(x, args.tokens)
            graph.append(1e3 * (time.perf_counter() - t) / args.tokens)
    res = {k: {'ms_per_token': v, 'median': statistics.median(v)} for k, v in times.items()}
    res['efficientgpt_graph_decoder_bf16'] = {'ms_per_token': graph, 'median': statistics.median(graph)}
    res['note'] = 'total generation time / new tokens (prefill included); same prompt, model, GPU, process'
    # every run of every variant (warm-up and timed) must produce the same greedy tokens
    res['all_outputs_identical'] = len(set().union(*outputs.values())) == 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1), encoding='utf-8')
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"{k:34s} {v['median']:.3f} ms/token  ({1e3 / v['median']:.1f} tok/s)")


if __name__ == '__main__':
    main()
