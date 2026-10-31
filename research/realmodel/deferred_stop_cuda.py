"""Should transformers' DeferredStopCheck be enabled on CUDA? Correctness and speed evidence for an upstream proposal.

    python -m research.realmodel.deferred_stop_cuda --out runs/<stamp>/deferred_stop_cuda.json
    python -m research.realmodel.deferred_stop_cuda --out runs/<stamp>/deferred_stop_cuda_smollm.json --model HuggingFaceTB/SmolLM2-135M-Instruct

transformers (5.21 and main as of 2026-10-31) defers the per-step stop-flag read only on MPS:
`DeferredStopCheck.is_supported` returns False unless `device.type == "mps"`. The proposed change widens that one check to
`device.type in ("mps", "cuda")` and keeps every other condition (assistant models excluded; a returned cache must be
croppable). This script applies exactly that change (`proposed_is_supported`, a copy of upstream's function with the device
check widened) and compares it with the unmodified library, in one process, on Qwen3-1.7B bf16 greedy:

Regimes
- eager_dynamic: default `generate()` (DynamicCache, no compile). The upstream PR expected no CUDA gain here, because the host
  is slower than the GPU and the sync costs little.
- compiled_static: `cache_implementation="static"` (generate compiles the decode step). The GPU step is shorter than the host's
  work, so the per-step sync stalls the launch queue.

Correctness (both regimes; default vs proposed must match exactly)
- 4 single prompts with max_new_tokens 160 (on Qwen3-1.7B two stop at EOS, two reach the limit) and 1 cut at 6
- a left-padded batch of 3 prompts whose sequences finish at different steps
- a streamer: the streamed text must equal the decoded returned sequence
- return_dict_in_generate + output_scores: sequences and every score tensor equal (a returned StaticCache is not croppable,
  so the deferred path must NOT engage there; that gating is recorded too)
For each case the script records whether DeferredStopCheck actually engaged.

Speed: 256 forced new tokens (min_new_tokens), batch 1, interleaved reps with alternating order after two warm-ups per variant;
ms/token = total generate time / 256 (prefill included, identical in both). Ratio of medians with a 95% bootstrap CI over
reps. GPU utilisation is sampled at start and before / after each timing block, each time after 3 s without GPU work from
this process (nvidia-smi averages over its last sample period); above 15% the run is flagged as contended.
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from .bench_int8 import MODEL, REVISION, gpu_util

PINNED = {MODEL: REVISION, 'HuggingFaceTB/SmolLM2-135M-Instruct': '12fd25f77366fa6b3b4b768ec3050bf629380bac'}

PROMPTS = ['Explain in detail how a GPU executes a matrix multiplication, step by step.',
           'Write a haiku about caches.',
           'What is 17 * 23? Answer with the number only.',
           'List three differences between TCP and UDP.']


def proposed_is_supported(device, cache, cache_is_returned, is_assistant):
    """Upstream `DeferredStopCheck.is_supported` with the device check widened from "mps" to ("mps", "cuda")."""
    if device.type not in ('mps', 'cuda') or is_assistant:
        return False
    if cache is None:
        return True
    return cache.is_croppable or not cache_is_returned


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--reps', type=int, default=8)
    ap.add_argument('--tokens', type=int, default=256)
    ap.add_argument('--regimes', nargs='*', default=['eager_dynamic', 'compiled_static'])
    ap.add_argument('--model', default=MODEL, help='any Hub causal LM with a chat template, e.g. HuggingFaceTB/SmolLM2-135M-Instruct')
    ap.add_argument('--revision', default=None, help='Hub revision (default: pinned for Qwen3-1.7B and SmolLM2-135M-Instruct, else main)')
    args = ap.parse_args()
    import inspect

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
    from transformers.generation import utils as gen_utils

    def idle_util():
        # nvidia-smi averages over its last sample period, so sample only after this process has been idle for a while
        torch.cuda.synchronize()
        time.sleep(3)
        return gpu_util()

    start_util = idle_util()
    revision = args.revision or PINNED.get(args.model, 'main')
    tok = AutoTokenizer.from_pretrained(args.model, revision=revision)
    tok.padding_side = 'left'
    if tok.pad_token is None:  # the padded-batch case needs one; generate() gets eos as pad_token_id anyway
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=revision, dtype=torch.bfloat16).cuda().eval()
    original = inspect.getattr_static(gen_utils.DeferredStopCheck, 'is_supported')
    original_init = gen_utils.DeferredStopCheck.__init__
    engaged = []

    def counting_init(self, *a, **k):
        engaged.append(True)
        original_init(self, *a, **k)

    def enc(prompts):
        texts = [tok.apply_chat_template([{'role': 'user', 'content': p}], add_generation_prompt=True,
                                         enable_thinking=False, tokenize=False) for p in prompts]
        return tok(texts, return_tensors='pt', padding=True).to('cuda')

    class Collect(TextStreamer):
        def __init__(self):
            super().__init__(tok, skip_prompt=True, skip_special_tokens=True)
            self.text = ''

        def on_finalized_text(self, text, stream_end=False):
            self.text += text

    def gen(variant, regime, batch, **kw):
        gen_utils.DeferredStopCheck.is_supported = (staticmethod(proposed_is_supported) if variant == 'proposed'
                                                    else original)
        engaged.clear()
        extra = {'cache_implementation': 'static'} if regime == 'compiled_static' else {}
        with torch.inference_mode():
            out = model.generate(**batch, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id,
                                 **extra, **kw)
        return out, bool(engaged)

    res = {'model': f'{args.model}@{revision}', 'torch': torch.__version__,
           'transformers': __import__('transformers').__version__, 'gpu': torch.cuda.get_device_name(),
           'patch': 'DeferredStopCheck.is_supported: device.type != "mps"  ->  device.type not in ("mps", "cuda")',
           'gpu_util_at_start_pct': start_util, 'regimes': {}}
    gen_utils.DeferredStopCheck.__init__ = counting_init
    try:
        for regime in args.regimes:
            r = res['regimes'][regime] = {'correctness': {}, 'timing': {}}
            cases = [(f'prompt{i}_eos', [p], {'max_new_tokens': 160}) for i, p in enumerate(PROMPTS)]
            cases += [('max_new_tokens_cutoff', [PROMPTS[0]], {'max_new_tokens': 6}),
                      ('left_padded_batch3', PROMPTS[:3], {'max_new_tokens': 160})]
            for name, prompts, kw in cases:
                batch = enc(prompts)
                outs = {v: gen(v, regime, batch, **kw) for v in ('default', 'proposed')}
                a, b = outs['default'][0], outs['proposed'][0]
                r['correctness'][name] = {
                    'identical': a.shape == b.shape and bool(torch.equal(a, b)),
                    'new_tokens': [int((row[batch['input_ids'].shape[1]:] != (tok.pad_token_id or -1)).sum())
                                   for row in b],
                    'deferred_engaged': {v: outs[v][1] for v in outs}}
            batch = enc([PROMPTS[0]])
            streamed = {}
            for v in ('default', 'proposed'):
                s = Collect()
                out, eng = gen(v, regime, batch, max_new_tokens=160, streamer=s)
                decoded = tok.decode(out[0, batch['input_ids'].shape[1]:], skip_special_tokens=True)
                streamed[v] = {'text': s.text, 'matches_returned': s.text.strip() == decoded.strip(), 'engaged': eng}
            r['correctness']['streamer'] = {
                'identical_across_variants': streamed['default']['text'] == streamed['proposed']['text'],
                'stream_matches_returned': {v: streamed[v]['matches_returned'] for v in streamed},
                'deferred_engaged': {v: streamed[v]['engaged'] for v in streamed}}
            outs = {v: gen(v, regime, batch, max_new_tokens=40, return_dict_in_generate=True, output_scores=True)
                    for v in ('default', 'proposed')}
            a, b = outs['default'][0], outs['proposed'][0]
            r['correctness']['return_dict_scores'] = {
                'identical': bool(torch.equal(a.sequences, b.sequences)) and len(a.scores) == len(b.scores)
                and all(torch.equal(x, y) for x, y in zip(a.scores, b.scores)),
                'deferred_engaged': {v: outs[v][1] for v in outs}}

            batch = enc([PROMPTS[0]])
            kw = {'max_new_tokens': args.tokens, 'min_new_tokens': args.tokens}
            for v in ('default', 'proposed'):  # warm-up (compile for the static regime)
                gen(v, regime, batch, **kw)
                gen(v, regime, batch, **kw)
            r['timing']['gpu_util_before_pct'] = idle_util()
            times = {'default': [], 'proposed': []}
            for i in range(args.reps):
                for v in (('default', 'proposed') if i % 2 == 0 else ('proposed', 'default')):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    out, _ = gen(v, regime, batch, **kw)
                    torch.cuda.synchronize()
                    assert out.shape[1] - batch['input_ids'].shape[1] == args.tokens
                    times[v].append(1e3 * (time.perf_counter() - t) / args.tokens)
            rng = random.Random(0)
            boots = sorted(statistics.median(rng.choices(times['default'], k=args.reps))
                           / statistics.median(rng.choices(times['proposed'], k=args.reps)) for _ in range(5000))
            med = {v: statistics.median(times[v]) for v in times}
            r['timing'].update({
                'ms_per_token': times, 'median_ms_per_token': med,
                'speedup_proposed_over_default': med['default'] / med['proposed'],
                'speedup_ci95': [boots[124], boots[4874]],
                'gpu_util_after_pct': idle_util(),
                'contended': max(r['timing']['gpu_util_before_pct'] or 0, start_util or 0) > 15})
            print(f"{regime:16s} default {med['default']:.3f} ms/tok, proposed {med['proposed']:.3f} ms/tok, "
                  f"x{r['timing']['speedup_proposed_over_default']:.3f} CI [{boots[124]:.3f}, {boots[4874]:.3f}]; "
                  f"all correct: {all(c.get('identical', c.get('identical_across_variants')) for c in r['correctness'].values())}",
                  flush=True)
            # release this regime's compiled graphs / caches before the next
            torch.cuda.empty_cache()
    finally:
        gen_utils.DeferredStopCheck.is_supported = original
        gen_utils.DeferredStopCheck.__init__ = original_init
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
