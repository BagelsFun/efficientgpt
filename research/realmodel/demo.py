"""EfficientGPT V3 demo: Qwen3-1.7B with weight-only int8 + the sync-free CUDA-graph decoder, streamed, with a bf16 check.

    python -m research.realmodel.demo [--prompt "..."] [--tokens 200] [--no-check]

1. Loads Qwen3-1.7B (pinned revision), quantises every linear to int8 (EfficientGPT Triton kernel), captures the decode
   step as one CUDA graph, and streams a greedy answer token by token, printing decode speed and GPU memory.
2. Unless --no-check: reloads the model in bf16 and generates the same prompt with Hugging Face `generate()` (static
   cache), then reports how many leading tokens agree. int8 is an approximate mode, so outputs may diverge after a while;
   measured quality is in BENCHMARK_REPORT.md (quality-v1 39/52 for both, dNLL -0.15%).
Needs the model in the local HF cache and an idle CUDA GPU (8 GB is enough).
"""

import argparse
import sys
import time

MODEL, REVISION = 'Qwen/Qwen3-1.7B', '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prompt', default='Explain in three short paragraphs why LLM decoding on a GPU is memory-bandwidth bound.')
    ap.add_argument('--tokens', type=int, default=200)
    ap.add_argument('--no-check', action='store_true')
    args = ap.parse_args()
    import torch

    from assistant.backends import HFBackend
    print(f'Loading {MODEL} -> weight-only int8 + CUDA-graph decoder ...', flush=True)
    t = time.perf_counter()
    b = HFBackend(MODEL, revision=REVISION, quant='int8', fast_decode=True)
    print(f'loaded in {time.perf_counter() - t:.1f} s; capturing the decode graph on the first request ...', flush=True)
    msg = [{'role': 'user', 'content': args.prompt}]
    b.chat(msg, max_tokens=8)  # compile + capture
    print(f'\n> {args.prompt}\n', flush=True)
    reply = b.chat(msg, max_tokens=args.tokens, on_token=lambda s: print(s, end='', flush=True))
    st = reply.stats
    print(f'\n\n[int8 + graph] {st.completion_tokens} tokens, TTFT {st.ttft_s * 1e3:.0f} ms, decode {st.decode_tok_s:.1f} tok/s, '
          f'GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB peak (incl. load)', flush=True)
    if args.no_check:
        return
    ids8 = b.tokenizer(reply.text).input_ids
    del b
    torch.cuda.empty_cache()
    print('\nChecking against bf16 (Hugging Face generate, static cache) ...', flush=True)
    ref = HFBackend(MODEL, revision=REVISION, quant='bf16', static_cache=True)
    r16 = ref.chat(msg, max_tokens=args.tokens)
    ids16 = ref.tokenizer(r16.text).input_ids
    same = 0
    for a, c in zip(ids8, ids16):
        if a != c:
            break
        same += 1
    # no bf16 speed here: this is the reference's first (compiling) call; measured speeds are in BENCHMARK_REPORT.md
    print(f'[bf16 reference] first {same} of {min(len(ids8), len(ids16))} tokens identical to int8 '
          f'({"identical" if ids8 == ids16 else "diverges later; int8 is approximate"})')
    sys.exit(0)


if __name__ == '__main__':
    main()
