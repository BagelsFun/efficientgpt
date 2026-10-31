"""End-to-end Qwen3-1.7B decode benchmark: EfficientGPT int8 vs bf16 (dynamic / static-compiled) vs bitsandbytes.

    python -m research.realmodel.bench_int8 --out runs/<stamp>/int8_e2e [--rounds 2] [--modes ...] [--quality]

Every mode runs in its own fresh process (weights of two modes do not fit next to each other reliably on 8 GB, and a fresh
process removes cache/compile carry-over); rounds alternate the mode order. Speed: greedy, 256 new tokens, unique prompt
prefixes (no prefix reuse), one warm-up + one compile-settling call, then `--reps` timed calls; decode tok/s excludes the
first token. Memory: torch peak allocated and process device memory. Quality (`--quality`, once per mode):
teacher-forced NLL on fixed windows of this repository's English docs vs the bf16 model (same tokens), greedy agreement of
the first 64 generated tokens with bf16 on 16 prompts, and assistant-quality-v1 (52 items, grading v1.1).
GPU utilisation is recorded before each child; a mode measured above --busy-pct is flagged.
Contracts: bf16 modes are the reference; int8 / bnb8 / nf4 are APPROXIMATE (quantised weights).
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

MODEL, REVISION = 'Qwen/Qwen3-1.7B', '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
MODES = {  # name: (quant, static_cache)
    'bf16_dynamic': ('bf16', False), 'bf16_static': ('bf16', True), 'int8_static': ('int8', True),
    'int8_dynamic': ('int8', False), 'bnb8_dynamic': ('bnb8', False), 'nf4_dynamic': ('nf4', False),
    'torchao_int8_static': ('torchao_int8', True),  # strong baseline: torchao int8 weight-only + same compiled path
    'llamacpp_q8': ('ollama:qwen3:1.7b-q8_0', None),
    # Inductor max-autotune variants (review: torchao's int8 weight-only mm fuses the dequantisation only under
    # max-autotune in torch 2.11; the bf16 baseline gets the same chance). Set via TORCHINDUCTOR_MAX_AUTOTUNE in the child.
    'torchao_int8_static_at': ('torchao_int8', True), 'bf16_static_at': ('bf16', True),
    # V3: sync-free graph decode loop (research/realmodel/fast_decode.GraphGreedyDecoder) for int8 and, for fairness, bf16
    'int8_graph': ('int8', 'graph'), 'bf16_graph': ('bf16', 'graph')}  # strong baseline: llama.cpp (via Ollama), same model, Q8_0
PROMPT = 'Explain in detail how a GPU executes a matrix multiplication, step by step.'
NLL_FILES = ['docs/PHASE5_CONTRACT.md', 'reports/PHASE5_REPORT.md', 'docs/ASSISTANT.md', 'reports/KERNELAGENT_ADA_REPORT.md']
AGREE_PROMPTS = ['Write a Python function that reverses a linked list.', 'What is a CUDA graph?',
                 'Summarise the causes of the French Revolution in three sentences.', 'Explain KV caching in transformers.',
                 'Give three tips for writing unit tests.', 'What does the softmax function do?',
                 'Translate "good morning" into Spanish and French.', 'How does binary search work?',
                 'Explain what a GPU warp is.', 'Write a haiku about autumn.', 'What is the derivative of x^3?',
                 'Describe the difference between TCP and UDP.', 'What is quantization in neural networks?',
                 'List the planets of the solar system.', 'Explain gradient descent briefly.', 'What is a hash map?']


def gpu_util():
    try:
        return float(subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=10).stdout.split()[0])
    except Exception:
        return float('nan')


def nll_windows(tok, n_windows=24, length=512):
    ids = []
    for f in NLL_FILES:
        ids += tok(Path(f).read_text(encoding='utf-8')).input_ids
    step = max(length, (len(ids) - length) // n_windows)
    return [ids[i:i + length] for i in range(0, len(ids) - length, step)][:n_windows]


def latency_fields(stats):
    """Per-request latency summary: inter-token gaps (ms) pooled over requests -> p50 / p95, plus end-to-end time."""
    gaps = sorted(1e3 * (b - a) for s in stats for a, b in zip(s.token_times, s.token_times[1:]))
    pct = lambda q: gaps[min(len(gaps) - 1, int(q * len(gaps)))] if gaps else float('nan')
    return {'itl_ms_p50': pct(0.50), 'itl_ms_p95': pct(0.95), 'itl_samples': len(gaps),
            'e2e_s_median': statistics.median(s.total_s for s in stats)}


def child(mode, reps, quality, out, parts=('nll', 'greedy', 'quality_v1')):
    quant, static = MODES[mode]
    if mode.endswith('_at'):
        import os
        os.environ['TORCHINDUCTOR_MAX_AUTOTUNE'] = '1'  # before torch is imported in this child
    if quant.startswith('ollama:'):
        return ollama_child(mode, quant.split(':', 1)[1], reps, quality, out, parts)
    import torch
    from assistant.backends import HFBackend
    res_env = {'max_autotune': bool(torch._inductor.config.max_autotune)}
    torch.cuda.init()
    free0, _ = torch.cuda.mem_get_info()
    t = time.perf_counter()
    b = HFBackend(MODEL, revision=REVISION, quant=quant, static_cache=static is True, fast_decode=static == 'graph')
    res = {'mode': mode, 'quant': quant, 'static_cache': static, 'load_s': round(time.perf_counter() - t, 2), **res_env,
           # load peak includes transient copies (int8 quantisation keeps bf16 + int8 + a float row block briefly)
           'load_peak_allocated_mib': round(torch.cuda.max_memory_allocated() / 2**20, 1),
           'weights_allocated_mib': round(torch.cuda.memory_allocated() / 2**20, 1)}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()  # torch_peak_allocated_mib below = steady-state serving peak
    msg = lambda: [{'role': 'user', 'content': f'[request {uuid.uuid4().hex}]\n{PROMPT}'}]
    t = time.perf_counter()
    b.chat(msg(), max_tokens=64)
    res['first_call_s'] = round(time.perf_counter() - t, 2)
    b.chat(msg(), max_tokens=256)  # settles compilation for the timed length
    stats = [b.chat(msg(), max_tokens=256).stats for _ in range(reps)]
    res['decode_tok_s'] = [round(s.decode_tok_s, 2) for s in stats]
    res['decode_tok_s_median'] = statistics.median(s.decode_tok_s for s in stats)
    res['ttft_s_median'] = statistics.median(s.ttft_s for s in stats)
    res['completion_tokens'] = [s.completion_tokens for s in stats]
    res['torch_peak_allocated_mib'] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    res['torch_peak_reserved_mib'] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    res |= latency_fields(stats)
    res['process_device_used_mib'] = round((free0 - torch.cuda.mem_get_info()[0]) / 2**20, 1)
    if quality:
        res['quality'] = quality_part(b, torch, parts)
    Path(out).write_text(json.dumps(res, indent=1), encoding='utf-8')


def ollama_child(mode, model, reps, quality, out, parts):
    """llama.cpp through Ollama: decode tok/s measured client-side like the HF modes ((n - 1) / (total - TTFT), including
    per-token streaming), with Ollama's server-side eval rate kept for reference. Memory: `ollama ps` size_vram."""
    from assistant.backends import OllamaBackend
    b = OllamaBackend(model, num_ctx=4096, options={'presence_penalty': 0.0})
    b.unload()
    time.sleep(2)
    used0 = _gpu_used()
    t = time.perf_counter()
    msg = lambda: [{'role': 'user', 'content': f'[request {uuid.uuid4().hex}]\n{PROMPT}'}]
    b.chat(msg(), max_tokens=64)
    res = {'mode': mode, 'quant': 'ollama ' + model, 'static_cache': None, 'load_s': None,
           'first_call_s': round(time.perf_counter() - t, 2)}
    b.chat(msg(), max_tokens=256)
    stats = [b.chat(msg(), max_tokens=256).stats for _ in range(reps)]
    client = [(s.completion_tokens - 1) / (s.total_s - s.ttft_s) for s in stats]
    res['decode_tok_s'] = [round(x, 2) for x in client]
    res['decode_tok_s_median'] = statistics.median(client)
    res['server_eval_tok_s'] = [round(s.decode_tok_s, 2) for s in stats]
    res['ttft_s_median'] = statistics.median(s.ttft_s for s in stats)
    res['completion_tokens'] = [s.completion_tokens for s in stats]
    ps = [m for m in b.loaded_models() if m.get('name', '').startswith(model.split(':')[0])]
    res['ollama_size_vram_mib'] = round(ps[0]['size_vram'] / 2**20, 1) if ps else None
    res |= latency_fields(stats)
    res['torch_peak_allocated_mib'] = res['ollama_size_vram_mib']  # nearest equivalent: model + KV + buffers in VRAM
    res['gpu_used_delta_mib'] = round(_gpu_used() - used0, 1)
    res['weights_allocated_mib'] = res['load_peak_allocated_mib'] = None
    if quality:
        from assistant.bench import quality_v1
        from assistant.bench.grading import grade
        q = {'greedy_64': [b.chat([{'role': 'user', 'content': p}], max_tokens=64).text for p in AGREE_PROMPTS]}
        if 'quality_v1' in parts:
            limits = {'code': 768, 'math': 512, 'mcq': 48}
            q['quality_v1'] = [{'id': i['id'], 'split': i['split'], 'passed': grade(i, b.chat(
                [{'role': 'user', 'content': i['prompt']}], max_tokens=limits[i['category']]).text)[0]} for i in quality_v1.items()]
        res['quality'] = q
    b.unload()
    Path(out).write_text(json.dumps(res, indent=1), encoding='utf-8')


def _gpu_used():
    try:
        return float(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=10).stdout.split()[0])
    except Exception:
        return float('nan')


def quality_part(b, torch, parts=('nll', 'greedy', 'quality_v1')):
    import hashlib
    from assistant.bench import quality_v1
    from assistant.bench.grading import grade
    q = {}
    windows = nll_windows(b.tokenizer)
    q['nll_windows_sha256'] = hashlib.sha256(str(windows).encode()).hexdigest()  # pairing check (review KA-07 #1)
    nlls = []
    with torch.inference_mode():
        for w in windows:
            x = torch.tensor([w], device='cuda')
            logits = b.model(x).logits.float()
            nlls.append(float(torch.nn.functional.cross_entropy(logits[0, :-1], x[0, 1:])))
    q['nll_windows'] = nlls
    q['greedy_64'] = [b.chat([{'role': 'user', 'content': p}], max_tokens=64).text for p in AGREE_PROMPTS]
    q['greedy_64_ids'] = [b.tokenizer(t).input_ids for t in q['greedy_64']]
    if 'quality_v1' not in parts:
        return q
    limits = {'code': 768, 'math': 512, 'mcq': 48}
    rows = []
    for item in quality_v1.items():
        text = b.chat([{'role': 'user', 'content': item['prompt']}], max_tokens=limits[item['category']]).text
        rows.append({'id': item['id'], 'split': item['split'], 'passed': grade(item, text)[0]})
    q['quality_v1'] = rows
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--modes', nargs='+', default=['bf16_dynamic', 'bf16_static', 'int8_static', 'bnb8_dynamic', 'nf4_dynamic'])
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--quality', action='store_true')
    ap.add_argument('--quality-modes', nargs='*', default=['bf16_static', 'int8_static'])
    ap.add_argument('--busy-pct', type=float, default=15.0)
    ap.add_argument('--child', default=None)
    ap.add_argument('--child-out', default=None)
    ap.add_argument('--child-quality', action='store_true')
    ap.add_argument('--quality-parts', default='nll,greedy,quality_v1')
    args = ap.parse_args()
    if args.child:
        child(args.child, args.reps, args.child_quality, args.child_out, tuple(args.quality_parts.split(',')))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    runs = []
    for r in range(args.rounds):
        order = args.modes if r % 2 == 0 else args.modes[::-1]
        for mode in order:
            out = args.out / f'{mode}_r{r}.json'
            if out.exists():
                runs.append(json.loads(out.read_text(encoding='utf-8')) | {'round': r})
                continue
            time.sleep(3)
            util = gpu_util()
            cmd = [sys.executable, '-m', 'research.realmodel.bench_int8', '--out', str(args.out), '--child', mode,
                   '--child-out', str(out), '--reps', str(args.reps)] + (['--child-quality'] if args.quality and r == 0 and mode in args.quality_modes else [])
            p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
            (args.out / f'{mode}_r{r}.log').write_text(p.stdout[-4000:] + p.stderr[-8000:], encoding='utf-8')
            if p.returncode or not out.exists():
                print(f'{mode} r{r}: FAILED rc={p.returncode} (see {mode}_r{r}.log)', flush=True)
                continue
            res = json.loads(out.read_text(encoding='utf-8')) | {'round': r, 'gpu_util_before': util,
                                                                  'contended': not util <= args.busy_pct}
            out.write_text(json.dumps(res, indent=1), encoding='utf-8')
            runs.append(res)
            print(f"{mode:13s} r{r} decode {res['decode_tok_s_median']:.1f} tok/s  ttft {res['ttft_s_median'] * 1e3:.0f} ms  "
                  f"peak {res['torch_peak_allocated_mib']:.0f} MiB  util {util}", flush=True)
    (args.out / 'summary.json').write_text(json.dumps(summarise(runs), indent=1), encoding='utf-8')


def summarise(runs):
    import random
    by = {}
    for r in runs:
        by.setdefault(r['mode'], []).append(r)
    out = {}
    for mode, rs in by.items():
        toks = [x for r in rs for x in r['decode_tok_s']]
        out[mode] = {'decode_tok_s_median': statistics.median(toks), 'per_round_medians': [r['decode_tok_s_median'] for r in rs],
                     'ttft_ms_median': 1e3 * statistics.median(r['ttft_s_median'] for r in rs),
                     'peak_mib': max(r['torch_peak_allocated_mib'] for r in rs), 'load_s': rs[0]['load_s'],
                     'first_call_s': rs[0]['first_call_s'], 'any_contended': any(r.get('contended') for r in rs)}
    if 'bf16_static' in by:  # bootstrap CI of decode speed ratio vs the fastest bf16 baseline (resample reps)
        base = [x for r in by['bf16_static'] for x in r['decode_tok_s']]
        rng = random.Random(0)
        for mode, rs in by.items():
            cand = [x for r in rs for x in r['decode_tok_s']]
            boots = sorted(statistics.median(rng.choices(cand, k=len(cand))) / statistics.median(rng.choices(base, k=len(base)))
                           for _ in range(5000))
            out[mode]['speedup_vs_bf16_static'] = {'ratio': statistics.median(cand) / statistics.median(base),
                                                   'ci': [boots[124], boots[4874]]}
    qual = {r['mode']: r['quality'] for r in runs if 'quality' in r}
    if 'bf16_static' in qual or 'bf16_dynamic' in qual:
        ref = qual.get('bf16_static') or qual['bf16_dynamic']
        for mode, q in qual.items():
            if 'nll_windows' not in q:  # llama.cpp via Ollama exposes no logits; see analyze_int8 for its quality rows
                continue
            d = [a - b for a, b in zip(q['nll_windows'], ref['nll_windows'])]
            rng = random.Random(1)
            boots = sorted(statistics.mean(rng.choices(d, k=len(d))) for _ in range(5000))
            agree = [_common_prefix(a, b) for a, b in zip(q['greedy_64'], ref['greedy_64'])]
            out[mode]['quality'] = {'nll_mean': statistics.mean(q['nll_windows']), 'delta_nll_vs_bf16': statistics.mean(d),
                                    'delta_nll_ci': [boots[124], boots[4874]],
                                    'greedy_identical_vs_bf16': f"{sum(a == b for a, b in zip(q['greedy_64'], ref['greedy_64']))}/{len(agree)}",
                                    'quality_v1': f"{sum(r['passed'] for r in q['quality_v1'])}/{len(q['quality_v1'])}",
                                    'quality_v1_test': f"{sum(r['passed'] for r in q['quality_v1'] if r['split'] == 'test')}/"
                                                       f"{sum(r['split'] == 'test' for r in q['quality_v1'])}"}
    return out


def _common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


if __name__ == '__main__':
    main()
