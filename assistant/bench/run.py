"""Measure a backend: speed (TTFT, decode tok/s, prefill), memory, and assistant-quality-v1 accuracy.

    python -m assistant.bench.run --backend hf:Qwen/Qwen3-1.7B --quant bf16 --out results/assistant/bench/<name>.json
    python -m assistant.bench.run --backend ollama:qwen3.5:4b --out ...

Greedy decoding (temperature 0), thinking off unless --think. Every response is
saved next to the summary so grades can be audited.
"""

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

from . import quality_v1
from .grading import grade

FILLER = Path('docs/PHASE5_CONTRACT.md')


def gpu_used_mib():
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return float(out[0]) if out else float('nan')


def speed_prompts(count_chars=4):
    ctx = FILLER.read_text(encoding='utf-8')
    return {'short': 'Explain in detail how a GPU executes a matrix multiplication, step by step.',
            'medium': 'Read this design note:\n\n' + ctx[:2400] + '\n\nExplain in detail what it describes and why it matters.',
            'long': 'Read this design note:\n\n' + (ctx * 3)[:8000] + '\n\nExplain in detail what it describes and why it matters.'}


def run_speed(backend, repeats, max_tokens):
    """Each request starts with a unique id so no backend can reuse a cached prompt prefix (cold prefill every time)."""
    import uuid
    rows = {}
    for name, prompt in speed_prompts().items():
        fresh = lambda: [{'role': 'user', 'content': f'[request {uuid.uuid4().hex}]\n{prompt}'}]
        backend.chat(fresh(), max_tokens=32)  # warmup (model load / kernels), different prefix
        stats = [backend.chat(fresh(), max_tokens=max_tokens).stats for _ in range(repeats)]
        rows[name] = {'prompt_tokens': stats[0].prompt_tokens,
                      'completion_tokens': [s.completion_tokens for s in stats],
                      'ttft_s_median': statistics.median(s.ttft_s for s in stats),
                      'decode_tok_s_median': statistics.median(s.decode_tok_s for s in stats),
                      'decode_tok_s_all': [round(s.decode_tok_s, 2) for s in stats],
                      'prompt_tok_s_median': statistics.median(s.prompt_tok_s for s in stats),
                      'prompt_tokens_per_ttft_s_median': statistics.median(s.prompt_tokens / s.ttft_s for s in stats),
                      'total_s_median': statistics.median(s.total_s for s in stats), 'timing_source': stats[0].source}
        print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[name].items()}), flush=True)
    return rows


def run_quality(backend, think, limits):
    records = []
    for item in quality_v1.items():
        t = time.perf_counter()
        reply = backend.chat([{'role': 'user', 'content': item['prompt']}], max_tokens=limits[item['category']], think=think)
        ok, why = grade(item, reply.text)
        records.append({'id': item['id'], 'category': item['category'], 'split': item['split'], 'passed': ok, 'reason': why,
                        'completion_tokens': reply.stats.completion_tokens, 'seconds': round(time.perf_counter() - t, 2),
                        'response': reply.text, 'thinking_chars': len(reply.thinking)})
    summary = {}
    for split in ('dev', 'test', 'all'):
        for cat in ('code', 'math', 'mcq', 'all'):
            sel = [r for r in records if (split == 'all' or r['split'] == split) and (cat == 'all' or r['category'] == cat)]
            summary[f'{split}/{cat}'] = f"{sum(r['passed'] for r in sel)}/{len(sel)}"
    return summary, records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--backend', required=True)
    ap.add_argument('--quant', default='bf16')
    ap.add_argument('--revision', default=None)
    ap.add_argument('--adapter', default=None, help='LoRA adapter directory (HF backends only)')
    ap.add_argument('--think', action='store_true')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--speed-tokens', type=int, default=256)
    ap.add_argument('--skip-speed', action='store_true')
    ap.add_argument('--skip-quality', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    from ..cli import make_backend
    out = {'backend': args.backend, 'quant': args.quant, 'revision': args.revision, 'adapter': args.adapter, 'think': args.think,
           'protocol': quality_v1.PROTOCOL, 'decoding': 'greedy (temperature 0)'}
    if args.backend.startswith('ollama:'):
        from ..backends import OllamaBackend
        backend = OllamaBackend(args.backend.split(':', 1)[1], options={'presence_penalty': 0.0})
        backend.unload()
        time.sleep(2)
        out['gpu_used_before_load_mib'] = gpu_used_mib()
        out['ollama_options'] = backend.options
    else:
        import torch
        torch.cuda.init()
        free0, _ = torch.cuda.mem_get_info()
        out['gpu_used_before_load_mib'] = gpu_used_mib()
        t = time.perf_counter()
        backend = make_backend(args.backend, args.quant, args.revision, adapter=args.adapter)
        out['load_s'] = round(time.perf_counter() - t, 2)
    if not args.skip_speed:
        out['speed'] = run_speed(backend, args.repeats, args.speed_tokens)
    if not args.skip_quality:
        limits = {'code': 768, 'math': 512, 'mcq': 48} if not args.think else {'code': 2048, 'math': 2048, 'mcq': 1024}
        out['quality_limits'] = limits
        out['quality'], records = run_quality(backend, args.think, limits)
        print('quality', json.dumps(out['quality']), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.with_suffix('.responses.jsonl').open('w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r) + '\n')
    out['gpu_used_after_mib'] = gpu_used_mib()
    if args.backend.startswith('ollama:'):
        out['ollama_ps'] = [{k: m.get(k) for k in ('name', 'size', 'size_vram', 'context_length')} for m in backend.loaded_models()]
        backend.unload()
    else:
        import torch
        free1, _ = torch.cuda.mem_get_info()
        out['torch_peak_allocated_mib'] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        out['torch_peak_reserved_mib'] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
        out['process_device_used_mib'] = round((free0 - free1) / 2**20, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in out.items() if k not in ('speed',)}, indent=1))


if __name__ == '__main__':
    main()
