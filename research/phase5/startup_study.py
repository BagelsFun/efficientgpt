"""Fresh-process startup cost and memory per mode, with cold or warm compiler caches.

    python -m research.phase5.startup_study --reps 2 --output research/phase5/ci_runs/<stamp>_startup/startup.json

Each measurement is a new Python process. `cold` points TORCHINDUCTOR_CACHE_DIR and
TRITON_CACHE_DIR at brand-new empty directories (nothing is deleted); `warm`
reuses one directory populated by an earlier process. Per process: import,
model load, then for each of the 12 benchmark workloads the time to obtain its
decoder (graph capture, plus compilation for compiled modes) and its first
request, then the steady-state median of one request type. Memory: PyTorch
peak allocated / reserved, and device memory used by the process measured as
the drop in cudaMemGetInfo free memory after CUDA init (context excluded).
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MARKER = 'RESULT_JSON:'
DEFAULT_MODES = ('full_graph', 'adaptive_graph', 'compiled_adaptive_graph', 'triton_full_graph',
                 'compiled_triton_full_graph', 'fp16_compiled_triton_full_graph')


def child(args):
    t0 = time.perf_counter()
    import torch
    from tokenizers import Tokenizer
    from ..phase4.benchmark_decode import load_model, make_prompts
    from .benchmark_requests import workloads
    from .benchmark_review import ModeRunner
    t_import = time.perf_counter()
    torch.cuda.init()
    free0, total = torch.cuda.mem_get_info()
    t_cuda = time.perf_counter()
    model = load_model(args.checkpoint)
    tok = Tokenizer.from_file(str(args.tokenizer))
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    torch.cuda.synchronize()
    t_model = time.perf_counter()
    rows = []
    with torch.inference_mode():
        runner = ModeRunner(model, args.mode)
        for batch, p, n in workloads(False):
            x = make_prompts(tok, model.config.vocab_size, batch, p, 1)[0]
            torch.cuda.synchronize()
            s = time.perf_counter()
            dec = runner.get(x, n)
            torch.cuda.synchronize()
            built = time.perf_counter()
            dec.generate(x, n)
            torch.cuda.synchronize()
            rows.append({'batch': batch, 'prompt_tokens': p, 'generated_tokens': n,
                         'get_decoder_ms': round(1000 * (built - s), 2),
                         'first_request_ms': round(1000 * (time.perf_counter() - built), 3)})
        t_ready = time.perf_counter()
        x = make_prompts(tok, model.config.vocab_size, 16, 64, 1)[0]
        dec = runner.get(x, 64)
        steady = []
        for _ in range(15):
            torch.cuda.synchronize()
            s = time.perf_counter()
            dec.generate(x, 64)
            torch.cuda.synchronize()
            steady.append(1000 * (time.perf_counter() - s))
    free1, _ = torch.cuda.mem_get_info()
    return {'mode': args.mode, 'cache': args.cache, 'import_s': round(t_import - t0, 3), 'cuda_init_s': round(t_cuda - t_import, 3),
            'model_load_s': round(t_model - t_cuda, 3), 'build_all_12_workloads_s': round(t_ready - t_model, 3),
            'time_to_ready_s': round(t_ready - t0, 3), 'graphs': runner.graph_count,
            'steady_B16_64_64_median_ms': round(statistics.median(steady), 3),
            'peak_allocated_mib': round(torch.cuda.max_memory_allocated() / 2**20, 1),
            'peak_reserved_mib': round(torch.cuda.max_memory_reserved() / 2**20, 1),
            'process_device_used_mib_excl_context': round((free0 - free1) / 2**20, 1),
            'device_total_mib': round(total / 2**20, 1), 'workloads': rows}


def run_all(args):
    work = Path(args.cache_root)
    work.mkdir(parents=True, exist_ok=True)
    warm_dir = Path(tempfile.mkdtemp(prefix='warm_', dir=work))
    results = []
    for mode in args.modes:
        # populate the warm cache once per mode (result discarded, logged as 'prime')
        plan = [('prime', warm_dir)] + [(c, None) for r in range(args.reps) for c in ('cold', 'warm')]
        for cache, fixed in plan:
            cdir = fixed or (warm_dir if cache == 'warm' else Path(tempfile.mkdtemp(prefix='cold_', dir=work)))
            env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=str(cdir / 'inductor'), TRITON_CACHE_DIR=str(cdir / 'triton'),
                       PYTHONDONTWRITEBYTECODE='1')
            cmd = [sys.executable, '-m', 'research.phase5.startup_study', '--child', '--mode', mode, '--cache', cache,
                   '--checkpoint', str(args.checkpoint), '--tokenizer', str(args.tokenizer), '--output', str(args.output)]
            s = time.perf_counter()
            proc = subprocess.run(cmd, text=True, capture_output=True, env=env, check=False)
            wall = time.perf_counter() - s
            if proc.returncode or MARKER not in proc.stdout:
                print(proc.stdout[-3000:], proc.stderr[-3000:], file=sys.stderr)
                raise RuntimeError(f'{mode} {cache} failed')
            r = json.loads(proc.stdout.rsplit(MARKER, 1)[1])
            r['process_wall_s'] = round(wall, 3)
            results.append(r)
            print(f"{mode:32} {cache:5} ready={r['time_to_ready_s']:7.2f}s build={r['build_all_12_workloads_s']:7.2f}s "
                  f"graphs={r['graphs']} steady={r['steady_B16_64_64_median_ms']:.2f}ms peak_alloc={r['peak_allocated_mib']}MiB "
                  f"reserved={r['peak_reserved_mib']}MiB device_used={r['process_device_used_mib_excl_context']}MiB", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--modes', nargs='+', default=list(DEFAULT_MODES))
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--cache-root', default='results/startup_caches')
    ap.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--mode')
    ap.add_argument('--cache', default='warm')
    ap.add_argument('--checkpoint', type=Path, default=Path('out-tinystories-v2/baseline-2500.pt'))
    ap.add_argument('--tokenizer', type=Path, default=Path('data/tinystories_v2/tokenizer.json'))
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.child:
        print(MARKER + json.dumps(child(args)), flush=True)
    else:
        run_all(args)


if __name__ == '__main__':
    main()
