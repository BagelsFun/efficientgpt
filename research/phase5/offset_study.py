"""Cross-process performance offsets: same mode in many fresh processes, with GPU-state telemetry.

    python -m research.phase5.offset_study --processes 16 --output research/phase5/ci_runs/<stamp>_offsets/offsets.json

Every child runs the SAME mode and workloads (B=1 and B=16, prompt 192, 32
tokens; both use the 256-capacity graph), timing each request with wall
clock and CUDA events. Meanwhile `nvidia-smi -lms 200` samples SM/memory
clocks, temperature, power state and throttle reasons. Per process:
medians, host overhead (wall - GPU), mean clocks during timing, buffer
addresses. The parent reports between-process spread and how much of it the
telemetry explains (correlations; within-process drift is reported too).
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

MARKER = 'RESULT_JSON:'
QUERY = 'clocks.sm,clocks.mem,temperature.gpu,pstate,clocks_throttle_reasons.active'


def child(args):
    import torch
    from tokenizers import Tokenizer
    from ..phase4.benchmark_decode import load_model, make_prompts
    from .benchmark_review import ModeRunner
    model = load_model(args.checkpoint)
    tok = Tokenizer.from_file(str(args.tokenizer))
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    out = {'mode': args.mode, 'cases': {}}
    with torch.inference_mode():
        runner = ModeRunner(model, args.mode)
        cases = {(b, 192, 32): make_prompts(tok, model.config.vocab_size, b, 192, 3) for b in (1, 16)}
        decs = {c: runner.get(p[0], c[2]) for c, p in cases.items()}
        for c, d in decs.items():
            for _ in range(5):
                d.generate(cases[c][0], c[2])
        torch.cuda.synchronize()
        smi = subprocess.Popen(['nvidia-smi', f'--query-gpu={QUERY}', '--format=csv,noheader,nounits', '-lms', '200'],
                               stdout=subprocess.PIPE, text=True)
        time.sleep(0.3)
        for c, d in decs.items():
            wall, gpu = [], []
            for i in range(args.requests):
                x = cases[c][i % 3]
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                t = time.perf_counter()
                start.record()
                d.generate(x, c[2])
                end.record()
                torch.cuda.synchronize()
                wall.append(1000 * (time.perf_counter() - t))
                gpu.append(start.elapsed_time(end))
            key0 = d.engine.keys[0].data_ptr()
            out['cases'][f'{c[0]}_{c[1]}_{c[2]}'] = {
                'wall_ms': wall, 'gpu_ms': gpu, 'wall_median': statistics.median(wall), 'gpu_median': statistics.median(gpu),
                'host_overhead_median_ms': statistics.median(w - g for w, g in zip(wall, gpu)),
                'first_half_vs_second_half': statistics.median(gpu[len(gpu) // 2:]) / statistics.median(gpu[:len(gpu) // 2]),
                'key0_ptr_mod_2MiB': key0 % (2 << 20), 'key0_ptr_mod_64KiB': key0 % (64 << 10)}
        smi.terminate()
        lines = [l.split(', ') for l in smi.communicate(timeout=5)[0].strip().splitlines() if l.count(',') >= 4]
    num = lambda i: [float(l[i]) for l in lines if l[i].replace('.', '').isdigit()]
    out['telemetry'] = {'samples': len(lines), 'sm_mhz_mean': statistics.mean(num(0)) if num(0) else None,
                        'sm_mhz_min': min(num(0)) if num(0) else None, 'mem_mhz_mean': statistics.mean(num(1)) if num(1) else None,
                        'mem_mhz_min': min(num(1)) if num(1) else None, 'temp_mean': statistics.mean(num(2)) if num(2) else None,
                        'pstates': sorted({l[3] for l in lines}), 'throttle_reasons': sorted({l[4] for l in lines})}
    return out


def corr(a, b):
    if len(a) < 3 or statistics.pstdev(a) == 0 or statistics.pstdev(b) == 0:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (len(a) * statistics.pstdev(a) * statistics.pstdev(b))


def run_all(args):
    procs = []
    for i in range(args.processes):
        cmd = [sys.executable, '-m', 'research.phase5.offset_study', '--child', '--mode', args.mode, '--requests', str(args.requests),
               '--checkpoint', str(args.checkpoint), '--tokenizer', str(args.tokenizer), '--output', str(args.output)]
        p = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if p.returncode or MARKER not in p.stdout:
            print(p.stdout[-2000:], p.stderr[-2000:], file=sys.stderr)
            raise RuntimeError(f'process {i} failed')
        r = json.loads(p.stdout.rsplit(MARKER, 1)[1])
        r['process'] = i
        procs.append(r)
        t = r['telemetry']
        print(f"proc {i:2}: " + ' '.join(f"{k}: gpu {v['gpu_median']:.3f} wall {v['wall_median']:.3f} host {v['host_overhead_median_ms']:.3f}"
                                        for k, v in r['cases'].items())
              + f" | sm {t['sm_mhz_mean']} mem {t['mem_mhz_mean']} (min {t['mem_mhz_min']}) temp {t['temp_mean']} {t['pstates']} {t['throttle_reasons']}", flush=True)
    summary = {}
    for case in procs[0]['cases']:
        gpu = [p['cases'][case]['gpu_median'] for p in procs]
        wall = [p['cases'][case]['wall_median'] for p in procs]
        host = [p['cases'][case]['host_overhead_median_ms'] for p in procs]
        within = [statistics.pstdev(p['cases'][case]['gpu_ms']) / p['cases'][case]['gpu_median'] for p in procs]
        tel = {k: [p['telemetry'][k] for p in procs] for k in ('sm_mhz_mean', 'mem_mhz_mean', 'temp_mean')}
        summary[case] = {
            'between_process_gpu_cv': statistics.pstdev(gpu) / statistics.mean(gpu),
            'between_process_wall_cv': statistics.pstdev(wall) / statistics.mean(wall),
            'gpu_min_max_ratio': max(gpu) / min(gpu), 'wall_min_max_ratio': max(wall) / min(wall),
            'within_process_gpu_cv_median': statistics.median(within),
            'host_overhead_ms_range': [min(host), max(host)],
            'corr_gpu_vs': {k: corr(gpu, [float(x) for x in v]) if None not in v else None for k, v in tel.items()},
            'corr_gpu_vs_key0_mod_2MiB': corr(gpu, [p['cases'][case]['key0_ptr_mod_2MiB'] for p in procs]),
            'corr_gpu_vs_process_index': corr(gpu, list(range(len(procs)))),
        }
        print(case, json.dumps(summary[case]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'processes': procs, 'summary': summary}, indent=2), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--processes', type=int, default=16)
    ap.add_argument('--requests', type=int, default=40)
    ap.add_argument('--mode', default='full_graph')
    ap.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
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
