"""Isolated-process variable-request CUDA Graph reuse benchmark.

Run from the existing nanoGPT ROOT with the trusted local checkpoint:
  python -m research.phase5.benchmark_requests --quick
  python -m research.phase5.benchmark_requests --all

Measures TOTAL greedy generation calls (prefill+decode+dispatch), not graph
capture, model load, tokenizer setup, network, production scheduling, or real
requests. Captures and resident graph memory are reported separately. This is
NOT an optimised industrial inference engine comparison. Only load your OWN
trusted checkpoint; it is loaded with weights_only=False.
"""

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from ..phase4.benchmark_decode import load_model, make_prompts
from ..phase4.decoder import generate_reference
from .adaptive_pool import DecoderPool

MIB = 1024*1024
BUCKETS = (80, 128, 192, 256)


def workloads(quick):
    cases = [(8,16), (64,16), (64,64), (128,32), (128,64), (192,32)]
    if quick:
        return [(1,8,16), (1,64,32), (1,128,32)]
    return [(batch,p,n) for batch in (1,16) for p,n in cases]


def p95(nums):
    nums = sorted(nums)
    return nums[min(len(nums)-1, math.ceil(len(nums)*0.95)-1)]


def run_mode(args):
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA GPU with working CUDA is required")
    if not args.tokenizer.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")
    model = load_model(args.checkpoint)
    if model.config.block_size < 256:
        raise ValueError("This planned workload requires a >=256-token context window")
    tok = Tokenizer.from_file(str(args.tokenizer))
    schedule = workloads(args.quick)
    input_sets = {}
    for batch, p, count in schedule:
        input_sets[(batch,p,count)] = make_prompts(
            tok, model.config.vocab_size, batch, p, args.variants)
    previous_precision = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    rows = []
    try:
        with torch.inference_mode():
            torch.cuda.synchronize()
            initial_alloc = torch.cuda.memory_allocated()
            initial_res = torch.cuda.memory_reserved()
            torch.cuda.reset_peak_memory_stats()
            pool = DecoderPool(model, args.mode,
                               capacities=tuple(c for c in BUCKETS if c <= model.config.block_size))
            # Capture each graph, then explicitly validate different input VALUES.
            # Caches are reused when multiple request lengths fall in a bucket.
            for key in schedule:
                batch,p,count = key
                prompts = input_sets[key]
                decoder,cap = pool.get(prompts[0], count)
                for prompt in prompts:
                    expected = generate_reference(model,prompt,count)
                    got = decoder.generate(prompt,count)
                    torch.cuda.synchronize()
                    if not torch.equal(expected,got):
                        raise AssertionError(f"CORRECTNESS FAIL: mode={args.mode}, workload={key}")
                    del expected,got
            for key in schedule:
                p = input_sets[key][0]
                d,_ = pool.get(p,key[2])
                for _ in range(2):
                    d.generate(p,key[2])
            torch.cuda.synchronize()

            for workload in schedule:
                batch,p,n = workload
                prompts = input_sets[workload]
                timings=[]
                for trial in range(args.trials):
                    order = range(len(prompts)) if trial%2==0 else reversed(range(len(prompts)))
                    for index in order:
                        prompt=prompts[index]
                        torch.cuda.synchronize()
                        start=time.perf_counter()
                        decoder,cap=pool.get(prompt,n)
                        out=decoder.generate(prompt,n)
                        torch.cuda.synchronize()
                        timings.append((time.perf_counter()-start)*1000)
                        del out
                rows.append({
                    'mode':args.mode,'batch':batch,'prompt_tokens':p,'generated_tokens':n,
                    'cache_capacity':cap,'num_requests':len(timings),
                    'p50_ms':round(statistics.median(timings),4),
                    'p95_ms':round(p95(timings),4),
                    'min_ms':round(min(timings),4),
                    'max_ms':round(max(timings),4),
                    'output_correctness':'PASS',
                })
            torch.cuda.synchronize()
            summary={
                'mode':args.mode,'workloads':len(schedule),'graphs_resident':pool.graph_count,
                'total_capture_ms':round(pool.total_capture_ms,4),
                'total_graph_initialization_ms':round(pool.total_initialization_ms,4),
                'baseline_allocated_mib':round(initial_alloc/MIB,3),
                'baseline_reserved_mib':round(initial_res/MIB,3),
                'peak_allocated_mib':round(torch.cuda.max_memory_allocated()/MIB,3),
                'peak_reserved_mib':round(torch.cuda.max_memory_reserved()/MIB,3),
                'total_steady_medians_ms':round(sum(x['p50_ms'] for x in rows),4),
                'correctness':'PASS',
            }
            return {'summary':summary,'rows':rows}
    finally:
        torch.backends.cuda.matmul.fp32_precision=previous_precision


def all_modes(args):
    import platform
    mode_results=[]
    for mode in ("full", "adaptive"):
        print(f"Running isolated {mode} CUDA-Graph pool experiment...", flush=True)
        cmd=[sys.executable,'-m','research.phase5.benchmark_requests',
             '--mode',mode,'--trials',str(args.trials),
             '--variants',str(args.variants),
             '--checkpoint',str(args.checkpoint),
             '--tokenizer',str(args.tokenizer)]
        if args.quick:
            cmd.append('--quick')
        result=subprocess.run(cmd, capture_output=True,text=True,check=False)
        if result.returncode:
            print(result.stdout,flush=True)
            print(result.stderr,file=sys.stderr,flush=True)
            raise RuntimeError(f"Mode {mode} failed with exit status {result.returncode}")
        marker='RESULT_JSON:'
        if marker not in result.stdout:
            raise RuntimeError("Missing structured result for "+mode)
        mode_results.append(json.loads(result.stdout.rsplit(marker,1)[1]))
    rows=[row for result in mode_results for row in result['rows']]
    summaries=[result['summary'] for result in mode_results]
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary_path=output.with_name(output.stem+'_summary.csv')
    with summary_path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]))
        writer.writeheader();writer.writerows(summaries)
    print('MODE SUMMARY (PyTorch allocation, not device-wide VRAM)')
    for s in summaries:
        print(f"{s['mode']}: graphs={s['graphs_resident']} "
              f"peak_alloc={s['peak_allocated_mib']}MiB "
              f"peak_reserved={s['peak_reserved_mib']}MiB "
              f"init={s['total_graph_initialization_ms']}ms "
              f"capture={s['total_capture_ms']}ms correctness={s['correctness']}")
    print('\nPER-WORKLOAD STEADY-STATE MEDIANS (lower ms is better)')
    for baseline in mode_results[0]['rows']:
        candidate=next(r for r in mode_results[1]['rows']
                       if (r['batch'],r['prompt_tokens'],r['generated_tokens']) ==
                       (baseline['batch'],baseline['prompt_tokens'],baseline['generated_tokens']))
        pct=(baseline['p50_ms']/candidate['p50_ms']-1)*100
        print(f"B={baseline['batch']:2}, prompt={baseline['prompt_tokens']:3}, "
              f"gen={baseline['generated_tokens']:2}: "
              f"full={baseline['p50_ms']:7.3f}ms adaptive={candidate['p50_ms']:7.3f}ms "
              f"throughput_change={pct:+.2f}%")
    print('\nFILES: ',output,summary_path)
    print('LIMITATIONS: one GPU, synthetic prompts, no EOS, fixed known token budgets, '
          'no network or serving overhead, capture/init reported separately. '
          'No claim of novelty or industry competitiveness.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--all',action='store_true')
    parser.add_argument('--mode',choices=('full','adaptive'))
    parser.add_argument('--checkpoint',type=Path,
                        default=Path('out-tinystories-v2/baseline-2500.pt'))
    parser.add_argument('--tokenizer',type=Path,
                        default=Path('data/tinystories_v2/tokenizer.json'))
    parser.add_argument('--output',type=Path,
                        default=Path('research/phase5/variable_requests.csv'))
    parser.add_argument('--trials',type=int,default=3)
    parser.add_argument('--variants',type=int,default=2)
    args=parser.parse_args()
    if args.trials<2 or args.variants<2:
        parser.error('Use at least two trials and two request variants')
    if args.mode is not None:
        print('RESULT_JSON:'+json.dumps(run_mode(args)),flush=True)
    else:
        all_modes(args)


if __name__=='__main__':
    main()
