"""Demo 2: a short, labelled performance comparison on this machine (Qwen3-1.7B, batch 1, greedy, 256 new tokens).

    python -m research.showcase.demo_compare [--rounds 1] [--with-torchao] [--out runs/demo_compare]

Runs the same audited harness as the published results (research.realmodel.bench_int8: every mode in a fresh process,
unique prompt prefixes, (n-1)/(total-TTFT) decode rate, per-token timestamps) for a few minutes instead of an hour, then
prints the analysis. One round gives no confidence interval; the published numbers use 4 rounds
(BENCHMARK_REPORT.md section 1a, reports/BENCHMARK_VERIFICATION.md).
Modes: EfficientGPT int8 + CUDA-graph decoder, llama.cpp Q8_0 via Ollama (needs `ollama pull qwen3:1.7b-q8_0`), HF bf16 +
static cache/torch.compile; --with-torchao adds torchao int8 weight-only with max-autotune (slow first compile).
Precision differs by design: bf16 is the reference; int8 and Q8_0 are approximate (see the quality rows of the report).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from ..console import utf8_stdout


def main():
    utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rounds', type=int, default=1)
    ap.add_argument('--with-torchao', action='store_true')
    ap.add_argument('--out', type=Path, default=Path('runs') / f"demo_compare_{time.strftime('%Y%m%d_%H%M%S')}")
    args = ap.parse_args()
    modes = ['int8_graph', 'llamacpp_q8', 'bf16_static'] + (['torchao_int8_static_at'] if args.with_torchao else [])
    print(f'Comparing {modes} for {args.rounds} round(s); each mode loads in its own process (about 1-2 min each).',
          flush=True)
    bench = subprocess.run([sys.executable, '-m', 'research.realmodel.bench_int8', '--out', str(args.out), '--modes', *modes,
                            '--rounds', str(args.rounds), '--reps', '3'])
    if bench.returncode:  # e.g. llamacpp_q8 without Ollama: compare the modes that did run
        print('Some modes failed (see the FAILED lines above); comparing the ones that ran.', flush=True)
    subprocess.run([sys.executable, '-m', 'research.realmodel.analyze_int8', '--dir', str(args.out)], check=True)
    print(f'\nRaw per-process results and REPORT.md: {args.out}')


if __name__ == '__main__':
    main()
