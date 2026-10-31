"""Phase 5 research: compare PyTorch SDPA-prefix eager and two CUDA Graph policies.

Run from nanoGPT root with YOUR trusted local checkpoint/tokenizer:
    python -m research.phase5.benchmark_review --quick
    python -m research.phase5.benchmark_review --all --rounds 2 --trials 5 --variants 3

The modes run in distinct child processes; rounds rotate process order to
mitigate thermal/order effects. This is a controlled PyTorch baseline, NOT a
production serving engine or evidence of novel technology.
"""

import argparse
import csv
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from ..phase4.benchmark_decode import load_model, make_prompts
from ..phase4.decoder import generate_reference, reference_step
from .adaptive_pool import DecoderPool
from .benchmark_requests import workloads
from .compiled_baseline import CompiledGraphDecoder, CompiledStaticDecoder
from .prefix_eager import PrefixSDPADecoder

MODES = ('prefix_sdpa_eager', 'full_graph', 'adaptive_graph')  # V1 report matrix
EXTRA_MODES = ('compiled_static', 'compiled_full_graph', 'compiled_adaptive_graph',
               'triton_full_graph', 'triton_adaptive_graph',
               'compiled_triton_full_graph', 'compiled_triton_adaptive_graph',
               'dispatch_triton_full_graph', 'dispatch_triton_adaptive_graph')  # opt-in via --modes
# APPROXIMATE modes: weights + KV cache in a low dtype (no token contract; quality per precision_study).
LOW_PRECISION = {'fp16_': torch.float16, 'bf16_': torch.bfloat16}
APPROX_MODES = tuple(prefix + m for prefix in LOW_PRECISION
                     for m in ('full_graph', 'triton_full_graph', 'compiled_triton_full_graph'))
MIB = 1024 * 1024
# Inductor reorders FP32 reductions (measured max |logit diff| 2.3e-3..6.1e-3 vs
# eager), so its greedy tokens may flip at near-ties. Accept a divergence only
# where the reference top-1/top-2 logit margin is below this bound.
NEAR_TIE_LOGIT_MARGIN = 1e-2


def nearest_p95(a):
    a = sorted(a)
    return a[min(len(a) - 1, math.ceil(len(a) * 0.95) - 1)]


def near_tie_match(model, ref, out, margin=NEAR_TIE_LOGIT_MARGIN):
    """(ok, near_ties): rows equal, or first divergence is a reference near-tie."""
    near_ties = 0
    for row in range(ref.size(0)):
        diff = (ref[row] != out[row]).nonzero()
        if len(diff) == 0:
            continue
        i = int(diff[0])
        logits, _ = reference_step(model, ref[row:row + 1, :i])
        top = logits[0, -1].topk(2).values
        if float(top[0] - top[1]) >= margin:
            return False, near_ties
        near_ties += 1
    return True, near_ties


class ModeRunner:
    """Decoder construction, lookup and validation for one mode (shared with benchmark_inprocess)."""

    def __init__(self, model, mode):
        self.reference_model, self.mode = model, mode
        prefix = next((p for p in LOW_PRECISION if mode.startswith(p)), '')
        self.approx = bool(prefix)
        if self.approx:  # a low-precision copy; validated against the fp32 model only for well-formed output
            import copy
            model = copy.deepcopy(model).to(LOW_PRECISION[prefix])
            mode = mode.removeprefix(prefix)
        self.model = model
        # compiled_* (Inductor) and triton_* (custom attention) reorder FP32 sums: near-tie rule
        self.exact = not self.approx and not mode.startswith(('compiled_', 'triton_', 'dispatch_'))
        self.decoders = {}
        self.pool = None
        if mode not in ('prefix_sdpa_eager', 'compiled_static'):
            factory = None
            if mode.startswith('compiled_triton_'):
                from .decode_attention import CompiledTritonGraphDecoder  # needs Triton; imported lazily
                factory = CompiledTritonGraphDecoder
            elif mode.startswith('compiled_'):
                factory = CompiledGraphDecoder
            elif mode.startswith('triton_'):
                from .decode_attention import TritonGraphDecoder
                factory = TritonGraphDecoder
            elif mode.startswith('dispatch_triton_'):  # KernelAgent-Ada single_pass behind the shape rule
                from .decode_attention import DispatchTritonGraphDecoder
                factory = DispatchTritonGraphDecoder
            kind = mode.removeprefix('dispatch_').removeprefix('compiled_').removeprefix('triton_').removesuffix('_graph')
            self.pool = DecoderPool(model, kind, factory=factory)

    def get(self, prompt, count):
        if self.pool is not None:
            return self.pool.get(prompt, count)[0]
        batch = prompt.size(0)
        if batch not in self.decoders:
            self.decoders[batch] = (PrefixSDPADecoder(self.model, batch) if self.mode == 'prefix_sdpa_eager'
                                    else CompiledStaticDecoder(self.model, prompt))
        return self.decoders[batch]

    def check(self, prompt, count):
        """Generate once and validate against the fp32 reference.

        Returns the near-tie count (exact/near-tie modes) or, for approximate
        modes, the number of rows that differ from the fp32 reference.
        """
        ref = generate_reference(self.reference_model, prompt, count)
        out = self.get(prompt, count).generate(prompt, count)
        if self.approx:
            vocab = self.model.config.vocab_size
            if out.shape != ref.shape or not torch.equal(out[:, :prompt.size(1)], prompt) or int(out.min()) < 0 or int(out.max()) >= vocab:
                raise AssertionError(f'Malformed output: {self.mode}')
            return int((out != ref).any(1).sum())
        ok, ties = (torch.equal(ref, out), 0) if self.exact else near_tie_match(self.reference_model, ref, out)
        if not ok:
            nwrong = int((ref != out).sum().item())
            raise AssertionError(f'Invalid greedy tokens: {self.mode}, {tuple(prompt.shape)}+{count}, {nwrong} differences')
        return ties

    @property
    def graph_count(self):
        return self.pool.graph_count if self.pool is not None else 0

    @property
    def capture_ms(self):
        return self.pool.total_capture_ms if self.pool is not None else 0.0

    @property
    def compile_ms(self):
        """compiled_static: compile + warmup; compiled graph modes: compile + warmup + capture."""
        if self.mode == 'compiled_static':
            return sum(d.compile_ms for d in self.decoders.values())
        return self.pool.total_initialization_ms if not self.exact else 0.0


def run_mode(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA hardware is required for the benchmark')
    if not args.tokenizer.is_file():
        raise FileNotFoundError('Tokenizer not found')
    model = load_model(args.checkpoint)
    tok = Tokenizer.from_file(str(args.tokenizer))
    if model.config.block_size < 256:
        raise ValueError('The Phase 5 worklist requires context length >=256')
    cases = workloads(args.quick)
    inputs = {case: make_prompts(tok, model.config.vocab_size,
                                case[0], case[1], args.variants) for case in cases}
    precision = torch.backends.cuda.matmul
    old_precision = precision.fp32_precision
    precision.fp32_precision = 'tf32'
    try:
        torch.cuda.synchronize()
        starting_alloc = torch.cuda.memory_allocated()
        starting_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            runner = ModeRunner(model, args.mode)
            get = runner.get
            def generate(decoder, prompt, count):
                return decoder.generate(prompt, count)
            exact = runner.exact

            setup_start = time.perf_counter()
            near_ties = 0
            # Every tested prompt must match reference output before timing.
            for case in cases:
                for prompt in inputs[case]:
                    near_ties += runner.check(prompt, case[2])
            # All kernels and graph instances have been initialized.
            torch.cuda.synchronize()
            initialization_ms = (time.perf_counter() - setup_start) * 1000
            for case in cases:
                dec = get(inputs[case][0], case[2])
                for _ in range(2):
                    generate(dec, inputs[case][0], case[2])
            torch.cuda.synchronize()
            setup_peak_alloc = torch.cuda.max_memory_allocated()
            setup_peak_reserved = torch.cuda.max_memory_reserved()
            # The setup stage includes reference validation and graph capture,
            # which would otherwise contaminate steady-state peak statistics.
            torch.cuda.reset_peak_memory_stats()
            rows = []
            raw = []  # one entry per timed request, for bootstrap CIs
            for batch, prompt_len, count in cases:
                case = (batch, prompt_len, count)
                dec = get(inputs[case][0], count)
                times = []
                for trial in range(args.trials):
                    indices = list(range(len(inputs[case])))
                    if trial % 2:
                        indices.reverse()
                    for ix in indices:
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        out = generate(dec, inputs[case][ix], count)
                        torch.cuda.synchronize()
                        times.append(1000 * (time.perf_counter() - start))
                        raw.append({'mode':args.mode, 'batch':batch, 'prompt_tokens':prompt_len,
                                    'generated_tokens':count, 'trial':trial, 'variant':ix,
                                    'ms':round(times[-1], 5)})
                        del out
                p50 = statistics.median(times)
                rows.append({
                    'mode':args.mode, 'batch':batch, 'prompt_tokens':prompt_len,
                    'generated_tokens':count, 'timed_requests':len(times),
                    'p50_ms':round(p50,4), 'p95_ms':round(nearest_p95(times),4),
                    'min_ms':round(min(times),4), 'max_ms':round(max(times),4),
                    'output_correctness':'PASS',
                })
            torch.cuda.synchronize()
            summary = {
                'mode':args.mode,
                'torch_version':str(torch.__version__),
                'python_version':platform.python_version(),
                'gpu':torch.cuda.get_device_name(0),
                'model_parameters':sum(t.numel() for t in model.parameters()),
                'context_limit':model.config.block_size,
                'precision':'fp32_matmul_tf32',
                'graphs_resident':runner.graph_count,
                'graph_capture_ms':round(runner.capture_ms,4),
                'compile_ms':round(runner.compile_ms,4),
                'correctness_rule':('approx_no_token_contract' if runner.approx else
                                    'exact' if exact else f'near_tie<{NEAR_TIE_LOGIT_MARGIN}'),
                'near_tie_divergences':near_ties,
                'initialization_and_correctness_ms':round(initialization_ms,4),
                'baseline_allocated_mib':round(starting_alloc/MIB,3),
                'baseline_reserved_mib':round(starting_reserved/MIB,3),
                'setup_peak_allocated_mib':round(setup_peak_alloc/MIB,3),
                'setup_peak_reserved_mib':round(setup_peak_reserved/MIB,3),
                'steady_peak_allocated_mib':round(torch.cuda.max_memory_allocated()/MIB,3),
                'steady_peak_reserved_mib':round(torch.cuda.max_memory_reserved()/MIB,3),
                'correctness':'PASS',
            }
            return {'rows':rows, 'summary':summary, 'raw':raw}
    finally:
        precision.fp32_precision = old_precision


def benchmark_all(args):
    rows=[]
    summaries=[]
    raw_rows=[]
    for rnd in range(args.rounds):
        order = list(args.modes)
        if rnd % 2:
            order.reverse()
        for mode in order:
            print(f'Round {rnd + 1}/{args.rounds}: {mode}', flush=True)
            cmd = [sys.executable, '-m', 'research.phase5.benchmark_review',
                   '--mode', mode, '--trials', str(args.trials),
                   '--variants', str(args.variants),
                   '--checkpoint', str(args.checkpoint),
                   '--tokenizer', str(args.tokenizer)]
            if args.quick:
                cmd.append('--quick')
            proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
            if proc.returncode != 0:
                print(proc.stdout, flush=True)
                print(proc.stderr, file=sys.stderr, flush=True)
                raise RuntimeError(f'Round {rnd+1}: {mode} failed; no comparison is valid')
            marker = 'RESULT_JSON:'
            if marker not in proc.stdout:
                raise RuntimeError(f'Missing results for {mode}')
            result = json.loads(proc.stdout.rsplit(marker,1)[1])
            for row in result['rows']:
                rows.append({'round':rnd+1, **row})
            for row in result['raw']:
                raw_rows.append({'round':rnd+1, **row})
            summaries.append({'round':rnd+1, **result['summary']})
            s=result['summary']
            print(f"  graphs={s['graphs_resident']} setup_peak_alloc={s['setup_peak_allocated_mib']}MiB "
                  f"steady_peak_alloc={s['steady_peak_allocated_mib']}MiB "
                  f"steady_peak_reserved={s['steady_peak_reserved_mib']}MiB correctness={s['correctness']} "
                  f"rule={s['correctness_rule']} near_ties={s['near_tie_divergences']}",flush=True)
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    summaryfile = out.with_name(out.stem+'_memory.csv')
    with summaryfile.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]))
        writer.writeheader();writer.writerows(summaries)
    rawfile = out.with_name(out.stem+'_raw.csv')
    with rawfile.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(raw_rows[0]))
        writer.writeheader();writer.writerows(raw_rows)
    print('\nAGGREGATED ACROSS PROCESS ROUNDS (same batch, prompt, generation)')
    for case in workloads(args.quick):
        values = {}
        for mode in args.modes:
            selected=[x['p50_ms'] for x in rows if x['mode']==mode and
                     (x['batch'],x['prompt_tokens'],x['generated_tokens'])==case]
            values[mode]=statistics.median(selected)
        line = ' '.join(f'{mode}={ms:.3f}ms' for mode, ms in values.items())
        if 'prefix_sdpa_eager' in values and 'adaptive_graph' in values:
            line += f"; adaptive/eager speedup={values['prefix_sdpa_eager']/values['adaptive_graph']:.3f}x"
        print(f'{case}: {line}')
    print(f'CSVs: {out}, {summaryfile} and {rawfile}')
    print('LIMITATIONS: one GPU; synthetic prompts; no industrial engine comparison. '
          'PyTorch SDPA kernel selected automatically, and FP32 may not use FlashAttention. '
          'No EOS, network, serving queue or unpredictable output lengths. '
          'Initialization includes correctness validation; graph_capture_ms is separately recorded.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--mode', choices=MODES + EXTRA_MODES + APPROX_MODES)
    parser.add_argument('--modes', nargs='+', choices=MODES + EXTRA_MODES + APPROX_MODES, default=list(MODES),
                        help='modes for --all/--quick (default: the V1 three)')
    parser.add_argument('--trials',type=int,default=5)
    parser.add_argument('--variants',type=int,default=3)
    parser.add_argument('--rounds',type=int,default=2)
    parser.add_argument('--checkpoint',type=Path,default=Path('out-tinystories-v2/baseline-2500.pt'))
    parser.add_argument('--tokenizer',type=Path,default=Path('data/tinystories_v2/tokenizer.json'))
    parser.add_argument('--output',type=Path,default=Path('research/phase5/review_comparison.csv'))
    args=parser.parse_args()
    if min(args.trials,args.variants,args.rounds) < 1 or args.trials<3 or args.variants<2:
        parser.error('Need >=3 trials, >=2 variants and >=1 rounds')
    if args.mode:
        print('RESULT_JSON:'+json.dumps(run_mode(args)),flush=True)
    else:
        benchmark_all(args)


if __name__=='__main__':
    main()
