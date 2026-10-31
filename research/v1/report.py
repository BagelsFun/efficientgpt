"""Create an evidence-based EfficientGPT V1 report from *real* benchmark CSVs.

Use only CSVs produced by research.phase5.benchmark_review. Missing, failed or
partial tests are labelled incomplete; no metrics are invented.
"""
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

EXPECTED_MODES = ('prefix_sdpa_eager','full_graph','adaptive_graph')
EXPECTED_CASES = {(batch,p,n) for batch in (1,16)
                    for p,n in ((8,16),(64,16),(64,64),(128,32),(128,64),(192,32))}


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing actual benchmark CSV: {path}")
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def summarize(result_path, memory_path):
    rows = read_csv(result_path)
    memories = read_csv(memory_path)
    if not rows or not memories:
        raise ValueError('Empty result CSV')
    by_mode = defaultdict(lambda: defaultdict(list))
    problems=[]
    for row in rows:
        if row.get('mode') not in EXPECTED_MODES:
            problems.append('Unknown mode: '+str(row.get('mode')))
            continue
        case=(int(row['batch']),int(row['prompt_tokens']),int(row['generated_tokens']))
        if row['output_correctness'] != 'PASS':
            problems.append(f'Correctness failure: {row["mode"]} {case}')
            continue
        by_mode[row['mode']][case].append(float(row['p50_ms']))
    for item in memories:
        if item.get('correctness') != 'PASS':
            problems.append('Mode-level correctness failure: '+str(item.get('mode')))
    all_modes = set(by_mode)
    measured_cases = set.intersection(*(set(by_mode[m]) for m in EXPECTED_MODES)) if all(m in all_modes for m in EXPECTED_MODES) else set()
    is_complete = (not problems and all(EXPECTED_CASES <= set(by_mode[m]) for m in EXPECTED_MODES)
                   and all(len(by_mode[m][case]) >= 2 for m in EXPECTED_MODES for case in EXPECTED_CASES)
                   and all(sum(1 for s in memories if s.get('mode')==m)>=2 for m in EXPECTED_MODES))
    lines=[
      '# EfficientGPT V1 — Reproducible benchmark report', '',
      '**Status:** ' + ('FULL PHASE 5 MATRIX VALIDATED ON THIS MACHINE' if is_complete else 'PARTIAL / NOT YET FULLY VALIDATED'), '',
      'Source: locally generated benchmark CSV files. No independent reproduction.',
      "Model: the owner's trained TinyStories GPT (not a general-purpose assistant).",
      'Reference: PyTorch SDPA eager prefix KV reuse, not vLLM or TensorRT-LLM.',
      'Measurements use synthetic prompts with fixed known maximum outputs; no EOS, serving or network overhead.', '',
      '## Median p50 generation latency across recorded rounds', '',
      '| Batch / prompt / generated | Eager prefix SDPA (ms) | Full graph (ms) | Adaptive graph (ms) | Adaptive vs eager throughput |',
      '|---|---:|---:|---:|---:|',
    ]
    for batch,p,n in sorted(measured_cases):
        key=(batch,p,n)
        v={m:statistics.median(by_mode[m][key]) for m in EXPECTED_MODES}
        ratio=v['prefix_sdpa_eager']/v['adaptive_graph']
        lines.append(f'| {batch} / {p} / {n} | {v["prefix_sdpa_eager"]:.4f} | {v["full_graph"]:.4f} | {v["adaptive_graph"]:.4f} | {ratio:.3f}x |')
    if not measured_cases:
        lines.append('| No comparable verified results yet | — | — | — | — |')
    lines.extend(['', '## Peak PyTorch GPU memory and graph cost per run', '',
      '| Round | Mode | Graphs | Setup peak alloc (MiB) | Steady peak alloc (MiB) | Steady peak reserved (MiB) | Capture (ms) | Init + correctness (ms) |',
      '|---|---|---:|---:|---:|---:|---:|---:|'])
    for s in sorted(memories,key=lambda x:(int(x['round']),x['mode'])):
        lines.append('| {round} | {mode} | {graphs_resident} | {setup_peak_allocated_mib} | {steady_peak_allocated_mib} | {steady_peak_reserved_mib} | {graph_capture_ms} | {initialization_and_correctness_ms} |'.format(**s))
    lines.extend(['', '## Acceptance gates', '',
      f'- Comparison modes present: {len(all_modes)}/{len(EXPECTED_MODES)}',
      f'- Matched workloads: {len(measured_cases)}/{len(EXPECTED_CASES)}',
      '- Required for full validation: exact reference-token match, all 12 workloads for all 3 modes in at least 2 process rounds, completed error-free.',
      '- Graph capture, initialization and memory are not amortized into steady-state p50 times; performance may reverse in low-reuse deployments.',
      '- No novel algorithm, production competitiveness, profitability or external hardware validation is demonstrated.',
    ])
    if problems:
        lines.extend(['', '### Validation problems', ''] + ['- '+p for p in problems])
    return '\n'.join(lines)+'\n', is_complete


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,required=True)
    ap.add_argument('--memory',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--require-full',action='store_true')
    args=ap.parse_args()
    report,complete=summarize(args.results,args.memory)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        raise SystemExit('Refusing to overwrite existing report: '+str(args.output))
    args.output.write_text(report,encoding='utf-8')
    print('Wrote report:',args.output.resolve())
    print('Status:', 'FULL MATRIX' if complete else 'PARTIAL')
    if args.require_full and not complete:
        raise SystemExit('Full matrix acceptance NOT met (report still saved)')

if __name__=='__main__':
    main()
