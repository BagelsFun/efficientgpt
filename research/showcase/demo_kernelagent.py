"""Demo 3: KernelAgent-Ada end to end on the real-model task `int8_gemv` (Qwen3-1.7B decode linear layers).

    python -m research.showcase.demo_kernelagent [--budget 6] [--out runs/demo_kernelagent]

Walks through the verification protocol in six visible steps:
1. Task: shapes taken from Qwen3-1.7B decode (dev shapes for search, disjoint held-out shapes for the final check).
2. Safety screen: an unsafe candidate (imports `os`) is rejected before it runs.
3. Correctness: the default template candidate passes the fp64-truth contract; a mutated candidate (per-channel scale
   dropped) runs in an isolated subprocess and is rejected as incorrect.
4. Search: `local` strategy at a small budget; every candidate is timed as a CUDA graph against eager, Inductor and the
   dense bf16 cuBLAS baseline; incorrect or failing candidates can never win; every result goes to a JSONL history.
5. Held-out: the selected candidate is evaluated once on shapes it was not selected on.
6. End-to-end caveat: microbenchmarks are warm-L2; the shipped config was chosen end to end
   (research/realmodel/sweep_configs.py), because the warm-L2 winner (split-K) was slower inside the model.
Needs an idle CUDA GPU; takes about 2-4 minutes.
"""

import argparse
import json
import time
from pathlib import Path

from ..console import utf8_stdout


def main():
    utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--budget', type=int, default=6)
    ap.add_argument('--out', type=Path, default=Path('runs') / f"demo_kernelagent_{time.strftime('%Y%m%d_%H%M%S')}")
    args = ap.parse_args()
    from research.kernelagent import templates
    from research.kernelagent.evaluate import environment, evaluate
    from research.kernelagent.screen import screen
    from research.kernelagent.search import Search, real_evaluator
    from research.kernelagent.tasks import task
    args.out.mkdir(parents=True, exist_ok=True)
    history = args.out / 'history.jsonl'
    t = task('int8_gemv')
    print('1. Task int8_gemv: y = x @ (W_int8 * s).T, shapes (M, K, N)')
    print(f"   dev (search):      {t['dev']}\n   held-out (final):  {t['heldout']}")

    default = templates.FAMILIES['int8_gemv']['dot16']['default']
    good = templates.render('int8_gemv', 'dot16', default)
    unsafe = 'import os\n' + good
    print(f'\n2. Safety screen on an unsafe candidate -> {screen(unsafe)}')

    res = evaluate('int8_gemv', good, 'dev', timing=False)
    cases = res.get('correctness', [])  # absent when the candidate timed out / crashed / failed to import
    print(f"\n3. Correctness (fp64 truth, edge cases, isolated subprocess)\n   default dot16 candidate: {res['status']} "
          f"({sum(r['ok'] for r in cases)}/{len(cases)} cases)")
    bad = good.replace('out = acc * s[None, :]', 'out = acc')
    assert bad != good, 'mutation pattern not found in the template'
    res_bad = evaluate('int8_gemv', bad, 'dev', timing=False)
    worst = max((r for r in res_bad.get('correctness', []) if not r['ok']), key=lambda r: r.get('err_candidate', 0),
                default={})
    print(f"   mutated candidate (scale dropped): {res_bad['status']} "
          f"(error {worst.get('err_candidate', float('nan')):.3g} > bound {worst.get('bound', float('nan')):.3g})")

    print(f'\n4. Search: strategy=local, budget={args.budget} (CUDA-graph timing, contention-checked) ...', flush=True)
    s = Search('int8_gemv', real_evaluator(history, 4, environment(), contention_ok=False, busy_pct=15.0), args.budget, seed=0)
    try:
        out = s.run('local')
    except SystemExit as e:  # the evaluator refuses timings taken while another program uses the GPU
        print(f'   stopped: {e}\n   Close other GPU programs and re-run; timings on a busy GPU are not meaningful.')
        return
    for row in out['trace']:
        score = f"{row['score_ms'] * 1e3:7.2f} us" if row['score_ms'] != float('inf') else '   rejected'
        print(f"   {row['family']:14s} {json.dumps(row['params']):55s} {row['status']:10s} {score}")
    best = out.get('best')
    if best is None:
        print('   no candidate passed correctness: nothing is selected (by design)')
        return
    print(f"   selected: {best['family']} {best['params']} ({best['score_ms'] * 1e3:.2f} us geometric mean over dev shapes)")

    print('\n5. Held-out evaluation of the selected candidate ...', flush=True)
    held = evaluate('int8_gemv', templates.render('int8_gemv', best['family'], best['params']), 'heldout', timing=True,
                    trials=4)
    hcases = held.get('correctness', [])
    print(f"   status: {held['status']} ({sum(r['ok'] for r in hcases)}/{len(hcases)} cases)")
    for shape, p in held.get('performance', {}).items():
        sp = p.get('speedup_vs_best_existing')
        vs = f"vs dense bf16 cuBLAS {sp['ratio']:.2f}x ({sp['class']})" if sp else 'dense bf16 baseline did not fit'
        print(f"   {shape:13s} {p['median_ms']['candidate'] * 1e3:8.1f} us  {vs}")
    (args.out / 'demo_result.json').write_text(json.dumps({'search': out, 'heldout': held}, indent=1, default=str),
                                               encoding='utf-8')
    print('\n6. Caveat: these are warm-L2 microbenchmarks. The config shipped in research/realmodel/int8_linear.py was chosen'
          '\n   end to end (research/realmodel/sweep_configs.py): split-K won here but was slower inside the model.'
          f'\n\nHistory: {history}\nResult:  {args.out / "demo_result.json"}')


if __name__ == '__main__':
    main()
