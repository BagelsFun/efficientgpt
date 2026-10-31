"""Final analysis of the R1 end-to-end benchmark (addresses the KA-07 review of bench_int8.summarise).

    python -m research.realmodel.analyze_int8 --dir runs/<stamp>/int8_e2e

Speed: per mode, the median of each fresh-process round; the speedup vs bf16_static is the ratio of medians of round
medians, with a bootstrap CI that resamples ROUNDS (the variance between processes; pooling reps as independent understates
it) and the min/max of per-round ratios (rounds paired by index). Quality: teacher-forced NLL difference vs bf16_static on
identical windows (asserted by token hash where recorded; note this forward uses the int8 PREFILL path, i.e. it measures the
quantised weights, not the decode kernel); greedy agreement in tokens (mean common prefix of the first 64 generated tokens,
and exact matches), reported next to a bf16-vs-bf16 noise floor (dynamic vs static cache) so int8's divergence can be
judged against the divergence bf16 already shows between two of its own implementations. Writes REPORT.md.
"""

import argparse
import json
import random
import statistics
from pathlib import Path

from .bench_int8 import MODEL, REVISION
from ..console import utf8_stdout


def rounds(d, mode):
    out = []
    for f in sorted(d.glob(f'{mode}_r*.json')):
        r = json.loads(f.read_text(encoding='utf-8'))
        out.append((r['round'], statistics.median(r['decode_tok_s']), r))
    return sorted(out)


def speed_table(d, modes):
    base = {rnd: med for rnd, med, _ in rounds(d, 'bf16_static')}
    rows = []
    rng = random.Random(0)
    for mode in modes:
        rs = rounds(d, mode)
        if not rs:
            continue
        meds = [m for _, m, _ in rs]
        paired = [m / base[rnd] for rnd, m, _ in rs if rnd in base]
        bmeds = list(base.values())
        boots = sorted(statistics.median(rng.choices(meds, k=len(meds))) / statistics.median(rng.choices(bmeds, k=len(bmeds)))
                       for _ in range(5000))
        last = rs[0][2]
        rows.append({'mode': mode, 'rounds': len(rs), 'tok_s_round_medians': [round(m, 2) for m in meds],
                     'tok_s': statistics.median(meds), 'ratio': statistics.median(meds) / statistics.median(bmeds),
                     'ci': [boots[124], boots[4874]], 'paired_min_max': [min(paired), max(paired)] if paired else None,
                     'ttft_ms': 1e3 * statistics.median(r['ttft_s_median'] for _, _, r in rs),
                     'serving_peak_mib': max(r['torch_peak_allocated_mib'] for _, _, r in rs),
                     # comparable across engines: device memory the process added (nvidia-smi delta for Ollama, whose
                     # size_vram = weights + 4096-ctx KV + buffers is not a torch allocator peak; review R2 #2)
                     'device_used_mib': max((r.get('process_device_used_mib') or r.get('gpu_used_delta_mib') or 0)
                                            for _, _, r in rs),
                     'weights_mib': last.get('weights_allocated_mib'), 'load_peak_mib': last.get('load_peak_allocated_mib'),
                     'first_call_s': last['first_call_s'], 'contended': any(r.get('contended') for _, _, r in rs),
                     'itl_ms_p50': statistics.median(r['itl_ms_p50'] for _, _, r in rs) if 'itl_ms_p50' in last else None,
                     'itl_ms_p95': statistics.median(r['itl_ms_p95'] for _, _, r in rs) if 'itl_ms_p95' in last else None,
                     'e2e_s': statistics.median(r['e2e_s_median'] for _, _, r in rs) if 'e2e_s_median' in last else None,
                     'reserved_mib': max(r.get('torch_peak_reserved_mib') or 0 for _, _, r in rs) or None,
                     'load_s': last.get('load_s'),
                     'all_256_tokens': all(set(r['completion_tokens']) == {256} for _, _, r in rs)})
    return rows


def pairwise(d, cand, bases):
    """cand / base speed per base: ratio of medians of round medians, bootstrap over rounds, paired per-round min-max."""
    c = {rnd: m for rnd, m, _ in rounds(d, cand)}
    out, rng = {}, random.Random(2)
    for b in bases:
        bm = {rnd: m for rnd, m, _ in rounds(d, b)}
        if not bm:
            continue
        cs, bs = list(c.values()), list(bm.values())
        boots = sorted(statistics.median(rng.choices(cs, k=len(cs))) / statistics.median(rng.choices(bs, k=len(bs)))
                       for _ in range(5000))
        paired = [c[r] / bm[r] for r in c if r in bm]
        out[b] = {'ratio': statistics.median(cs) / statistics.median(bs), 'ci': [boots[124], boots[4874]],
                  'paired': [min(paired), max(paired)] if paired else [float('nan')] * 2}
    return out


def prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def quality_table(d):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    q = {}
    for mode, f in (('bf16_static', 'bf16_static_r0.json'), ('int8_static', 'int8_static_r0.json'),
                    ('int8_graph', 'int8_graph_r0.json'), ('bf16_graph', 'bf16_graph_r0.json'),
                    ('torchao_int8_static', 'torchao_int8_static_r0.json'), ('llamacpp_q8', 'llamacpp_q8_r0.json'),
                    ('bf16_dynamic', 'bf16_dynamic_quality.json')):
        p = d / f
        if p.exists() and 'quality' in json.loads(p.read_text(encoding='utf-8')):
            q[mode] = json.loads(p.read_text(encoding='utf-8'))['quality']
    if 'bf16_static' not in q:  # speed-only run (e.g. the max-autotune fairness run)
        return {}
    ref = q['bf16_static']
    ids = lambda qq: qq.get('greedy_64_ids') or [tok(t).input_ids for t in qq['greedy_64']]
    out = {}
    for mode, qq in q.items():
        if mode == 'bf16_static':
            continue
        if 'nll_windows_sha256' in qq and 'nll_windows_sha256' in ref:
            assert qq['nll_windows_sha256'] == ref['nll_windows_sha256'], 'NLL windows differ between modes'
        pre = [prefix(a, b) for a, b in zip(ids(qq), ids(ref))]
        out[mode] = {'delta_nll': None, 'delta_nll_ci': None, 'relative_delta_nll': None}  # no logits via Ollama
        if 'nll_windows' in qq:
            diff = [a - b for a, b in zip(qq['nll_windows'], ref['nll_windows'])]
            rng = random.Random(1)
            boots = sorted(statistics.mean(rng.choices(diff, k=len(diff))) for _ in range(5000))
            out[mode] = {'delta_nll': statistics.mean(diff), 'delta_nll_ci': [boots[124], boots[4874]],
                         'relative_delta_nll': statistics.mean(diff) / statistics.mean(ref['nll_windows'])}
        out[mode] |= {
                     'greedy_mean_common_prefix_tokens': statistics.mean(pre), 'greedy_exact_64': sum(
                         a == b for a, b in zip(ids(qq), ids(ref))), 'n_prompts': len(pre),
                     'quality_v1': (f"{sum(r['passed'] for r in qq['quality_v1'])}/{len(qq['quality_v1'])}"
                                    if 'quality_v1' in qq else None)}
    out['bf16_static'] = {'quality_v1': f"{sum(r['passed'] for r in ref['quality_v1'])}/{len(ref['quality_v1'])}",
                          'nll_mean': statistics.mean(ref['nll_windows'])}
    return out


def main():
    utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', type=Path, required=True)
    args = ap.parse_args()
    speed = speed_table(args.dir, ['bf16_dynamic', 'bf16_static', 'bf16_static_at', 'bf16_graph', 'int8_static', 'int8_graph',
                                   'torchao_int8_static', 'torchao_int8_static_at', 'llamacpp_q8', 'bnb8_dynamic', 'nf4_dynamic'])
    cand = 'int8_graph' if list(args.dir.glob('int8_graph_r*.json')) else 'int8_static'
    pairs = pairwise(args.dir, cand, ['bf16_static', 'bf16_static_at', 'bf16_graph', 'int8_static', 'torchao_int8_static',
                                      'torchao_int8_static_at', 'llamacpp_q8'])
    quality = quality_table(args.dir)
    (args.dir / 'analysis.json').write_text(json.dumps({'speed': speed, 'quality': quality, 'pairs': pairs}, indent=1),
                                            encoding='utf-8')
    lines = ['# R1 end-to-end: Qwen3-1.7B batch-1 decode on RTX 4060 (analysis)', '',
             '| Mode | Rounds | Decode tok/s (median of round medians) | vs HF bf16 + static cache [95% CI over rounds] | paired per-round min-max | TTFT ms | Serving peak MiB (torch; Ollama: size_vram) | Device used MiB | Weights MiB | Load peak MiB |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for r in speed:
        pm = r['paired_min_max']
        lines.append(f"| {r['mode']} | {r['rounds']} | {r['tok_s']:.1f} | {r['ratio']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] | "
                     f"{f'{pm[0]:.3f}-{pm[1]:.3f}' if pm else '-'} | {r['ttft_ms']:.0f} | {r['serving_peak_mib']:.0f} | {r['device_used_mib']:.0f} | "
                     f"{r['weights_mib']} | {r['load_peak_mib']} |")
    lines += ['', f'| EfficientGPT {cand} vs | ratio of round medians [95% CI, round bootstrap] | paired per-round min-max |', '|---|---|---|']
    for k, v in pairs.items():
        lines.append(f"| {k} | {v['ratio']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}] | {v['paired'][0]:.3f}-{v['paired'][1]:.3f} |")
    lines += ['', '| Quality vs bf16_static | ΔNLL [95% CI] (relative) | greedy common prefix, mean tokens of 64 | exact 64-token matches | quality v1 |',
              '|---|---|---|---|---|']
    for mode, v in quality.items():
        if mode == 'bf16_static':
            lines.append(f"| bf16_static (reference) | NLL {v['nll_mean']:.4f} | - | - | {v['quality_v1']} |")
            continue
        nll = (f"{v['delta_nll']:+.4f} [{v['delta_nll_ci'][0]:+.4f}, {v['delta_nll_ci'][1]:+.4f}] "
               f"({100 * v['relative_delta_nll']:+.2f}%)") if v['delta_nll'] is not None else 'n/a (no logits)'
        lines.append(f"| {mode} | {nll} | {v['greedy_mean_common_prefix_tokens']:.1f} | "
                     f"{v['greedy_exact_64']}/{v['n_prompts']} | {v['quality_v1'] or '-'} |")
    if any(r.get('itl_ms_p50') is not None for r in speed):
        lines += ['', '| Mode | inter-token latency p50 / p95 ms | end-to-end request s (256 tok) | peak reserved MiB | load s | '
                      'first call s (compile/capture) |', '|---|---|---|---|---|---|']
        for r in speed:
            if r.get('itl_ms_p50') is None:
                continue
            lines.append(f"| {r['mode']} | {r['itl_ms_p50']:.2f} / {r['itl_ms_p95']:.2f} | {r['e2e_s']:.2f} | "
                         f"{r['reserved_mib'] or '-'} | {r['load_s'] if r['load_s'] is not None else '-'} | {r['first_call_s']} |")
    (args.dir / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
