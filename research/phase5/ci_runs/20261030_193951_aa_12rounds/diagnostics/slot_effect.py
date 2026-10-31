"""Slot-in-round effect: residual log-latency by process position (mode x case removed).
Usage: python slot_effect.py <run>/benchmark_raw.csv ...   (run from the nanoGPT root)"""
import csv, sys, math, collections, statistics as st
ORDERS = {
 '184148': ['prefix_sdpa_eager', 'full_graph', 'adaptive_graph'],
 'inductor': ['prefix_sdpa_eager', 'full_graph', 'adaptive_graph', 'compiled_static'],
 'compiled_graph': ['full_graph', 'adaptive_graph', 'compiled_full_graph', 'compiled_adaptive_graph'],
 'aa_12rounds': ['full_graph', 'adaptive_graph', 'compiled_full_graph', 'compiled_adaptive_graph'],
}
for path in sys.argv[1:]:
    order = next(v for k, v in ORDERS.items() if k in path)
    med = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        med[(r['mode'], r['batch'], r['prompt_tokens'], r['generated_tokens'], int(r['round']))].append(float(r['ms']))
    logm = {k: math.log(st.median(v)) for k, v in med.items()}
    base = collections.defaultdict(list)
    for (m, b, p, n, rnd), v in logm.items():
        base[(m, b, p, n)].append(v)
    base = {k: st.mean(v) for k, v in base.items()}
    by_slot = collections.defaultdict(list)
    for (m, b, p, n, rnd), v in logm.items():
        seq = order if (rnd - 1) % 2 == 0 else order[::-1]  # benchmark_review reverses odd 0-based rounds
        by_slot[seq.index(m) + 1].append(v - base[(m, b, p, n)])
    print(path, '  '.join(f'slot{s}={100*(math.exp(st.mean(v))-1):+.2f}%' for s, v in sorted(by_slot.items())))
