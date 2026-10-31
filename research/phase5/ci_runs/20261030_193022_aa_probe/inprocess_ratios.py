"""In-process eager-graph / compiled-graph ratios from aa_probe JSONs. Usage: python inprocess_ratios.py <probe dir>"""
import json, sys, pathlib, statistics as st
rows = {}
for f in sorted(pathlib.Path(sys.argv[1]).glob('probe_*.json')):
    for case, e in json.loads(f.read_text())['cases'].items():
        eager, comp = e['full_graph->adaptive_graph'], e['compiled_full_graph->compiled_adaptive_graph']
        rows.setdefault(case, []).append((eager['baseline_median_ms'] / comp['baseline_median_ms'],
                                          eager['candidate_median_ms'] / comp['candidate_median_ms']))
for case, v in rows.items():
    print(case, 'full/compiled_full mean %.3f' % st.mean(a for a, _ in v), 'adaptive/compiled_adaptive mean %.3f' % st.mean(b for _, b in v))
