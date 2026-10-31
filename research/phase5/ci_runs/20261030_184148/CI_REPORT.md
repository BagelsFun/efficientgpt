# Phase 5 bootstrap latency ratios

Source: `research\phase5\ci_runs\20261030_184148\benchmark_raw.csv`; 10000 two-level bootstrap resamples (rounds, then requests); 95% percentile CI.
Ratio = baseline median / candidate median (>1: candidate faster). WIN needs ratio >= 1.05 and CI low > 1. Prompt=192 full-vs-adaptive is the A/A control (identical capacity).

| B / prompt / gen | baseline -> candidate | base ms | cand ms | ratio | 95% CI | per-round range | rounds x n | verdict |
|---|---|---:|---:|---:|---|---|---|---|
| 1 / 8 / 16 | full_graph -> adaptive_graph | 11.896 | 10.060 | 1.182 | [1.097, 1.255] | 1.057-1.311 | 6 x 15 | WIN |
| 1 / 8 / 16 | prefix_sdpa_eager -> full_graph | 32.540 | 11.896 | 2.735 | [2.637, 2.881] | 2.553-2.850 | 6 x 15 | WIN |
| 1 / 8 / 16 | prefix_sdpa_eager -> adaptive_graph | 32.540 | 10.060 | 3.235 | [3.056, 3.407] | 3.012-3.548 | 6 x 15 | WIN |
| 1 / 64 / 16 | full_graph -> adaptive_graph | 12.035 | 9.969 | 1.207 | [1.139, 1.250] | 1.147-1.267 | 6 x 15 | WIN |
| 1 / 64 / 16 | prefix_sdpa_eager -> full_graph | 32.909 | 12.035 | 2.734 | [2.599, 2.880] | 2.522-2.891 | 6 x 15 | WIN |
| 1 / 64 / 16 | prefix_sdpa_eager -> adaptive_graph | 32.909 | 9.969 | 3.301 | [3.096, 3.410] | 2.992-3.400 | 6 x 15 | WIN |
| 1 / 64 / 64 | full_graph -> adaptive_graph | 42.174 | 35.079 | 1.202 | [1.186, 1.224] | 1.181-1.230 | 6 x 15 | WIN |
| 1 / 64 / 64 | prefix_sdpa_eager -> full_graph | 130.572 | 42.174 | 3.096 | [2.983, 3.181] | 2.942-3.209 | 6 x 15 | WIN |
| 1 / 64 / 64 | prefix_sdpa_eager -> adaptive_graph | 130.572 | 35.079 | 3.722 | [3.594, 3.799] | 3.527-3.790 | 6 x 15 | WIN |
| 1 / 128 / 32 | full_graph -> adaptive_graph | 22.017 | 20.126 | 1.094 | [1.079, 1.114] | 1.077-1.129 | 6 x 15 | WIN |
| 1 / 128 / 32 | prefix_sdpa_eager -> full_graph | 66.563 | 22.017 | 3.023 | [2.917, 3.086] | 2.846-3.129 | 6 x 15 | WIN |
| 1 / 128 / 32 | prefix_sdpa_eager -> adaptive_graph | 66.563 | 20.126 | 3.307 | [3.185, 3.383] | 3.103-3.400 | 6 x 15 | WIN |
| 1 / 128 / 64 | full_graph -> adaptive_graph | 41.730 | 38.310 | 1.089 | [1.047, 1.139] | 1.025-1.178 | 6 x 15 | WIN |
| 1 / 128 / 64 | prefix_sdpa_eager -> full_graph | 129.787 | 41.730 | 3.110 | [3.035, 3.211] | 3.024-3.232 | 6 x 15 | WIN |
| 1 / 128 / 64 | prefix_sdpa_eager -> adaptive_graph | 129.787 | 38.310 | 3.388 | [3.262, 3.592] | 3.224-3.717 | 6 x 15 | WIN |
| 1 / 192 / 32 | full_graph -> adaptive_graph | 21.630 | 22.026 | 0.982 | [0.956, 1.023] | 0.943-1.047 | 6 x 15 | AA_PASS |
| 1 / 192 / 32 | prefix_sdpa_eager -> full_graph | 65.304 | 21.630 | 3.019 | [2.916, 3.094] | 2.876-3.126 | 6 x 15 | WIN |
| 1 / 192 / 32 | prefix_sdpa_eager -> adaptive_graph | 65.304 | 22.026 | 2.965 | [2.848, 3.124] | 2.798-3.272 | 6 x 15 | WIN |
| 16 / 8 / 16 | full_graph -> adaptive_graph | 19.910 | 14.495 | 1.374 | [1.308, 1.416] | 1.256-1.429 | 6 x 15 | WIN |
| 16 / 8 / 16 | prefix_sdpa_eager -> full_graph | 33.340 | 19.910 | 1.675 | [1.628, 1.739] | 1.566-1.791 | 6 x 15 | WIN |
| 16 / 8 / 16 | prefix_sdpa_eager -> adaptive_graph | 33.340 | 14.495 | 2.300 | [2.212, 2.359] | 2.148-2.393 | 6 x 15 | WIN |
| 16 / 64 / 16 | full_graph -> adaptive_graph | 20.251 | 14.893 | 1.360 | [1.296, 1.413] | 1.275-1.432 | 6 x 15 | WIN |
| 16 / 64 / 16 | prefix_sdpa_eager -> full_graph | 32.685 | 20.251 | 1.614 | [1.554, 1.671] | 1.487-1.736 | 6 x 15 | WIN |
| 16 / 64 / 16 | prefix_sdpa_eager -> adaptive_graph | 32.685 | 14.893 | 2.195 | [2.104, 2.275] | 2.048-2.320 | 6 x 15 | WIN |
| 16 / 64 / 64 | full_graph -> adaptive_graph | 77.135 | 55.158 | 1.398 | [1.368, 1.457] | 1.333-1.483 | 6 x 15 | WIN |
| 16 / 64 / 64 | prefix_sdpa_eager -> full_graph | 131.860 | 77.135 | 1.709 | [1.677, 1.768] | 1.658-1.845 | 6 x 15 | WIN |
| 16 / 64 / 64 | prefix_sdpa_eager -> adaptive_graph | 131.860 | 55.158 | 2.391 | [2.331, 2.493] | 2.316-2.535 | 6 x 15 | WIN |
| 16 / 128 / 32 | full_graph -> adaptive_graph | 42.627 | 36.655 | 1.163 | [1.137, 1.197] | 1.124-1.213 | 6 x 15 | WIN |
| 16 / 128 / 32 | prefix_sdpa_eager -> full_graph | 65.523 | 42.627 | 1.537 | [1.491, 1.578] | 1.483-1.603 | 6 x 15 | WIN |
| 16 / 128 / 32 | prefix_sdpa_eager -> adaptive_graph | 65.523 | 36.655 | 1.788 | [1.728, 1.839] | 1.675-1.859 | 6 x 15 | WIN |
| 16 / 128 / 64 | full_graph -> adaptive_graph | 78.666 | 68.898 | 1.142 | [1.112, 1.175] | 1.111-1.212 | 6 x 15 | WIN |
| 16 / 128 / 64 | prefix_sdpa_eager -> full_graph | 131.720 | 78.666 | 1.674 | [1.638, 1.722] | 1.602-1.740 | 6 x 15 | WIN |
| 16 / 128 / 64 | prefix_sdpa_eager -> adaptive_graph | 131.720 | 68.898 | 1.912 | [1.876, 1.958] | 1.839-1.991 | 6 x 15 | WIN |
| 16 / 192 / 32 | full_graph -> adaptive_graph | 44.636 | 45.492 | 0.981 | [0.960, 1.016] | 0.932-1.056 | 6 x 15 | AA_PASS |
| 16 / 192 / 32 | prefix_sdpa_eager -> full_graph | 66.077 | 44.636 | 1.480 | [1.408, 1.531] | 1.391-1.599 | 6 x 15 | WIN |
| 16 / 192 / 32 | prefix_sdpa_eager -> adaptive_graph | 66.077 | 45.492 | 1.453 | [1.420, 1.504] | 1.421-1.555 | 6 x 15 | WIN |

## Verdict counts (full denominator)

- full_graph -> adaptive_graph: AA_PASS x2
- full_graph -> adaptive_graph: WIN x10
- prefix_sdpa_eager -> adaptive_graph: WIN x12
- prefix_sdpa_eager -> full_graph: WIN x12
