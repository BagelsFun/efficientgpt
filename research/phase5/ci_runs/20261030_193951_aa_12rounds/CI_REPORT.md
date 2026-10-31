# Phase 5 bootstrap latency ratios

Source: `research\phase5\ci_runs\20261030_193951_aa_12rounds\benchmark_raw.csv`; 10000 two-level bootstrap resamples (rounds, then requests); 95% percentile CI.
Ratio = baseline median / candidate median (>1: candidate faster). WIN needs ratio >= 1.05 and CI low > 1. Prompt=192 full-vs-adaptive is the A/A control (identical capacity).

| B / prompt / gen | baseline -> candidate | base ms | cand ms | ratio | 95% CI | per-round range | rounds x n | verdict |
|---|---|---:|---:|---:|---|---|---|---|
| 1 / 8 / 16 | full_graph -> adaptive_graph | 11.578 | 9.962 | 1.162 | [1.111, 1.212] | 1.070-1.277 | 12 x 15 | WIN |
| 1 / 8 / 16 | full_graph -> compiled_full_graph | 11.578 | 11.064 | 1.046 | [1.001, 1.098] | 0.947-1.193 | 12 x 15 | INCONCLUSIVE |
| 1 / 8 / 16 | adaptive_graph -> compiled_adaptive_graph | 9.962 | 9.039 | 1.102 | [1.061, 1.136] | 1.040-1.160 | 12 x 15 | WIN |
| 1 / 8 / 16 | full_graph -> compiled_adaptive_graph | 11.578 | 9.039 | 1.281 | [1.225, 1.320] | 1.191-1.355 | 12 x 15 | WIN |
| 1 / 8 / 16 | compiled_full_graph -> compiled_adaptive_graph | 11.064 | 9.039 | 1.224 | [1.157, 1.273] | 1.093-1.308 | 12 x 15 | WIN |
| 1 / 64 / 16 | full_graph -> adaptive_graph | 11.564 | 9.857 | 1.173 | [1.126, 1.214] | 1.012-1.265 | 12 x 15 | WIN |
| 1 / 64 / 16 | full_graph -> compiled_full_graph | 11.564 | 11.184 | 1.034 | [1.001, 1.068] | 0.909-1.157 | 12 x 15 | INCONCLUSIVE |
| 1 / 64 / 16 | adaptive_graph -> compiled_adaptive_graph | 9.857 | 9.116 | 1.081 | [1.037, 1.124] | 0.990-1.163 | 12 x 15 | WIN |
| 1 / 64 / 16 | full_graph -> compiled_adaptive_graph | 11.564 | 9.116 | 1.268 | [1.224, 1.301] | 1.167-1.377 | 12 x 15 | WIN |
| 1 / 64 / 16 | compiled_full_graph -> compiled_adaptive_graph | 11.184 | 9.116 | 1.227 | [1.183, 1.262] | 1.109-1.299 | 12 x 15 | WIN |
| 1 / 64 / 64 | full_graph -> adaptive_graph | 41.527 | 35.473 | 1.171 | [1.157, 1.215] | 1.127-1.253 | 12 x 15 | WIN |
| 1 / 64 / 64 | full_graph -> compiled_full_graph | 41.527 | 39.110 | 1.062 | [1.048, 1.077] | 1.029-1.129 | 12 x 15 | WIN |
| 1 / 64 / 64 | adaptive_graph -> compiled_adaptive_graph | 35.473 | 32.147 | 1.103 | [1.062, 1.117] | 1.027-1.148 | 12 x 15 | WIN |
| 1 / 64 / 64 | full_graph -> compiled_adaptive_graph | 41.527 | 32.147 | 1.292 | [1.276, 1.303] | 1.222-1.323 | 12 x 15 | WIN |
| 1 / 64 / 64 | compiled_full_graph -> compiled_adaptive_graph | 39.110 | 32.147 | 1.217 | [1.200, 1.230] | 1.150-1.262 | 12 x 15 | WIN |
| 1 / 128 / 32 | full_graph -> adaptive_graph | 21.925 | 20.375 | 1.076 | [1.062, 1.098] | 1.031-1.117 | 12 x 15 | WIN |
| 1 / 128 / 32 | full_graph -> compiled_full_graph | 21.925 | 20.439 | 1.073 | [1.058, 1.089] | 1.023-1.141 | 12 x 15 | WIN |
| 1 / 128 / 32 | adaptive_graph -> compiled_adaptive_graph | 20.375 | 18.664 | 1.092 | [1.075, 1.104] | 1.060-1.146 | 12 x 15 | WIN |
| 1 / 128 / 32 | full_graph -> compiled_adaptive_graph | 21.925 | 18.664 | 1.175 | [1.163, 1.194] | 1.128-1.280 | 12 x 15 | WIN |
| 1 / 128 / 32 | compiled_full_graph -> compiled_adaptive_graph | 20.439 | 18.664 | 1.095 | [1.081, 1.113] | 1.032-1.170 | 12 x 15 | WIN |
| 1 / 128 / 64 | full_graph -> adaptive_graph | 42.495 | 38.995 | 1.090 | [1.078, 1.104] | 1.057-1.156 | 12 x 15 | WIN |
| 1 / 128 / 64 | full_graph -> compiled_full_graph | 42.495 | 38.735 | 1.097 | [1.085, 1.108] | 1.067-1.140 | 12 x 15 | WIN |
| 1 / 128 / 64 | adaptive_graph -> compiled_adaptive_graph | 38.995 | 35.483 | 1.099 | [1.086, 1.115] | 1.061-1.145 | 12 x 15 | WIN |
| 1 / 128 / 64 | full_graph -> compiled_adaptive_graph | 42.495 | 35.483 | 1.198 | [1.183, 1.215] | 1.174-1.262 | 12 x 15 | WIN |
| 1 / 128 / 64 | compiled_full_graph -> compiled_adaptive_graph | 38.735 | 35.483 | 1.092 | [1.077, 1.109] | 1.030-1.155 | 12 x 15 | WIN |
| 1 / 192 / 32 | full_graph -> adaptive_graph | 22.242 | 22.182 | 1.003 | [0.991, 1.018] | 0.922-1.062 | 12 x 15 | AA_PASS |
| 1 / 192 / 32 | full_graph -> compiled_full_graph | 22.242 | 20.221 | 1.100 | [1.084, 1.118] | 1.057-1.155 | 12 x 15 | WIN |
| 1 / 192 / 32 | adaptive_graph -> compiled_adaptive_graph | 22.182 | 20.339 | 1.091 | [1.069, 1.109] | 1.021-1.157 | 12 x 15 | WIN |
| 1 / 192 / 32 | full_graph -> compiled_adaptive_graph | 22.242 | 20.339 | 1.094 | [1.078, 1.111] | 1.023-1.128 | 12 x 15 | WIN |
| 1 / 192 / 32 | compiled_full_graph -> compiled_adaptive_graph | 20.221 | 20.339 | 0.994 | [0.972, 1.014] | 0.937-1.049 | 12 x 15 | AA_PASS |
| 16 / 8 / 16 | full_graph -> adaptive_graph | 20.346 | 14.765 | 1.378 | [1.352, 1.409] | 1.262-1.497 | 12 x 15 | WIN |
| 16 / 8 / 16 | full_graph -> compiled_full_graph | 20.346 | 18.768 | 1.084 | [1.057, 1.113] | 1.005-1.144 | 12 x 15 | WIN |
| 16 / 8 / 16 | adaptive_graph -> compiled_adaptive_graph | 14.765 | 13.178 | 1.120 | [1.093, 1.147] | 1.033-1.162 | 12 x 15 | WIN |
| 16 / 8 / 16 | full_graph -> compiled_adaptive_graph | 20.346 | 13.178 | 1.544 | [1.506, 1.583] | 1.419-1.610 | 12 x 15 | WIN |
| 16 / 8 / 16 | compiled_full_graph -> compiled_adaptive_graph | 18.768 | 13.178 | 1.424 | [1.380, 1.465] | 1.294-1.541 | 12 x 15 | WIN |
| 16 / 64 / 16 | full_graph -> adaptive_graph | 20.609 | 15.152 | 1.360 | [1.332, 1.388] | 1.294-1.432 | 12 x 15 | WIN |
| 16 / 64 / 16 | full_graph -> compiled_full_graph | 20.609 | 19.225 | 1.072 | [1.045, 1.108] | 0.994-1.158 | 12 x 15 | WIN |
| 16 / 64 / 16 | adaptive_graph -> compiled_adaptive_graph | 15.152 | 13.572 | 1.116 | [1.092, 1.140] | 1.046-1.187 | 12 x 15 | WIN |
| 16 / 64 / 16 | full_graph -> compiled_adaptive_graph | 20.609 | 13.572 | 1.518 | [1.488, 1.548] | 1.419-1.580 | 12 x 15 | WIN |
| 16 / 64 / 16 | compiled_full_graph -> compiled_adaptive_graph | 19.225 | 13.572 | 1.416 | [1.379, 1.442] | 1.305-1.471 | 12 x 15 | WIN |
| 16 / 64 / 64 | full_graph -> adaptive_graph | 76.062 | 55.343 | 1.374 | [1.357, 1.400] | 1.329-1.467 | 12 x 15 | WIN |
| 16 / 64 / 64 | full_graph -> compiled_full_graph | 76.062 | 72.043 | 1.056 | [1.043, 1.068] | 1.013-1.104 | 12 x 15 | WIN |
| 16 / 64 / 64 | adaptive_graph -> compiled_adaptive_graph | 55.343 | 50.056 | 1.106 | [1.079, 1.132] | 1.050-1.155 | 12 x 15 | WIN |
| 16 / 64 / 64 | full_graph -> compiled_adaptive_graph | 76.062 | 50.056 | 1.520 | [1.492, 1.553] | 1.426-1.601 | 12 x 15 | WIN |
| 16 / 64 / 64 | compiled_full_graph -> compiled_adaptive_graph | 72.043 | 50.056 | 1.439 | [1.423, 1.463] | 1.407-1.476 | 12 x 15 | WIN |
| 16 / 128 / 32 | full_graph -> adaptive_graph | 42.112 | 36.765 | 1.145 | [1.131, 1.159] | 1.112-1.196 | 12 x 15 | WIN |
| 16 / 128 / 32 | full_graph -> compiled_full_graph | 42.112 | 39.293 | 1.072 | [1.058, 1.084] | 1.039-1.113 | 12 x 15 | WIN |
| 16 / 128 / 32 | adaptive_graph -> compiled_adaptive_graph | 36.765 | 34.702 | 1.059 | [1.044, 1.081] | 1.008-1.132 | 12 x 15 | WIN |
| 16 / 128 / 32 | full_graph -> compiled_adaptive_graph | 42.112 | 34.702 | 1.214 | [1.199, 1.231] | 1.176-1.269 | 12 x 15 | WIN |
| 16 / 128 / 32 | compiled_full_graph -> compiled_adaptive_graph | 39.293 | 34.702 | 1.132 | [1.119, 1.155] | 1.101-1.202 | 12 x 15 | WIN |
| 16 / 128 / 64 | full_graph -> adaptive_graph | 80.018 | 69.451 | 1.152 | [1.144, 1.161] | 1.119-1.180 | 12 x 15 | WIN |
| 16 / 128 / 64 | full_graph -> compiled_full_graph | 80.018 | 73.861 | 1.083 | [1.075, 1.099] | 1.034-1.115 | 12 x 15 | WIN |
| 16 / 128 / 64 | adaptive_graph -> compiled_adaptive_graph | 69.451 | 64.590 | 1.075 | [1.060, 1.088] | 1.025-1.122 | 12 x 15 | WIN |
| 16 / 128 / 64 | full_graph -> compiled_adaptive_graph | 80.018 | 64.590 | 1.239 | [1.224, 1.252] | 1.190-1.302 | 12 x 15 | WIN |
| 16 / 128 / 64 | compiled_full_graph -> compiled_adaptive_graph | 73.861 | 64.590 | 1.144 | [1.125, 1.156] | 1.081-1.186 | 12 x 15 | WIN |
| 16 / 192 / 32 | full_graph -> adaptive_graph | 44.166 | 45.492 | 0.971 | [0.958, 0.990] | 0.950-1.001 | 12 x 15 | AA_FAIL |
| 16 / 192 / 32 | full_graph -> compiled_full_graph | 44.166 | 42.748 | 1.033 | [1.011, 1.061] | 0.978-1.105 | 12 x 15 | INCONCLUSIVE |
| 16 / 192 / 32 | adaptive_graph -> compiled_adaptive_graph | 45.492 | 42.291 | 1.076 | [1.055, 1.090] | 1.012-1.125 | 12 x 15 | WIN |
| 16 / 192 / 32 | full_graph -> compiled_adaptive_graph | 44.166 | 42.291 | 1.044 | [1.026, 1.063] | 1.004-1.080 | 12 x 15 | INCONCLUSIVE |
| 16 / 192 / 32 | compiled_full_graph -> compiled_adaptive_graph | 42.748 | 42.291 | 1.011 | [0.985, 1.028] | 0.941-1.047 | 12 x 15 | AA_PASS |

## Verdict counts (full denominator)

- adaptive_graph -> compiled_adaptive_graph: WIN x12
- compiled_full_graph -> compiled_adaptive_graph: AA_PASS x2
- compiled_full_graph -> compiled_adaptive_graph: WIN x10
- full_graph -> adaptive_graph: AA_FAIL x1
- full_graph -> adaptive_graph: AA_PASS x1
- full_graph -> adaptive_graph: WIN x10
- full_graph -> compiled_adaptive_graph: INCONCLUSIVE x1
- full_graph -> compiled_adaptive_graph: WIN x11
- full_graph -> compiled_full_graph: INCONCLUSIVE x3
- full_graph -> compiled_full_graph: WIN x9
