# Phase 5 bootstrap latency ratios

Source: `research\phase5\ci_runs\20261030_191837_compiled_graph\benchmark_raw.csv`; 10000 two-level bootstrap resamples (rounds, then requests); 95% percentile CI.
Ratio = baseline median / candidate median (>1: candidate faster). WIN needs ratio >= 1.05 and CI low > 1. Prompt=192 full-vs-adaptive is the A/A control (identical capacity).

| B / prompt / gen | baseline -> candidate | base ms | cand ms | ratio | 95% CI | per-round range | rounds x n | verdict |
|---|---|---:|---:|---:|---|---|---|---|
| 1 / 8 / 16 | full_graph -> adaptive_graph | 11.909 | 10.049 | 1.185 | [1.142, 1.262] | 1.135-1.231 | 6 x 15 | WIN |
| 1 / 8 / 16 | full_graph -> compiled_full_graph | 11.909 | 11.367 | 1.048 | [1.012, 1.112] | 1.029-1.101 | 6 x 15 | INCONCLUSIVE |
| 1 / 8 / 16 | adaptive_graph -> compiled_adaptive_graph | 10.049 | 9.322 | 1.078 | [1.002, 1.130] | 0.974-1.141 | 6 x 15 | WIN |
| 1 / 8 / 16 | full_graph -> compiled_adaptive_graph | 11.909 | 9.322 | 1.278 | [1.212, 1.349] | 1.199-1.358 | 6 x 15 | WIN |
| 1 / 8 / 16 | compiled_full_graph -> compiled_adaptive_graph | 11.367 | 9.322 | 1.219 | [1.135, 1.284] | 1.153-1.312 | 6 x 15 | WIN |
| 1 / 64 / 16 | full_graph -> adaptive_graph | 12.164 | 10.160 | 1.197 | [1.157, 1.254] | 1.192-1.213 | 6 x 15 | WIN |
| 1 / 64 / 16 | full_graph -> compiled_full_graph | 12.164 | 11.438 | 1.064 | [1.018, 1.117] | 1.014-1.134 | 6 x 15 | WIN |
| 1 / 64 / 16 | adaptive_graph -> compiled_adaptive_graph | 10.160 | 9.216 | 1.102 | [1.035, 1.153] | 1.041-1.154 | 6 x 15 | WIN |
| 1 / 64 / 16 | full_graph -> compiled_adaptive_graph | 12.164 | 9.216 | 1.320 | [1.254, 1.379] | 1.257-1.384 | 6 x 15 | WIN |
| 1 / 64 / 16 | compiled_full_graph -> compiled_adaptive_graph | 11.438 | 9.216 | 1.241 | [1.177, 1.301] | 1.201-1.298 | 6 x 15 | WIN |
| 1 / 64 / 64 | full_graph -> adaptive_graph | 42.519 | 35.253 | 1.206 | [1.186, 1.228] | 1.184-1.252 | 6 x 15 | WIN |
| 1 / 64 / 64 | full_graph -> compiled_full_graph | 42.519 | 38.878 | 1.094 | [1.070, 1.111] | 1.050-1.134 | 6 x 15 | WIN |
| 1 / 64 / 64 | adaptive_graph -> compiled_adaptive_graph | 35.253 | 32.275 | 1.092 | [1.055, 1.115] | 1.029-1.119 | 6 x 15 | WIN |
| 1 / 64 / 64 | full_graph -> compiled_adaptive_graph | 42.519 | 32.275 | 1.317 | [1.286, 1.345] | 1.256-1.364 | 6 x 15 | WIN |
| 1 / 64 / 64 | compiled_full_graph -> compiled_adaptive_graph | 38.878 | 32.275 | 1.205 | [1.184, 1.219] | 1.178-1.208 | 6 x 15 | WIN |
| 1 / 128 / 32 | full_graph -> adaptive_graph | 22.347 | 20.342 | 1.099 | [1.062, 1.140] | 1.036-1.157 | 6 x 15 | WIN |
| 1 / 128 / 32 | full_graph -> compiled_full_graph | 22.347 | 20.382 | 1.096 | [1.076, 1.142] | 1.069-1.197 | 6 x 15 | WIN |
| 1 / 128 / 32 | adaptive_graph -> compiled_adaptive_graph | 20.342 | 18.799 | 1.082 | [1.048, 1.129] | 1.049-1.171 | 6 x 15 | WIN |
| 1 / 128 / 32 | full_graph -> compiled_adaptive_graph | 22.347 | 18.799 | 1.189 | [1.171, 1.215] | 1.163-1.245 | 6 x 15 | WIN |
| 1 / 128 / 32 | compiled_full_graph -> compiled_adaptive_graph | 20.382 | 18.799 | 1.084 | [1.053, 1.105] | 1.040-1.123 | 6 x 15 | WIN |
| 1 / 128 / 64 | full_graph -> adaptive_graph | 42.692 | 38.756 | 1.102 | [1.079, 1.124] | 1.069-1.138 | 6 x 15 | WIN |
| 1 / 128 / 64 | full_graph -> compiled_full_graph | 42.692 | 38.789 | 1.101 | [1.080, 1.134] | 1.077-1.140 | 6 x 15 | WIN |
| 1 / 128 / 64 | adaptive_graph -> compiled_adaptive_graph | 38.756 | 35.236 | 1.100 | [1.074, 1.120] | 1.061-1.131 | 6 x 15 | WIN |
| 1 / 128 / 64 | full_graph -> compiled_adaptive_graph | 42.692 | 35.236 | 1.212 | [1.193, 1.229] | 1.193-1.225 | 6 x 15 | WIN |
| 1 / 128 / 64 | compiled_full_graph -> compiled_adaptive_graph | 38.789 | 35.236 | 1.101 | [1.065, 1.126] | 1.052-1.124 | 6 x 15 | WIN |
| 1 / 192 / 32 | full_graph -> adaptive_graph | 21.942 | 22.006 | 0.997 | [0.960, 1.020] | 0.946-1.029 | 6 x 15 | AA_PASS |
| 1 / 192 / 32 | full_graph -> compiled_full_graph | 21.942 | 20.352 | 1.078 | [1.040, 1.110] | 1.037-1.112 | 6 x 15 | WIN |
| 1 / 192 / 32 | adaptive_graph -> compiled_adaptive_graph | 22.006 | 20.015 | 1.099 | [1.075, 1.128] | 1.055-1.152 | 6 x 15 | WIN |
| 1 / 192 / 32 | full_graph -> compiled_adaptive_graph | 21.942 | 20.015 | 1.096 | [1.054, 1.127] | 1.031-1.141 | 6 x 15 | WIN |
| 1 / 192 / 32 | compiled_full_graph -> compiled_adaptive_graph | 20.352 | 20.015 | 1.017 | [0.976, 1.053] | 0.951-1.068 | 6 x 15 | AA_PASS |
| 16 / 8 / 16 | full_graph -> adaptive_graph | 19.919 | 14.616 | 1.363 | [1.326, 1.419] | 1.310-1.441 | 6 x 15 | WIN |
| 16 / 8 / 16 | full_graph -> compiled_full_graph | 19.919 | 19.276 | 1.033 | [1.009, 1.082] | 1.007-1.110 | 6 x 15 | INCONCLUSIVE |
| 16 / 8 / 16 | adaptive_graph -> compiled_adaptive_graph | 14.616 | 12.990 | 1.125 | [1.095, 1.179] | 1.049-1.231 | 6 x 15 | WIN |
| 16 / 8 / 16 | full_graph -> compiled_adaptive_graph | 19.919 | 12.990 | 1.533 | [1.488, 1.597] | 1.430-1.612 | 6 x 15 | WIN |
| 16 / 8 / 16 | compiled_full_graph -> compiled_adaptive_graph | 19.276 | 12.990 | 1.484 | [1.438, 1.542] | 1.420-1.578 | 6 x 15 | WIN |
| 16 / 64 / 16 | full_graph -> adaptive_graph | 20.208 | 15.096 | 1.339 | [1.310, 1.370] | 1.280-1.369 | 6 x 15 | WIN |
| 16 / 64 / 16 | full_graph -> compiled_full_graph | 20.208 | 19.430 | 1.040 | [1.002, 1.083] | 0.988-1.117 | 6 x 15 | INCONCLUSIVE |
| 16 / 64 / 16 | adaptive_graph -> compiled_adaptive_graph | 15.096 | 13.478 | 1.120 | [1.074, 1.153] | 1.047-1.159 | 6 x 15 | WIN |
| 16 / 64 / 16 | full_graph -> compiled_adaptive_graph | 20.208 | 13.478 | 1.499 | [1.458, 1.542] | 1.433-1.530 | 6 x 15 | WIN |
| 16 / 64 / 16 | compiled_full_graph -> compiled_adaptive_graph | 19.430 | 13.478 | 1.442 | [1.397, 1.484] | 1.364-1.494 | 6 x 15 | WIN |
| 16 / 64 / 64 | full_graph -> adaptive_graph | 77.490 | 56.211 | 1.379 | [1.355, 1.401] | 1.339-1.433 | 6 x 15 | WIN |
| 16 / 64 / 64 | full_graph -> compiled_full_graph | 77.490 | 72.998 | 1.062 | [1.040, 1.071] | 1.003-1.084 | 6 x 15 | WIN |
| 16 / 64 / 64 | adaptive_graph -> compiled_adaptive_graph | 56.211 | 50.806 | 1.106 | [1.089, 1.122] | 1.075-1.133 | 6 x 15 | WIN |
| 16 / 64 / 64 | full_graph -> compiled_adaptive_graph | 77.490 | 50.806 | 1.525 | [1.499, 1.542] | 1.453-1.543 | 6 x 15 | WIN |
| 16 / 64 / 64 | compiled_full_graph -> compiled_adaptive_graph | 72.998 | 50.806 | 1.437 | [1.427, 1.450] | 1.421-1.454 | 6 x 15 | WIN |
| 16 / 128 / 32 | full_graph -> adaptive_graph | 43.194 | 37.448 | 1.153 | [1.139, 1.196] | 1.143-1.233 | 6 x 15 | WIN |
| 16 / 128 / 32 | full_graph -> compiled_full_graph | 43.194 | 39.322 | 1.098 | [1.071, 1.147] | 1.056-1.168 | 6 x 15 | WIN |
| 16 / 128 / 32 | adaptive_graph -> compiled_adaptive_graph | 37.448 | 35.019 | 1.069 | [1.037, 1.083] | 1.011-1.091 | 6 x 15 | WIN |
| 16 / 128 / 32 | full_graph -> compiled_adaptive_graph | 43.194 | 35.019 | 1.233 | [1.221, 1.257] | 1.221-1.253 | 6 x 15 | WIN |
| 16 / 128 / 32 | compiled_full_graph -> compiled_adaptive_graph | 39.322 | 35.019 | 1.123 | [1.083, 1.156] | 1.073-1.181 | 6 x 15 | WIN |
| 16 / 128 / 64 | full_graph -> adaptive_graph | 79.843 | 69.460 | 1.149 | [1.124, 1.172] | 1.121-1.173 | 6 x 15 | WIN |
| 16 / 128 / 64 | full_graph -> compiled_full_graph | 79.843 | 73.930 | 1.080 | [1.053, 1.117] | 1.040-1.141 | 6 x 15 | WIN |
| 16 / 128 / 64 | adaptive_graph -> compiled_adaptive_graph | 69.460 | 65.173 | 1.066 | [1.043, 1.094] | 1.006-1.136 | 6 x 15 | WIN |
| 16 / 128 / 64 | full_graph -> compiled_adaptive_graph | 79.843 | 65.173 | 1.225 | [1.199, 1.255] | 1.180-1.274 | 6 x 15 | WIN |
| 16 / 128 / 64 | compiled_full_graph -> compiled_adaptive_graph | 73.930 | 65.173 | 1.134 | [1.109, 1.159] | 1.097-1.194 | 6 x 15 | WIN |
| 16 / 192 / 32 | full_graph -> adaptive_graph | 44.995 | 45.612 | 0.986 | [0.963, 1.017] | 0.949-1.029 | 6 x 15 | AA_PASS |
| 16 / 192 / 32 | full_graph -> compiled_full_graph | 44.995 | 43.064 | 1.045 | [1.023, 1.071] | 1.015-1.068 | 6 x 15 | INCONCLUSIVE |
| 16 / 192 / 32 | adaptive_graph -> compiled_adaptive_graph | 45.612 | 41.545 | 1.098 | [1.061, 1.122] | 1.043-1.124 | 6 x 15 | WIN |
| 16 / 192 / 32 | full_graph -> compiled_adaptive_graph | 44.995 | 41.545 | 1.083 | [1.052, 1.114] | 1.046-1.116 | 6 x 15 | WIN |
| 16 / 192 / 32 | compiled_full_graph -> compiled_adaptive_graph | 43.064 | 41.545 | 1.037 | [1.010, 1.051] | 0.999-1.061 | 6 x 15 | AA_FAIL |

## Verdict counts (full denominator)

- adaptive_graph -> compiled_adaptive_graph: WIN x12
- compiled_full_graph -> compiled_adaptive_graph: AA_FAIL x1
- compiled_full_graph -> compiled_adaptive_graph: AA_PASS x1
- compiled_full_graph -> compiled_adaptive_graph: WIN x10
- full_graph -> adaptive_graph: AA_PASS x2
- full_graph -> adaptive_graph: WIN x10
- full_graph -> compiled_adaptive_graph: WIN x12
- full_graph -> compiled_full_graph: INCONCLUSIVE x4
- full_graph -> compiled_full_graph: WIN x8
