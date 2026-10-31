# Phase 5 in-process interleaved latency ratios

3 processes x 6 blocks x 12 requests per mode and workload; clusters = (process, block). Peak memory not comparable (all modes resident together).

Source: `research\phase5\ci_runs\20261030_195833_inprocess\benchmark_raw.csv`; 10000 two-level bootstrap resamples (rounds, then requests); 95% percentile CI.
Ratio = baseline median / candidate median (>1: candidate faster). WIN needs ratio >= 1.05 and CI low > 1. Prompt=192 full-vs-adaptive is the A/A control (identical capacity).

| B / prompt / gen | baseline -> candidate | base ms | cand ms | ratio | 95% CI | per-round range | rounds x n | verdict |
|---|---|---:|---:|---:|---|---|---|---|
| 1 / 8 / 16 | full_graph -> adaptive_graph | 12.003 | 10.087 | 1.190 | [1.143, 1.236] | 1.026-1.278 | 18 x 12 | WIN |
| 1 / 8 / 16 | full_graph -> compiled_full_graph | 12.003 | 11.079 | 1.083 | [1.051, 1.122] | 0.985-1.186 | 18 x 12 | WIN |
| 1 / 8 / 16 | adaptive_graph -> compiled_adaptive_graph | 10.087 | 9.234 | 1.092 | [1.051, 1.117] | 1.002-1.158 | 18 x 12 | WIN |
| 1 / 8 / 16 | full_graph -> compiled_adaptive_graph | 12.003 | 9.234 | 1.300 | [1.240, 1.343] | 1.173-1.429 | 18 x 12 | WIN |
| 1 / 8 / 16 | compiled_full_graph -> compiled_adaptive_graph | 11.079 | 9.234 | 1.200 | [1.138, 1.237] | 1.039-1.315 | 18 x 12 | WIN |
| 1 / 64 / 16 | full_graph -> adaptive_graph | 11.868 | 10.311 | 1.151 | [1.116, 1.183] | 1.054-1.234 | 18 x 12 | WIN |
| 1 / 64 / 16 | full_graph -> compiled_full_graph | 11.868 | 11.061 | 1.073 | [1.040, 1.107] | 0.978-1.134 | 18 x 12 | WIN |
| 1 / 64 / 16 | adaptive_graph -> compiled_adaptive_graph | 10.311 | 9.374 | 1.100 | [1.060, 1.137] | 0.983-1.213 | 18 x 12 | WIN |
| 1 / 64 / 16 | full_graph -> compiled_adaptive_graph | 11.868 | 9.374 | 1.266 | [1.215, 1.304] | 1.120-1.391 | 18 x 12 | WIN |
| 1 / 64 / 16 | compiled_full_graph -> compiled_adaptive_graph | 11.061 | 9.374 | 1.180 | [1.134, 1.215] | 1.123-1.295 | 18 x 12 | WIN |
| 1 / 64 / 64 | full_graph -> adaptive_graph | 42.395 | 35.467 | 1.195 | [1.187, 1.205] | 1.171-1.213 | 18 x 12 | WIN |
| 1 / 64 / 64 | full_graph -> compiled_full_graph | 42.395 | 38.728 | 1.095 | [1.086, 1.104] | 1.058-1.114 | 18 x 12 | WIN |
| 1 / 64 / 64 | adaptive_graph -> compiled_adaptive_graph | 35.467 | 32.150 | 1.103 | [1.094, 1.114] | 1.077-1.131 | 18 x 12 | WIN |
| 1 / 64 / 64 | full_graph -> compiled_adaptive_graph | 42.395 | 32.150 | 1.319 | [1.306, 1.333] | 1.305-1.348 | 18 x 12 | WIN |
| 1 / 64 / 64 | compiled_full_graph -> compiled_adaptive_graph | 38.728 | 32.150 | 1.205 | [1.195, 1.217] | 1.191-1.258 | 18 x 12 | WIN |
| 1 / 128 / 32 | full_graph -> adaptive_graph | 22.130 | 20.503 | 1.079 | [1.071, 1.095] | 1.057-1.127 | 18 x 12 | WIN |
| 1 / 128 / 32 | full_graph -> compiled_full_graph | 22.130 | 20.490 | 1.080 | [1.070, 1.096] | 1.064-1.131 | 18 x 12 | WIN |
| 1 / 128 / 32 | adaptive_graph -> compiled_adaptive_graph | 20.503 | 18.763 | 1.093 | [1.085, 1.101] | 1.072-1.121 | 18 x 12 | WIN |
| 1 / 128 / 32 | full_graph -> compiled_adaptive_graph | 22.130 | 18.763 | 1.179 | [1.170, 1.196] | 1.151-1.264 | 18 x 12 | WIN |
| 1 / 128 / 32 | compiled_full_graph -> compiled_adaptive_graph | 20.490 | 18.763 | 1.092 | [1.083, 1.102] | 1.076-1.117 | 18 x 12 | WIN |
| 1 / 128 / 64 | full_graph -> adaptive_graph | 42.442 | 39.131 | 1.085 | [1.077, 1.094] | 1.067-1.109 | 18 x 12 | WIN |
| 1 / 128 / 64 | full_graph -> compiled_full_graph | 42.442 | 38.975 | 1.089 | [1.082, 1.099] | 1.073-1.123 | 18 x 12 | WIN |
| 1 / 128 / 64 | adaptive_graph -> compiled_adaptive_graph | 39.131 | 35.524 | 1.102 | [1.093, 1.112] | 1.084-1.125 | 18 x 12 | WIN |
| 1 / 128 / 64 | full_graph -> compiled_adaptive_graph | 42.442 | 35.524 | 1.195 | [1.187, 1.205] | 1.176-1.218 | 18 x 12 | WIN |
| 1 / 128 / 64 | compiled_full_graph -> compiled_adaptive_graph | 38.975 | 35.524 | 1.097 | [1.089, 1.106] | 1.073-1.112 | 18 x 12 | WIN |
| 1 / 192 / 32 | full_graph -> adaptive_graph | 22.130 | 22.161 | 0.999 | [0.990, 1.011] | 0.973-1.040 | 18 x 12 | AA_PASS |
| 1 / 192 / 32 | full_graph -> compiled_full_graph | 22.130 | 20.362 | 1.087 | [1.078, 1.099] | 1.067-1.129 | 18 x 12 | WIN |
| 1 / 192 / 32 | adaptive_graph -> compiled_adaptive_graph | 22.161 | 20.444 | 1.084 | [1.075, 1.095] | 1.065-1.128 | 18 x 12 | WIN |
| 1 / 192 / 32 | full_graph -> compiled_adaptive_graph | 22.130 | 20.444 | 1.082 | [1.074, 1.095] | 1.063-1.111 | 18 x 12 | WIN |
| 1 / 192 / 32 | compiled_full_graph -> compiled_adaptive_graph | 20.362 | 20.444 | 0.996 | [0.989, 1.005] | 0.981-1.015 | 18 x 12 | AA_PASS |
| 16 / 8 / 16 | full_graph -> adaptive_graph | 20.326 | 14.593 | 1.393 | [1.360, 1.423] | 1.323-1.466 | 18 x 12 | WIN |
| 16 / 8 / 16 | full_graph -> compiled_full_graph | 20.326 | 19.066 | 1.066 | [1.056, 1.077] | 1.042-1.086 | 18 x 12 | WIN |
| 16 / 8 / 16 | adaptive_graph -> compiled_adaptive_graph | 14.593 | 13.379 | 1.091 | [1.068, 1.125] | 1.054-1.153 | 18 x 12 | WIN |
| 16 / 8 / 16 | full_graph -> compiled_adaptive_graph | 20.326 | 13.379 | 1.519 | [1.488, 1.566] | 1.455-1.629 | 18 x 12 | WIN |
| 16 / 8 / 16 | compiled_full_graph -> compiled_adaptive_graph | 19.066 | 13.379 | 1.425 | [1.398, 1.465] | 1.376-1.512 | 18 x 12 | WIN |
| 16 / 64 / 16 | full_graph -> adaptive_graph | 20.552 | 15.079 | 1.363 | [1.350, 1.379] | 1.315-1.403 | 18 x 12 | WIN |
| 16 / 64 / 16 | full_graph -> compiled_full_graph | 20.552 | 19.375 | 1.061 | [1.053, 1.069] | 1.044-1.084 | 18 x 12 | WIN |
| 16 / 64 / 16 | adaptive_graph -> compiled_adaptive_graph | 15.079 | 14.021 | 1.075 | [1.063, 1.091] | 1.052-1.111 | 18 x 12 | WIN |
| 16 / 64 / 16 | full_graph -> compiled_adaptive_graph | 20.552 | 14.021 | 1.466 | [1.453, 1.487] | 1.445-1.520 | 18 x 12 | WIN |
| 16 / 64 / 16 | compiled_full_graph -> compiled_adaptive_graph | 19.375 | 14.021 | 1.382 | [1.371, 1.401] | 1.361-1.422 | 18 x 12 | WIN |
| 16 / 64 / 64 | full_graph -> adaptive_graph | 77.388 | 55.857 | 1.385 | [1.373, 1.395] | 1.367-1.407 | 18 x 12 | WIN |
| 16 / 64 / 64 | full_graph -> compiled_full_graph | 77.388 | 71.977 | 1.075 | [1.066, 1.081] | 1.061-1.086 | 18 x 12 | WIN |
| 16 / 64 / 64 | adaptive_graph -> compiled_adaptive_graph | 55.857 | 50.691 | 1.102 | [1.094, 1.110] | 1.084-1.134 | 18 x 12 | WIN |
| 16 / 64 / 64 | full_graph -> compiled_adaptive_graph | 77.388 | 50.691 | 1.527 | [1.514, 1.537] | 1.505-1.549 | 18 x 12 | WIN |
| 16 / 64 / 64 | compiled_full_graph -> compiled_adaptive_graph | 71.977 | 50.691 | 1.420 | [1.412, 1.429] | 1.397-1.442 | 18 x 12 | WIN |
| 16 / 128 / 32 | full_graph -> adaptive_graph | 42.266 | 36.914 | 1.145 | [1.133, 1.155] | 1.111-1.164 | 18 x 12 | WIN |
| 16 / 128 / 32 | full_graph -> compiled_full_graph | 42.266 | 39.644 | 1.066 | [1.053, 1.078] | 1.036-1.098 | 18 x 12 | WIN |
| 16 / 128 / 32 | adaptive_graph -> compiled_adaptive_graph | 36.914 | 34.534 | 1.069 | [1.062, 1.076] | 1.047-1.115 | 18 x 12 | WIN |
| 16 / 128 / 32 | full_graph -> compiled_adaptive_graph | 42.266 | 34.534 | 1.224 | [1.212, 1.234] | 1.206-1.251 | 18 x 12 | WIN |
| 16 / 128 / 32 | compiled_full_graph -> compiled_adaptive_graph | 39.644 | 34.534 | 1.148 | [1.139, 1.158] | 1.130-1.176 | 18 x 12 | WIN |
| 16 / 128 / 64 | full_graph -> adaptive_graph | 80.378 | 69.605 | 1.155 | [1.144, 1.160] | 1.136-1.164 | 18 x 12 | WIN |
| 16 / 128 / 64 | full_graph -> compiled_full_graph | 80.378 | 74.867 | 1.074 | [1.065, 1.082] | 1.055-1.083 | 18 x 12 | WIN |
| 16 / 128 / 64 | adaptive_graph -> compiled_adaptive_graph | 69.605 | 64.619 | 1.077 | [1.072, 1.085] | 1.064-1.093 | 18 x 12 | WIN |
| 16 / 128 / 64 | full_graph -> compiled_adaptive_graph | 80.378 | 64.619 | 1.244 | [1.235, 1.249] | 1.231-1.248 | 18 x 12 | WIN |
| 16 / 128 / 64 | compiled_full_graph -> compiled_adaptive_graph | 74.867 | 64.619 | 1.159 | [1.150, 1.165] | 1.146-1.168 | 18 x 12 | WIN |
| 16 / 192 / 32 | full_graph -> adaptive_graph | 45.758 | 45.704 | 1.001 | [0.993, 1.012] | 0.979-1.025 | 18 x 12 | AA_PASS |
| 16 / 192 / 32 | full_graph -> compiled_full_graph | 45.758 | 43.290 | 1.057 | [1.047, 1.068] | 1.030-1.088 | 18 x 12 | WIN |
| 16 / 192 / 32 | adaptive_graph -> compiled_adaptive_graph | 45.704 | 43.146 | 1.059 | [1.048, 1.068] | 1.033-1.084 | 18 x 12 | WIN |
| 16 / 192 / 32 | full_graph -> compiled_adaptive_graph | 45.758 | 43.146 | 1.061 | [1.052, 1.069] | 1.035-1.090 | 18 x 12 | WIN |
| 16 / 192 / 32 | compiled_full_graph -> compiled_adaptive_graph | 43.290 | 43.146 | 1.003 | [0.993, 1.013] | 0.977-1.026 | 18 x 12 | AA_PASS |

## Verdict counts (full denominator)

- adaptive_graph -> compiled_adaptive_graph: WIN x12
- compiled_full_graph -> compiled_adaptive_graph: AA_PASS x2
- compiled_full_graph -> compiled_adaptive_graph: WIN x10
- full_graph -> adaptive_graph: AA_PASS x2
- full_graph -> adaptive_graph: WIN x10
- full_graph -> compiled_adaptive_graph: WIN x12
- full_graph -> compiled_full_graph: WIN x12
