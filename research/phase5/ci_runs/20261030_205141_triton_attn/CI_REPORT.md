# Phase 5 in-process interleaved latency ratios (re-analyzed with Triton comparisons)

3 processes x 6 blocks x 12 requests; clusters = (process, block).

Source: `research\phase5\ci_runs\20261030_205141_triton_attn\benchmark_raw.csv`; 10000 two-level bootstrap resamples (rounds, then requests); 95% percentile CI.
Ratio = baseline median / candidate median (>1: candidate faster). WIN needs ratio >= 1.05 and CI low > 1. Prompt=192 full-vs-adaptive is the A/A control (identical capacity).

| B / prompt / gen | baseline -> candidate | base ms | cand ms | ratio | 95% CI | per-round range | rounds x n | verdict |
|---|---|---:|---:|---:|---|---|---|---|
| 1 / 8 / 16 | full_graph -> adaptive_graph | 11.424 | 9.769 | 1.169 | [1.148, 1.210] | 1.089-1.313 | 18 x 12 | WIN |
| 1 / 8 / 16 | adaptive_graph -> compiled_adaptive_graph | 9.769 | 8.907 | 1.097 | [1.069, 1.124] | 0.956-1.204 | 18 x 12 | WIN |
| 1 / 8 / 16 | full_graph -> compiled_adaptive_graph | 11.424 | 8.907 | 1.283 | [1.254, 1.330] | 1.190-1.396 | 18 x 12 | WIN |
| 1 / 8 / 16 | full_graph -> triton_full_graph | 11.424 | 8.103 | 1.410 | [1.384, 1.464] | 1.274-1.515 | 18 x 12 | WIN |
| 1 / 8 / 16 | adaptive_graph -> triton_adaptive_graph | 9.769 | 8.027 | 1.217 | [1.184, 1.253] | 1.025-1.365 | 18 x 12 | WIN |
| 1 / 8 / 16 | compiled_adaptive_graph -> triton_adaptive_graph | 8.907 | 8.027 | 1.110 | [1.087, 1.140] | 1.070-1.186 | 18 x 12 | WIN |
| 1 / 8 / 16 | full_graph -> triton_adaptive_graph | 11.424 | 8.027 | 1.423 | [1.387, 1.479] | 1.275-1.577 | 18 x 12 | WIN |
| 1 / 8 / 16 | triton_full_graph -> triton_adaptive_graph | 8.103 | 8.027 | 1.009 | [0.981, 1.037] | 0.889-1.136 | 18 x 12 | INCONCLUSIVE |
| 1 / 64 / 16 | full_graph -> adaptive_graph | 11.702 | 10.017 | 1.168 | [1.132, 1.200] | 1.080-1.255 | 18 x 12 | WIN |
| 1 / 64 / 16 | adaptive_graph -> compiled_adaptive_graph | 10.017 | 9.073 | 1.104 | [1.075, 1.136] | 1.030-1.182 | 18 x 12 | WIN |
| 1 / 64 / 16 | full_graph -> compiled_adaptive_graph | 11.702 | 9.073 | 1.290 | [1.252, 1.323] | 1.243-1.345 | 18 x 12 | WIN |
| 1 / 64 / 16 | full_graph -> triton_full_graph | 11.702 | 8.157 | 1.435 | [1.396, 1.475] | 1.345-1.587 | 18 x 12 | WIN |
| 1 / 64 / 16 | adaptive_graph -> triton_adaptive_graph | 10.017 | 8.079 | 1.240 | [1.208, 1.273] | 1.189-1.349 | 18 x 12 | WIN |
| 1 / 64 / 16 | compiled_adaptive_graph -> triton_adaptive_graph | 9.073 | 8.079 | 1.123 | [1.094, 1.152] | 1.046-1.196 | 18 x 12 | WIN |
| 1 / 64 / 16 | full_graph -> triton_adaptive_graph | 11.702 | 8.079 | 1.448 | [1.409, 1.484] | 1.372-1.531 | 18 x 12 | WIN |
| 1 / 64 / 16 | triton_full_graph -> triton_adaptive_graph | 8.157 | 8.079 | 1.010 | [0.982, 1.032] | 0.929-1.057 | 18 x 12 | INCONCLUSIVE |
| 1 / 64 / 64 | full_graph -> adaptive_graph | 42.529 | 35.687 | 1.192 | [1.180, 1.206] | 1.145-1.226 | 18 x 12 | WIN |
| 1 / 64 / 64 | adaptive_graph -> compiled_adaptive_graph | 35.687 | 32.170 | 1.109 | [1.094, 1.122] | 1.074-1.138 | 18 x 12 | WIN |
| 1 / 64 / 64 | full_graph -> compiled_adaptive_graph | 42.529 | 32.170 | 1.322 | [1.306, 1.335] | 1.263-1.350 | 18 x 12 | WIN |
| 1 / 64 / 64 | full_graph -> triton_full_graph | 42.529 | 28.103 | 1.513 | [1.496, 1.526] | 1.473-1.550 | 18 x 12 | WIN |
| 1 / 64 / 64 | adaptive_graph -> triton_adaptive_graph | 35.687 | 27.892 | 1.279 | [1.255, 1.294] | 1.236-1.317 | 18 x 12 | WIN |
| 1 / 64 / 64 | compiled_adaptive_graph -> triton_adaptive_graph | 32.170 | 27.892 | 1.153 | [1.133, 1.168] | 1.123-1.177 | 18 x 12 | WIN |
| 1 / 64 / 64 | full_graph -> triton_adaptive_graph | 42.529 | 27.892 | 1.525 | [1.497, 1.540] | 1.470-1.561 | 18 x 12 | WIN |
| 1 / 64 / 64 | triton_full_graph -> triton_adaptive_graph | 28.103 | 27.892 | 1.008 | [0.990, 1.021] | 0.970-1.039 | 18 x 12 | INCONCLUSIVE |
| 1 / 128 / 32 | full_graph -> adaptive_graph | 22.093 | 20.580 | 1.073 | [1.064, 1.083] | 1.044-1.101 | 18 x 12 | WIN |
| 1 / 128 / 32 | adaptive_graph -> compiled_adaptive_graph | 20.580 | 18.745 | 1.098 | [1.089, 1.107] | 1.074-1.122 | 18 x 12 | WIN |
| 1 / 128 / 32 | full_graph -> compiled_adaptive_graph | 22.093 | 18.745 | 1.179 | [1.168, 1.190] | 1.137-1.216 | 18 x 12 | WIN |
| 1 / 128 / 32 | full_graph -> triton_full_graph | 22.093 | 15.284 | 1.445 | [1.431, 1.470] | 1.390-1.520 | 18 x 12 | WIN |
| 1 / 128 / 32 | adaptive_graph -> triton_adaptive_graph | 20.580 | 15.259 | 1.349 | [1.334, 1.364] | 1.298-1.386 | 18 x 12 | WIN |
| 1 / 128 / 32 | compiled_adaptive_graph -> triton_adaptive_graph | 18.745 | 15.259 | 1.228 | [1.215, 1.244] | 1.190-1.257 | 18 x 12 | WIN |
| 1 / 128 / 32 | full_graph -> triton_adaptive_graph | 22.093 | 15.259 | 1.448 | [1.433, 1.465] | 1.415-1.493 | 18 x 12 | WIN |
| 1 / 128 / 32 | triton_full_graph -> triton_adaptive_graph | 15.284 | 15.259 | 1.002 | [0.983, 1.015] | 0.960-1.042 | 18 x 12 | INCONCLUSIVE |
| 1 / 128 / 64 | full_graph -> adaptive_graph | 42.679 | 39.207 | 1.089 | [1.079, 1.096] | 1.066-1.118 | 18 x 12 | WIN |
| 1 / 128 / 64 | adaptive_graph -> compiled_adaptive_graph | 39.207 | 35.662 | 1.099 | [1.091, 1.108] | 1.084-1.114 | 18 x 12 | WIN |
| 1 / 128 / 64 | full_graph -> compiled_adaptive_graph | 42.679 | 35.662 | 1.197 | [1.186, 1.206] | 1.178-1.218 | 18 x 12 | WIN |
| 1 / 128 / 64 | full_graph -> triton_full_graph | 42.679 | 28.234 | 1.512 | [1.495, 1.531] | 1.470-1.566 | 18 x 12 | WIN |
| 1 / 128 / 64 | adaptive_graph -> triton_adaptive_graph | 39.207 | 28.258 | 1.387 | [1.371, 1.402] | 1.348-1.429 | 18 x 12 | WIN |
| 1 / 128 / 64 | compiled_adaptive_graph -> triton_adaptive_graph | 35.662 | 28.258 | 1.262 | [1.246, 1.277] | 1.238-1.303 | 18 x 12 | WIN |
| 1 / 128 / 64 | full_graph -> triton_adaptive_graph | 42.679 | 28.258 | 1.510 | [1.490, 1.526] | 1.466-1.582 | 18 x 12 | WIN |
| 1 / 128 / 64 | triton_full_graph -> triton_adaptive_graph | 28.234 | 28.258 | 0.999 | [0.982, 1.014] | 0.961-1.037 | 18 x 12 | INCONCLUSIVE |
| 1 / 192 / 32 | full_graph -> adaptive_graph | 22.171 | 22.352 | 0.992 | [0.981, 1.003] | 0.967-1.024 | 18 x 12 | AA_PASS |
| 1 / 192 / 32 | adaptive_graph -> compiled_adaptive_graph | 22.352 | 20.543 | 1.088 | [1.079, 1.098] | 1.071-1.116 | 18 x 12 | WIN |
| 1 / 192 / 32 | full_graph -> compiled_adaptive_graph | 22.171 | 20.543 | 1.079 | [1.070, 1.091] | 1.064-1.118 | 18 x 12 | WIN |
| 1 / 192 / 32 | full_graph -> triton_full_graph | 22.171 | 15.393 | 1.440 | [1.422, 1.458] | 1.390-1.512 | 18 x 12 | WIN |
| 1 / 192 / 32 | adaptive_graph -> triton_adaptive_graph | 22.352 | 15.317 | 1.459 | [1.441, 1.484] | 1.424-1.522 | 18 x 12 | WIN |
| 1 / 192 / 32 | compiled_adaptive_graph -> triton_adaptive_graph | 20.543 | 15.317 | 1.341 | [1.326, 1.361] | 1.305-1.394 | 18 x 12 | WIN |
| 1 / 192 / 32 | full_graph -> triton_adaptive_graph | 22.171 | 15.317 | 1.448 | [1.428, 1.472] | 1.410-1.517 | 18 x 12 | WIN |
| 1 / 192 / 32 | triton_full_graph -> triton_adaptive_graph | 15.393 | 15.317 | 1.005 | [0.992, 1.023] | 0.979-1.061 | 18 x 12 | AA_PASS |
| 16 / 8 / 16 | full_graph -> adaptive_graph | 20.180 | 14.740 | 1.369 | [1.352, 1.382] | 1.316-1.455 | 18 x 12 | WIN |
| 16 / 8 / 16 | adaptive_graph -> compiled_adaptive_graph | 14.740 | 13.530 | 1.089 | [1.074, 1.108] | 1.044-1.153 | 18 x 12 | WIN |
| 16 / 8 / 16 | full_graph -> compiled_adaptive_graph | 20.180 | 13.530 | 1.491 | [1.470, 1.517] | 1.424-1.584 | 18 x 12 | WIN |
| 16 / 8 / 16 | full_graph -> triton_full_graph | 20.180 | 9.520 | 2.120 | [2.080, 2.190] | 2.010-2.263 | 18 x 12 | WIN |
| 16 / 8 / 16 | adaptive_graph -> triton_adaptive_graph | 14.740 | 9.165 | 1.608 | [1.567, 1.643] | 1.478-1.711 | 18 x 12 | WIN |
| 16 / 8 / 16 | compiled_adaptive_graph -> triton_adaptive_graph | 13.530 | 9.165 | 1.476 | [1.432, 1.511] | 1.363-1.555 | 18 x 12 | WIN |
| 16 / 8 / 16 | full_graph -> triton_adaptive_graph | 20.180 | 9.165 | 2.202 | [2.146, 2.245] | 2.008-2.336 | 18 x 12 | WIN |
| 16 / 8 / 16 | triton_full_graph -> triton_adaptive_graph | 9.520 | 9.165 | 1.039 | [0.994, 1.068] | 0.888-1.133 | 18 x 12 | INCONCLUSIVE |
| 16 / 64 / 16 | full_graph -> adaptive_graph | 20.655 | 15.266 | 1.353 | [1.336, 1.365] | 1.313-1.382 | 18 x 12 | WIN |
| 16 / 64 / 16 | adaptive_graph -> compiled_adaptive_graph | 15.266 | 14.131 | 1.080 | [1.069, 1.098] | 1.037-1.127 | 18 x 12 | WIN |
| 16 / 64 / 16 | full_graph -> compiled_adaptive_graph | 20.655 | 14.131 | 1.462 | [1.447, 1.482] | 1.433-1.502 | 18 x 12 | WIN |
| 16 / 64 / 16 | full_graph -> triton_full_graph | 20.655 | 11.369 | 1.817 | [1.788, 1.860] | 1.759-1.965 | 18 x 12 | WIN |
| 16 / 64 / 16 | adaptive_graph -> triton_adaptive_graph | 15.266 | 11.133 | 1.371 | [1.352, 1.397] | 1.319-1.453 | 18 x 12 | WIN |
| 16 / 64 / 16 | compiled_adaptive_graph -> triton_adaptive_graph | 14.131 | 11.133 | 1.269 | [1.248, 1.288] | 1.219-1.319 | 18 x 12 | WIN |
| 16 / 64 / 16 | full_graph -> triton_adaptive_graph | 20.655 | 11.133 | 1.855 | [1.827, 1.882] | 1.773-1.925 | 18 x 12 | WIN |
| 16 / 64 / 16 | triton_full_graph -> triton_adaptive_graph | 11.369 | 11.133 | 1.021 | [0.993, 1.043] | 0.955-1.080 | 18 x 12 | INCONCLUSIVE |
| 16 / 64 / 64 | full_graph -> adaptive_graph | 77.872 | 56.169 | 1.386 | [1.377, 1.399] | 1.375-1.415 | 18 x 12 | WIN |
| 16 / 64 / 64 | adaptive_graph -> compiled_adaptive_graph | 56.169 | 51.207 | 1.097 | [1.090, 1.106] | 1.082-1.111 | 18 x 12 | WIN |
| 16 / 64 / 64 | full_graph -> compiled_adaptive_graph | 77.872 | 51.207 | 1.521 | [1.513, 1.535] | 1.505-1.544 | 18 x 12 | WIN |
| 16 / 64 / 64 | full_graph -> triton_full_graph | 77.872 | 39.444 | 1.974 | [1.961, 1.994] | 1.944-2.018 | 18 x 12 | WIN |
| 16 / 64 / 64 | adaptive_graph -> triton_adaptive_graph | 56.169 | 39.188 | 1.433 | [1.422, 1.449] | 1.406-1.459 | 18 x 12 | WIN |
| 16 / 64 / 64 | compiled_adaptive_graph -> triton_adaptive_graph | 51.207 | 39.188 | 1.307 | [1.296, 1.319] | 1.273-1.332 | 18 x 12 | WIN |
| 16 / 64 / 64 | full_graph -> triton_adaptive_graph | 77.872 | 39.188 | 1.987 | [1.973, 2.010] | 1.950-2.017 | 18 x 12 | WIN |
| 16 / 64 / 64 | triton_full_graph -> triton_adaptive_graph | 39.444 | 39.188 | 1.007 | [0.998, 1.016] | 0.978-1.024 | 18 x 12 | INCONCLUSIVE |
| 16 / 128 / 32 | full_graph -> adaptive_graph | 42.655 | 37.613 | 1.134 | [1.126, 1.149] | 1.115-1.168 | 18 x 12 | WIN |
| 16 / 128 / 32 | adaptive_graph -> compiled_adaptive_graph | 37.613 | 35.047 | 1.073 | [1.067, 1.080] | 1.058-1.100 | 18 x 12 | WIN |
| 16 / 128 / 32 | full_graph -> compiled_adaptive_graph | 42.655 | 35.047 | 1.217 | [1.208, 1.234] | 1.182-1.253 | 18 x 12 | WIN |
| 16 / 128 / 32 | full_graph -> triton_full_graph | 42.655 | 25.870 | 1.649 | [1.629, 1.682] | 1.581-1.740 | 18 x 12 | WIN |
| 16 / 128 / 32 | adaptive_graph -> triton_adaptive_graph | 37.613 | 25.672 | 1.465 | [1.447, 1.478] | 1.438-1.478 | 18 x 12 | WIN |
| 16 / 128 / 32 | compiled_adaptive_graph -> triton_adaptive_graph | 35.047 | 25.672 | 1.365 | [1.347, 1.377] | 1.334-1.383 | 18 x 12 | WIN |
| 16 / 128 / 32 | full_graph -> triton_adaptive_graph | 42.655 | 25.672 | 1.662 | [1.641, 1.687] | 1.618-1.726 | 18 x 12 | WIN |
| 16 / 128 / 32 | triton_full_graph -> triton_adaptive_graph | 25.870 | 25.672 | 1.008 | [0.989, 1.022] | 0.969-1.043 | 18 x 12 | INCONCLUSIVE |
| 16 / 128 / 64 | full_graph -> adaptive_graph | 81.492 | 70.544 | 1.155 | [1.148, 1.163] | 1.141-1.166 | 18 x 12 | WIN |
| 16 / 128 / 64 | adaptive_graph -> compiled_adaptive_graph | 70.544 | 65.433 | 1.078 | [1.068, 1.085] | 1.062-1.101 | 18 x 12 | WIN |
| 16 / 128 / 64 | full_graph -> compiled_adaptive_graph | 81.492 | 65.433 | 1.245 | [1.235, 1.251] | 1.216-1.257 | 18 x 12 | WIN |
| 16 / 128 / 64 | full_graph -> triton_full_graph | 81.492 | 48.398 | 1.684 | [1.673, 1.698] | 1.662-1.720 | 18 x 12 | WIN |
| 16 / 128 / 64 | adaptive_graph -> triton_adaptive_graph | 70.544 | 48.366 | 1.459 | [1.449, 1.472] | 1.447-1.483 | 18 x 12 | WIN |
| 16 / 128 / 64 | compiled_adaptive_graph -> triton_adaptive_graph | 65.433 | 48.366 | 1.353 | [1.345, 1.369] | 1.335-1.391 | 18 x 12 | WIN |
| 16 / 128 / 64 | full_graph -> triton_adaptive_graph | 81.492 | 48.366 | 1.685 | [1.675, 1.700] | 1.670-1.712 | 18 x 12 | WIN |
| 16 / 128 / 64 | triton_full_graph -> triton_adaptive_graph | 48.398 | 48.366 | 1.001 | [0.992, 1.010] | 0.993-1.018 | 18 x 12 | INCONCLUSIVE |
| 16 / 192 / 32 | full_graph -> adaptive_graph | 46.106 | 46.091 | 1.000 | [0.991, 1.014] | 0.980-1.030 | 18 x 12 | AA_PASS |
| 16 / 192 / 32 | adaptive_graph -> compiled_adaptive_graph | 46.091 | 43.226 | 1.066 | [1.053, 1.076] | 1.043-1.091 | 18 x 12 | WIN |
| 16 / 192 / 32 | full_graph -> compiled_adaptive_graph | 46.106 | 43.226 | 1.067 | [1.056, 1.077] | 1.034-1.085 | 18 x 12 | WIN |
| 16 / 192 / 32 | full_graph -> triton_full_graph | 46.106 | 31.972 | 1.442 | [1.429, 1.462] | 1.405-1.506 | 18 x 12 | WIN |
| 16 / 192 / 32 | adaptive_graph -> triton_adaptive_graph | 46.091 | 31.923 | 1.444 | [1.430, 1.457] | 1.404-1.474 | 18 x 12 | WIN |
| 16 / 192 / 32 | compiled_adaptive_graph -> triton_adaptive_graph | 43.226 | 31.923 | 1.354 | [1.344, 1.369] | 1.306-1.392 | 18 x 12 | WIN |
| 16 / 192 / 32 | full_graph -> triton_adaptive_graph | 46.106 | 31.923 | 1.444 | [1.431, 1.461] | 1.398-1.497 | 18 x 12 | WIN |
| 16 / 192 / 32 | triton_full_graph -> triton_adaptive_graph | 31.972 | 31.923 | 1.002 | [0.991, 1.009] | 0.978-1.037 | 18 x 12 | AA_PASS |

## Verdict counts (full denominator)

- adaptive_graph -> compiled_adaptive_graph: WIN x12
- adaptive_graph -> triton_adaptive_graph: WIN x12
- compiled_adaptive_graph -> triton_adaptive_graph: WIN x12
- full_graph -> adaptive_graph: AA_PASS x2
- full_graph -> adaptive_graph: WIN x10
- full_graph -> compiled_adaptive_graph: WIN x12
- full_graph -> triton_adaptive_graph: WIN x12
- full_graph -> triton_full_graph: WIN x12
- triton_full_graph -> triton_adaptive_graph: AA_PASS x2
- triton_full_graph -> triton_adaptive_graph: INCONCLUSIVE x10
