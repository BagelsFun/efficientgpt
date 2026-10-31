# Phase 5 bootstrap latency ratios

Source: `research\phase5\ci_runs\20261030_185752_inductor\benchmark_raw.csv`; 10000 two-level bootstrap resamples (rounds, then requests); 95% percentile CI.
Ratio = baseline median / candidate median (>1: candidate faster). WIN needs ratio >= 1.05 and CI low > 1. Prompt=192 full-vs-adaptive is the A/A control (identical capacity).

| B / prompt / gen | baseline -> candidate | base ms | cand ms | ratio | 95% CI | per-round range | rounds x n | verdict |
|---|---|---:|---:|---:|---|---|---|---|
| 1 / 8 / 16 | full_graph -> adaptive_graph | 12.246 | 9.634 | 1.271 | [1.194, 1.314] | 1.187-1.322 | 6 x 15 | WIN |
| 1 / 8 / 16 | prefix_sdpa_eager -> full_graph | 33.640 | 12.246 | 2.747 | [2.665, 2.862] | 2.685-2.886 | 6 x 15 | WIN |
| 1 / 8 / 16 | prefix_sdpa_eager -> adaptive_graph | 33.640 | 9.634 | 3.492 | [3.302, 3.618] | 3.260-3.655 | 6 x 15 | WIN |
| 1 / 8 / 16 | compiled_static -> full_graph | 28.088 | 12.246 | 2.294 | [2.230, 2.388] | 2.205-2.389 | 6 x 15 | WIN |
| 1 / 8 / 16 | compiled_static -> adaptive_graph | 28.088 | 9.634 | 2.915 | [2.775, 2.984] | 2.836-3.002 | 6 x 15 | WIN |
| 1 / 8 / 16 | prefix_sdpa_eager -> compiled_static | 33.640 | 28.088 | 1.198 | [1.167, 1.236] | 1.150-1.253 | 6 x 15 | WIN |
| 1 / 64 / 16 | full_graph -> adaptive_graph | 11.994 | 9.759 | 1.229 | [1.123, 1.318] | 1.110-1.356 | 6 x 15 | WIN |
| 1 / 64 / 16 | prefix_sdpa_eager -> full_graph | 33.144 | 11.994 | 2.763 | [2.646, 2.896] | 2.615-2.922 | 6 x 15 | WIN |
| 1 / 64 / 16 | prefix_sdpa_eager -> adaptive_graph | 33.144 | 9.759 | 3.396 | [3.202, 3.551] | 3.215-3.545 | 6 x 15 | WIN |
| 1 / 64 / 16 | compiled_static -> full_graph | 28.099 | 11.994 | 2.343 | [2.230, 2.469] | 2.214-2.444 | 6 x 15 | WIN |
| 1 / 64 / 16 | compiled_static -> adaptive_graph | 28.099 | 9.759 | 2.879 | [2.698, 3.036] | 2.686-3.053 | 6 x 15 | WIN |
| 1 / 64 / 16 | prefix_sdpa_eager -> compiled_static | 33.144 | 28.099 | 1.180 | [1.156, 1.207] | 1.156-1.205 | 6 x 15 | WIN |
| 1 / 64 / 64 | full_graph -> adaptive_graph | 42.420 | 35.362 | 1.200 | [1.172, 1.230] | 1.144-1.242 | 6 x 15 | WIN |
| 1 / 64 / 64 | prefix_sdpa_eager -> full_graph | 132.773 | 42.420 | 3.130 | [3.074, 3.182] | 3.049-3.197 | 6 x 15 | WIN |
| 1 / 64 / 64 | prefix_sdpa_eager -> adaptive_graph | 132.773 | 35.362 | 3.755 | [3.647, 3.856] | 3.585-3.931 | 6 x 15 | WIN |
| 1 / 64 / 64 | compiled_static -> full_graph | 109.624 | 42.420 | 2.584 | [2.539, 2.624] | 2.504-2.660 | 6 x 15 | WIN |
| 1 / 64 / 64 | compiled_static -> adaptive_graph | 109.624 | 35.362 | 3.100 | [3.032, 3.174] | 2.974-3.271 | 6 x 15 | WIN |
| 1 / 64 / 64 | prefix_sdpa_eager -> compiled_static | 132.773 | 109.624 | 1.211 | [1.196, 1.225] | 1.193-1.222 | 6 x 15 | WIN |
| 1 / 128 / 32 | full_graph -> adaptive_graph | 22.140 | 20.698 | 1.070 | [1.053, 1.090] | 1.048-1.091 | 6 x 15 | WIN |
| 1 / 128 / 32 | prefix_sdpa_eager -> full_graph | 66.706 | 22.140 | 3.013 | [2.954, 3.063] | 2.929-3.109 | 6 x 15 | WIN |
| 1 / 128 / 32 | prefix_sdpa_eager -> adaptive_graph | 66.706 | 20.698 | 3.223 | [3.172, 3.269] | 3.176-3.311 | 6 x 15 | WIN |
| 1 / 128 / 32 | compiled_static -> full_graph | 54.909 | 22.140 | 2.480 | [2.418, 2.542] | 2.379-2.568 | 6 x 15 | WIN |
| 1 / 128 / 32 | compiled_static -> adaptive_graph | 54.909 | 20.698 | 2.653 | [2.593, 2.706] | 2.567-2.748 | 6 x 15 | WIN |
| 1 / 128 / 32 | prefix_sdpa_eager -> compiled_static | 66.706 | 54.909 | 1.215 | [1.184, 1.249] | 1.163-1.269 | 6 x 15 | WIN |
| 1 / 128 / 64 | full_graph -> adaptive_graph | 41.954 | 39.273 | 1.068 | [1.044, 1.095] | 1.033-1.113 | 6 x 15 | WIN |
| 1 / 128 / 64 | prefix_sdpa_eager -> full_graph | 133.318 | 41.954 | 3.178 | [3.127, 3.246] | 3.113-3.279 | 6 x 15 | WIN |
| 1 / 128 / 64 | prefix_sdpa_eager -> adaptive_graph | 133.318 | 39.273 | 3.395 | [3.358, 3.447] | 3.362-3.516 | 6 x 15 | WIN |
| 1 / 128 / 64 | compiled_static -> full_graph | 109.280 | 41.954 | 2.605 | [2.556, 2.682] | 2.557-2.702 | 6 x 15 | WIN |
| 1 / 128 / 64 | compiled_static -> adaptive_graph | 109.280 | 39.273 | 2.783 | [2.750, 2.843] | 2.744-2.861 | 6 x 15 | WIN |
| 1 / 128 / 64 | prefix_sdpa_eager -> compiled_static | 133.318 | 109.280 | 1.220 | [1.191, 1.236] | 1.175-1.236 | 6 x 15 | WIN |
| 1 / 192 / 32 | full_graph -> adaptive_graph | 21.696 | 22.126 | 0.981 | [0.950, 1.020] | 0.945-1.048 | 6 x 15 | AA_PASS |
| 1 / 192 / 32 | prefix_sdpa_eager -> full_graph | 66.090 | 21.696 | 3.046 | [2.959, 3.186] | 2.947-3.307 | 6 x 15 | WIN |
| 1 / 192 / 32 | prefix_sdpa_eager -> adaptive_graph | 66.090 | 22.126 | 2.987 | [2.925, 3.110] | 2.876-3.146 | 6 x 15 | WIN |
| 1 / 192 / 32 | compiled_static -> full_graph | 55.305 | 21.696 | 2.549 | [2.465, 2.660] | 2.433-2.825 | 6 x 15 | WIN |
| 1 / 192 / 32 | compiled_static -> adaptive_graph | 55.305 | 22.126 | 2.499 | [2.422, 2.592] | 2.385-2.688 | 6 x 15 | WIN |
| 1 / 192 / 32 | prefix_sdpa_eager -> compiled_static | 66.090 | 55.305 | 1.195 | [1.166, 1.233] | 1.170-1.242 | 6 x 15 | WIN |
| 16 / 8 / 16 | full_graph -> adaptive_graph | 20.068 | 14.973 | 1.340 | [1.288, 1.355] | 1.256-1.349 | 6 x 15 | WIN |
| 16 / 8 / 16 | prefix_sdpa_eager -> full_graph | 34.216 | 20.068 | 1.705 | [1.671, 1.789] | 1.641-1.902 | 6 x 15 | WIN |
| 16 / 8 / 16 | prefix_sdpa_eager -> adaptive_graph | 34.216 | 14.973 | 2.285 | [2.236, 2.344] | 2.210-2.536 | 6 x 15 | WIN |
| 16 / 8 / 16 | compiled_static -> full_graph | 33.050 | 20.068 | 1.647 | [1.612, 1.732] | 1.600-1.752 | 6 x 15 | WIN |
| 16 / 8 / 16 | compiled_static -> adaptive_graph | 33.050 | 14.973 | 2.207 | [2.148, 2.255] | 2.114-2.287 | 6 x 15 | WIN |
| 16 / 8 / 16 | prefix_sdpa_eager -> compiled_static | 34.216 | 33.050 | 1.035 | [1.013, 1.066] | 1.000-1.148 | 6 x 15 | INCONCLUSIVE |
| 16 / 64 / 16 | full_graph -> adaptive_graph | 20.598 | 15.310 | 1.345 | [1.319, 1.367] | 1.297-1.367 | 6 x 15 | WIN |
| 16 / 64 / 16 | prefix_sdpa_eager -> full_graph | 34.457 | 20.598 | 1.673 | [1.601, 1.738] | 1.586-1.760 | 6 x 15 | WIN |
| 16 / 64 / 16 | prefix_sdpa_eager -> adaptive_graph | 34.457 | 15.310 | 2.251 | [2.156, 2.341] | 2.128-2.406 | 6 x 15 | WIN |
| 16 / 64 / 16 | compiled_static -> full_graph | 33.575 | 20.598 | 1.630 | [1.590, 1.660] | 1.526-1.666 | 6 x 15 | WIN |
| 16 / 64 / 16 | compiled_static -> adaptive_graph | 33.575 | 15.310 | 2.193 | [2.130, 2.241] | 2.070-2.278 | 6 x 15 | WIN |
| 16 / 64 / 16 | prefix_sdpa_eager -> compiled_static | 34.457 | 33.575 | 1.026 | [0.977, 1.065] | 0.959-1.082 | 6 x 15 | INCONCLUSIVE |
| 16 / 64 / 64 | full_graph -> adaptive_graph | 78.013 | 55.137 | 1.415 | [1.361, 1.459] | 1.307-1.473 | 6 x 15 | WIN |
| 16 / 64 / 64 | prefix_sdpa_eager -> full_graph | 134.016 | 78.013 | 1.718 | [1.691, 1.760] | 1.678-1.835 | 6 x 15 | WIN |
| 16 / 64 / 64 | prefix_sdpa_eager -> adaptive_graph | 134.016 | 55.137 | 2.431 | [2.377, 2.497] | 2.381-2.520 | 6 x 15 | WIN |
| 16 / 64 / 64 | compiled_static -> full_graph | 132.601 | 78.013 | 1.700 | [1.664, 1.743] | 1.618-1.809 | 6 x 15 | WIN |
| 16 / 64 / 64 | compiled_static -> adaptive_graph | 132.601 | 55.137 | 2.405 | [2.353, 2.472] | 2.365-2.515 | 6 x 15 | WIN |
| 16 / 64 / 64 | prefix_sdpa_eager -> compiled_static | 134.016 | 132.601 | 1.011 | [0.999, 1.025] | 0.998-1.037 | 6 x 15 | INCONCLUSIVE |
| 16 / 128 / 32 | full_graph -> adaptive_graph | 42.342 | 36.875 | 1.148 | [1.122, 1.186] | 1.103-1.203 | 6 x 15 | WIN |
| 16 / 128 / 32 | prefix_sdpa_eager -> full_graph | 67.262 | 42.342 | 1.589 | [1.567, 1.619] | 1.555-1.643 | 6 x 15 | WIN |
| 16 / 128 / 32 | prefix_sdpa_eager -> adaptive_graph | 67.262 | 36.875 | 1.824 | [1.792, 1.886] | 1.787-1.926 | 6 x 15 | WIN |
| 16 / 128 / 32 | compiled_static -> full_graph | 68.648 | 42.342 | 1.621 | [1.576, 1.651] | 1.554-1.665 | 6 x 15 | WIN |
| 16 / 128 / 32 | compiled_static -> adaptive_graph | 68.648 | 36.875 | 1.862 | [1.819, 1.909] | 1.837-1.937 | 6 x 15 | WIN |
| 16 / 128 / 32 | prefix_sdpa_eager -> compiled_static | 67.262 | 68.648 | 0.980 | [0.963, 1.015] | 0.956-1.032 | 6 x 15 | INCONCLUSIVE |
| 16 / 128 / 64 | full_graph -> adaptive_graph | 79.530 | 69.685 | 1.141 | [1.094, 1.187] | 1.063-1.207 | 6 x 15 | WIN |
| 16 / 128 / 64 | prefix_sdpa_eager -> full_graph | 134.656 | 79.530 | 1.693 | [1.667, 1.754] | 1.656-1.783 | 6 x 15 | WIN |
| 16 / 128 / 64 | prefix_sdpa_eager -> adaptive_graph | 134.656 | 69.685 | 1.932 | [1.899, 1.991] | 1.895-2.030 | 6 x 15 | WIN |
| 16 / 128 / 64 | compiled_static -> full_graph | 133.937 | 79.530 | 1.684 | [1.663, 1.744] | 1.657-1.760 | 6 x 15 | WIN |
| 16 / 128 / 64 | compiled_static -> adaptive_graph | 133.937 | 69.685 | 1.922 | [1.884, 1.987] | 1.871-2.059 | 6 x 15 | WIN |
| 16 / 128 / 64 | prefix_sdpa_eager -> compiled_static | 134.656 | 133.937 | 1.005 | [0.995, 1.015] | 0.986-1.013 | 6 x 15 | INCONCLUSIVE |
| 16 / 192 / 32 | full_graph -> adaptive_graph | 45.510 | 46.065 | 0.988 | [0.956, 1.007] | 0.938-1.037 | 6 x 15 | AA_PASS |
| 16 / 192 / 32 | prefix_sdpa_eager -> full_graph | 68.497 | 45.510 | 1.505 | [1.484, 1.561] | 1.444-1.569 | 6 x 15 | WIN |
| 16 / 192 / 32 | prefix_sdpa_eager -> adaptive_graph | 68.497 | 46.065 | 1.487 | [1.466, 1.516] | 1.462-1.532 | 6 x 15 | WIN |
| 16 / 192 / 32 | compiled_static -> full_graph | 72.231 | 45.510 | 1.587 | [1.552, 1.643] | 1.513-1.675 | 6 x 15 | WIN |
| 16 / 192 / 32 | compiled_static -> adaptive_graph | 72.231 | 46.065 | 1.568 | [1.548, 1.585] | 1.542-1.574 | 6 x 15 | WIN |
| 16 / 192 / 32 | prefix_sdpa_eager -> compiled_static | 68.497 | 72.231 | 0.948 | [0.935, 0.974] | 0.933-0.994 | 6 x 15 | SLOWER |

## Verdict counts (full denominator)

- compiled_static -> adaptive_graph: WIN x12
- compiled_static -> full_graph: WIN x12
- full_graph -> adaptive_graph: AA_PASS x2
- full_graph -> adaptive_graph: WIN x10
- prefix_sdpa_eager -> adaptive_graph: WIN x12
- prefix_sdpa_eager -> compiled_static: INCONCLUSIVE x5
- prefix_sdpa_eager -> compiled_static: SLOWER x1
- prefix_sdpa_eager -> compiled_static: WIN x6
- prefix_sdpa_eager -> full_graph: WIN x12
