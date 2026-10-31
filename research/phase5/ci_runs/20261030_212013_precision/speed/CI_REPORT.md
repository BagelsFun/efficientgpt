# Phase 5 in-process interleaved latency ratios

3 processes x 6 blocks x 12 requests per mode and workload; clusters = (process, block). Peak memory not comparable (all modes resident together).

Source: `research\phase5\ci_runs\20261030_212013_precision\speed\benchmark_raw.csv`; 10000 two-level bootstrap resamples (rounds, then requests); 95% percentile CI.
Ratio = baseline median / candidate median (>1: candidate faster). WIN needs ratio >= 1.05 and CI low > 1. Prompt=192 full-vs-adaptive is the A/A control (identical capacity).

| B / prompt / gen | baseline -> candidate | base ms | cand ms | ratio | 95% CI | per-round range | rounds x n | verdict |
|---|---|---:|---:|---:|---|---|---|---|
| 1 / 8 / 16 | adaptive_graph -> compiled_triton_full_graph | 9.303 | 7.256 | 1.282 | [1.256, 1.320] | 1.123-1.413 | 18 x 12 | WIN |
| 1 / 8 / 16 | compiled_triton_full_graph -> fp16_compiled_triton_full_graph | 7.256 | 5.188 | 1.399 | [1.349, 1.436] | 1.291-1.629 | 18 x 12 | WIN |
| 1 / 8 / 16 | compiled_triton_full_graph -> bf16_compiled_triton_full_graph | 7.256 | 5.257 | 1.380 | [1.340, 1.421] | 1.307-1.596 | 18 x 12 | WIN |
| 1 / 8 / 16 | fp16_compiled_triton_full_graph -> bf16_compiled_triton_full_graph | 5.188 | 5.257 | 0.987 | [0.962, 1.022] | 0.928-1.054 | 18 x 12 | INCONCLUSIVE |
| 1 / 8 / 16 | fp16_triton_full_graph -> fp16_compiled_triton_full_graph | 5.861 | 5.188 | 1.130 | [1.093, 1.157] | 1.034-1.237 | 18 x 12 | WIN |
| 1 / 8 / 16 | adaptive_graph -> fp16_compiled_triton_full_graph | 9.303 | 5.188 | 1.793 | [1.746, 1.841] | 1.704-1.945 | 18 x 12 | WIN |
| 1 / 8 / 16 | adaptive_graph -> fp16_triton_full_graph | 9.303 | 5.861 | 1.587 | [1.554, 1.632] | 1.417-1.684 | 18 x 12 | WIN |
| 1 / 8 / 16 | adaptive_graph -> bf16_compiled_triton_full_graph | 9.303 | 5.257 | 1.770 | [1.729, 1.823] | 1.706-1.902 | 18 x 12 | WIN |
| 1 / 64 / 16 | adaptive_graph -> compiled_triton_full_graph | 9.337 | 7.314 | 1.277 | [1.250, 1.317] | 1.097-1.475 | 18 x 12 | WIN |
| 1 / 64 / 16 | compiled_triton_full_graph -> fp16_compiled_triton_full_graph | 7.314 | 5.278 | 1.386 | [1.357, 1.423] | 1.312-1.587 | 18 x 12 | WIN |
| 1 / 64 / 16 | compiled_triton_full_graph -> bf16_compiled_triton_full_graph | 7.314 | 5.265 | 1.389 | [1.359, 1.426] | 1.331-1.534 | 18 x 12 | WIN |
| 1 / 64 / 16 | fp16_compiled_triton_full_graph -> bf16_compiled_triton_full_graph | 5.278 | 5.265 | 1.002 | [0.977, 1.032] | 0.925-1.083 | 18 x 12 | INCONCLUSIVE |
| 1 / 64 / 16 | fp16_triton_full_graph -> fp16_compiled_triton_full_graph | 5.905 | 5.278 | 1.119 | [1.090, 1.151] | 1.066-1.218 | 18 x 12 | WIN |
| 1 / 64 / 16 | adaptive_graph -> fp16_compiled_triton_full_graph | 9.337 | 5.278 | 1.769 | [1.731, 1.831] | 1.685-2.066 | 18 x 12 | WIN |
| 1 / 64 / 16 | adaptive_graph -> fp16_triton_full_graph | 9.337 | 5.905 | 1.581 | [1.540, 1.636] | 1.481-1.742 | 18 x 12 | WIN |
| 1 / 64 / 16 | adaptive_graph -> bf16_compiled_triton_full_graph | 9.337 | 5.265 | 1.773 | [1.731, 1.845] | 1.676-2.108 | 18 x 12 | WIN |
| 1 / 64 / 64 | adaptive_graph -> compiled_triton_full_graph | 33.513 | 24.177 | 1.386 | [1.374, 1.402] | 1.347-1.404 | 18 x 12 | WIN |
| 1 / 64 / 64 | compiled_triton_full_graph -> fp16_compiled_triton_full_graph | 24.177 | 15.918 | 1.519 | [1.494, 1.551] | 1.466-1.580 | 18 x 12 | WIN |
| 1 / 64 / 64 | compiled_triton_full_graph -> bf16_compiled_triton_full_graph | 24.177 | 15.986 | 1.512 | [1.488, 1.541] | 1.459-1.542 | 18 x 12 | WIN |
| 1 / 64 / 64 | fp16_compiled_triton_full_graph -> bf16_compiled_triton_full_graph | 15.918 | 15.986 | 0.996 | [0.976, 1.013] | 0.963-1.013 | 18 x 12 | INCONCLUSIVE |
| 1 / 64 / 64 | fp16_triton_full_graph -> fp16_compiled_triton_full_graph | 18.289 | 15.918 | 1.149 | [1.132, 1.171] | 1.121-1.173 | 18 x 12 | WIN |
| 1 / 64 / 64 | adaptive_graph -> fp16_compiled_triton_full_graph | 33.513 | 15.918 | 2.105 | [2.076, 2.153] | 2.053-2.157 | 18 x 12 | WIN |
| 1 / 64 / 64 | adaptive_graph -> fp16_triton_full_graph | 33.513 | 18.289 | 1.832 | [1.809, 1.860] | 1.805-1.881 | 18 x 12 | WIN |
| 1 / 64 / 64 | adaptive_graph -> bf16_compiled_triton_full_graph | 33.513 | 15.986 | 2.096 | [2.067, 2.141] | 2.013-2.128 | 18 x 12 | WIN |
| 1 / 128 / 32 | adaptive_graph -> compiled_triton_full_graph | 19.488 | 13.350 | 1.460 | [1.425, 1.497] | 1.385-1.494 | 18 x 12 | WIN |
| 1 / 128 / 32 | compiled_triton_full_graph -> fp16_compiled_triton_full_graph | 13.350 | 8.765 | 1.523 | [1.484, 1.574] | 1.428-1.667 | 18 x 12 | WIN |
| 1 / 128 / 32 | compiled_triton_full_graph -> bf16_compiled_triton_full_graph | 13.350 | 8.870 | 1.505 | [1.457, 1.552] | 1.364-1.576 | 18 x 12 | WIN |
| 1 / 128 / 32 | fp16_compiled_triton_full_graph -> bf16_compiled_triton_full_graph | 8.765 | 8.870 | 0.988 | [0.953, 1.016] | 0.928-1.087 | 18 x 12 | INCONCLUSIVE |
| 1 / 128 / 32 | fp16_triton_full_graph -> fp16_compiled_triton_full_graph | 10.060 | 8.765 | 1.148 | [1.111, 1.197] | 1.027-1.301 | 18 x 12 | WIN |
| 1 / 128 / 32 | adaptive_graph -> fp16_compiled_triton_full_graph | 19.488 | 8.765 | 2.223 | [2.170, 2.285] | 1.998-2.346 | 18 x 12 | WIN |
| 1 / 128 / 32 | adaptive_graph -> fp16_triton_full_graph | 19.488 | 10.060 | 1.937 | [1.875, 1.991] | 1.797-2.017 | 18 x 12 | WIN |
| 1 / 128 / 32 | adaptive_graph -> bf16_compiled_triton_full_graph | 19.488 | 8.870 | 2.197 | [2.119, 2.253] | 1.959-2.264 | 18 x 12 | WIN |
| 1 / 128 / 64 | adaptive_graph -> compiled_triton_full_graph | 37.266 | 24.523 | 1.520 | [1.499, 1.533] | 1.478-1.539 | 18 x 12 | WIN |
| 1 / 128 / 64 | compiled_triton_full_graph -> fp16_compiled_triton_full_graph | 24.523 | 16.207 | 1.513 | [1.494, 1.533] | 1.496-1.551 | 18 x 12 | WIN |
| 1 / 128 / 64 | compiled_triton_full_graph -> bf16_compiled_triton_full_graph | 24.523 | 16.329 | 1.502 | [1.487, 1.521] | 1.482-1.549 | 18 x 12 | WIN |
| 1 / 128 / 64 | fp16_compiled_triton_full_graph -> bf16_compiled_triton_full_graph | 16.207 | 16.329 | 0.993 | [0.981, 1.006] | 0.970-1.024 | 18 x 12 | INCONCLUSIVE |
| 1 / 128 / 64 | fp16_triton_full_graph -> fp16_compiled_triton_full_graph | 18.474 | 16.207 | 1.140 | [1.125, 1.153] | 1.119-1.163 | 18 x 12 | WIN |
| 1 / 128 / 64 | adaptive_graph -> fp16_compiled_triton_full_graph | 37.266 | 16.207 | 2.299 | [2.266, 2.322] | 2.236-2.350 | 18 x 12 | WIN |
| 1 / 128 / 64 | adaptive_graph -> fp16_triton_full_graph | 37.266 | 18.474 | 2.017 | [1.994, 2.034] | 1.970-2.093 | 18 x 12 | WIN |
| 1 / 128 / 64 | adaptive_graph -> bf16_compiled_triton_full_graph | 37.266 | 16.329 | 2.282 | [2.260, 2.303] | 2.235-2.349 | 18 x 12 | WIN |
| 1 / 192 / 32 | adaptive_graph -> compiled_triton_full_graph | 20.983 | 13.318 | 1.576 | [1.554, 1.618] | 1.524-1.626 | 18 x 12 | WIN |
| 1 / 192 / 32 | compiled_triton_full_graph -> fp16_compiled_triton_full_graph | 13.318 | 8.990 | 1.481 | [1.434, 1.520] | 1.386-1.631 | 18 x 12 | WIN |
| 1 / 192 / 32 | compiled_triton_full_graph -> bf16_compiled_triton_full_graph | 13.318 | 8.791 | 1.515 | [1.469, 1.548] | 1.426-1.592 | 18 x 12 | WIN |
| 1 / 192 / 32 | fp16_compiled_triton_full_graph -> bf16_compiled_triton_full_graph | 8.990 | 8.791 | 1.023 | [0.992, 1.054] | 0.953-1.082 | 18 x 12 | INCONCLUSIVE |
| 1 / 192 / 32 | fp16_triton_full_graph -> fp16_compiled_triton_full_graph | 10.214 | 8.990 | 1.136 | [1.098, 1.171] | 1.040-1.276 | 18 x 12 | WIN |
| 1 / 192 / 32 | adaptive_graph -> fp16_compiled_triton_full_graph | 20.983 | 8.990 | 2.334 | [2.269, 2.388] | 2.200-2.527 | 18 x 12 | WIN |
| 1 / 192 / 32 | adaptive_graph -> fp16_triton_full_graph | 20.983 | 10.214 | 2.054 | [2.020, 2.091] | 1.940-2.138 | 18 x 12 | WIN |
| 1 / 192 / 32 | adaptive_graph -> bf16_compiled_triton_full_graph | 20.983 | 8.791 | 2.387 | [2.325, 2.431] | 2.256-2.485 | 18 x 12 | WIN |
| 16 / 8 / 16 | adaptive_graph -> compiled_triton_full_graph | 13.428 | 8.140 | 1.650 | [1.576, 1.729] | 1.489-1.856 | 18 x 12 | WIN |
| 16 / 8 / 16 | compiled_triton_full_graph -> fp16_compiled_triton_full_graph | 8.140 | 5.328 | 1.528 | [1.463, 1.602] | 1.252-1.681 | 18 x 12 | WIN |
| 16 / 8 / 16 | compiled_triton_full_graph -> bf16_compiled_triton_full_graph | 8.140 | 5.476 | 1.487 | [1.429, 1.565] | 1.316-1.677 | 18 x 12 | WIN |
| 16 / 8 / 16 | fp16_compiled_triton_full_graph -> bf16_compiled_triton_full_graph | 5.328 | 5.476 | 0.973 | [0.945, 1.005] | 0.838-1.100 | 18 x 12 | INCONCLUSIVE |
| 16 / 8 / 16 | fp16_triton_full_graph -> fp16_compiled_triton_full_graph | 6.102 | 5.328 | 1.145 | [1.110, 1.178] | 0.993-1.249 | 18 x 12 | WIN |
| 16 / 8 / 16 | adaptive_graph -> fp16_compiled_triton_full_graph | 13.428 | 5.328 | 2.520 | [2.445, 2.597] | 2.305-2.721 | 18 x 12 | WIN |
| 16 / 8 / 16 | adaptive_graph -> fp16_triton_full_graph | 13.428 | 6.102 | 2.201 | [2.152, 2.263] | 2.029-2.339 | 18 x 12 | WIN |
| 16 / 8 / 16 | adaptive_graph -> bf16_compiled_triton_full_graph | 13.428 | 5.476 | 2.452 | [2.376, 2.550] | 1.997-2.609 | 18 x 12 | WIN |
| 16 / 64 / 16 | adaptive_graph -> compiled_triton_full_graph | 14.343 | 9.754 | 1.470 | [1.434, 1.542] | 1.347-1.668 | 18 x 12 | WIN |
| 16 / 64 / 16 | compiled_triton_full_graph -> fp16_compiled_triton_full_graph | 9.754 | 6.154 | 1.585 | [1.511, 1.626] | 1.488-1.644 | 18 x 12 | WIN |
| 16 / 64 / 16 | compiled_triton_full_graph -> bf16_compiled_triton_full_graph | 9.754 | 6.317 | 1.544 | [1.461, 1.594] | 1.346-1.682 | 18 x 12 | WIN |
| 16 / 64 / 16 | fp16_compiled_triton_full_graph -> bf16_compiled_triton_full_graph | 6.154 | 6.317 | 0.974 | [0.934, 1.010] | 0.863-1.050 | 18 x 12 | INCONCLUSIVE |
| 16 / 64 / 16 | fp16_triton_full_graph -> fp16_compiled_triton_full_graph | 6.939 | 6.154 | 1.128 | [1.078, 1.181] | 1.040-1.267 | 18 x 12 | WIN |
| 16 / 64 / 16 | adaptive_graph -> fp16_compiled_triton_full_graph | 14.343 | 6.154 | 2.331 | [2.252, 2.385] | 2.121-2.524 | 18 x 12 | WIN |
| 16 / 64 / 16 | adaptive_graph -> fp16_triton_full_graph | 14.343 | 6.939 | 2.067 | [1.965, 2.147] | 1.854-2.237 | 18 x 12 | WIN |
| 16 / 64 / 16 | adaptive_graph -> bf16_compiled_triton_full_graph | 14.343 | 6.317 | 2.270 | [2.176, 2.330] | 2.055-2.434 | 18 x 12 | WIN |
| 16 / 64 / 64 | adaptive_graph -> compiled_triton_full_graph | 52.806 | 33.532 | 1.575 | [1.567, 1.583] | 1.515-1.605 | 18 x 12 | WIN |
| 16 / 64 / 64 | compiled_triton_full_graph -> fp16_compiled_triton_full_graph | 33.532 | 20.445 | 1.640 | [1.626, 1.649] | 1.573-1.666 | 18 x 12 | WIN |
| 16 / 64 / 64 | compiled_triton_full_graph -> bf16_compiled_triton_full_graph | 33.532 | 21.029 | 1.595 | [1.585, 1.604] | 1.540-1.620 | 18 x 12 | WIN |
| 16 / 64 / 64 | fp16_compiled_triton_full_graph -> bf16_compiled_triton_full_graph | 20.445 | 21.029 | 0.972 | [0.967, 0.980] | 0.950-0.997 | 18 x 12 | SLOWER |
| 16 / 64 / 64 | fp16_triton_full_graph -> fp16_compiled_triton_full_graph | 23.963 | 20.445 | 1.172 | [1.156, 1.188] | 1.128-1.210 | 18 x 12 | WIN |
| 16 / 64 / 64 | adaptive_graph -> fp16_compiled_triton_full_graph | 52.806 | 20.445 | 2.583 | [2.564, 2.597] | 2.502-2.629 | 18 x 12 | WIN |
| 16 / 64 / 64 | adaptive_graph -> fp16_triton_full_graph | 52.806 | 23.963 | 2.204 | [2.176, 2.234] | 2.146-2.274 | 18 x 12 | WIN |
| 16 / 64 / 64 | adaptive_graph -> bf16_compiled_triton_full_graph | 52.806 | 21.029 | 2.511 | [2.496, 2.527] | 2.421-2.562 | 18 x 12 | WIN |
| 16 / 128 / 32 | adaptive_graph -> compiled_triton_full_graph | 34.928 | 22.273 | 1.568 | [1.553, 1.591] | 1.535-1.608 | 18 x 12 | WIN |
| 16 / 128 / 32 | compiled_triton_full_graph -> fp16_compiled_triton_full_graph | 22.273 | 13.010 | 1.712 | [1.682, 1.752] | 1.647-1.852 | 18 x 12 | WIN |
| 16 / 128 / 32 | compiled_triton_full_graph -> bf16_compiled_triton_full_graph | 22.273 | 13.294 | 1.675 | [1.653, 1.697] | 1.619-1.733 | 18 x 12 | WIN |
| 16 / 128 / 32 | fp16_compiled_triton_full_graph -> bf16_compiled_triton_full_graph | 13.010 | 13.294 | 0.979 | [0.961, 0.991] | 0.921-1.015 | 18 x 12 | SLOWER |
| 16 / 128 / 32 | fp16_triton_full_graph -> fp16_compiled_triton_full_graph | 14.652 | 13.010 | 1.126 | [1.115, 1.147] | 1.098-1.199 | 18 x 12 | WIN |
| 16 / 128 / 32 | adaptive_graph -> fp16_compiled_triton_full_graph | 34.928 | 13.010 | 2.685 | [2.663, 2.735] | 2.618-2.865 | 18 x 12 | WIN |
| 16 / 128 / 32 | adaptive_graph -> fp16_triton_full_graph | 34.928 | 14.652 | 2.384 | [2.374, 2.398] | 2.340-2.404 | 18 x 12 | WIN |
| 16 / 128 / 32 | adaptive_graph -> bf16_compiled_triton_full_graph | 34.928 | 13.294 | 2.627 | [2.614, 2.655] | 2.595-2.696 | 18 x 12 | WIN |
| 16 / 128 / 64 | adaptive_graph -> compiled_triton_full_graph | 65.902 | 41.366 | 1.593 | [1.581, 1.601] | 1.569-1.614 | 18 x 12 | WIN |
| 16 / 128 / 64 | compiled_triton_full_graph -> fp16_compiled_triton_full_graph | 41.366 | 23.841 | 1.735 | [1.716, 1.756] | 1.701-1.788 | 18 x 12 | WIN |
| 16 / 128 / 64 | compiled_triton_full_graph -> bf16_compiled_triton_full_graph | 41.366 | 24.514 | 1.687 | [1.672, 1.707] | 1.629-1.725 | 18 x 12 | WIN |
| 16 / 128 / 64 | fp16_compiled_triton_full_graph -> bf16_compiled_triton_full_graph | 23.841 | 24.514 | 0.973 | [0.961, 0.987] | 0.924-1.007 | 18 x 12 | SLOWER |
| 16 / 128 / 64 | fp16_triton_full_graph -> fp16_compiled_triton_full_graph | 27.327 | 23.841 | 1.146 | [1.129, 1.159] | 1.102-1.178 | 18 x 12 | WIN |
| 16 / 128 / 64 | adaptive_graph -> fp16_compiled_triton_full_graph | 65.902 | 23.841 | 2.764 | [2.736, 2.788] | 2.723-2.841 | 18 x 12 | WIN |
| 16 / 128 / 64 | adaptive_graph -> fp16_triton_full_graph | 65.902 | 27.327 | 2.412 | [2.393, 2.435] | 2.394-2.490 | 18 x 12 | WIN |
| 16 / 128 / 64 | adaptive_graph -> bf16_compiled_triton_full_graph | 65.902 | 24.514 | 2.688 | [2.663, 2.718] | 2.626-2.769 | 18 x 12 | WIN |
| 16 / 192 / 32 | adaptive_graph -> compiled_triton_full_graph | 43.077 | 27.991 | 1.539 | [1.528, 1.551] | 1.507-1.568 | 18 x 12 | WIN |
| 16 / 192 / 32 | compiled_triton_full_graph -> fp16_compiled_triton_full_graph | 27.991 | 15.537 | 1.802 | [1.789, 1.815] | 1.764-1.837 | 18 x 12 | WIN |
| 16 / 192 / 32 | compiled_triton_full_graph -> bf16_compiled_triton_full_graph | 27.991 | 15.822 | 1.769 | [1.758, 1.781] | 1.728-1.800 | 18 x 12 | WIN |
| 16 / 192 / 32 | fp16_compiled_triton_full_graph -> bf16_compiled_triton_full_graph | 15.537 | 15.822 | 0.982 | [0.977, 0.987] | 0.969-0.993 | 18 x 12 | SLOWER |
| 16 / 192 / 32 | fp16_triton_full_graph -> fp16_compiled_triton_full_graph | 17.078 | 15.537 | 1.099 | [1.092, 1.106] | 1.084-1.126 | 18 x 12 | WIN |
| 16 / 192 / 32 | adaptive_graph -> fp16_compiled_triton_full_graph | 43.077 | 15.537 | 2.772 | [2.755, 2.790] | 2.710-2.813 | 18 x 12 | WIN |
| 16 / 192 / 32 | adaptive_graph -> fp16_triton_full_graph | 43.077 | 17.078 | 2.522 | [2.509, 2.538] | 2.470-2.563 | 18 x 12 | WIN |
| 16 / 192 / 32 | adaptive_graph -> bf16_compiled_triton_full_graph | 43.077 | 15.822 | 2.723 | [2.709, 2.736] | 2.654-2.757 | 18 x 12 | WIN |

## Verdict counts (full denominator)

- adaptive_graph -> bf16_compiled_triton_full_graph: WIN x12
- adaptive_graph -> compiled_triton_full_graph: WIN x12
- adaptive_graph -> fp16_compiled_triton_full_graph: WIN x12
- adaptive_graph -> fp16_triton_full_graph: WIN x12
- compiled_triton_full_graph -> bf16_compiled_triton_full_graph: WIN x12
- compiled_triton_full_graph -> fp16_compiled_triton_full_graph: WIN x12
- fp16_compiled_triton_full_graph -> bf16_compiled_triton_full_graph: INCONCLUSIVE x8
- fp16_compiled_triton_full_graph -> bf16_compiled_triton_full_graph: SLOWER x4
- fp16_triton_full_graph -> fp16_compiled_triton_full_graph: WIN x12
