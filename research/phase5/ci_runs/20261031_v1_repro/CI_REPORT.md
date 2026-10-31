# Phase 5 in-process interleaved latency ratios

3 processes x 6 blocks x 12 requests per mode and workload; clusters = (process, block). Peak memory not comparable (all modes resident together).

Source: `research\phase5\ci_runs\20261031_v1_repro\benchmark_raw.csv`; 10000 two-level bootstrap resamples (rounds, then requests); 95% percentile CI.
Ratio = baseline median / candidate median (>1: candidate faster). WIN needs ratio >= 1.05 and CI low > 1. Prompt=192 full-vs-adaptive is the A/A control (identical capacity).

| B / prompt / gen | baseline -> candidate | base ms | cand ms | ratio | 95% CI | per-round range | rounds x n | verdict |
|---|---|---:|---:|---:|---|---|---|---|
| 1 / 8 / 16 | prefix_sdpa_eager -> adaptive_graph | 31.325 | 8.964 | 3.495 | [3.450, 3.541] | 3.385-3.680 | 18 x 12 | WIN |
| 1 / 8 / 16 | adaptive_graph -> compiled_triton_full_graph | 8.964 | 7.061 | 1.269 | [1.256, 1.280] | 1.236-1.300 | 18 x 12 | WIN |
| 1 / 8 / 16 | compiled_triton_full_graph -> fp16_compiled_triton_full_graph | 7.061 | 5.048 | 1.399 | [1.373, 1.437] | 1.339-1.452 | 18 x 12 | WIN |
| 1 / 8 / 16 | adaptive_graph -> fp16_compiled_triton_full_graph | 8.964 | 5.048 | 1.776 | [1.743, 1.822] | 1.731-1.839 | 18 x 12 | WIN |
| 1 / 64 / 16 | prefix_sdpa_eager -> adaptive_graph | 31.497 | 8.996 | 3.501 | [3.456, 3.533] | 3.403-3.724 | 18 x 12 | WIN |
| 1 / 64 / 16 | adaptive_graph -> compiled_triton_full_graph | 8.996 | 7.091 | 1.269 | [1.260, 1.284] | 1.237-1.303 | 18 x 12 | WIN |
| 1 / 64 / 16 | compiled_triton_full_graph -> fp16_compiled_triton_full_graph | 7.091 | 5.139 | 1.380 | [1.353, 1.409] | 1.324-1.440 | 18 x 12 | WIN |
| 1 / 64 / 16 | adaptive_graph -> fp16_compiled_triton_full_graph | 8.996 | 5.139 | 1.751 | [1.723, 1.789] | 1.711-1.803 | 18 x 12 | WIN |
| 1 / 64 / 64 | prefix_sdpa_eager -> adaptive_graph | 124.957 | 30.750 | 4.064 | [3.945, 4.130] | 3.868-4.519 | 18 x 12 | WIN |
| 1 / 64 / 64 | adaptive_graph -> compiled_triton_full_graph | 30.750 | 22.275 | 1.380 | [1.364, 1.400] | 1.313-1.401 | 18 x 12 | WIN |
| 1 / 64 / 64 | compiled_triton_full_graph -> fp16_compiled_triton_full_graph | 22.275 | 14.546 | 1.531 | [1.515, 1.556] | 1.506-1.685 | 18 x 12 | WIN |
| 1 / 64 / 64 | adaptive_graph -> fp16_compiled_triton_full_graph | 30.750 | 14.546 | 2.114 | [2.092, 2.155] | 2.082-2.212 | 18 x 12 | WIN |
| 1 / 128 / 32 | prefix_sdpa_eager -> adaptive_graph | 62.492 | 17.568 | 3.557 | [3.487, 3.603] | 3.377-3.767 | 18 x 12 | WIN |
| 1 / 128 / 32 | adaptive_graph -> compiled_triton_full_graph | 17.568 | 11.961 | 1.469 | [1.448, 1.486] | 1.422-1.491 | 18 x 12 | WIN |
| 1 / 128 / 32 | compiled_triton_full_graph -> fp16_compiled_triton_full_graph | 11.961 | 8.244 | 1.451 | [1.429, 1.468] | 1.425-1.494 | 18 x 12 | WIN |
| 1 / 128 / 32 | adaptive_graph -> fp16_compiled_triton_full_graph | 17.568 | 8.244 | 2.131 | [2.096, 2.151] | 2.105-2.161 | 18 x 12 | WIN |
| 1 / 128 / 64 | prefix_sdpa_eager -> adaptive_graph | 125.720 | 34.286 | 3.667 | [3.569, 3.763] | 3.503-3.923 | 18 x 12 | WIN |
| 1 / 128 / 64 | adaptive_graph -> compiled_triton_full_graph | 34.286 | 22.648 | 1.514 | [1.495, 1.535] | 1.497-1.535 | 18 x 12 | WIN |
| 1 / 128 / 64 | compiled_triton_full_graph -> fp16_compiled_triton_full_graph | 22.648 | 15.035 | 1.506 | [1.486, 1.532] | 1.486-1.554 | 18 x 12 | WIN |
| 1 / 128 / 64 | adaptive_graph -> fp16_compiled_triton_full_graph | 34.286 | 15.035 | 2.280 | [2.251, 2.326] | 2.253-2.346 | 18 x 12 | WIN |
| 1 / 192 / 32 | prefix_sdpa_eager -> adaptive_graph | 61.790 | 19.125 | 3.231 | [3.161, 3.299] | 3.105-3.387 | 18 x 12 | WIN |
| 1 / 192 / 32 | adaptive_graph -> compiled_triton_full_graph | 19.125 | 12.088 | 1.582 | [1.560, 1.597] | 1.552-1.608 | 18 x 12 | WIN |
| 1 / 192 / 32 | compiled_triton_full_graph -> fp16_compiled_triton_full_graph | 12.088 | 8.284 | 1.459 | [1.440, 1.485] | 1.377-1.493 | 18 x 12 | WIN |
| 1 / 192 / 32 | adaptive_graph -> fp16_compiled_triton_full_graph | 19.125 | 8.284 | 2.309 | [2.281, 2.348] | 2.214-2.356 | 18 x 12 | WIN |
| 16 / 8 / 16 | prefix_sdpa_eager -> adaptive_graph | 32.265 | 12.473 | 2.587 | [2.525, 2.637] | 2.468-2.705 | 18 x 12 | WIN |
| 16 / 8 / 16 | adaptive_graph -> compiled_triton_full_graph | 12.473 | 7.384 | 1.689 | [1.666, 1.717] | 1.655-1.722 | 18 x 12 | WIN |
| 16 / 8 / 16 | compiled_triton_full_graph -> fp16_compiled_triton_full_graph | 7.384 | 4.972 | 1.485 | [1.454, 1.510] | 1.422-1.521 | 18 x 12 | WIN |
| 16 / 8 / 16 | adaptive_graph -> fp16_compiled_triton_full_graph | 12.473 | 4.972 | 2.509 | [2.461, 2.555] | 2.410-2.565 | 18 x 12 | WIN |
| 16 / 64 / 16 | prefix_sdpa_eager -> adaptive_graph | 31.572 | 12.743 | 2.478 | [2.430, 2.505] | 2.315-2.560 | 18 x 12 | WIN |
| 16 / 64 / 16 | adaptive_graph -> compiled_triton_full_graph | 12.743 | 8.655 | 1.472 | [1.462, 1.479] | 1.426-1.486 | 18 x 12 | WIN |
| 16 / 64 / 16 | compiled_triton_full_graph -> fp16_compiled_triton_full_graph | 8.655 | 5.683 | 1.523 | [1.493, 1.537] | 1.514-1.582 | 18 x 12 | WIN |
| 16 / 64 / 16 | adaptive_graph -> fp16_compiled_triton_full_graph | 12.743 | 5.683 | 2.242 | [2.195, 2.264] | 2.226-2.350 | 18 x 12 | WIN |
| 16 / 64 / 64 | prefix_sdpa_eager -> adaptive_graph | 126.972 | 47.855 | 2.653 | [2.603, 2.693] | 2.488-2.794 | 18 x 12 | WIN |
| 16 / 64 / 64 | adaptive_graph -> compiled_triton_full_graph | 47.855 | 30.289 | 1.580 | [1.564, 1.591] | 1.563-1.587 | 18 x 12 | WIN |
| 16 / 64 / 64 | compiled_triton_full_graph -> fp16_compiled_triton_full_graph | 30.289 | 18.537 | 1.634 | [1.618, 1.647] | 1.632-1.667 | 18 x 12 | WIN |
| 16 / 64 / 64 | adaptive_graph -> fp16_compiled_triton_full_graph | 47.855 | 18.537 | 2.582 | [2.555, 2.602] | 2.580-2.617 | 18 x 12 | WIN |
| 16 / 128 / 32 | prefix_sdpa_eager -> adaptive_graph | 63.441 | 31.788 | 1.996 | [1.943, 2.019] | 1.879-2.056 | 18 x 12 | WIN |
| 16 / 128 / 32 | adaptive_graph -> compiled_triton_full_graph | 31.788 | 20.221 | 1.572 | [1.566, 1.599] | 1.551-1.578 | 18 x 12 | WIN |
| 16 / 128 / 32 | compiled_triton_full_graph -> fp16_compiled_triton_full_graph | 20.221 | 11.574 | 1.747 | [1.734, 1.754] | 1.735-1.772 | 18 x 12 | WIN |
| 16 / 128 / 32 | adaptive_graph -> fp16_compiled_triton_full_graph | 31.788 | 11.574 | 2.746 | [2.736, 2.794] | 2.721-2.774 | 18 x 12 | WIN |
| 16 / 128 / 64 | prefix_sdpa_eager -> adaptive_graph | 128.615 | 61.058 | 2.106 | [2.031, 2.161] | 2.000-2.317 | 18 x 12 | WIN |
| 16 / 128 / 64 | adaptive_graph -> compiled_triton_full_graph | 61.058 | 38.589 | 1.582 | [1.552, 1.614] | 1.574-1.600 | 18 x 12 | WIN |
| 16 / 128 / 64 | compiled_triton_full_graph -> fp16_compiled_triton_full_graph | 38.589 | 22.396 | 1.723 | [1.711, 1.786] | 1.732-1.770 | 18 x 12 | WIN |
| 16 / 128 / 64 | adaptive_graph -> fp16_compiled_triton_full_graph | 61.058 | 22.396 | 2.726 | [2.700, 2.815] | 2.748-2.803 | 18 x 12 | WIN |
| 16 / 192 / 32 | prefix_sdpa_eager -> adaptive_graph | 66.214 | 41.337 | 1.602 | [1.581, 1.623] | 1.545-1.820 | 18 x 12 | WIN |
| 16 / 192 / 32 | adaptive_graph -> compiled_triton_full_graph | 41.337 | 27.013 | 1.530 | [1.525, 1.534] | 1.507-1.567 | 18 x 12 | WIN |
| 16 / 192 / 32 | compiled_triton_full_graph -> fp16_compiled_triton_full_graph | 27.013 | 14.728 | 1.834 | [1.828, 1.842] | 1.772-1.890 | 18 x 12 | WIN |
| 16 / 192 / 32 | adaptive_graph -> fp16_compiled_triton_full_graph | 41.337 | 14.728 | 2.807 | [2.795, 2.817] | 2.768-2.874 | 18 x 12 | WIN |

## Verdict counts (full denominator)

- adaptive_graph -> compiled_triton_full_graph: WIN x12
- adaptive_graph -> fp16_compiled_triton_full_graph: WIN x12
- compiled_triton_full_graph -> fp16_compiled_triton_full_graph: WIN x12
- prefix_sdpa_eager -> adaptive_graph: WIN x12
