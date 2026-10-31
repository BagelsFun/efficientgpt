# Phase 5 in-process interleaved latency ratios

3 processes x 6 blocks x 12 requests per mode and workload; clusters = (process, block). Peak memory not comparable (all modes resident together).

Source: `research\phase5\ci_runs\20261031_dispatch_inprocess\benchmark_raw.csv`; 10000 two-level bootstrap resamples (rounds, then requests); 95% percentile CI.
Ratio = baseline median / candidate median (>1: candidate faster). WIN needs ratio >= 1.05 and CI low > 1. Prompt=192 full-vs-adaptive is the A/A control (identical capacity).

| B / prompt / gen | baseline -> candidate | base ms | cand ms | ratio | 95% CI | per-round range | rounds x n | verdict |
|---|---|---:|---:|---:|---|---|---|---|
| 1 / 8 / 16 | adaptive_graph -> triton_adaptive_graph | 9.080 | 7.582 | 1.198 | [1.181, 1.209] | 1.148-1.229 | 18 x 12 | WIN |
| 1 / 8 / 16 | triton_adaptive_graph -> dispatch_triton_adaptive_graph | 7.582 | 7.558 | 1.003 | [0.993, 1.016] | 0.971-1.026 | 18 x 12 | INCONCLUSIVE |
| 1 / 64 / 16 | adaptive_graph -> triton_adaptive_graph | 8.943 | 7.489 | 1.194 | [1.184, 1.206] | 1.161-1.226 | 18 x 12 | WIN |
| 1 / 64 / 16 | triton_adaptive_graph -> dispatch_triton_adaptive_graph | 7.489 | 7.488 | 1.000 | [0.993, 1.017] | 0.985-1.029 | 18 x 12 | INCONCLUSIVE |
| 1 / 64 / 64 | adaptive_graph -> triton_adaptive_graph | 32.082 | 25.253 | 1.270 | [1.268, 1.273] | 1.238-1.287 | 18 x 12 | WIN |
| 1 / 64 / 64 | triton_adaptive_graph -> dispatch_triton_adaptive_graph | 25.253 | 25.255 | 1.000 | [0.997, 1.002] | 0.986-1.009 | 18 x 12 | INCONCLUSIVE |
| 1 / 128 / 32 | adaptive_graph -> triton_adaptive_graph | 18.373 | 13.542 | 1.357 | [1.350, 1.364] | 1.321-1.377 | 18 x 12 | WIN |
| 1 / 128 / 32 | triton_adaptive_graph -> dispatch_triton_adaptive_graph | 13.542 | 13.507 | 1.003 | [0.997, 1.010] | 0.982-1.031 | 18 x 12 | INCONCLUSIVE |
| 1 / 128 / 64 | adaptive_graph -> triton_adaptive_graph | 35.274 | 25.346 | 1.392 | [1.387, 1.395] | 1.371-1.400 | 18 x 12 | WIN |
| 1 / 128 / 64 | triton_adaptive_graph -> dispatch_triton_adaptive_graph | 25.346 | 25.340 | 1.000 | [0.998, 1.003] | 0.985-1.020 | 18 x 12 | INCONCLUSIVE |
| 1 / 192 / 32 | adaptive_graph -> triton_adaptive_graph | 19.903 | 13.587 | 1.465 | [1.461, 1.473] | 1.454-1.478 | 18 x 12 | WIN |
| 1 / 192 / 32 | triton_adaptive_graph -> dispatch_triton_adaptive_graph | 13.587 | 13.549 | 1.003 | [0.997, 1.007] | 0.988-1.014 | 18 x 12 | INCONCLUSIVE |
| 16 / 8 / 16 | adaptive_graph -> triton_adaptive_graph | 12.956 | 8.352 | 1.551 | [1.532, 1.562] | 1.492-1.573 | 18 x 12 | WIN |
| 16 / 8 / 16 | triton_adaptive_graph -> dispatch_triton_adaptive_graph | 8.352 | 8.230 | 1.015 | [1.005, 1.027] | 0.961-1.066 | 18 x 12 | INCONCLUSIVE |
| 16 / 64 / 16 | adaptive_graph -> triton_adaptive_graph | 13.519 | 9.877 | 1.369 | [1.353, 1.379] | 1.350-1.405 | 18 x 12 | WIN |
| 16 / 64 / 16 | triton_adaptive_graph -> dispatch_triton_adaptive_graph | 9.877 | 9.888 | 0.999 | [0.989, 1.010] | 0.991-1.015 | 18 x 12 | INCONCLUSIVE |
| 16 / 64 / 64 | adaptive_graph -> triton_adaptive_graph | 50.776 | 35.456 | 1.432 | [1.430, 1.435] | 1.415-1.457 | 18 x 12 | WIN |
| 16 / 64 / 64 | triton_adaptive_graph -> dispatch_triton_adaptive_graph | 35.456 | 35.047 | 1.012 | [1.010, 1.017] | 0.998-1.028 | 18 x 12 | INCONCLUSIVE |
| 16 / 128 / 32 | adaptive_graph -> triton_adaptive_graph | 33.709 | 23.097 | 1.459 | [1.453, 1.469] | 1.422-1.500 | 18 x 12 | WIN |
| 16 / 128 / 32 | triton_adaptive_graph -> dispatch_triton_adaptive_graph | 23.097 | 22.881 | 1.009 | [1.002, 1.015] | 0.999-1.038 | 18 x 12 | INCONCLUSIVE |
| 16 / 128 / 64 | adaptive_graph -> triton_adaptive_graph | 63.325 | 43.101 | 1.469 | [1.467, 1.472] | 1.428-1.485 | 18 x 12 | WIN |
| 16 / 128 / 64 | triton_adaptive_graph -> dispatch_triton_adaptive_graph | 43.101 | 42.619 | 1.011 | [1.009, 1.013] | 1.006-1.025 | 18 x 12 | INCONCLUSIVE |
| 16 / 192 / 32 | adaptive_graph -> triton_adaptive_graph | 41.432 | 28.597 | 1.449 | [1.442, 1.456] | 1.424-1.476 | 18 x 12 | WIN |
| 16 / 192 / 32 | triton_adaptive_graph -> dispatch_triton_adaptive_graph | 28.597 | 28.488 | 1.004 | [0.998, 1.010] | 0.983-1.019 | 18 x 12 | INCONCLUSIVE |

## Verdict counts (full denominator)

- adaptive_graph -> triton_adaptive_graph: WIN x12
- triton_adaptive_graph -> dispatch_triton_adaptive_graph: INCONCLUSIVE x12
