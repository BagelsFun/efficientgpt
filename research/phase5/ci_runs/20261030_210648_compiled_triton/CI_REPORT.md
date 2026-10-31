# Phase 5 in-process interleaved latency ratios

3 processes x 6 blocks x 12 requests per mode and workload; clusters = (process, block). Peak memory not comparable (all modes resident together).

Source: `research\phase5\ci_runs\20261030_210648_compiled_triton\benchmark_raw.csv`; 10000 two-level bootstrap resamples (rounds, then requests); 95% percentile CI.
Ratio = baseline median / candidate median (>1: candidate faster). WIN needs ratio >= 1.05 and CI low > 1. Prompt=192 full-vs-adaptive is the A/A control (identical capacity).

| B / prompt / gen | baseline -> candidate | base ms | cand ms | ratio | 95% CI | per-round range | rounds x n | verdict |
|---|---|---:|---:|---:|---|---|---|---|
| 1 / 8 / 16 | adaptive_graph -> triton_adaptive_graph | 9.718 | 7.906 | 1.229 | [1.173, 1.280] | 1.104-1.324 | 18 x 12 | WIN |
| 1 / 8 / 16 | triton_full_graph -> triton_adaptive_graph | 8.057 | 7.906 | 1.019 | [0.966, 1.082] | 0.895-1.144 | 18 x 12 | INCONCLUSIVE |
| 1 / 8 / 16 | triton_full_graph -> compiled_triton_full_graph | 8.057 | 7.516 | 1.072 | [1.025, 1.126] | 0.990-1.247 | 18 x 12 | WIN |
| 1 / 8 / 16 | triton_adaptive_graph -> compiled_triton_adaptive_graph | 7.906 | 7.223 | 1.095 | [1.030, 1.130] | 0.956-1.178 | 18 x 12 | WIN |
| 1 / 8 / 16 | adaptive_graph -> compiled_triton_full_graph | 9.718 | 7.516 | 1.293 | [1.218, 1.351] | 1.171-1.417 | 18 x 12 | WIN |
| 1 / 8 / 16 | compiled_triton_full_graph -> compiled_triton_adaptive_graph | 7.516 | 7.223 | 1.041 | [0.975, 1.089] | 0.858-1.144 | 18 x 12 | INCONCLUSIVE |
| 1 / 8 / 16 | adaptive_graph -> triton_full_graph | 9.718 | 8.057 | 1.206 | [1.132, 1.263] | 1.034-1.341 | 18 x 12 | WIN |
| 1 / 8 / 16 | adaptive_graph -> compiled_triton_adaptive_graph | 9.718 | 7.223 | 1.345 | [1.248, 1.384] | 1.107-1.450 | 18 x 12 | WIN |
| 1 / 64 / 16 | adaptive_graph -> triton_adaptive_graph | 9.793 | 7.965 | 1.230 | [1.167, 1.261] | 1.143-1.313 | 18 x 12 | WIN |
| 1 / 64 / 16 | triton_full_graph -> triton_adaptive_graph | 8.027 | 7.965 | 1.008 | [0.941, 1.053] | 0.882-1.121 | 18 x 12 | INCONCLUSIVE |
| 1 / 64 / 16 | triton_full_graph -> compiled_triton_full_graph | 8.027 | 7.400 | 1.085 | [1.027, 1.132] | 0.921-1.195 | 18 x 12 | WIN |
| 1 / 64 / 16 | triton_adaptive_graph -> compiled_triton_adaptive_graph | 7.965 | 7.381 | 1.079 | [1.029, 1.154] | 0.975-1.184 | 18 x 12 | WIN |
| 1 / 64 / 16 | adaptive_graph -> compiled_triton_full_graph | 9.793 | 7.400 | 1.323 | [1.257, 1.367] | 1.147-1.405 | 18 x 12 | WIN |
| 1 / 64 / 16 | compiled_triton_full_graph -> compiled_triton_adaptive_graph | 7.400 | 7.381 | 1.003 | [0.957, 1.063] | 0.888-1.123 | 18 x 12 | INCONCLUSIVE |
| 1 / 64 / 16 | adaptive_graph -> triton_full_graph | 9.793 | 8.027 | 1.220 | [1.157, 1.274] | 1.064-1.350 | 18 x 12 | WIN |
| 1 / 64 / 16 | adaptive_graph -> compiled_triton_adaptive_graph | 9.793 | 7.381 | 1.327 | [1.259, 1.380] | 1.158-1.447 | 18 x 12 | WIN |
| 1 / 64 / 64 | adaptive_graph -> triton_adaptive_graph | 33.803 | 26.656 | 1.268 | [1.258, 1.279] | 1.234-1.301 | 18 x 12 | WIN |
| 1 / 64 / 64 | triton_full_graph -> triton_adaptive_graph | 26.710 | 26.656 | 1.002 | [0.992, 1.015] | 0.973-1.036 | 18 x 12 | INCONCLUSIVE |
| 1 / 64 / 64 | triton_full_graph -> compiled_triton_full_graph | 26.710 | 24.233 | 1.102 | [1.089, 1.117] | 1.050-1.126 | 18 x 12 | WIN |
| 1 / 64 / 64 | triton_adaptive_graph -> compiled_triton_adaptive_graph | 26.656 | 24.261 | 1.099 | [1.089, 1.110] | 1.081-1.129 | 18 x 12 | WIN |
| 1 / 64 / 64 | adaptive_graph -> compiled_triton_full_graph | 33.803 | 24.233 | 1.395 | [1.384, 1.402] | 1.367-1.410 | 18 x 12 | WIN |
| 1 / 64 / 64 | compiled_triton_full_graph -> compiled_triton_adaptive_graph | 24.233 | 24.261 | 0.999 | [0.992, 1.009] | 0.976-1.033 | 18 x 12 | INCONCLUSIVE |
| 1 / 64 / 64 | adaptive_graph -> triton_full_graph | 33.803 | 26.710 | 1.266 | [1.250, 1.277] | 1.241-1.302 | 18 x 12 | WIN |
| 1 / 64 / 64 | adaptive_graph -> compiled_triton_adaptive_graph | 33.803 | 24.261 | 1.393 | [1.385, 1.404] | 1.372-1.431 | 18 x 12 | WIN |
| 1 / 128 / 32 | adaptive_graph -> triton_adaptive_graph | 19.439 | 14.502 | 1.340 | [1.331, 1.352] | 1.303-1.369 | 18 x 12 | WIN |
| 1 / 128 / 32 | triton_full_graph -> triton_adaptive_graph | 14.596 | 14.502 | 1.007 | [0.999, 1.016] | 0.975-1.041 | 18 x 12 | INCONCLUSIVE |
| 1 / 128 / 32 | triton_full_graph -> compiled_triton_full_graph | 14.596 | 13.358 | 1.093 | [1.080, 1.104] | 1.063-1.108 | 18 x 12 | WIN |
| 1 / 128 / 32 | triton_adaptive_graph -> compiled_triton_adaptive_graph | 14.502 | 13.345 | 1.087 | [1.073, 1.103] | 1.065-1.124 | 18 x 12 | WIN |
| 1 / 128 / 32 | adaptive_graph -> compiled_triton_full_graph | 19.439 | 13.358 | 1.455 | [1.439, 1.470] | 1.420-1.487 | 18 x 12 | WIN |
| 1 / 128 / 32 | compiled_triton_full_graph -> compiled_triton_adaptive_graph | 13.358 | 13.345 | 1.001 | [0.989, 1.017] | 0.983-1.026 | 18 x 12 | INCONCLUSIVE |
| 1 / 128 / 32 | adaptive_graph -> triton_full_graph | 19.439 | 14.596 | 1.332 | [1.320, 1.342] | 1.286-1.367 | 18 x 12 | WIN |
| 1 / 128 / 32 | adaptive_graph -> compiled_triton_adaptive_graph | 19.439 | 13.345 | 1.457 | [1.443, 1.477] | 1.416-1.481 | 18 x 12 | WIN |
| 1 / 128 / 64 | adaptive_graph -> triton_adaptive_graph | 37.320 | 26.734 | 1.396 | [1.378, 1.411] | 1.344-1.430 | 18 x 12 | WIN |
| 1 / 128 / 64 | triton_full_graph -> triton_adaptive_graph | 26.969 | 26.734 | 1.009 | [0.990, 1.026] | 0.954-1.048 | 18 x 12 | INCONCLUSIVE |
| 1 / 128 / 64 | triton_full_graph -> compiled_triton_full_graph | 26.969 | 24.334 | 1.108 | [1.093, 1.122] | 1.053-1.144 | 18 x 12 | WIN |
| 1 / 128 / 64 | triton_adaptive_graph -> compiled_triton_adaptive_graph | 26.734 | 24.353 | 1.098 | [1.084, 1.113] | 1.066-1.145 | 18 x 12 | WIN |
| 1 / 128 / 64 | adaptive_graph -> compiled_triton_full_graph | 37.320 | 24.334 | 1.534 | [1.513, 1.545] | 1.467-1.563 | 18 x 12 | WIN |
| 1 / 128 / 64 | compiled_triton_full_graph -> compiled_triton_adaptive_graph | 24.334 | 24.353 | 0.999 | [0.986, 1.017] | 0.952-1.039 | 18 x 12 | INCONCLUSIVE |
| 1 / 128 / 64 | adaptive_graph -> triton_full_graph | 37.320 | 26.969 | 1.384 | [1.367, 1.399] | 1.348-1.426 | 18 x 12 | WIN |
| 1 / 128 / 64 | adaptive_graph -> compiled_triton_adaptive_graph | 37.320 | 24.353 | 1.532 | [1.517, 1.545] | 1.471-1.559 | 18 x 12 | WIN |
| 1 / 192 / 32 | adaptive_graph -> triton_adaptive_graph | 20.870 | 14.441 | 1.445 | [1.427, 1.487] | 1.407-1.487 | 18 x 12 | WIN |
| 1 / 192 / 32 | triton_full_graph -> triton_adaptive_graph | 14.512 | 14.441 | 1.005 | [0.979, 1.019] | 0.980-1.023 | 18 x 12 | AA_PASS |
| 1 / 192 / 32 | triton_full_graph -> compiled_triton_full_graph | 14.512 | 13.258 | 1.095 | [1.063, 1.119] | 1.061-1.137 | 18 x 12 | WIN |
| 1 / 192 / 32 | triton_adaptive_graph -> compiled_triton_adaptive_graph | 14.441 | 13.287 | 1.087 | [1.066, 1.109] | 1.068-1.120 | 18 x 12 | WIN |
| 1 / 192 / 32 | adaptive_graph -> compiled_triton_full_graph | 20.870 | 13.258 | 1.574 | [1.545, 1.628] | 1.512-1.626 | 18 x 12 | WIN |
| 1 / 192 / 32 | compiled_triton_full_graph -> compiled_triton_adaptive_graph | 13.258 | 13.287 | 0.998 | [0.975, 1.021] | 0.976-1.049 | 18 x 12 | AA_PASS |
| 1 / 192 / 32 | adaptive_graph -> triton_full_graph | 20.870 | 14.512 | 1.438 | [1.421, 1.488] | 1.405-1.502 | 18 x 12 | WIN |
| 1 / 192 / 32 | adaptive_graph -> compiled_triton_adaptive_graph | 20.870 | 13.287 | 1.571 | [1.550, 1.611] | 1.534-1.630 | 18 x 12 | WIN |
| 16 / 8 / 16 | adaptive_graph -> triton_adaptive_graph | 13.726 | 8.821 | 1.556 | [1.514, 1.594] | 1.484-1.672 | 18 x 12 | WIN |
| 16 / 8 / 16 | triton_full_graph -> triton_adaptive_graph | 8.758 | 8.821 | 0.993 | [0.963, 1.025] | 0.942-1.055 | 18 x 12 | INCONCLUSIVE |
| 16 / 8 / 16 | triton_full_graph -> compiled_triton_full_graph | 8.758 | 7.902 | 1.108 | [1.066, 1.141] | 1.047-1.190 | 18 x 12 | WIN |
| 16 / 8 / 16 | triton_adaptive_graph -> compiled_triton_adaptive_graph | 8.821 | 8.007 | 1.102 | [1.066, 1.140] | 1.008-1.167 | 18 x 12 | WIN |
| 16 / 8 / 16 | adaptive_graph -> compiled_triton_full_graph | 13.726 | 7.902 | 1.737 | [1.669, 1.780] | 1.600-1.833 | 18 x 12 | WIN |
| 16 / 8 / 16 | compiled_triton_full_graph -> compiled_triton_adaptive_graph | 7.902 | 8.007 | 0.987 | [0.955, 1.030] | 0.890-1.071 | 18 x 12 | INCONCLUSIVE |
| 16 / 8 / 16 | adaptive_graph -> triton_full_graph | 13.726 | 8.758 | 1.567 | [1.526, 1.598] | 1.454-1.613 | 18 x 12 | WIN |
| 16 / 8 / 16 | adaptive_graph -> compiled_triton_adaptive_graph | 13.726 | 8.007 | 1.714 | [1.669, 1.754] | 1.623-1.772 | 18 x 12 | WIN |
| 16 / 64 / 16 | adaptive_graph -> triton_adaptive_graph | 14.399 | 10.742 | 1.340 | [1.326, 1.364] | 1.321-1.478 | 18 x 12 | WIN |
| 16 / 64 / 16 | triton_full_graph -> triton_adaptive_graph | 10.707 | 10.742 | 0.997 | [0.976, 1.021] | 0.902-1.123 | 18 x 12 | INCONCLUSIVE |
| 16 / 64 / 16 | triton_full_graph -> compiled_triton_full_graph | 10.707 | 9.848 | 1.087 | [1.055, 1.145] | 0.975-1.155 | 18 x 12 | WIN |
| 16 / 64 / 16 | triton_adaptive_graph -> compiled_triton_adaptive_graph | 10.742 | 9.847 | 1.091 | [1.064, 1.131] | 1.002-1.194 | 18 x 12 | WIN |
| 16 / 64 / 16 | adaptive_graph -> compiled_triton_full_graph | 14.399 | 9.848 | 1.462 | [1.426, 1.542] | 1.398-1.580 | 18 x 12 | WIN |
| 16 / 64 / 16 | compiled_triton_full_graph -> compiled_triton_adaptive_graph | 9.848 | 9.847 | 1.000 | [0.947, 1.043] | 0.916-1.115 | 18 x 12 | INCONCLUSIVE |
| 16 / 64 / 16 | adaptive_graph -> triton_full_graph | 14.399 | 10.707 | 1.345 | [1.322, 1.373] | 1.303-1.478 | 18 x 12 | WIN |
| 16 / 64 / 16 | adaptive_graph -> compiled_triton_adaptive_graph | 14.399 | 9.847 | 1.462 | [1.437, 1.514] | 1.391-1.590 | 18 x 12 | WIN |
| 16 / 64 / 64 | adaptive_graph -> triton_adaptive_graph | 53.335 | 36.966 | 1.443 | [1.431, 1.452] | 1.431-1.475 | 18 x 12 | WIN |
| 16 / 64 / 64 | triton_full_graph -> triton_adaptive_graph | 37.027 | 36.966 | 1.002 | [0.994, 1.010] | 0.957-1.040 | 18 x 12 | INCONCLUSIVE |
| 16 / 64 / 64 | triton_full_graph -> compiled_triton_full_graph | 37.027 | 33.693 | 1.099 | [1.093, 1.109] | 1.057-1.133 | 18 x 12 | WIN |
| 16 / 64 / 64 | triton_adaptive_graph -> compiled_triton_adaptive_graph | 36.966 | 33.679 | 1.098 | [1.091, 1.106] | 1.063-1.121 | 18 x 12 | WIN |
| 16 / 64 / 64 | adaptive_graph -> compiled_triton_full_graph | 53.335 | 33.693 | 1.583 | [1.575, 1.595] | 1.563-1.608 | 18 x 12 | WIN |
| 16 / 64 / 64 | compiled_triton_full_graph -> compiled_triton_adaptive_graph | 33.693 | 33.679 | 1.000 | [0.993, 1.006] | 0.985-1.015 | 18 x 12 | INCONCLUSIVE |
| 16 / 64 / 64 | adaptive_graph -> triton_full_graph | 53.335 | 37.027 | 1.440 | [1.428, 1.451] | 1.409-1.503 | 18 x 12 | WIN |
| 16 / 64 / 64 | adaptive_graph -> compiled_triton_adaptive_graph | 53.335 | 33.679 | 1.584 | [1.572, 1.595] | 1.558-1.615 | 18 x 12 | WIN |
| 16 / 128 / 32 | adaptive_graph -> triton_adaptive_graph | 35.298 | 24.020 | 1.470 | [1.453, 1.487] | 1.432-1.489 | 18 x 12 | WIN |
| 16 / 128 / 32 | triton_full_graph -> triton_adaptive_graph | 24.231 | 24.020 | 1.009 | [0.994, 1.024] | 0.969-1.049 | 18 x 12 | INCONCLUSIVE |
| 16 / 128 / 32 | triton_full_graph -> compiled_triton_full_graph | 24.231 | 22.507 | 1.077 | [1.061, 1.093] | 1.042-1.117 | 18 x 12 | WIN |
| 16 / 128 / 32 | triton_adaptive_graph -> compiled_triton_adaptive_graph | 24.020 | 22.338 | 1.075 | [1.063, 1.090] | 1.026-1.106 | 18 x 12 | WIN |
| 16 / 128 / 32 | adaptive_graph -> compiled_triton_full_graph | 35.298 | 22.507 | 1.568 | [1.553, 1.586] | 1.528-1.591 | 18 x 12 | WIN |
| 16 / 128 / 32 | compiled_triton_full_graph -> compiled_triton_adaptive_graph | 22.507 | 22.338 | 1.008 | [0.995, 1.021] | 0.957-1.040 | 18 x 12 | INCONCLUSIVE |
| 16 / 128 / 32 | adaptive_graph -> triton_full_graph | 35.298 | 24.231 | 1.457 | [1.441, 1.475] | 1.410-1.487 | 18 x 12 | WIN |
| 16 / 128 / 32 | adaptive_graph -> compiled_triton_adaptive_graph | 35.298 | 22.338 | 1.580 | [1.567, 1.596] | 1.511-1.606 | 18 x 12 | WIN |
| 16 / 128 / 64 | adaptive_graph -> triton_adaptive_graph | 66.789 | 45.423 | 1.470 | [1.459, 1.481] | 1.455-1.481 | 18 x 12 | WIN |
| 16 / 128 / 64 | triton_full_graph -> triton_adaptive_graph | 45.375 | 45.423 | 0.999 | [0.992, 1.009] | 0.986-1.021 | 18 x 12 | INCONCLUSIVE |
| 16 / 128 / 64 | triton_full_graph -> compiled_triton_full_graph | 45.375 | 41.738 | 1.087 | [1.077, 1.097] | 1.072-1.106 | 18 x 12 | WIN |
| 16 / 128 / 64 | triton_adaptive_graph -> compiled_triton_adaptive_graph | 45.423 | 41.748 | 1.088 | [1.078, 1.095] | 1.073-1.102 | 18 x 12 | WIN |
| 16 / 128 / 64 | adaptive_graph -> compiled_triton_full_graph | 66.789 | 41.738 | 1.600 | [1.581, 1.610] | 1.576-1.623 | 18 x 12 | WIN |
| 16 / 128 / 64 | compiled_triton_full_graph -> compiled_triton_adaptive_graph | 41.738 | 41.748 | 1.000 | [0.992, 1.010] | 0.991-1.018 | 18 x 12 | INCONCLUSIVE |
| 16 / 128 / 64 | adaptive_graph -> triton_full_graph | 66.789 | 45.375 | 1.472 | [1.458, 1.481] | 1.444-1.480 | 18 x 12 | WIN |
| 16 / 128 / 64 | adaptive_graph -> compiled_triton_adaptive_graph | 66.789 | 41.748 | 1.600 | [1.583, 1.610] | 1.574-1.628 | 18 x 12 | WIN |
| 16 / 192 / 32 | adaptive_graph -> triton_adaptive_graph | 43.795 | 30.444 | 1.439 | [1.420, 1.457] | 1.412-1.474 | 18 x 12 | WIN |
| 16 / 192 / 32 | triton_full_graph -> triton_adaptive_graph | 30.264 | 30.444 | 0.994 | [0.986, 1.008] | 0.979-1.017 | 18 x 12 | AA_PASS |
| 16 / 192 / 32 | triton_full_graph -> compiled_triton_full_graph | 30.264 | 28.566 | 1.059 | [1.051, 1.083] | 1.040-1.099 | 18 x 12 | WIN |
| 16 / 192 / 32 | triton_adaptive_graph -> compiled_triton_adaptive_graph | 30.444 | 28.530 | 1.067 | [1.056, 1.082] | 1.049-1.093 | 18 x 12 | WIN |
| 16 / 192 / 32 | adaptive_graph -> compiled_triton_full_graph | 43.795 | 28.566 | 1.533 | [1.516, 1.554] | 1.496-1.590 | 18 x 12 | WIN |
| 16 / 192 / 32 | compiled_triton_full_graph -> compiled_triton_adaptive_graph | 28.566 | 28.530 | 1.001 | [0.989, 1.012] | 0.963-1.022 | 18 x 12 | AA_PASS |
| 16 / 192 / 32 | adaptive_graph -> triton_full_graph | 43.795 | 30.264 | 1.447 | [1.424, 1.456] | 1.415-1.481 | 18 x 12 | WIN |
| 16 / 192 / 32 | adaptive_graph -> compiled_triton_adaptive_graph | 43.795 | 28.530 | 1.535 | [1.519, 1.552] | 1.500-1.579 | 18 x 12 | WIN |

## Verdict counts (full denominator)

- adaptive_graph -> compiled_triton_adaptive_graph: WIN x12
- adaptive_graph -> compiled_triton_full_graph: WIN x12
- adaptive_graph -> triton_adaptive_graph: WIN x12
- adaptive_graph -> triton_full_graph: WIN x12
- compiled_triton_full_graph -> compiled_triton_adaptive_graph: AA_PASS x2
- compiled_triton_full_graph -> compiled_triton_adaptive_graph: INCONCLUSIVE x10
- triton_adaptive_graph -> compiled_triton_adaptive_graph: WIN x12
- triton_full_graph -> compiled_triton_full_graph: WIN x12
- triton_full_graph -> triton_adaptive_graph: AA_PASS x2
- triton_full_graph -> triton_adaptive_graph: INCONCLUSIVE x10
