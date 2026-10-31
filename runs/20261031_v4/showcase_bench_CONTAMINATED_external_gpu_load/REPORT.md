# R1 end-to-end: Qwen3-1.7B batch-1 decode on RTX 4060 (analysis)

| Mode | Rounds | Decode tok/s (median of round medians) | vs HF bf16 + static cache [95% CI over rounds] | paired per-round min-max | TTFT ms | Serving peak MiB (torch; Ollama: size_vram) | Device used MiB | Weights MiB | Load peak MiB |
|---|---|---|---|---|---|---|---|---|---|
| bf16_static | 4 | 44.1 | 1.000 [0.994, 1.006] | 1.000-1.000 | 65 | 3346 | 3424 | 3282.3 | 3282.3 |
| int8_graph | 4 | 86.6 | 1.963 [1.441, 2.499] | 1.444-2.505 | 67 | 2342 | 2474 | 2237.8 | 3426.3 |
| torchao_int8_static_at | 4 | 60.3 | 1.368 [1.351, 1.388] | 1.351-1.392 | 93 | 2876 | 3438 | 2238.5 | 4911.7 |
| llamacpp_q8 | 4 | 82.0 | 1.860 [1.846, 2.294] | 1.841-2.300 | 33 | 2300 | 2695 | None | None |

| EfficientGPT int8_graph vs | ratio of round medians [95% CI, round bootstrap] | paired per-round min-max |
|---|---|---|
| bf16_static | 1.963 [1.441, 2.499] | 1.444-2.505 |
| torchao_int8_static_at | 1.435 [1.054, 1.827] | 1.046-1.800 |
| llamacpp_q8 | 1.056 [0.773, 1.344] | 0.777-1.089 |

| Quality vs bf16_static | ΔNLL [95% CI] (relative) | greedy common prefix, mean tokens of 64 | exact 64-token matches | quality v1 |
|---|---|---|---|---|

| Mode | inter-token latency p50 / p95 ms | end-to-end request s (256 tok) | peak reserved MiB | load s | first call s (compile/capture) |
|---|---|---|---|---|---|
| bf16_static | 21.61 / 26.38 | 5.85 | 3378.0 | 6.3 | 6.99 |
| int8_graph | 12.48 / 13.13 | 3.01 | 2424.0 | 7.05 | 6.36 |
| torchao_int8_static_at | 16.62 / 17.83 | 4.32 | 3360.0 | 6.2 | 25.84 |
| llamacpp_q8 | 12.99 / 14.30 | 3.15 | - | - | 3.93 |
