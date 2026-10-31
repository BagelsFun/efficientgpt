# R1 end-to-end: Qwen3-1.7B batch-1 decode on RTX 4060 (analysis)

| Mode | Rounds | Decode tok/s (median of round medians) | vs HF bf16 + static cache [95% CI over rounds] | paired per-round min-max | TTFT ms | Serving peak MiB (torch; Ollama: size_vram) | Device used MiB | Weights MiB | Load peak MiB |
|---|---|---|---|---|---|---|---|---|---|
| bf16_static | 4 | 63.2 | 1.000 [0.994, 1.006] | 1.000-1.000 | 44 | 3346 | 3438 | 3282.3 | 3282.3 |
| int8_graph | 4 | 117.3 | 1.858 [1.854, 1.869] | 1.856-1.866 | 53 | 2342 | 2474 | 2237.8 | 3426.3 |
| torchao_int8_static_at | 4 | 90.5 | 1.433 [1.411, 1.443] | 1.412-1.441 | 66 | 2876 | 3430 | 2238.5 | 4911.7 |
| llamacpp_q8 | 4 | 116.0 | 1.837 [1.834, 1.849] | 1.836-1.846 | 32 | 2300 | 2423 | None | None |

| EfficientGPT int8_graph vs | ratio of round medians [95% CI, round bootstrap] | paired per-round min-max |
|---|---|---|
| bf16_static | 1.858 [1.854, 1.869] | 1.856-1.866 |
| torchao_int8_static_at | 1.297 [1.291, 1.316] | 1.291-1.317 |
| llamacpp_q8 | 1.011 [1.009, 1.013] | 1.011-1.011 |

| Quality vs bf16_static | ΔNLL [95% CI] (relative) | greedy common prefix, mean tokens of 64 | exact 64-token matches | quality v1 |
|---|---|---|---|---|

| Mode | inter-token latency p50 / p95 ms | end-to-end request s (256 tok) | peak reserved MiB | load s | first call s (compile/capture) |
|---|---|---|---|---|---|
| bf16_static | 15.72 / 16.37 | 4.08 | 3378.0 | 4.93 | 5.5 |
| int8_graph | 8.49 / 8.55 | 2.23 | 2424.0 | 6.61 | 5.87 |
| torchao_int8_static_at | 10.85 / 12.02 | 2.88 | 3360.0 | 5.11 | 32.79 |
| llamacpp_q8 | 8.57 / 8.93 | 2.23 | - | - | 7.21 |
