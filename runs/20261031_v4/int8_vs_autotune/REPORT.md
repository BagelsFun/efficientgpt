# R1 end-to-end: Qwen3-1.7B batch-1 decode on RTX 4060 (analysis)

| Mode | Rounds | Decode tok/s (median of round medians) | vs HF bf16 + static cache [95% CI over rounds] | paired per-round min-max | TTFT ms | Serving peak MiB (torch; Ollama: size_vram) | Device used MiB | Weights MiB | Load peak MiB |
|---|---|---|---|---|---|---|---|---|---|
| bf16_static | 3 | 62.7 | 1.000 [0.940, 1.064] | 1.000-1.000 | 44 | 3346 | 3442 | 3282.3 | 3282.3 |
| bf16_static_at | 3 | 63.1 | 1.006 [0.946, 1.070] | 1.005-1.007 | 44 | 3352 | 3470 | 3282.3 | 3282.3 |
| int8_static | 3 | 97.0 | 1.548 [1.395, 1.648] | 1.484-1.550 | 65 | 2358 | 2490 | 2237.8 | 3426.3 |
| torchao_int8_static_at | 3 | 88.5 | 1.412 [1.384, 1.516] | 1.384-1.501 | 66 | 2875 | 3438 | 2238.5 | 4911.7 |

| EfficientGPT int8 vs | ratio of round medians [95% CI, round bootstrap] | paired per-round min-max |
|---|---|---|
| bf16_static | 1.548 [1.395, 1.648] | 1.484-1.550 |
| bf16_static_at | 1.539 [1.387, 1.637] | 1.475-1.540 |
| torchao_int8_static_at | 1.096 [0.979, 1.119] | 0.989-1.118 |

| Quality vs bf16_static | ΔNLL [95% CI] (relative) | greedy common prefix, mean tokens of 64 | exact 64-token matches | quality v1 |
|---|---|---|---|---|
