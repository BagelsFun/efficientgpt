# R1 end-to-end: Qwen3-1.7B batch-1 decode on NVIDIA L4 (analysis)

| Mode | Rounds | Decode tok/s (median of round medians) | vs HF bf16 + static cache [95% CI over rounds] | paired per-round min-max | TTFT ms | Serving peak MiB (torch; Ollama: size_vram) | Device used MiB | Weights MiB | Load peak MiB |
|---|---|---|---|---|---|---|---|---|---|
| bf16_static | 4 | 59.9 | 1.000 [0.993, 1.007] | 1.000-1.000 | 47 | 3376 | 3496 | 3282.3 | 3282.3 |
| int8_graph | 4 | 106.4 | 1.777 [1.767, 1.797] | 1.774-1.789 | 61 | 2367 | 2502 | 2237.8 | 3426.3 |

| EfficientGPT int8_graph vs | ratio of round medians [95% CI, round bootstrap] | paired per-round min-max |
|---|---|---|
| bf16_static | 1.777 [1.767, 1.797] | 1.774-1.789 |

| Quality vs bf16_static | ΔNLL [95% CI] (relative) | greedy common prefix, mean tokens of 64 | exact 64-token matches | quality v1 |
|---|---|---|---|---|

| Mode | inter-token latency p50 / p95 ms | end-to-end request s (256 tok) | peak reserved MiB | load s | first call s (compile/capture) |
|---|---|---|---|---|---|
| bf16_static | 16.56 / 16.96 | 4.31 | 3424.0 | 6.49 | 35.85 |
| int8_graph | 9.40 / 9.46 | 2.46 | 2500.0 | 6.88 | 11.34 |
