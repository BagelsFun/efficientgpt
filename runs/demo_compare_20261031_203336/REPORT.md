# R1 end-to-end: Qwen3-1.7B batch-1 decode on RTX 4060 (analysis)

| Mode | Rounds | Decode tok/s (median of round medians) | vs HF bf16 + static cache [95% CI over rounds] | paired per-round min-max | TTFT ms | Serving peak MiB (torch; Ollama: size_vram) | Device used MiB | Weights MiB | Load peak MiB |
|---|---|---|---|---|---|---|---|---|---|
| bf16_static | 1 | 59.6 | 1.000 [1.000, 1.000] | 1.000-1.000 | 45 | 3346 | 3416 | 3282.3 | 3282.3 |
| int8_graph | 1 | 111.2 | 1.864 [1.864, 1.864] | 1.864-1.864 | 54 | 2342 | 2474 | 2237.8 | 3426.3 |
| llamacpp_q8 | 1 | 109.8 | 1.842 [1.842, 1.842] | 1.842-1.842 | 38 | 2300 | 2422 | None | None |

| EfficientGPT int8_graph vs | ratio of round medians [95% CI, round bootstrap] | paired per-round min-max |
|---|---|---|
| bf16_static | 1.864 [1.864, 1.864] | 1.864-1.864 |
| llamacpp_q8 | 1.012 [1.012, 1.012] | 1.012-1.012 |

| Quality vs bf16_static | ΔNLL [95% CI] (relative) | greedy common prefix, mean tokens of 64 | exact 64-token matches | quality v1 |
|---|---|---|---|---|

| Mode | inter-token latency p50 / p95 ms | end-to-end request s (256 tok) | peak reserved MiB | load s | first call s (compile/capture) |
|---|---|---|---|---|---|
| bf16_static | 16.97 / 17.48 | 4.32 | 3376.0 | 5.06 | 5.68 |
| int8_graph | 8.93 / 9.35 | 2.35 | 2424.0 | 5.23 | 5.47 |
| llamacpp_q8 | 9.10 / 9.65 | 2.36 | - | - | 6.97 |
