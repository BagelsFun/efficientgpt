# R1 end-to-end: Qwen3-1.7B batch-1 decode on RTX 4060 (analysis)

| Mode | Rounds | Decode tok/s (median of round medians) | vs HF bf16 + static cache [95% CI over rounds] | paired per-round min-max | TTFT ms | Serving peak MiB (torch; Ollama: size_vram) | Device used MiB | Weights MiB | Load peak MiB |
|---|---|---|---|---|---|---|---|---|---|
| bf16_static | 1 | 59.7 | 1.000 [1.000, 1.000] | 1.000-1.000 | 44 | 3350 | 3440 | 3282.3 | 3282.3 |
| int8_graph | 1 | 109.0 | 1.826 [1.826, 1.826] | 1.826-1.826 | 55 | 2342 | 2474 | 2237.8 | 3426.3 |
| llamacpp_q8 | 1 | 110.5 | 1.850 [1.850, 1.850] | 1.850-1.850 | 29 | 2300 | 2388 | None | None |

| EfficientGPT int8_graph vs | ratio of round medians [95% CI, round bootstrap] | paired per-round min-max |
|---|---|---|
| bf16_static | 1.826 [1.826, 1.826] | 1.826-1.826 |
| llamacpp_q8 | 0.987 [0.987, 0.987] | 0.987-0.987 |

| Quality vs bf16_static | ΔNLL [95% CI] (relative) | greedy common prefix, mean tokens of 64 | exact 64-token matches | quality v1 |
|---|---|---|---|---|

| Mode | inter-token latency p50 / p95 ms | end-to-end request s (256 tok) | peak reserved MiB | load s | first call s (compile/capture) |
|---|---|---|---|---|---|
| bf16_static | 16.93 / 17.48 | 4.31 | 3402.0 | 4.72 | 18.65 |
| int8_graph | 9.17 / 9.41 | 2.39 | 2424.0 | 5.35 | 5.59 |
| llamacpp_q8 | 9.06 / 9.61 | 2.34 | - | - | 2.36 |
