# R1 end-to-end: Qwen3-1.7B batch-1 decode on RTX 4060 (analysis)

| Mode | Rounds | Decode tok/s (median of round medians) | vs HF bf16 + static cache [95% CI over rounds] | paired per-round min-max | TTFT ms | Serving peak MiB (torch; Ollama: size_vram) | Device used MiB | Weights MiB | Load peak MiB |
|---|---|---|---|---|---|---|---|---|---|
| bf16_static | 3 | 58.5 | 1.000 [0.978, 1.023] | 1.000-1.000 | 48 | 3346 | 3442 | 3282.3 | 3282.3 |
| int8_graph | 3 | 111.1 | 1.899 [1.882, 2.012] | 1.889-2.012 | 56 | 2342 | 2474 | 2237.8 | 3426.3 |
| llamacpp_q8 | 3 | 107.9 | 1.845 [1.829, 1.901] | 1.837-1.879 | 29 | 2300 | 2426 | None | None |

| EfficientGPT int8_graph vs | ratio of round medians [95% CI, round bootstrap] | paired per-round min-max |
|---|---|---|
| bf16_static | 1.899 [1.882, 2.012] | 1.889-2.012 |
| llamacpp_q8 | 1.029 [1.013, 1.076] | 1.018-1.071 |

| Quality vs bf16_static | ΔNLL [95% CI] (relative) | greedy common prefix, mean tokens of 64 | exact 64-token matches | quality v1 |
|---|---|---|---|---|
