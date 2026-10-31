# R1 end-to-end: Qwen3-1.7B batch-1 decode on RTX 4060 (analysis)

| Mode | Rounds | Decode tok/s (median of round medians) | vs HF bf16 + static cache [95% CI over rounds] | paired per-round min-max | TTFT ms | Serving peak MiB (torch; Ollama: size_vram) | Device used MiB | Weights MiB | Load peak MiB |
|---|---|---|---|---|---|---|---|---|---|
| bf16_static | 4 | 59.3 | 1.000 [0.992, 1.008] | 1.000-1.000 | 46 | 3346 | 3426 | 3282.3 | 3282.3 |
| bf16_graph | 4 | 63.7 | 1.074 [1.074, 1.119] | 1.074-1.128 | 41 | 3389 | 3536 | 3282.3 | 3282.3 |
| int8_static | 4 | 99.7 | 1.682 [1.670, 1.699] | 1.670-1.711 | 65 | 2307 | 2440 | 2237.8 | 3426.3 |
| int8_graph | 4 | 111.9 | 1.888 [1.885, 1.948] | 1.885-1.964 | 54 | 2342 | 2474 | 2237.8 | 3426.3 |
| torchao_int8_static_at | 4 | 84.4 | 1.423 [1.417, 1.436] | 1.418-1.442 | 66 | 2876 | 3432 | 2238.5 | 4911.7 |
| llamacpp_q8 | 4 | 109.9 | 1.854 [1.852, 1.869] | 1.851-1.874 | 33 | 2300 | 2425 | None | None |

| EfficientGPT int8_graph vs | ratio of round medians [95% CI, round bootstrap] | paired per-round min-max |
|---|---|---|
| bf16_static | 1.888 [1.885, 1.948] | 1.885-1.964 |
| bf16_graph | 1.758 [1.687, 1.814] | 1.741-1.759 |
| int8_static | 1.122 [1.112, 1.158] | 1.121-1.148 |
| torchao_int8_static_at | 1.326 [1.320, 1.369] | 1.324-1.362 |
| llamacpp_q8 | 1.018 [1.015, 1.051] | 1.017-1.048 |

| Quality vs bf16_static | ΔNLL [95% CI] (relative) | greedy common prefix, mean tokens of 64 | exact 64-token matches | quality v1 |
|---|---|---|---|---|
| int8_graph | -0.0054 [-0.0076, -0.0032] (-0.15%) | 18.8 | 1/16 | 39/52 |
| bf16_graph | +0.0000 [+0.0000, +0.0000] (+0.00%) | 45.8 | 10/16 | 39/52 |
| bf16_static (reference) | NLL 3.6831 | - | - | 39/52 |
