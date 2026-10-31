# R1 end-to-end: Qwen3-1.7B batch-1 decode on RTX 4060 (analysis)

| Mode | Rounds | Decode tok/s (median of round medians) | vs HF bf16 + static cache [95% CI over rounds] | paired per-round min-max | TTFT ms | Serving peak MiB (torch; Ollama: size_vram) | Device used MiB | Weights MiB | Load peak MiB |
|---|---|---|---|---|---|---|---|---|---|
| bf16_static | 4 | 59.2 | 1.000 [0.995, 1.005] | 1.000-1.000 | 46 | 3350 | 3450 | 3282.3 | 3282.3 |
| int8_static | 4 | 91.3 | 1.543 [1.537, 1.559] | 1.543-1.556 | 67 | 2358 | 2598 | 2237.8 | 3426.3 |
| torchao_int8_static | 4 | 82.8 | 1.400 [1.393, 1.406] | 1.392-1.404 | 68 | 2875 | 3428 | 2238.5 | 4911.7 |
| llamacpp_q8 | 4 | 109.7 | 1.853 [1.846, 1.864] | 1.845-1.864 | 38 | 2300 | 2424 | None | None |

| EfficientGPT int8 vs | ratio of round medians [95% CI, round bootstrap] | paired per-round min-max |
|---|---|---|
| bf16_static | 1.543 [1.537, 1.559] | 1.543-1.556 |
| torchao_int8_static | 1.103 [1.098, 1.113] | 1.100-1.118 |
| llamacpp_q8 | 0.833 [0.828, 0.840] | 0.828-0.844 |

| Quality vs bf16_static | ΔNLL [95% CI] (relative) | greedy common prefix, mean tokens of 64 | exact 64-token matches | quality v1 |
|---|---|---|---|---|
| int8_static | -0.0054 [-0.0076, -0.0032] (-0.15%) | 22.9 | 4/16 | 39/52 |
| torchao_int8_static | -0.0058 [-0.0075, -0.0043] (-0.16%) | 34.4 | 7/16 | 38/52 |
| llamacpp_q8 | n/a (no logits) | 32.2 | 7/16 | 39/52 |
| bf16_dynamic | +0.0000 [+0.0000, +0.0000] (+0.00%) | 42.9 | 10/16 | - |
| bf16_static (reference) | NLL 3.6831 | - | - | 39/52 |
