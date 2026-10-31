# R1 end-to-end: Qwen3-1.7B batch-1 decode on RTX 4060 (analysis)

| Mode | Rounds | Decode tok/s (median of round medians) | vs HF bf16 + static cache [95% CI over rounds] | paired per-round min-max | TTFT ms | Serving peak MiB (torch; Ollama: size_vram) | Device used MiB | Weights MiB | Load peak MiB |
|---|---|---|---|---|---|---|---|---|---|
| bf16_static | 4 | 58.9 | 1.000 [0.969, 1.032] | 1.000-1.000 | 45 | 3346 | 3426 | 3282.3 | 3282.3 |
| bf16_graph | 4 | 39.6 | 0.672 [0.669, 0.694] | 0.669-0.694 | 43 | 3802 | 3870 | 3282.3 | 3282.3 |
| int8_static | 4 | 99.0 | 1.680 [1.671, 1.734] | 1.677-1.733 | 66 | 2307 | 2440 | 2237.8 | 3426.3 |
| int8_graph | 4 | 55.2 | 0.937 [0.933, 0.968] | 0.932-0.969 | 56 | 2720 | 2826 | 2237.8 | 3426.3 |
| torchao_int8_static_at | 4 | 83.3 | 1.415 [1.406, 1.461] | 1.406-1.464 | 68 | 2875 | 3430 | 2238.5 | 4911.7 |
| llamacpp_q8 | 4 | 109.8 | 1.865 [1.854, 1.925] | 1.857-1.918 | 30 | 2300 | 2423 | None | None |

| EfficientGPT int8_graph vs | ratio of round medians [95% CI, round bootstrap] | paired per-round min-max |
|---|---|---|
| bf16_static | 0.937 [0.933, 0.968] | 0.932-0.969 |
| bf16_graph | 1.394 [1.393, 1.396] | 1.393-1.396 |
| int8_static | 0.558 [0.556, 0.560] | 0.556-0.559 |
| torchao_int8_static_at | 0.662 [0.659, 0.665] | 0.659-0.664 |
| llamacpp_q8 | 0.503 [0.502, 0.505] | 0.501-0.505 |

| Quality vs bf16_static | ΔNLL [95% CI] (relative) | greedy common prefix, mean tokens of 64 | exact 64-token matches | quality v1 |
|---|---|---|---|---|
| int8_graph | -0.0054 [-0.0076, -0.0032] (-0.15%) | 23.1 | 3/16 | 39/52 |
| bf16_graph | +0.0000 [+0.0000, +0.0000] (+0.00%) | 48.2 | 12/16 | 40/52 |
| bf16_static (reference) | NLL 3.6831 | - | - | 39/52 |
