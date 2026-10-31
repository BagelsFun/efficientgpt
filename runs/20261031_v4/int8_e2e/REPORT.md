# R1 end-to-end: Qwen3-1.7B batch-1 decode on RTX 4060 (analysis)

| Mode | Rounds | Decode tok/s (median of round medians) | vs HF bf16 + static cache [95% CI over rounds] | paired per-round min-max | TTFT ms | Serving peak MiB | Weights MiB | Load peak MiB |
|---|---|---|---|---|---|---|---|---|
| bf16_dynamic | 2 | 27.3 | 0.463 [0.458, 0.473] | 0.462-0.470 | 41 | 3336 | 3282.3 | 3282.3 |
| bf16_static | 4 | 58.8 | 1.000 [0.979, 1.021] | 1.000-1.000 | 45 | 3350 | 3282.3 | 3282.3 |
| int8_static | 4 | 91.3 | 1.552 [1.525, 1.585] | 1.543-1.558 | 65 | 2358 | 2237.8 | 6846.8 |
| bnb8_dynamic | 2 | 7.8 | 0.132 [0.130, 0.135] | 0.131-0.134 | 150 | 2019 | 1940.9 | 2831.2 |
| nf4_dynamic | 2 | 22.0 | 0.374 [0.370, 0.382] | 0.370-0.382 | 56 | 1342 | 1288.5 | 2178.7 |

| Quality vs bf16_static | ΔNLL [95% CI] (relative) | greedy common prefix, mean tokens of 64 | exact 64-token matches | quality v1 |
|---|---|---|---|---|
| int8_static | -0.0054 [-0.0076, -0.0032] (-0.15%) | 22.9 | 4/16 | 39/52 |
| bf16_dynamic | +0.0000 [+0.0000, +0.0000] (+0.00%) | 42.9 | 10/16 | - |
| bf16_static (reference) | NLL 3.6831 | - | - | 39/52 |
