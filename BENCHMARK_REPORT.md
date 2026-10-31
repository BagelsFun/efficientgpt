# Benchmark report: verified headline numbers

One RTX 4060 (8 GB), Windows 11, torch 2.11.0+cu128, triton-windows 3.7.1. Every line links to raw evidence; "idle" means
GPU utilisation ≈ 0% before timing with no other GPU job of ours running. Full details: `reports/PHASE5_REPORT.md`,
`reports/KERNELAGENT_ADA_REPORT.md`, `docs/ASSISTANT.md`, `docs/EXPERIMENT_LOG.md`. Not compared with vLLM / TensorRT-LLM.

## 1. EfficientGPT Phase 5 decoder (12.19M TinyStories GPT; inference research, not a general LLM)
| Comparison | Contract | Result (95% CI, all workloads) | Evidence |
|---|---|---|---|
| eager prefix SDPA → `adaptive_graph` (CUDA Graph, adaptive KV buckets) | exact (bit-identical tokens and logits) | B16 p64 n64: 131.9 → 52.8–56.2 ms (four runs) | PHASE5_REPORT §1 ladder |
| `adaptive_graph` → `compiled_triton_full_graph` | near-tie | 1.29–1.74× | PHASE5_REPORT §5 |
| `compiled_triton_full_graph` → fp16 weights | approximate (ΔNLL −4e-6) | 1.39–1.80× | `research/phase5/ci_runs/20261030_212013_precision/` |
| split-K Triton → KernelAgent-Ada single-pass (shape dispatch) | near-tie | batch 16: 0.999–1.015× (≈ 1%, below 5% bar); batch 1: 1.000–1.003× | `research/phase5/ci_runs/20261031_dispatch_inprocess/` |

Bottleneck: GEMMs ≈ 65% of the B16 step at 78–83% of measured DRAM bandwidth (211 GB/s), so kernel work on this model is
near its limit; fewer weight bytes (fp16) is the lever.

## 1a. V3: Qwen3-1.7B batch-1 decode on RTX 4060 (int8 + sync-free CUDA-graph decoder)
4 fresh-process interleaved rounds, GPU 0-1%, greedy, 256 new tokens; `runs/20261031_v4/v3_final/REPORT.md`
(re-verified on the final code: `runs/20261031_v4/v3_verify/`, 1.029x vs llama.cpp).

| Engine | tok/s | EfficientGPT V3 vs it [95% CI over rounds] | TTFT | Device memory | Quality v1 |
|---|---|---|---|---|---|
| **EfficientGPT V3 (int8 + graph decoder)** | **111.9** | - | 54 ms | 2474 MiB | 39/52 |
| llama.cpp Q8_0 (Ollama 0.40.12 defaults) | 109.9 | **1.018 [1.015, 1.051]: parity** | 33 ms | 2425 MiB (nvidia-smi delta) | 39/52 |
| EfficientGPT int8 via HF generate (V3 kernel config) | 99.7 | 1.12 | 65 ms | 2440 MiB | - |
| torchao int8 weight-only, max-autotune | 84.4 | 1.33 | 66 ms | 3432 MiB | 38/52 (default compile, R2) |
| HF bf16 + the same graph decoder | 63.7 | 1.76 | 41 ms | 3536 MiB | 39/52 |
| HF bf16 + static cache / torch.compile | 59.3 | **1.89** | 46 ms | 3426 MiB | 39/52 |

## 1b. V2 (superseded by 1a): Qwen3-1.7B batch-1 decode, weight-only int8 (R1/R2; approximate contract)
Fresh-process rounds, interleaved, GPU 0-1%, greedy, 256 new tokens, unique prompts. Evidence: `runs/20261031_v4/int8_vs_baselines/`
(4 rounds) and `runs/20261031_v4/int8_vs_autotune/` (5 rounds, max-autotune), each with `REPORT.md`.

| Engine | Decode tok/s | Ours vs it (round-level 95% CI) | Device memory | Quality v1 (52) | dNLL vs bf16 |
|---|---|---|---|---|---|
| HF bf16 + static cache / torch.compile (with max-autotune: +0.6%) | 59.2 | **1.54x [1.54, 1.56]** | 3.45 GB | 39/52 | - |
| bitsandbytes LLM.int8 / nf4 | 7.8 / 22.0 | 11.7x / 4.2x | - | - / 36/52 | - |
| torchao int8 weight-only, default compile | 82.8 | **1.10x [1.10, 1.11]** | 3.43 GB | 38/52 | -0.16% |
| torchao int8 weight-only, max-autotune | 78.9-89.4 (per process) | **1.10x [1.08, 1.16]** on 4 clean rounds (1.09x [0.99, 1.16] incl. a round perturbed by our own git push) | 3.43 GB | - | - |
| **EfficientGPT int8 (KernelAgent-Ada kernel)** | **91.3** | - | **2.49 GB** | 39/52 | -0.15% |
| **llama.cpp Q8_0 (Ollama 0.40.12)** | **109.7** | **0.83x [0.83, 0.84] (llama.cpp is faster)** | 2.42 GB | 39/52 | n/a |

TTFT: ours 67 ms, torchao 68 ms, bf16 46 ms, llama.cpp 38 ms. Greedy agreement with bf16 (common prefix of 64 tokens): ours 22.9,
torchao 34.4, llama.cpp 32.2, bf16-vs-bf16 floor 42.9; ours diverges earlier than torchao on 4/16 prompts.

## 2. KernelAgent-Ada (template Triton search; warm-L2 microbenchmarks, eval-v2)
| Result | Value | Evidence |
|---|---|---|
| Held-out shapes where the task winner beats `torch.compile` | 19/19 (1.07–9.3×) | `runs/20261031_v4/ka_study_evalv2/heldout_*.json` |
| Decode attention vs existing Phase 5 Triton kernel (held-out) | 3 WIN (1.16–1.58×), 1 SLOWER (0.83×) | same |
| Candidates evaluated / correct / contended | 138 / 138 / 0 | `runs/20261031_v4/ka_study_evalv2/history.jsonl` |
| Real-model task `int8_gemv` (Qwen3-1.7B shapes) | 62/62 correct, held-out 20/20; winner = shipped kernel | `runs/20261031_v4/int8_study/` |
| Mutation and review-regression tests | 13 GPU + 10 CPU OK (+ 4 R1 int8 GPU tests) | `research/kernelagent/test_*.py` |
| Single-pass numerics vs fp64 (model, 17,408 row-steps) | max dlogit 1.3e-5 (SDPA 1.7e-5), 0 argmax changes | `runs/20261031_v4/attention_error_audit.json` |

## 3. Local assistant (inference speed ≠ capability; quality listed separately)
| Backend | Decode tok/s (idle) | TTFT cold prefill | Memory | Quality v1.1 (52) | Evidence |
|---|---|---|---|---|---|
| Ollama qwen3.5:4b Q4_K_M | 69.6–71.2 | 84 / 253 / 706 ms @ 61 / 713 / 2,229 tok | 3.13 GiB | 47/52 (test 22/26) | `runs/20261031_v4/assistant_speed_idle/` |
| HF Qwen3-1.7B bf16, static cache | 57.7 (2.15× [2.14, 2.16] vs dynamic) | 45 ms (short prompt) | 3.35 GiB peak | 39/52 | same; ASSIST-03 for equivalence |
| HF Qwen3-1.7B bf16, dynamic cache | 26.8 | 43 ms (short prompt) | 4.3 GiB | 39/52 (test 18/26) | same |
| HF Qwen3-1.7B nf4 | 20 (under load) | — | 2.3 GiB | 36/52 (test 17/26) | ASSIST-01 |

Agent on repo questions (v1.1, 8 items): qwen3.5:4b 0/8 model-only → 8/8 with retrieval; Qwen3-1.7B 0/8 → 6/8 (ASSIST-02).
Agent v1.2 (20 items, audited): with tools qwen3.5:4b 20/20, Qwen3-1.7B nf4 17/20; model-only unsupported answers with the
calibrated prompt (now the CLI default): Qwen3-1.7B 11 → 4 (p = 0.016), qwen3.5:4b 3 → 0 (ASSIST-05).

## Known measurement caveats
- Phase 5 cross-process runs show ≈ ±3% per-process offsets; differences below 5% use the in-process interleaved benchmark.
- KernelAgent-Ada timings are warm-L2 (inputs reused); not end-to-end.
- Assistant speeds before 2026-10-31 were measured under an external GPU load and are superseded by the idle numbers above.
