# EfficientGPT V3: closing the gap to llama.cpp on a consumer GPU

**Summary.** On one RTX 4060 (8 GB), EfficientGPT decodes Qwen3-1.7B (batch 1, greedy) at **111.9 tok/s** with weight-only
int8 and a sync-free CUDA-graph decode loop: **1.89× Hugging Face bf16 + static cache/torch.compile, 1.33× torchao int8
weight-only (max-autotune), and 1.02× llama.cpp Q8_0 via Ollama** (95% CI [1.015, 1.051]; below the project's 5% bar for
claiming a win, so reported as parity). V2 was 0.83× llama.cpp. The gain came from profiling, not from a faster kernel:
the int8 GEMV was already at the DRAM limit; the remaining time was GPU idle between steps and an extra kernel per linear.
Time-to-first-token remains worse than llama.cpp (54 vs 33 ms). All numbers: fresh-process interleaved rounds on an idle
GPU, evidence in `runs/20261031_v4/`.

## 1. Problem
Batch-1 LLM decode is bound by weight streaming: Qwen3-1.7B in bf16 reads ≈ 3.4 GB per token, so an RTX 4060 (measured
≈ 211–222 GB/s) cannot exceed ≈ 62 tok/s in bf16. Weight-only int8 halves the bytes (ceiling ≈ 120 tok/s), but a PyTorch /
Hugging Face stack loses much of that to framework overhead. The question for V3: why did EfficientGPT int8 (91.3 tok/s)
trail llama.cpp Q8_0 (109.7), and what is the cheapest change that closes it, measured end to end?

## 2. Diagnosis (profile first)
`research/realmodel/profile_decode.py` (torch.profiler incl. kernels replayed inside CUDA graphs), per decoded token:

| | int8 (V2) | bf16 |
|---|---|---|
| GPU kernel time | 9.0 ms | 14.65 ms |
| wall time (unprofiled) | 10.95 ms | 16.9 ms |
| int8 GEMV (`_part`) | 7.76 ms ≈ 222 GB/s, **at the DRAM limit** | — |

Two findings: (1) the kernel was not the problem; (2) ≈ 2 ms per token (18%) was the GPU idle. *(Refined in the industry
showcase, `streamer_overhead.py`: ≈ 0.6 ms of that was HF `generate()`'s host loop, removed by the graph loop; the rest sat at
kernel boundaries inside the step, reduced by the kernel config; the streamer itself costs ≤ 0.07 ms/token.)* A second, smaller cost appeared only end to end: the split-K kernel config
chosen by KernelAgent-Ada on warm-L2 microbenchmarks adds a reduce kernel per linear (≈ 197 per token).

## 3. Changes (each measured)
| Step | ms / token | tok/s | Evidence |
|---|---|---|---|
| V2: HF `generate()` + split-K int8 | 10.95 | 91.3 | `int8_vs_baselines/` |
| + sync-free decode loop (device-resident tokens, pinned-memory streaming) | 10.32 | 96.8 | `v3_profile/config_sweep.json` |
| + one manual CUDA graph per step (forward + argmax + token/position update) | 10.32 | 96.8 | same (host was no longer the limit) |
| + end-to-end-chosen kernel config (no split-K) | 9.51 | 105.1 | same |
| + adaptive cache-capacity buckets in the backend (fixed 4096-slot cache was about 2× slower) | — | 111.9 | `v3_final/` |

Tried and dropped (measured, not shipped): fusing gate/up projections into one GEMV (bit-identical, +0.3%: launch count was
not the remaining cost).

**What is reused from earlier EfficientGPT work:** CUDA-graph capture of the whole decode step and adaptive capacity buckets
(Phase 5, TinyStories), the KernelAgent-Ada int8 kernel and its verification protocol, the fresh-process / bootstrap
benchmark methodology. **What is new in V3:** applying them to a real HF model (StaticCache with device-resident position),
pinned-memory token streaming with bounded run-ahead so per-token streaming survives without per-token synchronisation, and
the end-to-end config selection that overrode the microbenchmark winner.

## 4. Results (Qwen3-1.7B, batch 1, 256 new tokens, 4 fresh-process rounds, GPU 0–1%)
| Engine | tok/s | EfficientGPT V3 vs it [95% CI over rounds] | TTFT | Device memory | Quality v1 (52) |
|---|---|---|---|---|---|
| **EfficientGPT V3 (int8 + graph decoder)** | **111.9** | — | 54 ms | 2.47 GB | 39/52 |
| llama.cpp Q8_0 (Ollama 0.40.12 defaults) | 109.9 | 1.018 [1.015, 1.051] | 33 ms | 2.43 GB | 39/52 |
| EfficientGPT int8 via HF `generate()` (V3 kernel config) | 99.7 | 1.122 [1.112, 1.158] | 65 ms | 2.44 GB | — |
| torchao int8 weight-only, max-autotune | 84.4 | 1.326 [1.320, 1.369] | 66 ms | 3.43 GB | 38/52 (R2) |
| HF bf16 + the same graph decoder | 63.7 | 1.758 [1.687, 1.814] | 41 ms | 3.54 GB | 39/52 |
| HF bf16 + static cache / torch.compile | 59.3 | 1.888 [1.885, 1.948] | 46 ms | 3.43 GB | 39/52 |

Re-verification on the final code after the audit fixes (`runs/20261031_v4/v3_verify/`, 3 rounds): 111.1 vs llama.cpp 107.9
tok/s, **1.029× [1.013, 1.076]**; 1.90× HF bf16 static.

## 5. Correctness and precision
- **Contracts.** bf16 paths are the reference. int8 is approximate: ΔNLL −0.15% [−0.21%, −0.09%] vs bf16 (teacher-forced,
  24 × 512-token windows; through the prefill path), quality-v1 39/52 = bf16. Greedy outputs diverge from bf16 earlier
  than bf16 paths diverge from each other (common prefix of 64 tokens: int8 18.8, bf16 graph vs bf16 static 45.8, bf16
  dynamic vs static 42.9).
- **Decoder equivalence.** On tiny random Qwen3 models (bf16 and int8) the graph decoder produces exactly HF `generate()`'s
  greedy tokens (non-degenerate outputs asserted), stops exactly at EOS without leaking run-ahead tokens, and resets its
  cache between requests (`research/realmodel/test_fast_decode.py`). On the real model, compiled-kernel near-ties cause the
  divergence measured above.
- **Kernel.** fp64-truth contract, mutation tests (missing scale, skipped K block, unsigned reinterpretation rejected),
  held-out shapes (`research/kernelagent/`).

## 6. Reproducibility and fairness
- Every mode runs in fresh processes, rounds interleave with alternating order, GPU utilisation is recorded before each
  process; bootstrap CIs resample rounds. Same prompt (unique prefix per request), same 256 tokens, same (n−1)/(total−TTFT)
  decode formula for HF and Ollama paths (Ollama's server-side rate is ≈ 0.4% higher than its client-side rate, which makes
  the margin ≈ 1.4% instead of 1.8%).
- Baselines got their best available configuration: torchao with Inductor max-autotune (its default is ≈ 7% slower); bf16
  with max-autotune (+0.6%, not material); llama.cpp is Ollama's build with default settings, not a hand-tuned llama.cpp.
- Clean checkout: `python -m research.release.clean_checkout` runs every suite from a fresh `git worktree` with only
  hash-checked inputs copied in. Its first run found an undeclared test input (the TinyStories validation split), now declared.
- Pinned environment: `requirements-lock.txt`, `docs/ENVIRONMENT.md`.

## 7. Independent review and verification
Independent code reviews by a separate agent at every milestone (V3: 7 findings; none changed a published number; fixed:
incomplete EOS set, broken streaming of multi-byte characters, batch-size guard, unbounded per-bucket decoders, capture time
in TTFT, an overstated docstring, baseline wording). Final regression from a clean `git worktree` of the final commit:
149/149 OK, 0 skipped (`runs/20261031_v4/v3_clean_checkout_final.json`).

## 8. Limitations
- One GPU (RTX 4060, Windows); one model; batch 1, greedy. Sampling and batching use the slower `generate()` path.
- TTFT is worse than llama.cpp (prefill dequantises int8 weights; not optimised).
- llama.cpp was run through Ollama defaults, not tuned; vLLM / TensorRT-LLM were not run (8 GB Windows setup).
- Established techniques throughout (weight-only int8, CUDA graphs, static KV caches); the contribution is the measured,
  verified engineering path and its evidence, not a new algorithm.
