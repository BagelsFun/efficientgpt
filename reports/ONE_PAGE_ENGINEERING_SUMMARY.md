# EfficientGPT: one-page engineering summary

**What it is.** A PyTorch/Triton LLM inference stack and a verification-first GPU-kernel search tool (KernelAgent-Ada),
built on nanoGPT and applied to Qwen3-1.7B on a consumer RTX 4060 (8 GB). Single-author open-source project (MIT).

**Headline result (verified).** Batch-1 greedy decoding of Qwen3-1.7B, 4 fresh-process rounds, 95% CIs over rounds:

| | tok/s | EfficientGPT vs it | GPU memory |
|---|---|---|---|
| **EfficientGPT (int8 + CUDA-graph decoder)** | **111.9** | — | 2474 MiB |
| llama.cpp Q8_0 (via Ollama) | 109.9 | 1.02× [1.015, 1.051]: **parity** (below our 5% bar) | 2425 MiB |
| torchao int8 weight-only (max-autotune) | 84.4 | **1.33×** | 3432 MiB |
| Hugging Face bf16 + static cache / torch.compile | 59.3 | **1.89×** | 3426 MiB |

Quality is equal to bf16: 39/52 on the project's task suite, perplexity ΔNLL −0.15%. Time to first token is 54 ms
against llama.cpp's 33 ms, where we are worse.

A second verification run after closing other GPU applications (`reports/BENCHMARK_VERIFICATION.md`) reproduced parity:
117.3 vs 116.0 tok/s (1.011 [1.009, 1.013]). It also measured the steadiest inter-token latency of the four engines,
p50 / p95 8.49 / 8.55 ms (llama.cpp 8.57 / 8.93 ms, per streamed chunk).

**How (the engineering).**
1. **Profiled first.** The int8 GEMV already streamed weights at ≈ 222 GB/s (the DRAM limit); about 18% of each token was
   GPU idle time, partly HF `generate()`'s per-step host loop (≈ 0.6 ms) and partly kernel boundaries inside the step.
2. **Sync-free streaming decoder:** one manual CUDA graph per token (forward + argmax + in-place token/position update),
   tokens streamed through pinned memory and CUDA events with bounded run-ahead, with exact EOS semantics (tested
   against HF `generate()`).
3. **End-to-end kernel selection:** the microbenchmark winner (split-K) was 8% slower inside the model, so the shipped
   config was chosen end to end.
4. **Adaptive KV-cache buckets:** they fixed a 2× slowdown caused by a fixed-size static cache.

Path: 91.3 → 96.8 → 105.1 → 111.9 tok/s.

**What's original vs established.** Weight-only int8, CUDA graphs, static KV caches and split-K attention are
established. The contribution is engineering:
- the measured diagnosis of a PyTorch-vs-C++ gap and its closure;
- a streaming graph decoder for HF models;
- a verification protocol: fp64-truth kernel contracts, mutation-tested evaluator, held-out shapes, contention-aware
  fresh-process benchmarks with CIs and a promotion bar, clean-checkout regression, and independent reviews with every
  finding fixed or disclosed.

**Evidence quality.**
- **Tests:** 156 tests pass from a pristine `git worktree` and in the public tree (0 skipped; 47 need a GPU).
- **Raw evidence:** per-process JSON for every number, including contaminated and invalid runs, kept and labelled.
- **Integrity:** checkpoints are SHA-256 verified, with no secrets or model files in history.

**Limits.**
- Batch 1, greedy only (sampling and batching fall back to HF).
- One model, one Windows GPU.
- llama.cpp only via Ollama defaults; vLLM / TensorRT-LLM not run.
- Int8 prefill is unoptimised.

**Try it:** `python -m research.realmodel.demo` · `python -m research.showcase.demo_kernelagent` · see `docs/DEMO.md`.

**Next step with the largest expected gain:** an int8 GEMM for prefill, to remove the dequantised weight copy (the main
TTFT cost).
