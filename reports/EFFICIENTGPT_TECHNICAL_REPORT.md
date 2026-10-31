# EfficientGPT: verified small-batch LLM decoding on a consumer GPU

*Engineering research report, 2026-10-31. Single-author open-source project (MIT). All numbers are measured
on one NVIDIA RTX 4060 8 GB (Windows 11) and link to raw evidence in the repository.*

## 1. Abstract
EfficientGPT is a PyTorch/Triton inference stack and a verification-first kernel-search framework (KernelAgent-Ada), built
on nanoGPT and applied to Qwen3-1.7B. For batch-1 greedy decoding it reaches **111.9 tok/s, within 2% of llama.cpp Q8_0
(1.018× [1.015, 1.051]; 109.9 tok/s; the CI excludes 1 but the margin is below the project's 5% bar, so it is reported as
parity)**, 1.33× torchao's int8 weight-only path under Inductor max-autotune
and 1.89× Hugging Face bf16 with static cache and torch.compile, using 2474 MiB of GPU memory, with quality-v1 (39/52) and
perplexity (ΔNLL −0.15%) equal to bf16. The improvements came from profiling rather than from a novel kernel: the int8
GEMV already streamed weights at the DRAM limit, and the remaining time was GPU idle time: HF `generate()`'s per-step host
loop and per-kernel overhead inside the step, the latter mostly from one extra reduce kernel per linear layer. Time to first token remains worse than llama.cpp (54 vs 33 ms). The techniques are established. The
contribution is a reproducible, statistically controlled engineering path, and evidence that a PyTorch-native stack can
match a C++ engine at batch 1.

## 2. Problem statement
Autoregressive decoding at batch 1 is memory-bandwidth bound. Every token reads all weights once. Qwen3-1.7B in bf16
reads ≈ 3.4 GB per token, so an RTX 4060 (measured 211–222 GB/s) cannot exceed ≈ 62 tok/s, and int8 halves the bytes
(ceiling ≈ 120 tok/s). In practice PyTorch/Hugging Face stacks lose a large share of that ceiling to framework overhead,
while C++ engines such as llama.cpp get close to it. The questions were:
1. Where exactly does a PyTorch-native int8 path lose time?
2. Can it reach the C++ engine's decode rate without leaving PyTorch?
3. Can every claim be verified independently, with numerical contracts and fair baselines?

## 3. Prior approaches
| Approach | Examples | Relation to this work |
|---|---|---|
| Weight-only quantisation | llama.cpp (Q8_0 and k-quants), torchao int8/int4 weight-only, bitsandbytes LLM.int8, GPTQ / AWQ, TensorRT-LLM W8A16 | EfficientGPT uses plain per-channel symmetric int8, an established scheme |
| CUDA-graph decode | vLLM, TensorRT-LLM, gpt-fast, HF static cache with torch.compile ("reduce-overhead") | EfficientGPT captures the whole step, including argmax and the token/position update, in one manual graph |
| Avoiding per-token host sync in generation | transformers `DeferredStopCheck` (reads the stop flag one step late via pinned memory + event; enabled upstream for Apple MPS only); vLLM / TensorRT-LLM schedulers | EfficientGPT uses the same idea with a deeper bounded run-ahead; enabling transformers' own check on CUDA recovers most of the gain here (+4.3% on an RTX 4060 under Windows; upstream measured no gain on an H100; `docs/UPSTREAM_CONTRIBUTION_CANDIDATES.md`) |
| Split-K / FlashDecoding attention | FlashAttention, FlashDecoding, FlashInfer | Phase 5 reimplemented split-K decode attention in Triton on a small model |
| Kernel autotuning and search | Triton autotune, AutoTVM / Ansor, LLM-generated kernels (KernelBench-style) | KernelAgent-Ada is template search plus a verification protocol, not a new search algorithm |

## 4. System architecture
Details: `docs/ARCHITECTURE.md`.
```
request ─► chat template / tokenize ─► pick cache bucket (256 … 4096 slots, smallest ≥ prompt + new tokens)
        ─► prefill: eager HF forward (int8 prefill = dequantise + cuBLAS)  ─► first token
        ─► decode: ONE CUDA graph per step = compiled HF forward (int8 Triton GEMV per linear, SDPA over the static cache)
                   + argmax + in-place token / position update      (replayed per token; no host sync in the loop)
        ─► token stream: non-blocking copy to pinned host memory + CUDA event per token, bounded run-ahead (8)
        ─► incremental detokenisation ─► EOS (all generation_config ids) ─► text
```
KernelAgent-Ada: task (real shapes, dev / held-out) → template candidates → AST safety screen → isolated subprocess (fp64
truth contract, input-mutation guard, CUDA-graph timing vs eager / Inductor / best existing) → search strategies → JSONL
history → held-out evaluation → end-to-end confirmation.

## 5. Engineering contributions
**Established techniques, reimplemented and verified here:** CUDA-graph decode with static KV caches, per-channel int8
weight-only quantisation with an fp32-accumulating tensor-core GEMV, split-K decode attention, Inductor compilation, and
bootstrap confidence intervals.

**Contributions of this project (engineering, not algorithmic):**
1. **Diagnosis of a PyTorch-vs-C++ gap by measurement.** The kernel profile showed the int8 GEMV at ≈ 222 GB/s (the DRAM
   limit), so no faster kernel could help; the GPU was idle ≈ 2 ms per token (wall 10.95 vs 9.0 ms of kernels); ≈ 0.6 ms of it was host-side loop overhead in HF `generate()` (removed by the sync-free graph loop: 91.3 → 96.8 tok/s), the rest sat at kernel boundaries inside the step (reduced by removing ≈ 197 split-K reduce kernels: → 105.1 tok/s). The text streamer itself costs ≤ 0.07 ms/token (`runs/20261031_v4/streamer_overhead.json`).
2. **A sync-free, streaming graph decoder for HF models** (`research/realmodel/fast_decode.py`). The whole step plus
   argmax and position update sit in one manual CUDA graph. Per-token streaming survives through pinned-memory copies and
   CUDA events with bounded run-ahead, and EOS never leaks run-ahead tokens (tested). The deferred-stop idea itself is prior
   art: transformers' `DeferredStopCheck`, which is MPS-only upstream. The contribution here is the CUDA measurement:
   forcing transformers' check on CUDA gives +4.3% with identical outputs, within 0.4% of the full graph loop. Upstream
   measured no gain on an H100 (transformers #47975), so this is a candidate only if it also holds on Linux.
3. **End-to-end selection overriding microbenchmarks.** KernelAgent-Ada's warm-L2 winner (split-K) added ≈ 197 reduce
   kernels per token and was slower in the model (10.32 vs 9.51 ms/step). The shipped config was chosen end to end. The
   same discipline earlier rejected a 1.1–1.6× attention-kernel win that was worth ≈ 1% end to end.
4. **Adaptive cache-capacity buckets for graph decoding** (Phase 5 idea, applied to HF StaticCache). A fixed 4096-slot
   cache made decoding about 2× slower (55.2 vs 111.9 tok/s), because static-cache attention scans every slot.
5. **A verification protocol around everything:**
   - fp64-truth contracts by kernel kind, with mutation tests of the evaluator itself and input-mutation guards;
   - fresh-process interleaved benchmarking with round-level bootstrap CIs, contention detection, and a 5% promotion bar;
   - clean-checkout regression from a pristine `git worktree`;
   - independent code reviews at every milestone, with every finding fixed or disclosed.

## 6. Experimental methodology
- **Hardware and software:** RTX 4060 8 GB, Windows 11, driver and versions in `reports/BENCHMARK_VERIFICATION.md`.
  Software is pinned in `requirements-lock.txt` (torch 2.11.0+cu128, triton-windows 3.7.1, transformers 5.21.0, torchao
  0.18.0). Model: Qwen3-1.7B rev 70d244cc.
- **Workload:** chat-formatted prompt (~61 tokens) with a unique prefix per request to defeat prefix caching; greedy;
  exactly 256 new tokens (verified for every rep of every mode); batch 1.
- **Design:**
  - every mode runs in its own fresh process, after 1 warm-up and 1 compile-settling call, then 5 timed requests;
  - rounds interleave the modes and alternate their order;
  - GPU utilisation is recorded before each process, and anything above 15% is flagged as contended.
- **Metrics:** decode tok/s = (n − 1) / (total − TTFT), identical for the HF and Ollama paths; TTFT; inter-token latency
  from per-token timestamps; end-to-end time; torch peak allocated and reserved memory; process device memory.
- **Statistics:** the ratio of medians of round medians, with a 95% bootstrap CI over rounds; paired per-round min–max.
  A "win" needs a ratio ≥ 1.05 with the CI excluding 1.
- **Baselines at their best available configuration:** torchao with max-autotune (+2–7% over its default across separate runs;
  no same-run comparison), bf16
  with and without max-autotune, and llama.cpp through Ollama defaults.

## 7. Benchmark results
Qwen3-1.7B, batch 1, 256 new tokens, 4 fresh-process rounds, idle GPU (`runs/20261031_v4/v3_final/`):

| Engine | tok/s | EfficientGPT vs it | TTFT | Device memory |
|---|---|---|---|---|
| **EfficientGPT (int8 + graph decoder)** | **111.9** | — | 54 ms | 2474 MiB |
| llama.cpp Q8_0 (Ollama 0.40.12) | 109.9 | **1.018 [1.015, 1.051]: parity** | 33 ms | 2425 MiB (nvidia-smi delta) |
| torchao int8 weight-only, max-autotune | 84.4 | 1.33 [1.32, 1.37] | 66 ms | 3432 MiB |
| HF bf16 + same graph decoder | 63.7 | 1.76 [1.69, 1.81] | 41 ms | 3536 MiB |
| HF bf16 + static cache / torch.compile | 59.3 | 1.89 [1.89, 1.95] | 46 ms | 3426 MiB |

The final code was re-verified after the audit fixes (`v3_verify/`): 111.1 vs 107.9 tok/s, 1.029× [1.013, 1.076].

**Showcase verification run** (`runs/20261031_v4/showcase_bench/`, 4 rounds, all processes at 0–1% GPU, after other GPU
applications were closed; every engine is somewhat faster than in v3_final and the ratios agree):

| Engine | tok/s | EfficientGPT vs it [95% CI over rounds] | inter-token latency p50 / p95 | 256-token request | TTFT | peak reserved / device memory | first call (compile / capture) |
|---|---|---|---|---|---|---|---|
| **EfficientGPT (int8 + graph decoder)** | **117.3** | — | **8.49 / 8.55 ms** | 2.23 s | 53 ms | 2424 / 2474 MiB | 5.9 s |
| llama.cpp Q8_0 (Ollama 0.40.12) | 116.0 | 1.011 [1.009, 1.013]: parity | 8.57 / 8.93 ms (per streamed chunk) | 2.23 s | 32 ms | – / 2423 MiB | 7.2 s |
| torchao int8 weight-only, max-autotune | 90.5 | 1.30 [1.29, 1.32] | 10.85 / 12.02 ms | 2.88 s | 66 ms | 3360 / 3430 MiB | 32.8 s |
| HF bf16 + static cache / torch.compile | 63.2 | 1.86 [1.85, 1.87] | 15.72 / 16.38 ms | 4.08 s | 44 ms | 3378 / 3438 MiB | 5.5 s |


**Optimisation ladder** (`reports/EFFICIENTGPT_V3_RESEARCH_REPORT.md` §3):

| Step | tok/s |
|---|---|
| HF `generate()` with split-K int8 | 91.3 |
| Sync-free loop | 96.8 |
| Manual whole-step graph | 96.8 (host was no longer the limit) |
| End-to-end kernel config | 105.1 |
| Adaptive buckets (backend) | 111.9 |

## 8. Accuracy and numerical considerations
- **Contracts.** bf16 is the reference; int8 (ours and torchao) and Q8_0 are approximate.
- **Perplexity:** teacher-forced ΔNLL vs bf16 on 24 × 512 tokens is −0.15% [−0.21%, −0.09%] for ours and −0.16% for
  torchao (default compile; `int8_vs_baselines/`). This is measured through the prefill path, so it tests the quantised weights rather than the decode kernel.
- **Task quality:** quality-v1 (52 items) is 39/52 for ours, bf16 and llama.cpp, and 38/52 for torchao (default compile).
- **Divergence:** greedy common prefix with bf16 over 64 tokens is 18.8 tokens for ours, against 45.8 for the bf16 graph
  decoder and a 42.9-token floor between two bf16 implementations. Quantisation changes texts sooner, without a measured
  quality loss on these suites.
- **Decoder equivalence:** on tiny random Qwen3 models the graph decoder reproduces HF `generate()` tokens exactly (bf16
  and int8). On the real model, compiled-kernel near-ties cause the divergence above.
- **Kernels:** fp64-truth bounds (≤ 4× the eager error + floor), mutation tests, held-out shapes.

## 9. Memory performance
Weights are 2238 MiB (int8 linears plus bf16 embedding) against 3282 MiB in bf16. Process device memory: 2474 MiB for
ours, 2425 MiB for llama.cpp (nvidia-smi delta), 3432 MiB for torchao, 3426 MiB for bf16. The load peak was 6847 MiB in the
first version, then cut to 3426 MiB by quantising in row chunks and releasing each bf16 layer as soon as it is replaced. KV memory depends on the bucket
(≈ 112 KiB per slot); at most 3 bucket decoders are kept.

## 10. Limitations
Full list: `reports/BENCHMARK_LIMITATIONS.md`.
- **Time to first token** (53 vs 32 ms): profiled as int8→bf16 weight copies in prefill (21.3 ms of GPU time) plus
  ≈ 21–25 ms of eager-prefill launch overhead shared with bf16 (`BENCHMARK_VERIFICATION.md` §TTFT profile).
- **Scope:** batch 1 and greedy only on the fast path; one model; one Windows GPU.
- **Baseline:** llama.cpp only via Ollama defaults; vLLM / TensorRT-LLM not run.
- **Statistics:** the llama.cpp margin is below the 5% bar.
- **Microbenchmarks:** warm-L2 only.

## 11. Reproducibility
- **Commands:** `REPRODUCE.md` and `docs/DEMO.md`.
- **Clean checkout:** a fresh `git worktree` with hash-checked inputs passed 149/149 tests at V3 and 156/156 at the public
  release (`runs/20261031_v4/test_gpu_census.json` lists which need a GPU).
- **Evidence:** raw per-process JSON is committed for every result, and so are the contaminated and invalid runs, labelled.

## 12. Future research directions
1. A Triton int8 GEMM for prefill (M > 16), to remove the dequantised copies (21.3 ms GPU), and graph capture of prefill
   per prompt-length bucket (≈ 21–25 ms host overhead): the two measured TTFT levers.
2. A graph-captured sampling path (inverse-CDF with pre-drawn uniforms; Phase 5 has a tested implementation for the small
   model).
3. Batched decoding with per-row positions (Phase 5 ragged batches), and int4 weight-only.
4. Same-model comparisons against a tuned llama.cpp build, vLLM and TensorRT-LLM on Linux.
5. Upstreaming measured findings (candidates with evidence in `docs/UPSTREAM_CONTRIBUTION_CANDIDATES.md`; none submitted
   yet).

## 13. References and acknowledgments
nanoGPT (A. Karpathy, MIT); PyTorch (BSD-3) and TorchInductor; Triton (MIT; triton-windows build); Hugging Face
transformers (Apache-2.0); torchao (BSD-3); bitsandbytes (MIT); llama.cpp and Ollama (MIT); Qwen3 (Alibaba Qwen team,
Apache-2.0). Background techniques: FlashAttention / FlashDecoding (Dao et al.), LLM.int8 (Dettmers et al.), GPTQ
(Frantar et al.), AWQ (Lin et al.), vLLM (Kwon et al.), TensorRT-LLM (NVIDIA). Development used an AI coding assistant; all
results were produced by code in this repository and verified by the procedures above.
