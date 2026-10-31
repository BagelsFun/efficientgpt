# Benchmark limitations and known weaknesses

What the published comparisons do not show, and why EfficientGPT is still behind llama.cpp on time to first token.
Numbers are from `runs/20261031_v4/v3_final/` and `v3_verify/` unless stated.

## 1. Time to first token: 54 ms (EfficientGPT V3) vs 33 ms (llama.cpp Q8_0)
TTFT here is prefill of a ~61-token chat prompt plus the first decode step, measured from the request until the first
token is available to the client.

| Contributor | Evidence | Estimated share |
|---|---|---|
| **int8 prefill dequantises every weight** | `int8_linear.int8_linear`: prompts > 16 rows take `x @ w.to(bf16).t()`, materialising a bf16 copy of each weight matrix per request: read 1.72 GB int8, write and re-read 3.44 GB bf16 | **profiled:** the int8→bf16 copy kernels take 21.3 ms of GPU time per prefill (196 calls) vs 1.0 ms of copies in bf16 (`BENCHMARK_VERIFICATION.md` §TTFT profile); the earlier 40 ms arithmetic bound overestimated it |
| Prefill runs eagerly (no CUDA graph, no compile) | `GraphGreedyDecoder.generate` calls the HF forward eagerly for the prompt | **profiled:** wall time exceeds GPU kernel time by ≈ 21–25 ms in both bf16 and int8 (host launch overhead of ≈ 1,000 kernels); a second TTFT lever: capture prefill per prompt-length bucket |
| bf16 prefill for comparison | HF bf16 TTFT 41–46 ms: reads 3.4 GB once | sets the floor the int8 path currently exceeds |
| llama.cpp | quantised matmul for prompt processing (no dequantised copy), C++ host path | 33 ms |

**Highest-value fix (not done; future work):** a Triton int8 GEMM for M > 16 (the `dot16` kernel tiled over M), which
would read 1.72 GB once instead of moving 8.6 GB. Expected: TTFT below the bf16 path, likely 20–30 ms for short prompts
(estimate, not measured).

## 2. Scope of the fast path
- **Batch 1, greedy only.** `GraphGreedyDecoder` has [1, 1] token/position buffers and argmax in the graph. Sampling
  (temperature, top-p) and batched requests fall back to HF `generate()` (EfficientGPT int8 via generate: 99.7 tok/s).
- **One model** (Qwen3-1.7B, rev 70d244cc), **one GPU** (RTX 4060 8 GB, Windows 11, driver in `BENCHMARK_VERIFICATION.md`).
- **Prompt length:** published runs use a ~61-token prompt and 256 new tokens; long-context decode (cache buckets up to
  4096) is tested for correctness but not benchmarked against llama.cpp.

## 3. Baseline differences that cannot be removed
| Difference | Effect |
|---|---|
| llama.cpp Q8_0 uses 32-weight blocks with fp16 scales (8.5 bits/weight) and also quantises the token embedding / output tensor; EfficientGPT uses per-output-channel int8 (8 bits) and keeps the embedding in bf16 | similar byte counts; different numerics (both approximate vs bf16) |
| llama.cpp runs through Ollama 0.40.12 with default settings (num_ctx 4096, its own threading) over HTTP streaming | not a tuned llama.cpp build; Ollama's server-side rate is ≈ 0.4% above our client-side measurement |
| Ollama occasionally streams two tokens in one chunk | its inter-token latency is per streamed chunk (504 of 510 expected gaps in the smoke run); a merged chunk yields one double-length gap, biasing its p95 upward |
| EfficientGPT's graph decoder uses request-sized cache buckets (384 slots here); HF static uses ~prompt + 256 | the bucket is larger than HF's cache, i.e. slightly unfavourable to EfficientGPT |
| torchao timings vary 79–89 tok/s across processes under max-autotune | its CI is wider; we report its best configuration |

## 4. Statistical limits
- 4 fresh-process rounds per mode (plus a 3-round re-verification); bootstrap over rounds. The llama.cpp comparison
  (1.018× [1.015, 1.051]) is **below the project's 5% promotion bar**, so it is reported as parity.
- Microbenchmarks in `reports/KERNELAGENT_ADA_REPORT.md` are warm-L2 and are never used as end-to-end claims.

## 5. Measurement hygiene incidents (disclosed)
- An external GPU workload (other applications at 19–30% utilisation) invalidated a showcase run:
  `runs/20261031_v4/showcase_bench_CONTAMINATED_external_gpu_load/` (kept, flagged, not used).
- Earlier incidents: a `git push` during one torchao round (R2), a 9% utilisation spike from a commit (V3 verification
  round 0). Each is reported where it occurred.

## 6. Not measured
vLLM / TensorRT-LLM (not run on this Windows 8 GB setup); Linux; other GPUs; power/energy; models other than Qwen3-1.7B;
a fresh-virtualenv install from `requirements-lock.txt` (≈ 3 GB download).
