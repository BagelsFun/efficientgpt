# Benchmark verification (Phase B)

## Environment (recorded 2026-10-31)
| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060, 8 GB (8188 MiB), compute capability 8.9, 24 SMs, 24 MiB L2, PCIe Gen4 x8, max SM clock 3135 MHz |
| Driver | 617.14 |
| OS | Windows 11 Pro, build 10.0.26200 (Python reports `Windows-10-10.0.26200`) |
| Python | 3.11.9 (`.venv`) |
| PyTorch / CUDA / cuDNN | 2.11.0+cu128 / 12.8 / 9.19 |
| Triton | triton-windows 3.7.1.post27 |
| transformers / torchao | 5.21.0 / 0.18.0 (`--no-deps`, torch unchanged) |
| Full pin | `requirements-lock.txt` (83 packages) |
| Model | `Qwen/Qwen3-1.7B` revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (HF); `qwen3:1.7b-q8_0` digest 29b6a7a420a4 (Ollama 0.40.12; modelfile: repeat_penalty 1, our requests set temperature 0) |
| EfficientGPT quantisation | per-output-channel symmetric int8 for every linear incl. lm_head (untied), bf16 embedding; Triton `dot16` BLOCK_N 32, BLOCK_K 128, 4 warps (`research/realmodel/int8_linear.py` DEFAULT) |
| torchao quantisation | `quantize_(model, Int8WeightOnlyConfig())` after untying lm_head (same layer set), Inductor max-autotune |
| Code | V3 code of the private development history (commit shown as <private-commit>); this verification on the code published here |

## Workload and controls
Qwen3-1.7B chat prompt (~61 tokens, unique `[request <uuid>]` prefix per request), greedy (temperature 0, no sampling), batch 1,
exactly 256 new tokens (checked for every rep), thinking disabled on both HF and Ollama sides.

Each mode runs in a fresh process: 1 warm-up, 1 compile-settling call, then 5 timed requests. Rounds interleave the
modes in alternating order, and GPU utilisation is recorded before each process. Threading: HF/PyTorch defaults;
Ollama defaults (its own CPU threads, irrelevant at full GPU offload).

**Differences that remain and are disclosed** (`BENCHMARK_LIMITATIONS.md` §3): quantisation formats (per-channel int8 vs
Q8_0 blocks), Ollama's HTTP streaming, cache sizes (bucket 384 vs Ollama num_ctx 4096 vs HF ≈ 317), and precision
(bf16 reference vs approximate int8/Q8_0).

## Commands
```
python -m research.realmodel.bench_int8 --out runs/<stamp>/bench --modes int8_graph llamacpp_q8 bf16_static torchao_int8_static_at --rounds 4 --reps 5 [--quality --quality-modes bf16_static int8_graph]
python -m research.realmodel.analyze_int8 --dir runs/<stamp>/bench
python -m research.realmodel.profile_decode --quant int8 --out runs/<stamp>/profile_int8.json
```

## Verified results
Throughput, TTFT and memory come from `runs/20261031_v4/v3_final/` (4 rounds, GPU 0–1%), re-verified on the final V3
code in `runs/20261031_v4/v3_verify/` (3 rounds).

| Claim | Status | Evidence |
|---|---|---|
| EfficientGPT V3 111.9 tok/s | **VERIFIED** (v3_final; 111.1 in v3_verify) | `v3_final/REPORT.md`, `v3_verify/REPORT.md` |
| llama.cpp Q8_0 109.9 tok/s; ratio 1.018 [1.015, 1.051] → parity | **VERIFIED** (re-verified 1.029 [1.013, 1.076]) | same |
| torchao int8 (max-autotune) 84.4 tok/s; ratio 1.33 | **VERIFIED** (process medians 78.9–89.4 tok/s across the R2 / V3 runs; autotuning picks differ per process) | `v3_final/`, `int8_vs_autotune/` |
| HF bf16 + compile 59.3 tok/s; ratio 1.89 | **VERIFIED** (1.90 on the final code) | same |
| TTFT 54 vs 33 ms | **VERIFIED** | same |
| 149/149 tests at V3; 156/156 now, 0 skipped | VERIFIED | `runs/20261031_v4/v3_clean_checkout_final.json`, `runs/20261031_v4/recognition_clean_checkout.json`, `runs/20261031_v4/test_gpu_census.json` |
| Quality (quality-v1 39/52 = bf16, ΔNLL −0.15%) | **VERIFIED** | `v3_final/` (quality rows) |

## Additional metrics (showcase run)
Inter-token latency p50/p95, end-to-end request time, peak reserved memory and compile/capture overhead per mode come
from the showcase run `runs/20261031_v4/showcase_bench/` (same design, 4 rounds, all 16 processes started at 0–1% GPU
utilisation, every rep 256 tokens). It ran after other GPU applications were closed, and every engine is somewhat faster
than in `v3_final`; the ratios are consistent with it. An earlier attempt ran while other applications used the GPU (19–30%)
and is kept as `showcase_bench_CONTAMINATED_external_gpu_load/`. It is not used.

| Engine | tok/s | EfficientGPT vs it [95% CI over rounds] | inter-token latency p50 / p95 | 256-token request | TTFT | peak reserved / device memory | first call (compile / capture) |
|---|---|---|---|---|---|---|---|
| **EfficientGPT (int8 + graph decoder)** | **117.3** | — | **8.49 / 8.55 ms** | 2.23 s | 53 ms | 2424 / 2474 MiB | 5.9 s |
| llama.cpp Q8_0 (Ollama 0.40.12) | 116.0 | 1.011 [1.009, 1.013]: parity | 8.57 / 8.93 ms (per streamed chunk) | 2.23 s | 32 ms | – / 2423 MiB | 7.2 s |
| torchao int8 weight-only, max-autotune | 90.5 | 1.30 [1.29, 1.32] | 10.85 / 12.02 ms | 2.88 s | 66 ms | 3360 / 3430 MiB | 32.8 s |
| HF bf16 + static cache / torch.compile | 63.2 | 1.86 [1.85, 1.87] | 15.72 / 16.38 ms | 4.08 s | 44 ms | 3378 / 3438 MiB | 5.5 s |

Inter-token latency comes from per-token timestamps (HF streamer `put()` per token; graph decoder event per token; Ollama per
streamed chunk, occasionally two tokens). End-to-end is the median request time for 256 tokens. The 1-round Demo 2 run
(`runs/20261031_v4/demo_compare/`) agrees: 117.3 / 115.6 / 63.0 tok/s.

## TTFT profile
`python -m research.realmodel.profile_prefill` (`runs/20261031_v4/showcase_profile_prefill.json`). The benchmark prompt is
57–64 tokens (with its unique request prefix); the profile uses the same prompt without the prefix (28 tokens). Prefill is
weight-bound at these lengths, so the kernel breakdown carries over.

| Eager prefill (28-token prompt; profile) | wall | GPU kernel time | int8→bf16 copy kernels | GEMM | other |
|---|---|---|---|---|---|
| bf16 | 40.0 ms | 18.5 ms | 1.0 ms | 15.2 ms | 2.3 ms |
| int8 | 59.6 ms | 34.9 ms | **21.3 ms (196 calls)** | 7.9 ms | 4.0 ms |

Two costs: (1) the int8 prefill path materialises bf16 copies of every weight matrix (21.3 ms of GPU time); (2) prefill runs
eagerly, so wall time exceeds GPU time by ≈ 21–25 ms in both precisions (host launch overhead for ≈ 1,000 kernels). The
standalone wall times are larger than the in-benchmark TTFT (53 / 44 ms); the kernel breakdown is the reliable part.

## Unverified
- Fresh-virtualenv installation from `requirements-lock.txt`: not executed (≈ 3 GB download).
- Any result on Linux, other GPUs, other models, batch > 1 or sampling.
