# EfficientGPT V3 — release report, 2026-10-31

## Declaration
**V3 COMPLETE — all defined release criteria are satisfied.** (Historical record. At the time V2 and V3 were tagged in the
private development repository; both are included in the public release.)

## 1. Competitive performance
Qwen3-1.7B, batch-1 greedy decode, RTX 4060 8 GB, 4 fresh-process interleaved rounds, idle GPU (`runs/20261031_v4/v3_final/`):
**EfficientGPT V3 111.9 tok/s: parity with llama.cpp Q8_0 (1.018× [1.015, 1.051]; below the 5% bar, so not claimed as a
win), 1.33× torchao int8 max-autotune, 1.89× HF bf16 + static cache/torch.compile.** Re-verified on the final code:
111.1 vs 107.9 tok/s, 1.029× [1.013, 1.076] (`runs/20261031_v4/v3_verify/`). Found by profiling: the int8 GEMV was already
at the DRAM limit; the gap was ≈ 2 ms/token of GPU idle time: ≈ 0.6 ms in HF `generate()`'s host loop, the rest at kernel
boundaries (one extra reduce kernel per linear) — refined later (`reports/EFFICIENTGPT_TECHNICAL_REPORT.md` §2). Not closed: time to
first token (54 vs 33 ms).

## 2. Independent reproducibility
| Check | Result | Evidence |
|---|---|---|
| Clean checkout (fresh `git worktree` of the final commit, hash-checked inputs) | 149/149 OK, 0 skipped | `runs/20261031_v4/v3_clean_checkout_final.json` |
| Fresh-process trials with CIs | 4 rounds + 3-round re-verification, round-bootstrap CIs | `v3_final/`, `v3_verify/` |
| Memory | device memory per process (int8 2474 MiB, llama.cpp 2425 MiB, torchao 3432 MiB, bf16 3426 MiB) | same |
| Output quality / precision | quality-v1 39/52 = bf16; dNLL −0.15%; greedy prefix vs bf16 18.8 tokens (bf16-vs-bf16 45.8) | same, research report §5 |
| Benchmark fairness | baselines at their best config (torchao max-autotune), same prompts / tokens / formula, Ollama identity recorded | research report §6 |
| Pinned environment | `requirements-lock.txt` (a fresh-venv install was not run: ≈ 3 GB of downloads) | `REPRODUCE.md` |

## 3. Documentation
| Item | Where |
|---|---|
| Technical architecture | `README.md` (diagrams), research report §3 |
| Original engineering contributions | research report §3; `reports/EFFICIENTGPT_TECHNICAL_REPORT.md` §5 |
| Reproducible benchmarks + strong baselines | `BENCHMARK_REPORT.md` §1a; `REPRODUCE.md` |
| Code correctness evidence | tests (149), mutation tests, decoder equivalence tests, research report §5 |
| Known limitations | `README.md` §Limitations; research report §8 |
| Short demonstration | `python -m research.realmodel.demo` (`runs/20261031_v4/v3_demo_run.txt`) |
| Concise engineering research report | `reports/EFFICIENTGPT_V3_RESEARCH_REPORT.md` |

## 4. Audit
Independent review of all V3 code (separate agent): 7 findings, none changed a published number; all fixed with regression
tests (EOS set, multi-byte streaming, batch/length guards, bounded per-bucket decoders, capture time, contract wording,
baseline wording). Defects found by the release process itself: a fixed-size cache that made the integrated graph path 1.9×
slower (benchmark), and an undeclared test input (clean checkout). Both fixed and re-verified.

## 5. Not claimed
Faster than llama.cpp (parity only, via Ollama defaults); anything on other GPUs, Linux, larger models, batched or sampled
decoding; vLLM / TensorRT-LLM comparisons; novelty of the underlying techniques.
