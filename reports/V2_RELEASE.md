# EfficientGPT V2 — release report, 2026-10-31

## Declaration
**V2 COMPLETE — all mandatory release checks passed.**
A passed check means the evidence exists and was independently reviewed, not that every comparison was won: on the real
model, llama.cpp Q8_0 is faster than EfficientGPT's int8 path, and that result is reported as measured.

## Gate evidence
| # | Gate | Status | Evidence |
|---|---|---|---|
| 1 | Verified inference implementation | PASS | Phase 5 exact / near-tie / approximate contracts, 86 tests; `reports/PHASE5_REPORT.md` |
| 2 | Fair end-to-end benchmark, practical open-weight model, strong baselines | PASS | Qwen3-1.7B batch-1 decode, fresh-process interleaved rounds, idle GPU: **ours 91.3 tok/s = 1.54× HF bf16 + static cache/compile [1.54, 1.56], 1.10× torchao int8 weight-only (default [1.10, 1.11]; max-autotune [1.08, 1.16] on 4 clean rounds), 11.7× bitsandbytes LLM.int8, 0.83× llama.cpp Q8_0 [0.83, 0.84]**; device memory 2.49 GB (bf16 3.45, torchao 3.43, llama.cpp 2.42); quality-v1 39/52 = bf16. `BENCHMARK_REPORT.md` §1b |
| 3 | KernelAgent-Ada demonstration | PASS | 6-task study (held-out 19/19 vs torch.compile); attention end-to-end check (≈ 1%, correctly not promoted); real-model `int8_gemv` task (62/62 correct, held-out 20/20) whose winner is the shipped kernel |
| 4 | Reproducible measurements + correctness | PASS | Phase 5 headline reproduced in fresh processes (48/48 WIN); R1/R2 round-level CIs; per-process idle checks; one perturbed round identified and reported |
| 5 | Independent review + regression | PASS | 6 independent reviews (KA-02, KA-06, V1 pre-release, R1 code, R1 benchmark, R2 fairness), findings fixed or stated; final regression **144/144 OK, 0 skipped** on the final code (`runs/20261031_v4/v2_final2/`) |
| 6 | README, architecture, portfolio | PASS | `README.md` (results, architecture diagrams, methodology, map), `BENCHMARK_REPORT.md`, `CAREER_VALUE_REPORT.md`, `REPRODUCE.md` |
| 7 | Honest limitations + original contributions | PASS | `README.md` §Limitations; `reports/EFFICIENTGPT_TECHNICAL_REPORT.md` §3, §5; this report |

## What a reviewer should know
- **Established techniques, verified implementations:** CUDA-Graph decode, split-K attention, weight-only int8. The
  differentiators are the verification protocol (fp64-truth contracts, mutation-tested evaluator, held-out shapes,
  contention-aware timing) and honest end-to-end accounting.
- **Where we lose:** llama.cpp Q8_0 decodes 1.20× faster than our int8 path on this GPU (ours ≈ 74% of the DRAM bound end to
  end, llama.cpp ≈ 93%); our int8 TTFT is slower than bf16 (67 vs 46 ms; prefill dequantises); our greedy outputs diverge from
  bf16 earlier than torchao's on 4/16 prompts (NLL and quality-v1 equal).
- **Measurement incidents, disclosed:** a `git push` overlapped one benchmark process (round 0 of the max-autotune run; shown
  both with and without it); torchao's max-autotune speed varies 79–89 tok/s across processes.
- **Not verified:** other GPUs, Linux, larger models, vLLM / TensorRT-LLM. No external users.
- **Publication state (at the time):** private development repository only; published later as part of the public release.

## Deferred (outside V2)
Closing the gap to llama.cpp (better int8 GEMV, int8 prefill), distillation / QLoRA (no training GPU time without a demonstrated
benefit), cold-L2 kernel timing, other hardware, upstream contributions.
