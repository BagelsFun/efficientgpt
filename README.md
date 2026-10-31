# EfficientGPT

EfficientGPT is a personal GPU inference-engineering project on a single RTX 4060 8 GB (Windows 11): a verified CUDA-Graph
decoder with adaptive KV buckets, KernelAgent-Ada (a template-based Triton kernel search with a strict verification protocol),
and a weight-only int8 decode path for a real open-weight model (Qwen3-1.7B) built from KernelAgent-Ada's kernel.
It is a fork of [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) and uses the nanoGPT model code as its starting point.

> Every number below is taken from `BENCHMARK_REPORT.md` or the report named beside it, each with raw evidence in the repo.
> No company has expressed interest in or endorsed this work, and no upstream contribution has been submitted.

**Start here (reviewers):**
- [`reports/ONE_PAGE_ENGINEERING_SUMMARY.md`](reports/ONE_PAGE_ENGINEERING_SUMMARY.md): the work in one page.
- [`reports/EFFICIENTGPT_INDUSTRY_RESEARCH_REPORT.md`](reports/EFFICIENTGPT_INDUSTRY_RESEARCH_REPORT.md): the full report.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): how the pipeline works.
- [`docs/DEMO.md`](docs/DEMO.md): four runnable demos.
- [`reports/BENCHMARK_VERIFICATION.md`](reports/BENCHMARK_VERIFICATION.md) and
  [`reports/BENCHMARK_LIMITATIONS.md`](reports/BENCHMARK_LIMITATIONS.md): evidence and limits.
- [`REPRODUCE.md`](REPRODUCE.md): every command;
  [`docs/REPRODUCE_ON_ANOTHER_MACHINE.md`](docs/REPRODUCE_ON_ANOTHER_MACHINE.md): independent reproduction on another GPU / Linux.
- [`docs/UPSTREAM_CONTRIBUTION_CANDIDATES.md`](docs/UPSTREAM_CONTRIBUTION_CANDIDATES.md): measured upstream candidates (none submitted).
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md): upstream and dependency licences.

## Results at a glance

Hardware for all rows: one RTX 4060 8 GB, torch 2.11.0+cu128, triton-windows 3.7.1. Not compared with vLLM or TensorRT-LLM.

| Result | Contract | Evidence |
|---|---|---|
| **Qwen3-1.7B batch-1 decode (V3): EfficientGPT int8 + sync-free CUDA-graph decoder 111.9 tok/s = parity with llama.cpp Q8_0 (1.02x [1.015, 1.051]), 1.33x torchao int8 (max-autotune), 1.89x HF bf16 + static cache/torch.compile; device memory 2474 MiB; quality-v1 39/52 = bf16** | approximate (dNLL -0.15%) | `reports/EFFICIENTGPT_V3_RESEARCH_REPORT.md`; `BENCHMARK_REPORT.md` s1a |
| Eager prefix SDPA to `adaptive_graph` (CUDA Graph, adaptive KV buckets), B16 prompt 64, 64 new tokens: 131.9 ms to 52.8-56.2 ms | exact (bit-identical tokens and logits) | `BENCHMARK_REPORT.md` s1; `reports/PHASE5_REPORT.md` s1 |
| `adaptive_graph` to `compiled_triton_full_graph`: 1.29-1.74x (95% CI, all 12 workloads); 33.5 ms at B16 | near-tie | `reports/PHASE5_REPORT.md` s1, s5 |
| `compiled_triton_full_graph` to fp16 weights: 1.39-1.80x; 20.4 ms at B16; dNLL -4e-6 | approximate | `research/phase5/ci_runs/20261030_212013_precision/` |
| KernelAgent-Ada task winners vs `torch.compile`: 19/19 held-out shapes won (1.07-9.3x), warm-L2 microbenchmarks | correct vs fp64 truth | `runs/20261031_v4/ka_study_evalv2/heldout_*.json`; `reports/KERNELAGENT_ADA_REPORT.md` |
| Decode attention vs the existing Phase 5 Triton kernel (held-out): 3 WIN (1.16-1.58x), 1 SLOWER (0.83x); end to end only 0.999-1.015x at batch 16 (about 1%, below the 5% bar) | near-tie | `reports/KERNELAGENT_ADA_REPORT.md` (KA-04); `research/phase5/ci_runs/20261031_dispatch_inprocess/` |
| HF Qwen3-1.7B bf16 batch-1 decode: 26.8 to 57.7 tok/s with a static cache (2.15x, CI [2.14, 2.16]), about 93% of measured DRAM bandwidth | speed only; outputs 40/52 identical, equal quality | `docs/ASSISTANT.md`; `runs/20261031_v4/assistant_speed_idle/` |
| Local assistant, repo questions with tools (20 audited items): qwen3.5:4b 20/20, Qwen3-1.7B nf4 17/20 | graded against fixed answers | `BENCHMARK_REPORT.md` s3 (ASSIST-05) |

Bottleneck on the 12.19M TinyStories model: GEMMs are about 65% of the B16 step at 78-83% of the measured 211 GB/s DRAM
bandwidth, so fewer weight bytes (fp16) is the lever (`BENCHMARK_REPORT.md` s1).

## Architecture

Decoder (`research/phase5/`, details in `reports/PHASE5_REPORT.md` s2):

```
request: prompt [B, p], budget n   (ragged: right-padded + lengths)
   |
   v
DecoderPool.get(B, capacity)   capacity = 256 (full) | smallest of 80/128/192/256 >= p+n (adaptive)
   |                           one cached decoder per key
   v
prefill (eager) -> K/V copied into cache slots [0, p) -> first token
   |
   v
CUDA Graph replayed n-1 times (all state on GPU)
   step = Inductor-compiled step + Triton split-K attention over slots <= position
   embed -> per layer: write K/V, attend, MLP -> logits
   |
   v
sampling (argmax, or inverse CDF with caller-drawn uniforms) -> EOS / ragged handling -> tokens
```

KernelAgent-Ada (`research/kernelagent/`, `reports/KERNELAGENT_ADA_REPORT.md`):

```
task (decode hot path + dev shapes + held-out shapes)
   |
   v
template candidates (families x parameters, 4-20 points; no LLM generation)
   |
   v
AST screen (rejects e.g. aliased-import bypasses, input writes)
   |
   v
isolated subprocess:
   correctness vs fp64 truth + edge cases (candidates get cloned inputs)
   CUDA-graph timing vs eager / Inductor / existing kernel, bootstrap CI
   |
   v
search strategies (single_shot, random, local, local_meta; equal budget 12)
   |
   v
held-out shapes, evaluated once  ->  end-to-end check (Amdahl) in the decoder
```

## What is original vs established

Established (reimplemented and verified here, not new): CUDA-Graph decode with static KV, split-K decode attention
(FlashDecoding family), fp16 weights, template search in the spirit of Triton autotune. Positioning against vLLM,
TensorRT-LLM, gpt-fast, FlashInfer, transformers' `DeferredStopCheck` and LLM kernel agents is in `reports/EFFICIENTGPT_INDUSTRY_RESEARCH_REPORT.md` §3: the systems here are not
claimed to be faster than those (not compared).

Modest original contributions (`reports/EFFICIENTGPT_INDUSTRY_RESEARCH_REPORT.md` §5):
- The verification protocol around automated kernel search: fp64-truth contracts by task kind, mutation testing of the
  evaluator itself, input-mutation guards, contention-aware timing, held-out selection, and an end-to-end check that
  correctly reduced a 1.1-1.6x kernel win to about 1% of the system.
- Adaptive KV-capacity buckets with exact CUDA-Graph replay as a combined, tested design (the components are known).
- A measured interaction between a calibration prompt and tool use in small local models (an engineering finding, not a
  research claim).

## Correctness and measurement methodology

- **Contracts** (`docs/PHASE5_CONTRACT.md`): exact (bit-identical tokens and logits vs an independent reference; the
  default modes), near-tie, and approximate (for example fp16 weights). Each mode states its contract and has tests.
- **Statistics:** raw per-request timings; two-level bootstrap CIs on ratios of medians; A/A controls (identical work)
  to expose measurement offsets; promotion rule is ratio >= 1.05 with the CI excluding 1, reproduced in fresh processes.
- **Offsets:** cross-process runs showed about +-3% per-process offsets, so sub-5% questions use the in-process
  interleaved benchmark (typical CI half-width 0.5-1%; `reports/PHASE5_REPORT.md` s3).
- **Contention detection:** timing requires an idle GPU; the harnesses flag or refuse contended runs. The KernelAgent-Ada
  study had 138 evaluations, 0 contended, 138/138 correct (`reports/KERNELAGENT_ADA_REPORT.md`).
- **Mutation tests:** 12/12 deliberately wrong candidates rejected; 12 GPU + 10 CPU kernel-search tests (`BENCHMARK_REPORT.md` s2).
- **Independent reviews and self-caught errors:** four independent code reviews by a separate agent (KA-02, KA-06, the V1
  pre-release audit, and the R1 int8 review), every finding fixed with a regression test or stated as a limitation. Errors kept on record: benchmark answer leakage (ASSIST-02),
  launch-bound microbenchmarks (KA-03; eval-v1 discarded), contention artifacts (ASSIST-04), classifier mislabels
  (ASSIST-05, KA-06). See `docs/EXPERIMENT_LOG.md`.
- **Test counts:** final run 144/144 OK, 0 skipped (Phase 5 86, KernelAgent-Ada + int8 27, assistant 31; `runs/20261031_v4/v2_final/`; pre-release run 140/140,
  `runs/20261031_v4/v1_release/`); the Phase 5 headline benchmark was re-run in fresh processes and reproduced (48/48 WIN,
  `research/phase5/ci_runs/20261031_v1_repro/`).

## Repository map

| Path | Contents |
|---|---|
| `research/phase5/` | Graph decoder, DecoderPool, EOS, sampling, ragged batches, Triton attention, in-process benchmark, CI analysis; `ci_runs/` holds raw timings and CI reports |
| `research/kernelagent/` | KernelAgent-Ada: templates, AST screen, isolated evaluator, search strategies, study runner, mutation tests |
| `research/realmodel/` | Weight-only int8 linear (torch.library custom op + Triton kernel), model surgery, sync-free CUDA-graph greedy decoder (`fast_decode.py`), decode profiler, config sweep, end-to-end benchmark and analysis for Qwen3-1.7B |
| `research/showcase/` | Demo 2 (performance comparison) and Demo 3 (KernelAgent-Ada walkthrough) |
| `research/release/` | `clean_checkout.py` (every test suite from a pristine `git worktree`, hash-checked inputs), `gpu_census.py` (which tests need the GPU), `make_public_export.py` (the sanitiser that produced this public tree) |
| `assistant/` | Local assistant (Ollama / Hugging Face backends, retrieval, permissioned tools) and its benchmarks |
| `reports/` | Research report, one-page summary, benchmark verification and limitations, `PHASE5_REPORT.md`, `KERNELAGENT_ADA_REPORT.md`, V2/V3 release records |
| `docs/` | `ASSISTANT.md`, `PHASE5_CONTRACT.md`, `ENVIRONMENT.md`, `EXPERIMENT_LOG.md` (append-only), `ROADMAP.md`, checkpoint checksums |
| `runs/` | Evidence directories (for example `runs/20261031_v4/`) |
| `model.py`, `train.py`, `sample.py`, ... | nanoGPT lineage and early benchmark scripts |

## Reproduce

Full instructions, inputs not in git and checksums: `REPRODUCE.md`; environment: `docs/ENVIRONMENT.md` (Python 3.11 venv,
torch 2.11.0+cu128, triton-windows 3.7.1). Check `nvidia-smi` shows about 0% utilisation before any timing.
On another GPU or on Linux: `docs/REPRODUCE_ON_ANOTHER_MACHINE.md` (Qwen3-1.7B benchmark; no private inputs needed).

```
python -m unittest research.phase5.test_pool research.phase5.test_graph_gpu research.kernelagent.test_evaluate research.kernelagent.test_search
python -m research.phase5.benchmark_inprocess --modes adaptive_graph compiled_triton_full_graph prefix_sdpa_eager --output research/phase5/ci_runs/<stamp>/benchmark_raw.csv
python -m research.kernelagent.run_study --out results/kernelagent/study_<stamp> --busy-pct 15
python -m research.realmodel.bench_int8 --out runs/<stamp>/int8_e2e --rounds 4 --quality && python -m research.realmodel.analyze_int8 --dir runs/<stamp>/int8_e2e
python -m research.realmodel.demo        # Qwen3-1.7B int8 + graph decoder, streamed, with a bf16 agreement check
python -m research.release.clean_checkout --ref HEAD --out runs/<stamp>/clean_checkout.json
python -m assistant.demo
```

## Limitations

- On the real model, V3 decode is at parity with llama.cpp Q8_0 (via Ollama defaults, not a tuned build; +2%, below the
  5% bar), but llama.cpp's time to first token is better (33 vs 54 ms). Batch 1 and greedy only; sampling and batching use
  the slower HF `generate()` path. vLLM and TensorRT-LLM were not run. Weight-only int8, CUDA graphs and static KV caches
  are established techniques.
- int8 prefill is not optimised (TTFT 65 vs 45 ms) and int8 outputs diverge from bf16 more than bf16 implementations do from
  each other (greedy common prefix 22.9 vs 42.9 tokens), without a measured quality loss on these suites.
- Phase 5 decoder results are on a 12.19M-parameter TinyStories model, which is research scaffolding and not a general LLM.
- One machine, one GPU (RTX 4060), Windows 11; nothing verified on other GPUs or Linux. No external users yet.
- KernelAgent-Ada timings are warm-L2 microbenchmarks (inputs reused), not end-to-end; search spaces are small (4-20
  points), budget 12, two seeds; the profiler-guided ordering showed no benefit (inconclusive).
- The KernelAgent-Ada attention winner is about 1% end to end at batch 16; the GEMMs, not attention, dominate that step.
- The assistant uses small local models (Qwen3-1.7B, qwen3.5:4b) and a 52-item project benchmark; it measures relative
  progress on this machine, not general capability. Some earlier assistant speeds were measured under external GPU load
  and are superseded by the idle numbers.

## Status

V3 complete (`reports/V3_RELEASE.md`). This repository is a single-commit, sanitised snapshot of the private research
repository: its development history, the `v2` / `v3` tags and the private TinyStories checkpoint are not included (local
paths in logs are shown as `<HOME>`, and commit ids of the private history as `<private-commit>`). The evidence files
referenced throughout are included.

## Credits

Built on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy (MIT); the original README is preserved in
`docs/NANOGPT_README.md`. Uses PyTorch, Triton (triton-windows), Hugging Face transformers, Ollama, and Qwen models (Apache-2.0).

## Licence

MIT (`LICENSE`): nanoGPT © 2022 Andrej Karpathy; EfficientGPT additions © 2026 BagelsFun. Dependency licences: `THIRD_PARTY_NOTICES.md`.
