# KernelAgent-Ada: first study (2026-10-31)

Template-based Triton kernel search with isolated evaluation, fp64-truth correctness, CUDA-graph timing against
eager / Inductor / the existing Phase 5 kernel, equal-budget strategy comparison, and one held-out evaluation per task.
No LLM generates kernels; "agent" here means the search controller. Evidence: `runs/20261031_v4/ka_study_evalv2/`
(per-run JSON, history, held-out JSON, study.log); log entries KA-01..KA-03 in `docs/EXPERIMENT_LOG.md`.

## Setup
- RTX 4060 8 GB, Windows 11, torch 2.11.0+cu128, triton 3.7.1 (triton-windows). GPU utilisation 0–1% before
  every timing (owner's apps closed; distillation paused). 138 evaluations, 0 contended, 138/138 correct.
- 6 tasks from EfficientGPT decode hot paths: decode attention (6 heads x 64), add+LayerNorm (384), temperature softmax,
  top-k(50) softmax, inverse-CDF sampling, argmax (vocab 4096/8192). Dev shapes for search, disjoint held-out shapes once.
- Correctness: float tasks err <= 4x eager-fp32 err vs fp64 + floor; argmax exact; sampling within fp64 CDF interval;
  plus edge cases (len-1 / full cache, zeros, x50 logits, duplicated maxima, sparse probabilities, u at 0 and 1-2^-24),
  candidates get cloned inputs (writes to inputs fail). Mutation tests: 12/12 rejected as intended.
- Timing (protocol eval-v2): each callable captured as 20 back-to-back calls in one CUDA graph; 8 trials x 5 reps x 10
  replays, rotating interleaved order; bootstrap CI of the median ratio (`phase5.analyze_ci.ratio_ci`). WIN = ratio >= 1.05
  and CI low > 1.
- **Warm-L2:** inputs are reused across calls, so working sets under the 24 MB L2 stay cache-resident. The "roofline"
  column (vs measured 211 GB/s DRAM) exceeds 1 for many shapes for this reason; read it as an L2-residency indicator.
  Within one decode step the same tensors are often L2-resident too, but this is not an end-to-end decode measurement.

## Held-out results (selected on dev, evaluated once)
| Task | Winner (family, params) | vs eager | vs Inductor | vs existing Phase 5 kernel |
|---|---|---|---|---|
| decode attention | single_pass, BLOCK_N 64, 2 warps | 7.8–11.1x WIN 4/4 | 7.1–9.3x WIN 4/4 | **WIN 3/4 (1.16–1.58x), SLOWER 1/4 (4x256: 0.83x)** |
| add+LayerNorm | row_per_program, 1 warp | 3.3–3.5x WIN 3/3 | 1.07–1.16x WIN 3/3 | — |
| temperature softmax | row_whole, 16 warps | 1.8–2.6x WIN 3/3 | 1.19–1.22x WIN 3/3 | — |
| top-k softmax | bisect_whole_row, 8 warps | 2.9–4.2x WIN 3/3 | 2.5–3.5x WIN 3/3 | — |
| inverse-CDF sample | scan_whole_row, 4 warps | 9.3–13.1x WIN 3/3 | 2.4–2.6x WIN 3/3 | — |
| argmax | row_whole, 8 warps | 2.0–5.4x WIN 3/3 | 1.28–1.42x WIN 3/3 | — |

Per-call medians are 1–16 µs (`heldout_*.json`). The Inductor baseline is `torch.compile(reference, dynamic=False)`
captured the same way. The eager and Inductor decode-attention baselines use SDPA with a materialised boolean mask
(the reference contract); the meaningful comparison for decode attention is the existing Phase 5 Triton kernel.

## Strategy comparison (dev score, geometric-mean µs; budget 12; seeds 0/1 where stochastic)
| Task | single_shot | random s0 / s1 | local s0 / s1 | local_meta |
|---|---|---|---|---|
| decode attention | 5.04 | 4.13 / 4.70 | 4.33 / 4.33 | 4.33 |
| add+LayerNorm | 1.20 | 1.02 / 1.02 | 1.02 / 1.20 | 1.20 |
| temperature softmax | 1.92 | 1.92 / 1.92 | 1.63 / 1.63 | 1.63 |
| top-k softmax | 10.34 | 9.96 / 9.96 | 9.96 / 9.96 | 9.96 |
| inverse-CDF sample | 1.64 | 1.64 / 1.87 | 1.64 / 1.64 | 1.64 |
| argmax | 1.22 | 1.22 / 1.22 | 1.19 / 1.19 | 1.19 |

Findings: searching beyond the family defaults improved the dev score on 5/6 tasks (2–18%; none on inverse-CDF sampling). Local search was as good as
or better than random in 10 of 12 seed-matched comparisons. The profiler-guided ordering (`local_meta`) never beat plain
`local` and was worse on add+LayerNorm: **no evidence that metadata guidance helps** at this budget (one seed only,
small spaces of 4–20 points; inconclusive, not negative).

## Failures and corrections (kept as evidence)
- **eval-v1 was launch-bound** (`runs/20261031_v4/ka_study_evalv1_LAUNCHBOUND/`): one graph launch per timed call put a
  ≈ 7 µs floor under every kernel and compressed all ratios toward 1 (e.g. decode attention "tied" the Phase 5 kernel).
  Detected from roofline fractions > 1 alongside a uniform floor; fixed by capturing 20 calls per graph (eval-v2, protocol
  bump so v1 rows are never reused). v1 numbers are not results.
- 8 evaluator/screen bugs found by independent review and fixed with regression tests (KA-02), including candidates
  writing to their inputs and an `import torch as tl` screen bypass.

## Limitations
- Warm-L2 microbenchmarks, not end-to-end decode; no integration of the winners into the Phase 5 decoder yet.
- Decode attention: the winner is slower than the existing kernel at 4x256, so adopting it needs a shape-based dispatch
  validated on its own held-out set (the current held-out set has now been used once and is spent for selection).
- Small search spaces (template families with 4–20 points); budget 12; two seeds; no LLM-generated candidates.
- One machine, one session; per-process GPU speed offsets of ≈ 3% (Phase 5 OFFSET-02) are below these effect sizes but not
  controlled here.

## End-to-end follow-up (KA-04)
The decode-attention winner was ported into `research/phase5/decode_attention.py` behind a shape rule (single_pass iff
B·H >= 48 or cap <= 64, derived from a fresh 24-shape grid) and benchmarked end to end in-process (3 processes, near-tie
contract PASS): **batch 16: 0.999–1.015x vs split-K Triton (INCONCLUSIVE; ≈ 1%), batch 1: 1.000–1.003x (rule keeps split-K;
A/A-like control)**. A 1.1–1.6x kernel gain is ≈ 1% of a GEMM-bound decode step. The dispatch stays opt-in. Lesson for the
agent: rank kernel targets by their share of end-to-end time before searching (the GEMMs, not attention, dominate here).

## Next
1. Point the search at the step's dominant cost (GEMM / fused MLP) or at the assistant model's hot ops, chosen by an
   end-to-end profile share, not by kernel-level speedup.
2. Cold-L2 timing variant (rotate input buffers larger than L2) to bound the warm-L2 effect.
3. Larger spaces and more seeds before any claim about strategy quality.
