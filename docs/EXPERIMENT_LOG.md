# Experiment log (append-only)

## 2026-10-30 — AUDIT-01: baseline truth recovery
- **Git:** EfficientGPT work was entirely untracked on `master` with `origin` = karpathy/nanoGPT. It's now committed on local
  branch `efficientgpt/main` (`<private-commit>`); `origin` renamed to `upstream`. Nothing pushed.
- **CPU tests:** phase4 test_decoder/test_compact, phase5 test_pool/test_prefix, v1 test_demo/test_report: 23/23 OK. **VERIFIED**
- **Prior V1 run `research/v1/runs/20261029_214112150`:** log stops at Round 2 `prefix_sdpa_eager`, no CSV, no error, so it
  was interrupted, not failed. Left untouched as evidence.
- **Phase 5 quick:** `python -m research.phase5.benchmark_requests --quick` rc=0, correctness PASS. **VERIFIED**
- **Adversarial GPU stress (scratch script):** 288/288 exact greedy matches across full/adaptive × tf32/ieee × 2 seeds, plus
  TF32-graph vs IEEE-reference agreement of 3072/3072 tokens. **VERIFIED**

## 2026-10-30 — P5-TEST-01: GPU graph exactness test (issue I4)
- Added `research/phase5/test_graph_gpu.py`: uniform random prompts, long-then-short reuse in a bucket, exact bucket edges,
  n=1, B>1, tf32 and ieee, on a tiny model and on `out-tinystories-v2/baseline-2500.pt`. Skips without CUDA or checkpoint.
- `python -m unittest research.phase5.test_graph_gpu -v`: 2 tests OK in 16.8 s. **VERIFIED**
- Mutation check (scratch only, no repo edit): replacing the attention mask with `locations >= 0` makes `test_tiny_model`
  fail. So the test does detect stale-KV leakage. Log: `results/audit-2026-10-30/mutant.log` (git-ignored).

## 2026-10-30 — V1-FULL-02: full V1 benchmark re-run (issue I2)
- Command: `python -m research.phase5.benchmark_review --all --rounds 2 --trials 5 --variants 3 --output research/v1/runs/20261030_183607000/benchmark.csv`,
  then `python -m research.v1.report ... --require-full`. Commit `<private-commit>` plus the uncommitted test file (no runtime code changed).
- Result: rc=0, 3/3 modes × 12/12 workloads × 2 rounds, exact reference tokens throughout; report status FULL MATRIX. **VERIFIED**
- Highlights (median of round p50s, ms; eager / full / adaptive): B1 8+16: 33.18 / 12.28 / 9.67; B16 64+64: 133.77 / 78.56 / 54.51;
  B16 192+32: 68.44 / 45.76 / 44.57. Peak allocated: eager 267 MiB, full 291 MiB, adaptive 461 MiB.
- GPU 46→52 °C, SM 2805 MHz, ~2.9 GB VRAM used by other processes.
- **Performance classification: INCONCLUSIVE (formal).** Each cell has n=15 requests per round over 2 rounds, and raw timings
  aren't saved, so there's no CI. At prompt=192 both graph modes use the same 256-capacity graph (identical work) yet differed
  by up to 8% across runs. That's the current noise floor. Graph-vs-eager ratios of 1.5–3.8x and adaptive-vs-full gains of
  18–47% at capacity 80/128 repeated across 4 independent runs, so they're likely real, but they're unconfirmed under the
  5% + CI rule. Inductor baseline NOT RUN (no Triton).

## 2026-10-30 — P5-CI-01: raw timings + bootstrap CIs (issue I3)
- `benchmark_review.py` now also writes `<output>_raw.csv` (one row per timed request: round, mode, case, trial, variant, ms).
  The main and memory CSV formats are unchanged. New `research/phase5/analyze_ci.py` runs a two-level bootstrap
  (resample process rounds, paired across modes, then requests within each round) on median(baseline)/median(candidate),
  with a 95% percentile CI. WIN = ratio ≥ 1.05 and CI low > 1. Prompt=192 full-vs-adaptive (both 256 capacity) is an A/A
  control whose CI must contain 1. CPU tests: `research.phase5.test_analyze_ci`, 7/7 OK.
- Run: `python -m research.phase5.benchmark_review --all --rounds 6 --trials 5 --variants 3 --output research/phase5/ci_runs/20261030_184148/benchmark.csv`
  then `python -m research.phase5.analyze_ci --raw .../benchmark_raw.csv --output .../CI_REPORT.md --boots 10000`.
  Commit `<private-commit>` plus the I3 changes (no decoder code changed). rc=0, 304 s, 18/18 mode-rounds with exact tokens,
  3240 raw timings (6 rounds × 15 requests per cell). GPU 46→53 °C, SM 2805–2820 MHz, ~2.9 GB used by other processes.
  Post-run throttle-reasons bitmask 0x400 (sampled while idle, not decoded).
- **A/A control: AA_PASS 2/2.** B1 192+32: 0.982 [0.956, 1.023]; B16 192+32: 0.981 [0.960, 1.016]. Noise floor is now ≈ ±3–4%.
- **Adaptive vs full graph: WIN 10/10** (all capacity-80/128 cells). Range 1.089 [1.047, 1.139] (B1 128+64) to
  1.398 [1.368, 1.457] (B16 64+64). **VERIFIED WIN** under the contract: ≥5%, CI excludes 1, fresh process per round.
- **Graph vs prefix-SDPA eager: WIN 24/24.** Full: 1.48–3.11×; adaptive: 1.45–3.72×.
- Caveats: steady-state only. Capture/init (adaptive ≈ 4.3 s including correctness checks, 8 graphs) and +170 MiB peak
  allocation vs full aren't amortized. Synthetic prompts, greedy, known budgets, one GPU. Inductor baseline still NOT RUN.

## 2026-10-30 — P5-INDUCTOR-01: torch.compile baseline (issue I5)
- Installed `triton-windows==3.7.1.post27` into `.venv` (torch 2.11 pairs with Triton 3.7; dry run first; pip freeze diff = only
  that package; before/after freezes in `research/phase5/ci_runs/20261030_185752_inductor/diagnostics/`). Inductor works without MSVC on PATH. **VERIFIED**
- New `research/phase5/compiled_baseline.py`: `torch.compile(StaticKVDecoder.step, fullgraph=True, dynamic=False)`. This is
  the same masked full-cache step the graph modes replay, with eager prefill. Opt-in mode `compiled_static` via
  `benchmark_review --modes`; the default modes stay the V1 three, so `research/v1/report.py` is unaffected.
- `mode="reduce-overhead"`: **UNSUPPORTED as-is**. Inductor logs "skipping cudagraphs due to mutated inputs"
  (`token.copy_`, `position.add_`) and falls back to plain kernels (`research/phase5/ci_runs/20261030_185752_inductor/diagnostics/smoke_model.log`).
- **Numerics:** compiled greedy tokens aren't bit-exact. Diagnosis (`research/phase5/ci_runs/20261030_185752_inductor/diagnostics/margins.log`): 8/288 random B=16 sequences
  diverged, every one at a reference near-tie (top-1/top-2 margin 1.7e-5..9.7e-4). Compiled-vs-eager max |logit diff| was
  2.3e-3..6.1e-3; typical margins are p1 2.0e-2, median 1.3. Not order- or cache-dependent (`research/phase5/ci_runs/20261030_185752_inductor/diagnostics/repro_order.py`).
  Contract for this mode: pass if rows are equal, or the first divergence has a reference margin < 1e-2
  (`near_tie_match`, CPU tests `test_near_tie` 4/4). Graph modes remain bit-exact.
- Run: `research/phase5/ci_runs/20261030_185752_inductor` (6 rounds × 4 modes × 12 workloads × 15 requests; rc=0, 464 s, 24/24 mode-rounds PASS; compiled_static 3
  near-tie divergences per round out of 306 validation sequences; compile+warmup ≈ 2.0 s per round with a warm
  on-disk Inductor cache, 6.6 s cold in the smoke test; steady peak alloc 267 MiB).
- **Results (bootstrap 95% CI, 10k resamples):** A/A PASS 2/2. Graph modes beat Inductor 24/24: full 1.59–2.61×,
  adaptive 1.57–3.10×. Inductor vs prefix-SDPA eager: B=1 1.18–1.22× faster (WIN 6/6); B=16 INCONCLUSIVE 5, SLOWER 1
  (0.948 [0.935, 0.974] at 192+32). Adaptive vs full WIN 10/10 again (reproduces P5-CI-01).
- **Interpretation:** at this model size decode is launch/dispatch-bound. Fused Inductor kernels still pay one Python
  call per step (~1 ms/step at B=1), and a captured CUDA graph removes it. Next candidate: Inductor kernels captured
  inside the manual CUDA graph (compile, then `torch.cuda.graph`).

## 2026-10-30 — P2-CGRAPH-01: Inductor-compiled step inside the manual CUDA Graph
- New `CompiledGraphDecoder` (`research/phase5/compiled_baseline.py`): `torch.compile` of the step, warmed and compiled on
  the side stream inside `GraphDecoder._capture`, then captured with `torch.cuda.graph`. Plugged into `DecoderPool` via
  `factory=`. Opt-in modes `compiled_full_graph` and `compiled_adaptive_graph`. Dynamo `recompile_limit` raised to 64 and
  `fail_on_recompile_limit_hit` set, because adaptive needs exactly 8 specializations (the default limit) and silent eager
  fallback isn't acceptable.
- **Design 1 (unchanged scatter_ step): FAILED at B=16.** Compiled-in-graph was 1.28–1.6× *slower* than the eager-kernel graph
  (smoke, `research/phase5/ci_runs/20261030_191837_compiled_graph/diagnostics/smoke.log`). The profile (`research/phase5/ci_runs/20261030_191837_compiled_graph/diagnostics/profile_step.log`) shows Inductor functionalizing each `scatter_` cache write
  into a full-cache out-of-place scatter + `copy_` (12 + 12 kernels/step, 0.62 of 1.79 ms GPU time at B=16; negligible at B=1).
- **Design 2 (`step_index_copy`: `index_copy_` cache/history writes):** bit-identical to the scatter_ step in eager (CPU test
  `test_compiled_step`; 40-step GPU check in `research/phase5/ci_runs/20261030_191837_compiled_graph/diagnostics/variant.log`). Inductor reinplaces it into a 0.005 ms row write. B=16 compiled
  step 0.995 ms GPU vs 1.048 ms eager (`research/phase5/ci_runs/20261030_191837_compiled_graph/diagnostics/variant.log`). The `compiled_static` baseline keeps the original step.
- Run `research/phase5/ci_runs/20261030_191837_compiled_graph`: 6 rounds × 4 graph modes × 12 workloads × 15 requests; rc=0, 382 s, 24/24 mode-rounds PASS. Compiled modes
  pass the near-tie rule with 3 near-tie divergences per round / 306 validation sequences. Init incl. compile + capture +
  correctness: compiled_full ≈ 2.2 s, compiled_adaptive ≈ 4.3 s per process (warm Inductor cache). Steady peak alloc is
  unchanged (291 / 461 MiB).
- **A/A controls: 3/4 PASS, 1 FAIL.** Eager-graph pair 0.997 [0.960, 1.020] and 0.986 [0.963, 1.017]; compiled pair B1
  1.017 [0.976, 1.053] PASS, but **B16 192+32 1.037 [1.010, 1.051] AA_FAIL**. That's identical shapes and work in separate
  processes (compiled_full vs compiled_adaptive pools), cause unexplained (candidates: graph-pool memory layout, other
  resident graphs). So the bootstrap CI understates noise by ≈ 4% for compiled-vs-compiled comparisons. Treat gains < 5%
  there as unresolved (the WIN threshold already requires ≥ 5%).
- **Results (95% CI):** adaptive_graph → compiled_adaptive_graph **WIN 12/12**, 1.066 [1.043, 1.094] .. 1.125 [1.095, 1.179].
  full_graph → compiled_full_graph WIN 8 / INCONCLUSIVE 4 (1.033–1.101). full_graph → compiled_adaptive_graph WIN 12/12,
  1.083–1.533. The previous best (adaptive_graph) is beaten in every workload, including prompt=192 where the cache sizes match.
- **Classification:** compiled_adaptive_graph is the fastest measured mode (VERIFIED WIN on steady-state latency). But it's
  **not bit-exact** (near-tie contract), needs Triton, and costs ≈ 4.3 s more init. It stays opt-in. Making it the default
  would change the Phase 5 exactness contract and needs owner approval.

## 2026-10-30 — AA-01: investigation of the P2-CGRAPH-01 A/A failure
- **In-process probe** (`research/phase5/aa_probe.py`; run `research/phase5/ci_runs/20261030_193022_aa_probe`): all four pools (eager/compiled × full/adaptive) built in
  one process, every bucket warmed so the adaptive pools hold their 8 graphs, the 256-capacity graphs timed interleaved
  (ABBA), 10 blocks × 30 requests, block bootstrap. 6 processes alternating pool creation order. **24/24 AA_PASS**, ratios
  0.9959–1.0041, CIs ≈ ±1%. Key-buffer alignment varied (0 to 1.6 MiB mod 2 MiB) with no effect. Pool type, creation order
  and per-decoder memory layout are **ruled out**.
- **12-round separate-process repeat** (run `research/phase5/ci_runs/20261030_193951_aa_12rounds`; 4 graph modes; rc=0, 759 s, 48/48 mode-rounds PASS): the compiled pair now
  passes (B1 0.994 [0.972, 1.014], B16 1.011 [0.985, 1.028]), but the **eager** pair fails at B16 192+32: 0.971 [0.958, 0.990].
  So the failure isn't specific to compiled graphs. Per-round cf/ca ratios for identical work range 0.94–1.05 (SD 0.037)
  at both batch sizes.
- **Slot-in-round effect ruled out:** residual log-latency by process slot, with mode × case removed, is < 0.7% and
  inconsistent in sign across all four multi-round runs.
- **Persisting pair offsets:** eager full/adaptive at B16 192+32 came out 0.981, 0.988, 0.986 and 0.971 in four independent
  runs; compiled pair 1.017, 1.037, 0.994 and 1.011. These are small systematic offsets (≈ 1.5–3%) between separate processes
  running different pools, in opposite directions for the two pairs, and absent in-process. **Cause unresolved** (remaining
  candidates: process-wide state such as total resident graphs/allocations or driver/clock state). Not pursued further.
  It's below the 5% decision threshold.
- **Headline claim re-checked in-process:** at 192+32 (same capacity), compiled vs eager graph is 1.083× at B1
  (1.081–1.088 over 6 processes) and 1.063× at B16 (1.057–1.069), identical for full and adaptive pools. Cross-process runs
  gave 1.091–1.100× (B1) and 1.033–1.098× (B16) for the same comparison. The cross-process offset is ≈ ±3%, as the A/A data says.
- 12-round results: adaptive_graph → compiled_adaptive_graph WIN 12/12, 1.059 [1.044, 1.081] .. 1.120 [1.093, 1.147]
  (reproduces P2-CGRAPH-01). full_graph → adaptive_graph WIN 10/10 at capacity 80/128 (1.076–1.378).
- **Method conclusion:** the round-clustered bootstrap with 6–12 processes doesn't cover process-level offsets of ≈ 2–3%. Even
  a round-level t-interval on the B16 eager pair excludes 1 (per-round mean 0.979, SD 0.019). Claims between modes that differ
  by < 5% need in-process interleaved measurement (aa_probe-style, ≈ ±1% resolution); cross-process CIs remain fine for the
  ≥ 5% gates used so far.

## 2026-10-30 — INPROC-01: in-process interleaved benchmark
- New `research/phase5/benchmark_inprocess.py`: each child process builds every requested mode, validates all workloads
  (exact, or the near-tie rule for compiled modes), then times them interleaved request by request. The mode order rotates
  each request, so positions balance, and build order rotates per process. Clusters = (process, block); CIs via
  `analyze_ci`. Peak memory isn't reported (all modes resident together). Mode construction is now shared through
  `benchmark_review.ModeRunner` (refactor; summary fields unchanged, quick-run check in `research/phase5/ci_runs/20261030_195833_inprocess/review_q.log`).
  CPU tests `test_inprocess` 4/4. Full suite: 43 tests OK.
- Run `research/phase5/ci_runs/20261030_195833_inprocess`: 4 graph modes, 3 processes × 6 blocks × 12 requests per mode and workload, 12 workloads; rc=0, 428 s; validation
  PASS in all 3 processes (compiled modes 3 near-ties each, exact modes 0).
- **A/A 4/4 PASS:** full→adaptive 0.999 [0.990, 1.011] (B1), 1.001 [0.993, 1.012] (B16); compiled_full→compiled_adaptive
  0.996 [0.989, 1.005], 1.003 [0.993, 1.013]. Typical CI half-width is ≈ 0.5–1%, versus ≈ 2–3% (and miscalibrated) across processes.
- **Results (VERIFIED, in-process):**
  - adaptive_graph → compiled_adaptive_graph WIN 12/12: 1.059 [1.048, 1.068] (B16 192+32) .. 1.103 [1.094, 1.114] (B1 64+64).
  - full_graph → compiled_full_graph WIN 12/12: 1.057 .. 1.095.
  - full_graph → adaptive_graph WIN 10/10 at capacity 80/128: 1.079 .. 1.393. compiled_full → compiled_adaptive WIN 10/10: 1.092 .. 1.425.
  - full_graph → compiled_adaptive_graph WIN 12/12: 1.061 .. 1.527.
  - Agrees with the cross-process estimates to within their ≈ ±3% offsets. The Inductor gain is ≈ 6–10% and roughly
    independent of capacity; the adaptive gain depends on capacity as before.

## 2026-10-30 — EOS-01: EOS stopping (plan option A, owner-approved)
- Decisions (owner, 2026-10-30): pad = EOS id 0; fixed steps + mask first; early exit only if measurement justifies it.
- Implementation `research/phase5/eos.py`: `mask_after_eos` + `EOSDecoder` wrapper (works for every mode). Masking
  *after* generation instead of feeding pad inside the captured step is exactly equivalent, because batch rows never
  interact and post-EOS tokens only affect that row's masked outputs. So the captured step and all existing paths are
  unchanged. `generate_reference_fed_pad` implements the plan's feed-pad semantics step by step as the test oracle.
- Tests: CPU `test_eos` 5/5: mask semantics (first EOS kept, later slots pad, prompt EOS ignored, separate pad id,
  no input mutation) and post-mask == feed-pad reference on a tiny model, with a frequent token as a stand-in EOS so stops
  happen (asserts > 0 early stops). GPU `TestGraphPoolEOS` 2/2: full + adaptive pools == feed-pad reference exactly, for
  real EOS 0 and a forced EOS, tiny model and trained checkpoint (budgets up to 240). Full suite: 50 tests OK
  (`research/phase5/ci_runs/20261030_201510_eos/regression.log`).
- **Study** (`research/phase5/eos_study.py`, run `research/phase5/ci_runs/20261030_201510_eos`; rc=0, 94 s): greedy adaptive graph + EOSDecoder on the first p tokens
  of 128 validation stories, budget 256 − p. Exactness spot checks vs the feed-pad reference passed in every config.

  | p (budget) | distinct prompts | rows emitting EOS | length p10/p50/p90 | early-exit upper bound, B=1 | B=16 |
  |---|---|---|---|---|---|
  | 8 (248) | **11 / 128** | 127/128 | 222/222/222 | 12.8% | 9.2% |
  | 32 (224) | 128 | 119/128 (93%) | 94/139/213 | 36.6% | 3.1% |
  | 64 (192) | 128 | 117/128 (91%) | 52/101/179 | 44.6% | 1.4% |

  The upper bound is the mean fraction of decode steps after the slowest row in a batch finished, ignoring prefill and
  check cost. p=8 is low-diversity (short openings repeat; greedy is deterministic), so its numbers reflect 11 prompts.
- **Mask overhead:** B16 8+248, 40 interleaved reps: 297.57 vs 298.15 ms plain (−0.2%, noise). **VERIFIED ≈ 0.**
- **Implication for option B (early exit):** worthwhile only at small batch. B=1 could save up to ≈ 37–45% of decode steps
  for diverse prompts. At B=16 the slowest of 16 rows nearly always runs to budget, so the bound is ≤ 3% before check
  cost: not worth it without per-row eviction (plan option C, out of scope).

## 2026-10-30 — SAMPLE-01: sampling (temperature → top-k → top-p, pre-drawn uniforms)
- Owner decisions (2026-10-30): pre-drawn uniforms + inverse CDF; temperature, top-k and top-p; no repetition penalty.
- **Precondition checked first** (`research/phase5/ci_runs/20261030_202509_sampling/diagnostics/logit_identity.log`): the graph's fixed-cache masked step gives logits
  **bit-identical** to the incremental concat-KV `reference_step` (100/100 steps, TF32 and IEEE). So "token-exact given u"
  is achievable bitwise. (A full-sequence recompute differs by up to 5.1e-3 under TF32, a different GEMM shape; that's not
  the reference used.)
- Implementation `research/phase5/sampling.py`: `SamplingConfig`; `filtered_probs` (top-k masking identical to
  `GPT.generate`, ties at the k-th value kept; top-p = smallest stable-sorted prefix reaching p, top-1 always kept);
  `sample_from` (`searchsorted` on the CDF with the target scaled by total mass, guarded so a zero-mass token can never be
  chosen); `generate_reference_sampled`; `SampledEngine` (first token sampled from prefill with u[:,0], then an on-device
  `step_index` selects u per step) and `SampledGraphDecoder` / `sampled_factory(cfg)` for `DecoderPool`. The step body
  is shared: `decode_logits` was factored out of `step_index_copy` (bit-identity test still passes). `EOSDecoder`
  passes uniforms through. Compiled (Inductor) sampling is **not** implemented: it would need a near-tie rule at CDF,
  top-k and top-p boundaries.
- **Tests:** CPU `test_sampling` 10/10 (transforms, top-k parity with `GPT.generate` masking, inverse-CDF edge cases incl.
  u = 1 − 2⁻²⁴, eager engine == reference given u, k=1 and tiny-p equal greedy). GPU `TestSampledGraphPool` 2/2 (full +
  adaptive pools == reference token for token, also wrapped in EOSDecoder; tiny model with 4 configs and the trained
  checkpoint with 2 configs, budgets up to 200; different u gives different text). Full suite 62 OK.
- **Distribution:** V=8 tiny model, all 64 two-token continuations enumerated exactly, 100k samples, chi-square with cells
  < 5 expected merged, gate p > 1e-3. Our sampler passes 4 configs; nanoGPT's `GPT.generate` (multinomial) passes against the
  same enumerated probabilities (temperature/top-k configs), so semantics agree. Over 20 seeds per config, p-values look
  uniform (median 0.42–0.57; 0–2/20 below 0.05; `research/phase5/ci_runs/20261030_202509_sampling/diagnostics/pvalue_uniformity.log`). **Power** (`research/phase5/ci_runs/20261030_202509_sampling/diagnostics/mutation_power.log`):
  ignoring temperature, top-k + 1 and top-p keeping one extra token are each rejected (p ≈ 0, plus 4.7k–19k samples in
  zero-probability cells for the masking mutants) in every config where that transform binds.
- **Cost vs greedy** (`sampling_study.py`, run `research/phase5/ci_runs/20261030_202509_sampling`, 2 processes, in-process interleaved, uniform draws included, adaptive
  graphs, 32-token real story prompts; ratio CIs ≈ ±0.5%): top-k 50: +3.1–7.5%; top-p 0.9: +5.6–9.5%; top-k 50 + top-p
  0.95: +7.8–13.2%. Relative cost is largest at B=1 and short budgets (≈ 0.05–0.08 ms extra per step); the two processes
  agree to within ≈ 1 point. Exactness spot checks on real prompts passed in every config.
- **EOS under sampling** (B=16, budget 224, 128 stories): rows emitting EOS 85–96% (greedy 93%); median length 145–162 (greedy
  139). It varies with the seed: T1.0_p0.9 gave 90.6% in one run and 85.2% in the other.

## 2026-10-30 — RAGGED-01: ragged batches with per-row positions (plan option A, greedy)
- Implementation `research/phase5/ragged.py`: right-padded prompts [B, Pmax] plus `lengths` [B]; output is the generated
  tokens [B, count]. Cache slot = absolute position per row. `RaggedEngine.step` uses per-row `wpe(position[b])`, per-row
  `scatter_` writes and mask `slot <= position[b]`. `ragged_prefill` is one batched causal forward plus a gather of each
  row's last real token. With right padding, causal attention alone keeps real tokens from seeing pads; the explicit length
  mask I first wrote was redundant (an equivalent mutant exposed it) and was removed. `RaggedGraphDecoder` is one graph per
  (batch, capacity) for any length mix, via `DecoderPool(factory=RaggedGraphDecoder)` with buckets on Pmax + count.
  Not implemented: ragged + sampling, ragged + compiled, early exit.
- **Contract:** graph == eager engine exactly; equal lengths == existing uniform decoders exactly; each row == batch-1
  `generate_reference` under the near-tie rule (batch-1 vs batch-16 GEMMs are not bit-identical).
- **Tests:** CPU `test_ragged` 5/5: validation; prefill logits/KV vs batch 1 (atol/rtol 1e-5); **pad invariance** (tokens
  and every valid K/V slot bit-identical for 3 pad fills); uniform == CompactStaticKVDecoder; per-row near-tie with > 50%
  exact rows. GPU `TestRaggedGraphPool` 2/2: full/adaptive pools, 7 (tiny) and 5 (trained) length mixes reusing graphs;
  exact vs eager; uniform == existing pool; per-row near-tie. Full suite: 69 tests OK.
- **Mutation check** (`research/phase5/ci_runs/20261030_204002_ragged/diagnostics/mutants.log`): prefill gathering the last padded slot, finish ignoring lengths, a shared position
  embedding and a mask using the batch-max position are all killed. The mask mutant was first caught only by a weak
  token-only pad test (1/5 rows changed); the pad test now compares K/V state and catches it.
- **Study** (`research/phase5/ragged_study.py`, run `research/phase5/ci_runs/20261030_204002_ragged`, 2 processes × 8 batches of 16 real story prompts, lengths
  U[8, 128], budget 128, adaptive pools, in-process interleaved; block = batch):
  - ragged vs 16 sequential batch-1 calls: **7.96× [7.87, 8.06]** and **7.97× [7.89, 8.09]** (≈ 12.9k vs 1.62k generated tok/s).
  - ragged vs a uniform B=16 call at Pmax (cost reference, different output): 0.993 [0.983, 1.000] and 0.995 [0.983, 1.004],
    so ragged costs within ≈ 1% of a same-shape uniform batch.
  - agreement with per-row decoding: 123/128 rows bit-exact + 5 near-ties (run 1), 128/128 bit-exact (run 2); 0 rows outside the rule.

## 2026-10-30 — KERNEL-01: Triton split-K decode attention (/architect)
- **Bottleneck characterization** (`research/phase5/ci_runs/20261030_205141_triton_attn/diagnostics/profile_matrix.log`; model: 6 layers, 6 heads, D=64): per-step GPU time and attention
  share. B1: 0.473 ms / 21% at cap 80 up to 0.577 ms / 35% at cap 256. B16: 0.735 ms / 42% up to 1.043 ms / **60%** at cap 256.
  The masked SDPA (cutlass mem-efficient FP32, 64-query tiles for 1 query) scales with **capacity, not valid length**
  (profiled at positions 8–30). Measured achievable DRAM bandwidth: copy and read 211 GB/s (512 MiB, cache-cold); used as
  the roofline instead of the spec sheet. At B16 cap 256, K/V is 75.5 MB/step, so ≥ 0.357 ms at 211 GB/s (theoretical);
  SDPA took 0.626 ms. At B1, 4.7 MB/step (0.022 ms) vs 0.200 ms, so latency/occupancy bound (6 CTAs on 24 SMs).
- **Design 1** (`research/phase5/decode_attention.py`): Triton split-K (flash-decoding). Grid (B·H, NSPLIT); each
  program reads only slots ≤ position[b] with masked loads, keeps an online softmax in FP32 (elementwise dot, no TF32), and
  writes (m, l, acc) partials; a combine kernel merges them. Config (SPLIT, BLOCK_N, warps) = (64, 32, 4) when B·H ≥ 48,
  else (16, 16, 2), chosen from a small sweep (`research/phase5/ci_runs/20261030_205141_triton_attn/diagnostics/microbench.log`). Wired in via `decode_logits(self, attention=...)`
  (the default path is unchanged; bit-identity tests still pass), `TritonGraphDecoder` (manual CUDA graph) and opt-in modes
  `triton_full_graph` / `triton_adaptive_graph` (near-tie rule). Position may be [1] or [B], so per-row lengths work in the
  kernel (not yet wired into ragged).
- **Numerics** (`research/phase5/ci_runs/20261030_205141_triton_attn/diagnostics/kernel_check.log`, GPU test `TestDecodeAttention`): max |err| vs fp64 1.7e-5 vs SDPA's 7.1e-5 over 40
  cases (L = 1, cap, odd caps, ragged positions, inputs ×8); err ≤ 2× SDPA's (or 1e-6) in all. NaN-poisoned slots past the
  position leave the output bit-identical, so they're never loaded. Triton pools stay within the near-tie rule (tiny + trained
  model). Full suite: 72 tests OK.
- **Kernel-level** (`research/phase5/ci_runs/20261030_205141_triton_attn/diagnostics/microbench_graph.log`; times from CUDA-graph replays). An earlier event-timed run without graphs
  measured host launch latency for the "hot" case and is **discarded** (`research/phase5/ci_runs/20261030_205141_triton_attn/diagnostics/microbench.log`). Per layer call: B1 cap 256 SDPA
  35 µs vs Triton 2.7–3.4 µs (hot). B16 cap 256 SDPA 103–109 µs vs Triton 4.1 µs (L=16), 8.9 µs (L=128), 13.7 µs (L=256),
  hot. Cold (minus a 297 µs flush; ≈ ±5 µs noise, small values unreliable, one implies an impossible 302 GB/s): B16 L=256
  67 µs = 187 GB/s useful (89% of measured copy bandwidth).
- **End-to-end** (run `research/phase5/ci_runs/20261030_205141_triton_attn`, `benchmark_inprocess`, 5 graph modes, 3 processes × 6 blocks × 12 requests; rc=0, 468 s;
  validation PASS, Triton modes 0 near-ties; `CI_REPORT.md` re-analyzed from the raw CSV after adding the Triton comparisons
  to analyze_ci; the original report is kept as `CI_REPORT_initial.md`):
  - adaptive_graph → triton_adaptive_graph **WIN 12/12**: 1.217 [1.184, 1.253] .. 1.608 [1.567, 1.643].
  - compiled_adaptive_graph → triton_adaptive_graph **WIN 12/12**: 1.110 .. 1.476 (previous fastest mode).
  - full_graph → triton_full_graph WIN 12/12: 1.410 .. 2.120; full_graph → triton_adaptive_graph WIN 12/12: 1.423 .. 2.202.
  - triton_full_graph → triton_adaptive_graph: INCONCLUSIVE 10/10 (0.999 .. 1.039), A/A 2/2 PASS. **Because the kernel
    reads only valid slots, capacity bucketing no longer matters.** triton_full_graph gets the same speed with 2 graphs
    instead of 8 (less graph memory; not measured in-process).
  - The earlier A/A pairs still pass: full→adaptive 0.992 [0.981, 1.003], 1.000 [0.991, 1.014].
- **Classification:** VERIFIED WIN on steady-state latency; opt-in, near-tie contract (not bit-exact), FP32 and power-of-two
  head dim only, needs Triton. Next measured bottleneck: GEMMs (≈ 0.28–0.31 ms/step) and unfused elementwise ops (≈ 0.07–0.09
  ms/step eager; Inductor removes most). Candidate: Inductor-fused step with the Triton attention inside.

## 2026-10-30 — KERNEL-02: Inductor-fused step with the Triton attention inside
- `CompiledTritonGraphDecoder` (`research/phase5/decode_attention.py`): `torch.compile(triton_attention_step, fullgraph=True,
  dynamic=False)`, compiled during the side-stream warmup and captured in the manual CUDA graph. Opt-in modes
  `compiled_triton_full_graph` / `compiled_triton_adaptive_graph` (near-tie rule).
- **First attempt failed to compile:** Inductor passes the Python-float `scale` as fp64, which promoted the loop-carried
  softmax state ("m_i ... re-assigned to fp64"). Fix: cast the scaled scores to fp32 in `_split_kernel`; this is a no-op in
  eager, so the `triton_*` modes are unchanged.
- **Warning, checked:** Inductor's mutation analysis of `_combine_kernel` raises "Function argument index out of range" and
  falls back to "assuming every input is mutated". The full kernel list (`research/phase5/ci_runs/20261030_210648_compiled_triton/diagnostics/kernel_list.log`, B16 cap 256) shows no added
  copy kernels: 100 → 83 kernels/step, GPU 0.488 → 0.435 ms. Inductor fused the layer norms, adds, GELU, cache writes and
  argmax; the GEMMs and both attention kernels are unchanged.
- Tests: GPU `test_compiled_triton_pool_within_near_tie_rule` (full + adaptive pools, tiny model); suite 73 OK; Triton and
  analysis tests re-run after a cleanup edit: 15/15 OK (`research/phase5/ci_runs/20261030_210648_compiled_triton/diagnostics/post_cleanup_tests.log`).
- **In-process** (run `research/phase5/ci_runs/20261030_210648_compiled_triton`; adaptive_graph, triton_full/adaptive, compiled_triton_full/adaptive; 3 processes × 6 blocks ×
  12 requests; rc=0, 403 s; validation PASS with 0 near-ties in every mode):
  - triton → compiled_triton **WIN 12/12** for both pools: full 1.059 [1.051, 1.083] .. 1.108 [1.093, 1.122]; adaptive 1.067 .. 1.102.
  - adaptive_graph (best exact) → compiled_triton_full_graph **WIN 12/12: 1.293 [1.218, 1.351] .. 1.737 [1.669, 1.780]**.
  - compiled_triton_full → compiled_triton_adaptive INCONCLUSIVE 10/10 (0.987 .. 1.041), A/A 2/2 PASS (and triton pair A/A
    2/2 PASS), so the full-capacity variant is sufficient.
- **Where the step stands** (B16 cap 256, compiled_triton step 0.435 ms GPU): GEMMs 0.28 ms (65%), attention 0.10 ms, fused
  elementwise ≈ 0.05 ms. GEMM roofline: each step reads all linear weights, 12.2M fp32 params ≈ 48.7 MB, which is 0.231 ms
  at the measured 211 GB/s (theoretical floor). GEMMs measured at 0.28–0.30 ms (B1 gemv and B16 cutlass) are already ≈ 78–83%
  of achievable bandwidth. Further large gains need fewer weight bytes (BF16/FP16 weights: an approximate mode needing a
  quality evaluation and owner approval) or more rows per weight read (larger batches).

## 2026-10-30 — PREC-01: low-precision weights (fp16 / bf16), APPROXIMATE modes
- Mode definition: weights, and therefore the KV cache, stored in fp16 or bf16 (`copy.deepcopy(model).to(dtype)`). cuBLAS
  GEMMs accumulate in fp32; the Triton attention now accepts bf16/fp16 inputs, computes in fp32 and stores in the input dtype
  (GPU test `test_low_precision_inputs`: error vs fp64 ≤ 2× same-dtype SDPA's or 1e-3; the fp32 path is unchanged). Not
  autocast: earlier `benchmark_precision.py` autocast re-casts fp32 weights per op and saves no weight bytes.
- Opt-in modes `{fp16,bf16}_{full_graph,triton_full_graph,compiled_triton_full_graph}` (`benchmark_review.APPROX_MODES`).
  ModeRunner validates them only for well-formed output and reports rows differing from the fp32 reference; no token contract.
  Suite: 74 tests OK (`research/phase5/ci_runs/20261030_212013_precision/diagnostics/regression.log`).
- **Quality** (`research/phase5/precision_study.py`, `research/phase5/ci_runs/20261030_212013_precision/quality.json`):
  - Teacher-forced, 1,024 val windows × 256 tokens: fp32 NLL 2.16393 (ppl 8.7053). **fp16** ΔNLL −4.1e-6 [−1.2e-5, +3.3e-6]
    (−0.0002%), argmax agreement 99.88%, KL 1.8e-6. **bf16** ΔNLL +1.12e-4 [+5.2e-5, +1.7e-4] (+0.005%; detectable, tiny),
    agreement 99.04%, KL 1.1e-4.
  - Free-running greedy, 128 real prompts × 224 tokens vs bit-exact fp32. Control (fp32 compiled_triton): 124/128 identical,
    4/4 divergences near-ties. **fp16**: 105/128 identical, 23/23 near-ties, median first divergence at token 119. **bf16**:
    30/128 identical, 28/98 near-ties, median first divergence at token 64. Continuation NLL scored by the fp32 model shows no
    detectable change for either (fp16 +6e-5 [−3.5e-3, +3.4e-3]; bf16 −7e-5 [−8.2e-3, +7.7e-3]). EOS share: fp16 93% (= fp32),
    bf16 95%.
  - Near-tie rule check (`research/phase5/ci_runs/20261030_212013_precision/diagnostics/near_tie_check.log`): fp16 divergences within the fp32 near-tie rule 9/9 (benchmark prompts) and
    14/14 (random prompts), max margin at divergence 7.8e-3 (< 1e-2, but close). bf16: 28/89 and 29/84, max 6.9e-2, so it fails.
    **fp16 satisfied the near-tie rule on all 46 divergences seen (≈ 580 sequences); this is empirical, not a guarantee.**
- **Speed** (`research/phase5/ci_runs/20261030_212013_precision/speed/CI_REPORT.md`; in-process, 5 modes, 3 processes × 6 blocks × 12 requests; rc=0, 327 s):
  - compiled_triton_full (fp32, previous fastest) → **fp16_compiled_triton_full WIN 12/12: 1.386 [1.357, 1.423] .. 1.802
    [1.789, 1.815]** (largest at B16 / long contexts: weights and KV cache both halve).
  - → bf16_compiled_triton_full WIN 12/12: 1.380 .. 1.769. fp16 → bf16: INCONCLUSIVE 8, SLOWER 4 (0.972–0.982 at B16).
  - fp16_triton_full → fp16_compiled_triton_full WIN 12/12: 1.099 .. 1.172. adaptive_graph (best exact) → fp16_compiled_triton_full
    WIN 12/12: **1.769 .. 2.772**.
- **Classification:** fp16 dominates bf16 on quality and speed. Both are VERIFIED speed wins as **approximate** modes; neither
  is a default. Promoting fp16 to the near-tie contract (or a default) is an owner decision. Overflow wasn't observed (all
  losses finite) but isn't excluded for other checkpoints or inputs.
- **Owner decision (2026-10-30):** fp16 (and bf16) stay opt-in APPROXIMATE modes with no token contract. They aren't put under
  the near-tie rule and aren't a default.

## 2026-10-30 — VERIFY-02: evidence re-check before Phase 5 closure
- Git clean on `efficientgpt/main`; all 7 original checkpoints have unchanged sizes and mtimes; SHA-256 manifest written to
  `docs/CHECKPOINT_MANIFEST.sha256`. Full suite 74/74 OK (`research/phase5/ci_runs/20261030_221132_combinations/baseline_suite.log`).
- **All 636 committed CI rows recompute exactly from their raw CSVs** (max |Δ| 0, no verdict changes;
  `research/phase5/ci_runs/20261030_221132_combinations/verify_reports.py`). No expensive re-runs were needed.

## 2026-10-30 — COMBO-01: Triton attention × ragged and × sampling; KV state and EOS across lengths
- Hooks: `RaggedEngine/RaggedGraphDecoder(..., attention=None)` and `SampledEngine/SampledGraphDecoder(..., attention=None)`,
  `sampled_factory(cfg, attention=None)`. The default (None) path is unchanged, so the exact contracts hold.
- **Triton ragged** (`research/phase5/ci_runs/20261030_221132_combinations/measure_triton_paths.log`, 64 real-prompt rows, lengths U[8,128], budget 128): 0 rows outside
  the near-tie rule; 62/64 identical to the exact ragged engine, 60/64 to batch-1 decoding.
- **Triton sampling: distribution, not token agreement.** Teacher-forced on the trained model (64 real prompts × 159
  steps = 10,176 steps per config; the same state fed to both paths each step), I measured the total-variation (TV) distance
  between filtered next-token distributions. Calibration: exact SDPA with TF32 vs IEEE matmul, two equally "exact"
  implementations (`research/phase5/ci_runs/20261030_221132_combinations/measure_tv_control.json`):

  | config | TF32 vs IEEE (exact): TV median / p99 / max / kept-set changes | exact vs Triton (TF32): median / p99 / max / changes |
  |---|---|---|
  | T1.0 | 2.0e-4 / 6.5e-4 / 1.9e-3 / 0 | 1.3e-4 / 4.1e-4 / 9.4e-4 / 0 |
  | T0.8 k50 | 1.7e-4 / 7.9e-4 / 3.3e-3 / 137 | 1.1e-4 / 5.0e-4 / 2.7e-3 / 72 |
  | T1.0 p0.9 | 1.8e-4 / 1.1e-3 / 5.8e-2 / 339 | 1.1e-4 / 7.5e-4 / 7.8e-2 / 246 |
  | T0.8 k50 p0.95 | 1.4e-4 / 8.2e-4 / 4.3e-2 / 44 | 9.2e-5 / 5.1e-4 / 2.2e-2 / 31 |

  Triton's perturbation is below the precision-policy noise floor in median, p99 and kept-set changes. The max is set by
  rare top-k/top-p cutoff flips in both columns. (A first run mislabeled the Triton column "ieee"; the script was fixed and
  re-run, and the archived JSON is the corrected one.)
- **Tests** (`research/phase5/test_gpu_combinations.py`, 6/6 OK):
  - Triton ragged: tiny + trained model, near-tie vs exact engine and batch 1, pad invariance bit-identical.
  - Triton sampling chi-square through the real graph path: V=8 enumerated two-token distribution, 100k samples, 4 configs,
    p > 1e-3, **df ≥ 5 asserted**. The first version used a sharpened tiny model where 3/4 configs collapsed to a single
    cell (df = 0, vacuous); fixed (scale 1.0, ≥ 9 cells per config), and the CPU test gained the same df guard (its df is
    8–63; it was meaningful).
  - Triton sampling TV vs noise floor (median and p99 ≤ TF32-vs-IEEE on the same states; 2 configs).
  - KV state across lengths: for exact, Triton, compiled-Triton, sampled-exact, sampled-Triton, ragged-exact and
    ragged-Triton decoders, a request after a longer one == the same request on a fresh decoder, bit for bit.
  - Ragged + EOS: on rows identical to per-row decoding (≥ 12/16 required; stops > 0 required), EOS lengths and masked
    tokens == the feed-pad reference.
- **Power** (`research/phase5/ci_runs/20261030_221132_combinations/triton_sampling_power.log`): an off-by-one Triton attention (drops the current key) fails both the
  chi-square and the TV test; the real kernel passes both.

## 2026-10-30 — STARTUP-01: fresh-process memory and cold-start cost per mode
- `research/phase5/startup_study.py` (run `research/phase5/ci_runs/20261030_221343_startup`; rc=0, 314 s): one new process per measurement; **cold** = brand-new empty
  TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR (nothing deleted), **warm** = a cache populated by a priming process; 2 cold
  + 2 warm per mode. "Ready" = process start until decoders for all 12 workloads are built and have served one request.

  | mode | ready cold | ready warm | build 12 workloads cold / warm | graphs | peak alloc / reserved | device used (excl. CUDA context) |
  |---|---|---|---|---|---|---|
  | full_graph | 3.11–3.39 s | 3.11–3.18 s | 0.64–0.68 / 0.65–0.69 s | 2 | 291 / 442 MiB | 478 MiB |
  | adaptive_graph | 3.21–3.26 s | 3.28–3.32 s | 0.76–0.77 / 0.76 s | 8 | 461 / 770 MiB | 658 MiB |
  | compiled_adaptive_graph | **30.91–30.92 s** | 7.15–7.16 s | 28.43–28.44 / 4.66–4.68 s | 8 | 461 / 770–794 MiB | 656 MiB |
  | triton_full_graph | 4.25–4.29 s | 3.17 s | 1.76–1.77 / 0.69–0.70 s | 2 | 291 / 442 MiB | 478 MiB |
  | compiled_triton_full_graph | 14.72–14.74 s | 5.19–5.24 s | 12.26–12.28 / 2.69–2.74 s | 2 | 291 / 442 MiB | 478 MiB |
  | fp16_compiled_triton_full_graph | 15.19–15.21 s | 5.45–5.47 s | 12.59–12.62 / 2.91–2.93 s | 2 | 211 / 286 MiB* | 328 MiB* |

  \*Includes the fp32 reference model that ModeRunner keeps for validation (48.7 MB); fp16-only usage is lower (not
  separately measured). triton_full_graph has full_graph's memory with 2 graphs instead of adaptive's 8 (−180 MiB peak
  vs adaptive) at Triton-attention speed. Steady medians in these single-process runs (n=15, B16 64+64) are indicative only
  (e.g. fp16 21.5–24.6 ms); speed claims stay with the in-process CI runs.

## 2026-10-30 — OFFSET-02: cross-process offsets, same mode in 16 fresh processes
- `research/phase5/offset_study.py` (run `research/phase5/ci_runs/20261030_221921_offsets`; rc=0, 110 s): every process runs **full_graph** only (B1 and B16, 192+32,
  40 requests each), timed by wall clock and CUDA events, with `nvidia-smi -lms 200` telemetry during timing.
- Between-process spread for **identical work**: CV 2.8% (B1) / 2.9% (B16), min–max 1.10×. It's in **GPU time** (CUDA-event CV
  matches wall CV); host overhead is constant at 0.07–0.09 ms. Within-process CV of requests: 2.7–4.4%.
- **Ruled out:** clocks (SM 2805 MHz and memory 8251 MHz in every sample, P2, constant throttle bitmask 0x400); mode/pool (one
  mode only); buffer alignment (key buffer at the same offset mod 2 MiB in every process); host dispatch. **Weight
  placement** is ruled out in-process too (`research/phase5/ci_runs/20261030_221921_offsets/weight_copies.json`): six independent model copies with their own graphs,
  timed interleaved, differ by CV 0.23% (B1) / 0.14% (B16).
- **Process-wide:** a process's B1 and B16 offsets correlate at r = 0.88. Weak correlation with temperature (0.45–0.54, 52→56 °C)
  is confounded with run order (r = 0.44–0.47 with process index) and isn't monotonic.
- **Conclusion:** a per-process GPU-side speed level of ≈ ±3% that affects all kernels and sits below anything the
  application controls (CUDA context / driver state; root cause not identifiable without driver-level tools). This explains
  the AA-01 failures. Policy unchanged: decisions use the in-process interleaved benchmark; absolute cross-process numbers
  need many processes (CV/√n).

## 2026-10-30 — ASSIST-01: local assistant prototype and measured baseline (new primary mission)
- **Stack audit before any install:** Ollama 0.40.12 running with `qwen3.5:4b` (4.2B, Q4_K_M) and `nomic-embed-text` already
  pulled; `.venv` already had transformers 5.21.0. With owner approval: downloaded `Qwen/Qwen3-1.7B` (Apache-2.0, revision
  `70d244cc…`, 3.80 GiB) and installed peft 0.21.2, accelerate 1.15.0, bitsandbytes 0.50.2 and psutil 7.2.2 (dry run first;
  torch unchanged; freezes in `assistant/bench/runs/20261030_224206/`).
- **Code** (`assistant/`): Ollama and HF backends; token-budgeted context; BM25 + optional dense retrieval; permissioned
  tools (read-only auto, execute per-call approval, no write tools, workspace confinement); agent loop with trace; CLI;
  benchmarks. Tests `assistant.test_assistant` (14) + `assistant.bench.test_quality_suite` (3), 17/17 OK. Building the tests
  surfaced and fixed a crash on an unknown tool name requested by the model.
- **assistant-quality-v1:** 52 original items (16 code with hidden tests, 16 integer math, 20 PyTorch/CUDA MCQ), dev/test
  split; reference solutions verified. **Grading v1.1:** after the run, all three models failed `softmax` with a correct
  algorithm because the prompt asked for "only the function" and none imported `math`. v1.1 prepends common stdlib
  imports and was re-applied to the **saved** responses of every model (no new generations; `assistant/bench/runs/20261030_224206/regrade.json`). It
  changed exactly that one item for each model. Two other shared failures (`count_params`, `conv_out_size`) are genuine
  model errors.
- **Results (greedy, thinking off; clean run on an idle GPU, `assistant/bench/runs/20261030_224206/`):**

  | model / backend | quality v1.1 all | held-out test | code / math / mcq | decode tok/s (short / medium / long prompt) | TTFT s (28 / 659 / 2143-token prompt) | memory |
  |---|---|---|---|---|---|---|
  | qwen3.5:4b Q4_K_M (Ollama) | **47/52** | **22/26** | 14/16 · 14/16 · 19/20 | **50.3 / 50.2 / 49.8** | not valid (prefix-cache hits; see below) | 3,209 MiB VRAM (ollama ps); +3,992 MiB device |
  | Qwen3-1.7B bf16 (HF) | 39/52 | 18/26 | 10/16 · 12/16 · 17/20 | 23.2 / 23.1 / 16.9 | 0.087 / 0.201 / 1.191 | 4,292 MiB peak alloc; 4,452 MiB device |
  | Qwen3-1.7B nf4 (HF, bitsandbytes) | 36/52 | 17/26 | 8/16 · 11/16 · 17/20 | 20.2 / 20.2 / 19.6 | 0.159 / 0.272 / 1.205 | 2,300 MiB peak alloc; 2,454 MiB device |

  52 items carry wide uncertainty (one item ≈ 2 points); differences of 1–3 items aren't meaningful.
- **Measurement problems found and handled:**
  1. Ollama's medium/long TTFT (≈ 0.07 s, implied prefill 13.5k–44.9k tok/s) came from Ollama reusing the KV cache of an
     identical prompt prefix across warmup/repeats. Fixed: every speed request now starts with a unique id.
  2. The re-run with unique prefixes (`assistant/bench/runs/20261030_224206/speed_cold_CONTAMINATED/`) coincided with an **external GPU workload**: 65–72%
     utilization and 69–71 °C with none of my jobs running, SW power cap active; decode roughly halved for every backend.
     That run is marked CONTAMINATED and not used. Cold Ollama TTFT is still **NOT MEASURED** (needs an idle GPU).
- **Observations:** the HF path decodes at 17–23 tok/s, well below the ≈ 60 tok/s that reading 3.4 GB of bf16 weights per
  token at 211 GB/s would allow, so it's host/dispatch-bound like the Phase 5 decoder was. nf4 halves memory but isn't faster
  in transformers. Ollama (llama.cpp kernels) on a 2.5× larger 4-bit model is 2.2× faster and scores higher.

## 2026-10-31 — ASSIST-02: agent end-to-end (model-only vs RAG vs RAG+tools) and a benchmark-leakage fix
- Suite `assistant/bench/agent_e2e.py`: 8 questions whose answers are facts in this repo's source/docs, graded by regex
  (pattern self-test in `assistant.test_assistant`). Configs: model_only (no context), rag (retrieved excerpts), rag_tools
  (retrieval + read-only tools; execute tools never approved).
- **Leakage found and fixed:** the v1 run let retrieval and `search_code` see `assistant/bench/` and
  `assistant/test_assistant.py`, which contain the questions and expected answers. Evidence: a tool call surfaced the
  benchmark file's own constant instead of the asked-about source. Fix: exclusion globs in `Retriever` and `Toolbox`;
  protocol **v1.1** hides those files; a unit test proves they are invisible while the real sources stay findable. The v1
  results are kept as `*_agent_v1_LEAKED.json` in `assistant/bench/runs/20261030_224206/` and are **not used**.
- **v1.1 results (accuracy; timings not reported because an external GPU workload, 100% utilization, was running):**

  | model | model_only | rag | rag_tools | tool calls / question (rag_tools) |
  |---|---|---|---|---|
  | qwen3.5:4b Q4_K_M (Ollama) | 0/8 | **8/8** | **8/8** | 1.4 |
  | Qwen3-1.7B bf16 (HF) | 0/8 | 6/8 | 6/8 | 0.5 |

  The leak had inflated Qwen3-1.7B's rag score from 6/8 to 8/8. Model-only is 0/8 as expected for repo-specific facts. The
  behaviours differ: qwen3.5:4b mostly declines ("cannot read your files"), while Qwen3-1.7B often states invented values
  confidently. That's a calibration gap relevant for any fine-tuning target (prefer "use a tool / say unknown").
- Note for future logging: don't write these questions' answers into `docs/` (it is retrievable context).

## 2026-10-31 — DATA-01 / QLORA-00: distillation pilots and QLoRA fit test (Phase C groundwork)
- **Teacher/data pipeline** `assistant/train/distill_data.py`: local `qwen3.5:4b` (Apache-2.0) invents one function task per
  request across 14 topics; kept only if the solution passes the teacher's own tests in the sandboxed grader, survives the
  dangerous-code screen, isn't a near-duplicate, and doesn't collide with any assistant-quality-v1 item (both splits).
- **v1 (JSON output), 80 requests, 1,296 s:** kept 11 (14%); invalid JSON 30; solution failed its tests 39. Data-quality
  bug found on inspection: the teacher sometimes put the function body into the "signature", so the student prompt
  contained the answer. **v1 data is not used for training.**
- **v2 (NAME + solution block + tests block; the student prompt is built from the solution's AST, def line + docstring only,
  so it can't contain the body; tests in `assistant.train.test_train`, 5/5 OK), 80 requests, 1,003 s:** kept **18 (22.5%)**;
  unparseable 14; failed tests 47; duplicate name 1. ≈ 56 s per kept example on this GPU, so 500–1,000 examples would take
  ≈ 8–15 h of generation. Not started (needs the owner's go-ahead; it occupies the GPU for hours).
- Known limitation: kept examples are verified only against the teacher's own tests (correct-by-tests, not by spec). The
  docstring/examples can still be imprecise. A second test-generation pass or stronger teacher would tighten this.
- **QLoRA fit test** (`assistant/train/train_qlora.py`, 5 steps on v1 data, smoke only, adapter in
  `out-assistant/SMOKE-…`, never to be evaluated): Qwen3-1.7B NF4 + LoRA r=16 on all attention/MLP projections =
  **17.4M trainable params; peak 3,436 MiB allocated / 3,816 MiB reserved; ≈ 1.4 s/step** (accum 2, short sequences). It
  fits the 8 GB GPU alongside ≈ 1.7–3 GB of other processes. Loss numbers from 5 steps and 2 validation examples aren't
  evidence of anything. Evidence: `assistant/bench/runs/20261030_224206/phase_c_pilot/`.

## 2026-10-31 — ASSIST-03: HF backend robustness fix and static-cache decode (Qwen3-1.7B bf16)
- **Bug:** `HFBackend` ran `generate()` in a worker thread feeding a streamer. When the thread raised, the caller waited on
  the streamer forever: the first static-cache A/B hung for its full 3,600 s timeout (log kept as
  `static_cache_ab_attempt1_HUNG.log`). The trigger was Inductor's CUDA-graph trees, used by transformers' static cache,
  which assert on thread-local state when called from another thread. **Fix:** `generate()` runs on the calling thread
  with a synchronous streamer that timestamps the first generated token. A forced error now raises promptly. Suite 23/23 OK.
- **A/B** (`assistant/bench/static_cache_ab.py`, one process, interleaved, unique-prefix prompts, 256 new tokens;
  `assistant/bench/runs/20261030_224206/static_cache_ab.json`). External GPU load 29–38% was present (logged). The dynamic baseline matched the earlier
  idle-GPU decode rate (22.6 vs 23.1–23.2 tok/s), so interference was small, but treat the numbers as indicative:
  - decode tok/s: dynamic 22.2/22.6/22.6/22.7 (median 22.6); static 19.4*/34.3/33.9/34.3 (median **34.1**) ⇒ **≈ 1.5×**.
    \*The first static rep recompiles for the new length.
  - first call: 3.3 s dynamic vs **23.4 s** static (compilation). Peak 3,410 MiB allocated.
  - **Equivalence:** 40/52 greedy responses identical; 12 differ (different compiled kernels). Quality v1.1: 39/52 for both.
    Not an exact mode: same class as the Phase 5 near-tie modes; opt-in via `HFBackend(static_cache=True)`.
- Status: INCONCLUSIVE under the 5% + CI rule (4 reps, no bootstrap, partial external load), but a large, consistent
  effect. Re-run on an idle GPU with more reps before making it a default for the HF path.

## KA-01 (2026-10-31) KernelAgent-Ada evaluator + search controller (correctness verified; timings not yet valid)
- New: `research/kernelagent/evaluate.py` (protocol `kernelagent-ada/eval-v1`): fresh subprocess per candidate (timeout,
  `set_per_process_memory_fraction(0.45)`), AST screen, fp64-truth correctness (float: err ≤ 4× eager-fp32 err + floor;
  index: exact; sample: token's fp64 CDF interval contains u·total ± 1e-5), CUDA-graph timing interleaved across
  candidate / eager / Inductor / best existing, bootstrap via `phase5.analyze_ci.ratio_ci`, Triton launch metadata, JSONL history
  keyed by sha256(task+source). `research/kernelagent/search.py`: single_shot / random / local / local_meta at equal budget,
  history reuse, refuses contended timings unless `--allow-contended`, held-out only via `--final`.
- **Correctness smoke** (`runs/20261031_v4/ka_smoke_correctness.json`): all 14 default templates correct on every dev shape and
  edge case (8/8 cases per row task, 15/15 for both decode-attention families). Correctness is load-independent.
- **Mutation tests** (`research.kernelagent.test_evaluate`, GPU, 10/10 OK): argmin-for-argmax, `>=` tie-break (caught by the
  dupmax edge), softmax ×1.001, sampler off-by-one, attention reading one slot past `position`, `import os`, reference
  smuggling, import error, infinite loop (timeout) are all rejected. Search controller: `research.kernelagent.test_search`, CPU, 8/8 OK.
- **Bug found and fixed:** the AST screen let tensor-method reference calls through (`inp['logits'].argmax(-1)` passed as a
  "Triton" candidate). Added `BANNED_METHODS` (allowed only under `tl.`/`triton.`); regression test `test_reference_smuggling_rejected`;
  all 14 templates still pass the screen.
- **Timing path smoke only** (`runs/20261031_v4/ka_timing_smoke_CONTENDED.json`): runs end to end (graphs for all four callables,
  CIs, roofline). GPU was at 100% utilisation from the concurrent distillation job, so the record is flagged `contended`;
  **no speed number from it is a result.**
- Concurrent: `assistant.train.distill_data --n 3000 --seed 2` started 2026-10-31 (owner-approved overnight run), ≈ 13 s/request.

## KA-02 (2026-10-31) Independent review of KA-01: 8 findings, all fixed with regression tests
Reviewer (separate agent, code only) findings and fixes:
1. Screen bypass via `import torch as tl` / any torch alias → only `import triton.language as tl` may alias; `tl`/`triton`/`torch`/`math` cannot be rebound.
2. Method blacklist holes (`.aminmax`, `.exp2`, `.nansum`, …) → tensor attributes are now an **allowlist** (shape/launch plumbing only).
3. **Candidate could overwrite its inputs** (a kernel writing to an input pointer) so eager and fp64 truth were computed on forged
   inputs → the candidate gets clones; any change to the pristine inputs fails the case (`inputs modified`).
4. History cache keyed on family+params only (held-out rows and stale template versions reusable) → reuse requires dev split,
   protocol, sha256 of the current rendering, trial count, uncontended timings.
5. Transient `crash`/`timeout` rows cached forever → only deterministic outcomes are reused.
6. Unknown GPU utilisation (NaN) counted as idle in the evaluator → now counts as contended; `wait_idle` stops at once.
7. `local` could finish under budget (unequal comparison) → leftover budget spent on random untried points; `unused_budget` reported.
8. Held-out winner could be chosen from partial searches → held-out deferred until every search of the task succeeded.
Checked fine by the reviewer: speedup direction/labels, sample-kind CDF rule, index rule, CUDA-graph warm-up/capture, Inductor outside capture.
Tests: `research.kernelagent.test_evaluate` 12/12 (GPU), `test_search` 10/10 (CPU); correctness smoke re-run 14/14
(`runs/20261031_v4/ka_smoke_correctness_postreview.json`). The queued study imports this code when it starts.

## KA-03 (2026-10-31) First KernelAgent-Ada study on an idle GPU; eval-v1 launch-bound artifact found and fixed
- Study 1 (eval-v1, `runs/20261031_v4/ka_study_evalv1_LAUNCHBOUND/`): 137 evaluations, all correct, 0 contended, but every
  kernel sat on a ≈ 7 µs floor and roofline fractions exceeded 1. Cause: one CUDA-graph launch per timed call (launch overhead
  dominates µs kernels; inputs reused ⇒ L2-resident). **Not results.**
- Fix: eval-v2 captures 20 calls per graph (protocol bump; v1 rows never reused). Study 2 (`runs/20261031_v4/ka_study_evalv2/`):
  138 evaluations, 138 correct, 0 contended (GPU 0–1%). Held-out: every task winner beats Inductor on every held-out shape
  (1.07–9.3×, 19/19 WIN); decode attention beats the existing Phase 5 Triton kernel on 3/4 held-out shapes (1.16–1.58×) and is
  SLOWER on 4x256 (0.83×). Search beat the family defaults on 5/6 tasks (2–18% dev); profiler-guided ordering showed no benefit.
- Report: `reports/KERNELAGENT_ADA_REPORT.md` (warm-L2 microbenchmarks; not end-to-end decode).
- Distillation was paused for both studies (kept rows preserved) and resumed with `--resume --start 60`.

## KA-04 (2026-10-31) KernelAgent-Ada decode-attention winner integrated behind a shape dispatch; end-to-end effect ≈ 1%
- Code: `research/phase5/decode_attention.py` gains `decode_attention_single_pass` (production port of the KA-03 winner:
  full strides, bf16/fp16 inputs with fp32 math, fp32 scale), `use_single_pass` / `decode_attention_dispatch`, and
  `DispatchTritonGraphDecoder`; benchmark modes `dispatch_triton_{full,adaptive}_graph` (opt-in, near-tie contract).
- Dispatch grid (`research/phase5/dispatch_study.py`, `runs/20261031_v4/dispatch_grid.json`; idle GPU ≤ 6%, eval-v2 timing,
  max |single − split| 3.6e-7): single_pass WIN at all 12 shapes with B·H ≥ 48 (1.10–1.84×) and at cap ≤ 64 for every batch
  (1.25–1.48×); SLOWER at B·H < 48 with cap 256 (0.73–0.85×) and B=2/cap 128 (0.94×). Rule: single_pass iff B·H ≥ 48 or
  cap ≤ 64. It never selects a measured-slower kernel (misses one win: B=1/cap 128).
- Tests (`TestSinglePassDispatch`, 4 GPU tests incl. near-tie pools on the real 12.19M checkpoint): one fp64 case failed the
  per-case "2× masked SDPA" bound (fp32, B=1, cap 256, ×8 logits, pos 37: 1.68e-5 > 9.6e-6). **The accepted split-K kernel has
  exactly the same error on that input** (1 of 168 cases for both); over 90 random ×8 cases both kernels' worst error is
  2.54e-5 vs SDPA 9.6e-5. The single-pass bound is therefore max(2× SDPA, 1.25× split-K, floor). Regression after the change:
  46/46 OK (`test_graph_gpu`, `test_gpu_combinations`, `test_inprocess`, `test_analyze_ci`, kernelagent search).
- End to end (`research/phase5/ci_runs/20261031_dispatch_inprocess/CI_REPORT.md`; 3 processes, interleaved, correctness PASS,
  0 near-ties): dispatch vs split-K Triton adaptive graph — batch 1: 1.000–1.003× (the rule picks split-K for every batch-1
  bucket ≥ 80, so these rows act as an A/A control: noise ≤ 0.3%); batch 16: 0.999–1.015×, CI above 1 on 4/6 workloads but
  all < 5% ⇒ **INCONCLUSIVE under the WIN rule; a real but ≈ 1% gain**. Reason: attention is a small share of the GEMM-bound
  decode step, so 1.1–1.6× on the kernel moves the step by ≈ 1% (Amdahl). Split-K Triton stays the reference Triton mode;
  dispatch stays opt-in.

## KA-05 (2026-10-31) Independent audit of the KA-04 single-pass tolerance change
- `research/phase5/attention_error_audit.py` → `runs/20261031_v4/attention_error_audit.json`. Kernel: 240 cases per dtype
  (8 shapes × 3 logit scales × 10): fp32 max error single 2.75e-5 = split-K 2.75e-5 vs masked SDPA 1.09e-4 (median 2.1e-6 both
  vs 6.5e-6); bf16/fp16 identical to split-K (output rounding dominates). Per-case single/split ratio p99 1.69, max 1.90 — but
  an extra 480-case sweep shows every ratio > 1.25 has absolute error < 1e-6 (fp32 rounding level), covered by the 1e-6 floor:
  0 violations of max(2× SDPA, 1.25× split, 1e-6), and 0 of the symmetric "each within 2× the other or 2× SDPA" rule.
- Model: teacher-forced decoding of the 12.19M checkpoint, 8 sequences × 256 tokens (batch 1 and 16; 17,408 row-steps) vs the
  same model in fp64: max |Δlogit| SDPA 1.67e-5, split-K 9.8e-6, single_pass 1.30e-5; **0 argmax disagreements** for all three.
- Verdict: the tolerance change is justified (single_pass is in split-K's error class, both better than SDPA), but the KA-04
  test proved it on one seed only. Added `test_bound_is_seed_robust` (3 seeds) and `test_model_logits_vs_fp64` (real
  checkpoint; argmax agreement + logit error within 2× the SDPA path's). `TestSinglePassDispatch` 6/6 OK.

## Decision (2026-10-31): stop kernel search on the TinyStories decoder
Phase 5 profile (2026-10-30, lines above): GEMMs 65% of the B16 step at 78–83% of measured DRAM bandwidth ⇒ a perfect GEMM
kernel buys ≤ ≈ 13% end to end, against cuBLAS; attention dispatch measured ≈ 1% (KA-04). The remaining lever, fp16 weights
(1.39–1.80×), is an approximate mode kept opt-in by owner decision. Further kernel work here fails the Amdahl/stop criteria;
not re-profiled because the step code is unchanged apart from the opt-in dispatch.

## ASSIST-04 (2026-10-31) Idle-GPU assistant speed (supersedes the loaded ASSIST-01/03 speed numbers)
- Distillation paused (resumed at request 260); GPU 0% before. Evidence: `runs/20261031_v4/assistant_speed_idle/`.
- **HF Qwen3-1.7B bf16, static vs dynamic KV cache** (`static_cache_ab --repeats 10 --skip-equivalence`, interleaved, unique
  prefixes, 256 tokens): decode 57.68 vs 26.85 tok/s, **2.15× [2.14, 2.16]** (paired bootstrap); TTFT 45 vs 43 ms; first call
  6.2 s vs 2.8 s (compile; 23 s in ASSIST-03, now cached); peak allocated 3,346 MiB. ≈ 93% of measured DRAM bandwidth for the
  bf16 weights. The utilisation samples inside the script (66% / 58%) are the script's own load (sampled after model load /
  at the end), not external work. Promoted to the CLI default for HF backends (`--no-static-cache` opts out); not exact
  (40/52 identical responses, equal quality score, ASSIST-03).
- **Ollama qwen3.5:4b** (`assistant.bench.run --skip-quality --repeats 5`): decode 71.2 / 70.6 / 69.6 tok/s for
  61 / 713 / 2,229-token prompts; cold-prefill TTFT 84 / 253 / 706 ms; prefill up to 3,507 tok/s; 3.13 GiB VRAM. The ≈ 50 tok/s
  of ASSIST-01 was depressed by an external GPU load.

## ASSIST-05 (2026-10-31) Calibration: unsupported-answer rate, default vs calibrated system prompt (assistant-agent-v1.2)
- Suite v1.2: 20 repo-fact questions (8 from v1.1 + 12 new, answers verified in non-excluded files), configs model_only
  (cannot know the answers) and rag_tools; answers classified correct / abstained / unsupported (wrong value, false claim that
  the fact is absent, fabricated file content, or deflection). Accuracy-only runs, GPU shared with distillation (no timings).
  Evidence: `runs/20261031_v4/agent_v12/` (raw `*.json`, re-graded `*.regraded.json`, `run_all.sh`).
- **Classifier bug found by hand audit and fixed before reporting:** v1 missed "I cannot *directly* read…" abstentions and
  counted "X is not defined in the file" (false) as abstention; qwen3.5 default model-only showed 10 "hallucinated" that were
  really 2. Classifier v2 (`agent_e2e.classify`, regression tests with the real mislabelled answers) re-graded the saved answers
  (`assistant.bench.regrade_agent`). Manual audit of all 40 default model-only answers: 38/40 agree; both disagreements are
  classifier-lenient (hedge + wrong value; "I don't see any function…" false claim) and are counted strictly below, as is one
  fabricated "simulated check" in the calibrated run. Three calibrated nf4 answers abstain while also falsely saying the name
  does not appear in a file never shown: borderline, counted as abstained.
- Results (audited, unsupported / 20; correct / 20 with tools):
  | Model | model_only default → calibrated | rag_tools correct default → calibrated |
  |---|---|---|
  | Qwen3-1.7B nf4 (HF) | **11 → 4** (7 fixed, 0 new; exact McNemar p = 0.016) | 17 → 16 (unsupported 3 → 3) |
  | qwen3.5:4b (Ollama) | 3 → 0 (p = 0.25) | 20 → 20 |
  The single model-only "correct" (qwen3.5, LoRA rank 16) is a guess of a common default, not knowledge.
- Decision: the calibration rule is now the CLI system prompt (`agent.INTERACTIVE_SYSTEM`); benchmarks name their prompt
  (`agent_e2e --system default|calibrated`) so v1.1/v1.2 numbers stay reproducible. n = 20 is small; the rag_tools change of
  −1 for nf4 is within noise and should be re-checked on a larger suite (backlog D3).

## KA-06 (2026-10-31) Second independent review (<private-commit>..<private-commit>): 10 findings, fixed or re-measured
- Agent classifier (5 findings): unanchored number regexes ("4.5", "line 4" matched 4), false-absence claims beating
  abstention, hedged guesses counted as abstained, regrading 600-char truncated answers, crashes counted as unsupported.
  Fix: anchored numbers; `review` and `error` verdicts; full answers saved; truncation guard; human decisions recorded in
  `runs/20261031_v4/agent_v12/manual_labels.json` (7 rows, each with its reason). ASSIST-05 re-graded with every review
  item resolved: **Qwen3-1.7B nf4 model-only unsupported 11 → 4 (7 fixed, 0 new, McNemar p = 0.016)** unchanged;
  qwen3.5:4b 4 → 0 (p = 0.125; was 3 → 0, one lucky right guess now counted as unsupported); with tools 20 → 20 and 17 → 16.
- Distillation resume: original seeding restored for fresh runs; request index `k` stored per kept row; a torn last line
  is skipped *and terminated*, so the next append is not glued to it (regression test `TestResume`).
- Tests: the tiny-model dispatch pool now exercises the split-K branch (block 128, caps 96/128 at batch 1–2); the single-
  pass fp64 bound additionally checks ≤ 4× SDPA independent of split-K. `TestSinglePassDispatch` 6/6, assistant 25/25 OK.
- `static_cache_ab --repeats` default 4 → 10 (bootstrap of medians over 4 reps is too coarse).
- **Dispatch grid re-measured under the deployed condition** (one shared scalar position for all rows, swept over
  cap/8 … 7cap/8; the KA-04 grid used independent per-row positions, where a kernel waits for its longest row):
  `runs/20261031_v4/dispatch_grid_shared_position.json`, idle (≤ 4%), max |single − split| 4.8e-7. Lifetime-average single/
  split speedup: B ≥ 8 (≥ 48 programs) 1.22–1.59× at every cap, WIN at all 4 positions; cap ≤ 64 1.30–1.39× for every batch;
  B·H < 48 with cap 128–256 mixed (0.82–1.20×; single loses at late positions). The rule (single iff B·H ≥ 48 or cap ≤ 64)
  still never selects a slower kernel; it forgoes 2×128 (1.14×) and 4×128 (1.20×). Unchanged, since attention is ≈ 1% of the
  end-to-end step (KA-04).

## ASSIST-06 (2026-10-31) Coding-agent evaluation (assistant-coding-v1) and two agent-loop fixes
- Suite: 10 self-written bug-fix tasks (moving average, median, KV bucket choice, top-k ties, ring buffer, LRU, config
  parsing, byte formatting, overlapping windows, percentile); visible failing tests + hidden edge-case tests; reference fixes
  validated by `TestCodingTasks` (each bug real; the first top-k "bug" was not, because Python's reverse sort is stable,
  and was replaced). `given` = file + failing output in the prompt; `agent` = tools in a throwaway workspace.
- Baseline (`runs/20261031_v4/coding_v1/`, accuracy only): qwen3.5:4b given **8/10**, agent **5/10**; Qwen3-1.7B nf4 given
  **5/10**, agent **0/10**.
- Failure analysis: qwen3.5 agent — 4/5 failures hit the 8-step limit re-reading files and never answered, 1 broken code.
  Qwen3-1.7B agent — 6/10 replied "I don't have access to the files" with no tool call (the calibration rule "say you do not
  know and name the file to read" is read as "do not look"), 4 returned wrong code.
- Fixes (commit <private-commit>): `TOOL_RULE` appended on turns that offer tools ("you do have access … use read_file / search_code /
  run_unittests"), and a forced final no-tool answer at the step limit. Benches gained `--no-tool-rule --no-force-final` to
  reproduce earlier runs. Re-measurement in `runs/20261031_v4/assist06/` (below).
- **Re-measurement with both fixes** (`runs/20261031_v4/assist06/`, accuracy only):
  | | before (coding_v1) | after |
  |---|---|---|
  | qwen3.5:4b coding agent | 5/10 (4 failures: no answer at step limit) | 5/10 (0 no-answer; all 5 failures are wrong code) |
  | Qwen3-1.7B nf4 coding agent | 0/10 (6 "no access" refusals) | **3/10** (3 fixed, 0 lost; McNemar p = 0.25, n = 10) |
  | qwen3.5:4b agent v1.2 calibrated: model_only unsupported / rag_tools correct | 0/20 / 20/20 | 0/20 / 20/20 |
  | Qwen3-1.7B nf4 agent v1.2 calibrated | 4/20 / 16/20 | 3/20 / 15/20 (1 fixed, 2 lost, p = 1.0) |
  One review item (nf4 rag_tools #1, abstention) labelled in `runs/20261031_v4/assist06/manual_labels.json`. Verdict: the
  fixes remove the two mechanical failure modes (no answer at the step limit; refusing available tools) without hurting
  calibration; the remaining coding failures are model capability (wrong code). Qwen3-1.7B still answers most Q&A from
  retrieval (tool use 2 → 4 of 20 answers); qwen3.5:4b uses tools on 20/20. Demo (`python -m assistant.demo`, qwen3.5:4b):
  2/2 tasks passed (`runs/20261031_v4/demo_run1.txt`).

## KA-07 / R1 (2026-10-31) Real model: weight-only int8 decode for Qwen3-1.7B (APPROXIMATE contract)
- Why: HF bf16 static-cache decode (57.7 tok/s) streams ≈ 3.4 GB of weights per token, ≈ 94% of the step at 211 GB/s, so
  fewer weight bytes is the only large lever (Amdahl). Owner priority (2026-10-31): real-model results vs strong baselines.
- KernelAgent-Ada task `int8_gemv` (Qwen3-1.7B decode linear shapes; dev: gate/up, down, k/v, batch-4; held-out: q/o, the
  151,936-wide lm_head, batches 2/4/8), 2 Triton families (tensor-core dot with M padded to 16; deterministic split-K).
  Mutation tests (no scale, skipped last K block, unsigned reinterpretation) rejected. Study (`runs/20261031_v4/int8_study/`):
  62/62 candidates correct; winner `dot16_splitk` BLOCK_N 64, BLOCK_K 64, SPLITK 8, 4 warps (local, local_meta, both local
  seeds); held-out 20/20 correct. The first held-out attempt crashed (OOM in *my* input generator: int32 → int64 temporaries
  for the lm_head; kept as `heldout_int8_gemv_CRASH_OOM_generator.json`), fixed to generate int8 directly. Held-out
  microbenchmarks are warm-L2 for the small shapes (roofline > 1) and are not claimed; the cache-cold lm_head runs at 1.44 ms,
  103% of the measured DRAM bandwidth.
- Integration: `research/realmodel/int8_linear.py` (torch.library custom op; decode ≤ 16 rows → Triton kernel, prefill →
  bf16 cuBLAS on the exact bf16 image of q with the scale in fp32), `quantize_model` (every nn.Linear incl. lm_head; per-channel
  symmetric int8), `HFBackend(quant='int8')`, CLI `--quant int8` (opt-in). GPU tests 4/4 (accuracy vs fp64 no worse than 2×
  bf16 dequant + cuBLAS, split-K, compile parity, CUDA-graph capture/replay, > 16-row guard, chunked quantisation identity).
- **End to end** (`research.realmodel.bench_int8` + `analyze_int8`, `runs/20261031_v4/int8_e2e/REPORT.md`): Qwen3-1.7B rev
  70d244cc, greedy, batch 1, 256 new tokens, unique prompt prefixes, each mode in fresh processes with alternating order,
  GPU 0–1% before every process, all reps 256 tokens:
  | Mode | tok/s | vs HF bf16 + static cache [95% CI over rounds] | serving peak | quality v1 |
  |---|---|---|---|---|
  | HF bf16 dynamic | 27.3 | 0.46× | 3.34 GB | 39/52 (ASSIST-01) |
  | HF bf16 + static cache (torch.compile) | 58.8 | 1.00 | 3.35 GB | 39/52 |
  | **EfficientGPT int8 + static cache** | **91.3** | **1.55× [1.53, 1.59]** (paired rounds 1.54–1.56; 4 rounds) | **2.36 GB** | **39/52 (test 18/26)** |
  | bitsandbytes LLM.int8 | 7.8 | 0.13× | 2.02 GB | — |
  | bitsandbytes nf4 | 22.0 | 0.37× | 1.34 GB | 36/52 (ASSIST-01) |
  Two later standalone int8 processes (load-peak checks, not interleaved with bf16) gave 87.4 and 88.2 tok/s, so across all 6
  int8 processes the ratio to bf16's 58.8 spans 1.49–1.56×.
- Quality (approximate contract): teacher-forced ΔNLL vs bf16 on 24 × 512-token windows of repo docs **−0.0054 [−0.0076,
  −0.0032] (−0.15%)** — measured through the prefill path, i.e. it tests the quantised weights, not the decode kernel;
  quality v1 equal (39/52, held-out 18/26); greedy: int8 vs bf16 share a 22.9-token common prefix on average (4/16 identical
  64-token outputs), against a bf16-vs-bf16 noise floor (dynamic vs static cache) of 42.9 tokens (10/16). Quantisation changes
  outputs more than bf16 implementation noise does, without a measured quality loss on these suites.
- Costs: TTFT 65 vs 45 ms (prefill dequantises; not optimised); first call 18 s (compile, same as bf16 static); load peak
  6.8 GB → **3.4 GB** after fixing `quantize_weight` (row chunks) and `quantize_model` (it held every bf16 Linear until the
  end), both bit-identical.
- Independent review (separate agent, code only): no finding invalidated the speed claim; fixed before reporting — NLL
  window pairing hash, round-level CI (instead of pooled reps), token-level agreement + noise floor, baseline wording
  ("vs HF bf16 + static cache", not "vs bf16"), memory split (load vs serving).
- Not compared (needs owner approval): torchao int8 weight-only, llama.cpp / Ollama Q8_0 of the same model. Until then the
  claim is limited to the HF stack on this GPU.

## R2 (2026-10-31) Strong same-model baselines: torchao and llama.cpp (owner-approved install / download)
- Installed `torchao==0.18.0` with `--no-deps` (torch 2.11.0 unchanged; `pip freeze` before: `runs/20261031_v4/pip_freeze_before_torchao.txt`);
  pulled `qwen3:1.7b-q8_0` (Ollama 0.40.12, digest 29b6a7a420a4, modelfile in `runs/20261031_v4/int8_vs_baselines/`).
- Modes: `torchao_int8_static` (HF bf16, lm_head untied like ours, `quantize_(model, Int8WeightOnlyConfig())`, same static-cache
  compile path), `torchao_int8_static_at` / `bf16_static_at` (Inductor max-autotune), `llamacpp_q8` (Ollama, greedy, think off,
  num_ctx 4096; decode tok/s client-side (n-1)/(total-TTFT) like the HF modes, within 0.4% of Ollama's server rate).
- Run A (`runs/20261031_v4/int8_vs_baselines/`, 4 interleaved fresh-process rounds, GPU 0-1%): bf16 static 59.2, **ours 91.3**,
  torchao default 82.8, **llama.cpp Q8_0 109.7 tok/s**. Ours vs torchao default 1.103x [1.098, 1.113]; **ours vs llama.cpp 0.833x
  [0.828, 0.840]**. Device memory (process / nvidia-smi delta): ours 2.49 GB, torchao 3.43 GB, llama.cpp 2.42 GB, bf16 3.45 GB.
  Quality: quality-v1 39 / 38 / 39 of 52 (ours / torchao / llama.cpp; bf16 39); dNLL ours -0.15%, torchao -0.16%; greedy common
  prefix with bf16 ours 22.9, torchao 34.4, llama.cpp 32.2 tokens (bf16-vs-bf16 floor 42.9). Ours diverges earlier than torchao on
  4/16 prompts and never later (paired 95% CI of the difference [-23.2, -0.9] tokens; vs llama.cpp not significant).
- Independent fairness review (separate agent): torchao's weight-only mm fuses its dequantisation only under max-autotune in
  torch 2.11, so the default run may under-state it; memory columns were not comparable; ratios between engines lacked CIs.
  Fixed: max-autotune variants re-run, device-memory column, pairwise round-bootstrap CIs, Ollama model identity recorded.
- Run B (`runs/20261031_v4/int8_vs_autotune/`, 5 rounds): bf16 max-autotune gains 0.5-0.7% (ours vs it 1.54x). torchao
  max-autotune 78.9-89.4 tok/s across processes (autotuning picks differ per process). **Ours vs max-autotuned torchao: 4/4 clean
  rounds WIN, 1.08-1.16x, median 1.10x [1.08, 1.16]**. Round 0 (0.99x) is excluded from that statement because my own `git push`
  ran during our int8 process in that round (its reps spread 86.7-90.3; every other round within 1.2%); with round 0 included the
  median is 1.09x and the round-bootstrap CI [0.99, 1.16] touches 1. Both are reported.
- Interpretation: our path streams weights at about 74% of the measured DRAM bound end to end (10.95 ms/token vs an 8.15 ms floor
  for 1.72 GB of int8 weights); llama.cpp reaches about 93% of its own floor (Q8_0, 8.5 bits/weight) with a C++ engine. Closing
  that gap needs a better GEMV and less per-token framework overhead; out of V2 scope.
- Final regression on the final code: 144/144 OK, 0 skipped (`runs/20261031_v4/v2_final2/`).

## V3 (2026-10-31) Closing the gap to llama.cpp: profile, sync-free graph decoding, end-to-end kernel config
- V2 preserved as local tag `v2` (<private-commit>). Full write-up: `reports/EFFICIENTGPT_V3_RESEARCH_REPORT.md`.
- Profile (`research/realmodel/profile_decode.py`, `runs/20261031_v4/v3_profile/`): int8 V2 decode = 9.0 ms GPU kernels per
  token, of which the int8 GEMV is 7.76 ms at about 222 GB/s (the DRAM limit); wall 10.95 ms, so about 2 ms per token was GPU
  idle while HF `generate()` synchronised every token. bf16 shows the same host gap (14.65 ms kernels, 16.9 ms wall).
- `research/realmodel/fast_decode.py`: sync-free greedy loop (device-resident tokens, pinned-memory token stream with CUDA
  events, bounded run-ahead) and `GraphGreedyDecoder` (compiled HF forward + argmax + in-place token/position update in ONE
  manual CUDA graph; StaticCache with device-resident position). Both 96.8 tok/s: host was no longer the limit.
- End-to-end config sweep (`research/realmodel/sweep_configs.py`, GPU time per step = decode time exactly): split-K 8
  (the KA-07 warm-L2 microbenchmark winner) 10.32 ms/step; plain `dot16` BLOCK_N 32 / BLOCK_K 128 **9.51 ms** (105.1 tok/s):
  the reduce kernel per linear (about 197 per token) cost more than split-K gained. New default. Lesson recorded: select
  kernels on end-to-end time, not warm-L2 microbenchmarks. Gate/up fusion: bit-identical, +0.3%; not shipped.
- Backend integration bug found by the benchmark: a fixed 4096-slot cache made the graph path 1.9x slower (static-cache
  attention scans the whole cache): `runs/20261031_v4/v3_bench/` (int8_graph 55.2 tok/s, kept as evidence). Fixed with
  adaptive capacity buckets (256, 384, 512, 1024, 2048, 4096; Phase 5 idea).
- Definitive run (`runs/20261031_v4/v3_final/`, 4 fresh-process rounds, GPU 0-1%): **int8 + graph 111.9 tok/s; vs llama.cpp
  Q8_0 109.9: 1.018x [1.015, 1.051]** (below the 5% bar: parity); vs torchao max-autotune 1.33x; vs HF bf16 static 1.89x;
  vs HF bf16 with the same graph decoder 1.76x; vs int8 through HF generate (same kernel config) 1.12x. TTFT 54 ms (llama.cpp
  33). Quality: quality-v1 39/52, dNLL -0.15%; greedy common prefix with bf16 18.8 tokens (bf16 graph vs static 45.8).
- Independent V3 audit (separate agent): no finding changed a published number; fixed with regression tests: EOS set now
  includes every generation_config id (<|endoftext|> was missing), incremental streaming decode (per-token decode broke 6 of
  8 tokens of a multi-byte test string), batch/length guards, at most 3 bucket decoders, capture time reported separately,
  decoder contract worded as measured near-tie divergence, baseline named "llama.cpp via Ollama defaults".
- Re-verification on the final code (`runs/20261031_v4/v3_verify/`, 3 rounds): int8 + graph 111.1 vs llama.cpp 107.9 tok/s,
  **1.029x [1.013, 1.076]**; vs bf16 static 1.90x (round 0's bf16 process started at 9% GPU utilisation from a git commit).
- Clean checkout (`research/release/clean_checkout.py`; fresh `git worktree` of the final commit, inputs hash-checked):
  first run found an undeclared test input (`data/tinystories_v2/val.bin`), now declared; final run **149/149 OK, 0 skipped**
  (`runs/20261031_v4/v3_clean_checkout_final.json`). Lockfile `requirements-lock.txt`. Demo `python -m research.realmodel.demo`
  (`runs/20261031_v4/v3_demo_run.txt`).
