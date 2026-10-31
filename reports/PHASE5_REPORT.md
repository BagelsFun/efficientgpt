# EfficientGPT Phase 5 — report (2026-10-30)

A greedy/stochastic decoder for a 12.19M-parameter TinyStories GPT (nanoGPT, 6 layers, 6 heads, d=384, context 256),
optimized and verified on one RTX 4060. Every number below comes from a committed run directory under
`research/phase5/ci_runs/` or `research/v1/runs/`, and is cross-referenced to `docs/EXPERIMENT_LOG.md`. The contracts are in
`docs/PHASE5_CONTRACT.md`. This is a research decoder for a small model, **not** a production serving engine. It has not
been compared with vLLM, TensorRT-LLM or similar.

## 1. Summary

- **What Phase 5 is:** not speculative decoding. A decode step (masked attention over a fixed-size, GPU-resident KV cache)
  is captured once as a CUDA Graph and replayed per token. `adaptive` pools pick the smallest cache capacity bucket
  (80/128/192/256) that fits prompt + budget; `full` uses 256.
- **Exact by default.** The default graph modes produce greedy tokens bit-identical to an independent concat-KV reference,
  and their decode-step logits are bit-identical too (100/100 steps, TF32 and IEEE).
- **Added on top, opt-in, each with its own contract and tests:** EOS stopping, temperature/top-k/top-p sampling (exact
  given caller-drawn uniforms), ragged batches, an Inductor-compiled step, a Triton split-K decode-attention kernel, and
  fp16/bf16 weights (approximate).
- **Measurement method:** raw per-request timings; bootstrap CIs; A/A controls (identical-work pairs); an in-process
  interleaved benchmark after cross-process runs showed unexplained ≈ ±3% per-process offsets. Promotion rule: ≥ 5% faster,
  CI excludes 1, reproduced in fresh processes.

**Latency ladder** (B=16, prompt 64, 64 generated tokens = 1,024 tokens per request; median ms; in-process numbers unless
noted; values from different runs are labeled and only roughly comparable):

| Mode | Contract | Median ms | Generated tok/s | Source |
|---|---|---:|---:|---|
| `prefix_sdpa_eager` (no graph) | exact | 131.86 | 7.8k | cross-process 6-round run `20261030_184148` |
| `full_graph` | **exact** (default family) | 77.13 | 13.3k | same |
| `adaptive_graph` | **exact** (default family) | 52.81–56.17 | 18.2k–19.4k | four runs |
| `compiled_adaptive_graph` | near-tie | 51.21 | 20.0k | `20261030_205141_triton_attn` |
| `triton_full_graph` | near-tie | 37.03–39.44 | 26.0k–27.7k | `…_triton_attn`, `…_compiled_triton` |
| `compiled_triton_full_graph` | near-tie | 33.53–33.69 | 30.4k–30.5k | `…_compiled_triton`, `…_precision` |
| `fp16_compiled_triton_full_graph` | **approximate** | 20.44 | 50.1k | `20261030_212013_precision` |

Headline in-process ratios (all 12 workloads, 95% CIs; §5): `adaptive_graph` → `compiled_triton_full_graph`
**1.29–1.74×**. `compiled_triton_full_graph` → fp16 **1.39–1.80×**. `adaptive_graph` → fp16 1.77–2.77×.

## 2. Mechanism

```
request: prompt [B, p] (or right-padded + lengths for ragged), budget n
   │
   ├─ DecoderPool.get ── key (B, capacity) ── capacity = 256 (full) | smallest of 80/128/192/256 ≥ p+n (adaptive)
   │        └─ cached decoder per key; factory selects the step implementation
   │
   ├─ reset (eager): prefill forward → K/V copied into cache slots [0, p); first token from prefill logits
   │
   ├─ replay CUDA Graph n−1 times; each replay = one step, all state on the GPU:
   │     x = wte(token) + wpe(position)
   │     per layer: write K/V at slot `position` → attention over slots ≤ position → MLP
   │     token ← argmax(logits)        (sampling: inverse CDF of filtered probs with u[:, step_index])
   │     history[position] ← token; position += 1
   │
   └─ finish: tokens [B, p+n]  (ragged: generated [B, n]; EOS: post-mask + lengths)

attention implementations:
   default        masked SDPA over the whole fixed cache (cost ∝ capacity)
   triton_*       split-K kernel reading only slots ≤ position[b] (cost ∝ valid length)
step implementations:
   eager kernels  StaticKVDecoder.step (bit-exact)
   compiled_*     torch.compile of step_index_copy (index_copy_ writes; Inductor-fused elementwise)
```

## 3. Environment and method

- **Hardware/software (observed):** RTX 4060 8 GB (CC 8.9, 24 SMs, 24 MiB L2), driver 617.14, Windows 11, Python 3.11.9,
  torch 2.11.0+cu128, triton-windows 3.7.1.post27. About 2.9 GB of VRAM was used by other processes during runs. Measured
  copy bandwidth is 211 GB/s; this is the roofline used, not the spec sheet. Details: `docs/ENVIRONMENT.md`.
- **Workloads:** batch {1, 16} × (prompt, budget) ∈ {(8,16), (64,16), (64,64), (128,32), (128,64), (192,32)}. The
  benchmark prompts are a repeated sentence with varied last tokens; the EOS, sampling, ragged and precision studies use
  real validation-story prompts.
- **Precision policy:** matmul `fp32_precision = 'tf32'` in all benchmarks.
- **Timing:** synchronized wall-clock per `generate` call (prefill + replays + host dispatch), steady state. Capture,
  compile and correctness validation are reported separately, not amortized. Raw timings: `benchmark_raw.csv` in every run.
- **Statistics** (`research/phase5/analyze_ci.py`): two-level bootstrap (clusters, then requests) on the ratio of medians;
  95% percentile CI; WIN = ratio ≥ 1.05 and CI low > 1; prompt-192 pairs at equal capacity are A/A controls.
- **Cross-process vs in-process.** Separate-process rounds showed per-process offsets of ≈ ±3% even for identical work (one
  A/A failure in each of two runs; EXPERIMENT_LOG AA-01). In-process A/A agreed within ±0.4%. Comparisons after
  INPROC-01 therefore use `benchmark_inprocess.py`: all modes built in one process, interleaved request by request with
  rotating order, (process, block) clusters, typical CI half-width ≈ 0.5–1%. Peak memory can't be measured that way, so
  memory numbers come from cross-process runs.

## 4. Contracts by mode

| Mode family | Token contract | Default | Tests |
|---|---|---|---|
| `full_graph`, `adaptive_graph` | bit-exact vs `generate_reference` (same precision policy) | **yes** | `test_graph_gpu.TestGraphPoolExact` (+ mask-removal mutant) |
| + EOS (`EOSDecoder`) | first EOS kept, later slots pad (0), lengths; == feed-pad reference | opt-in | `test_eos`, `TestGraphPoolEOS` |
| + sampling (`SampledGraphDecoder`) | token-exact given uniforms; distribution = declared transforms | opt-in | `test_sampling` (incl. chi-square), `TestSampledGraphPool` |
| ragged (`RaggedGraphDecoder`) | == own eager engine exactly; equal lengths == uniform path; per-row near-tie vs batch 1 | opt-in | `test_ragged`, `TestRaggedGraphPool` |
| `compiled_*`, `triton_*`, `compiled_triton_*` | near-tie: equal, or first divergence at reference top-2 margin < 1e-2 | opt-in | `test_near_tie`, `TestCompiledBaseline`, `TestDecodeAttention` |
| `fp16_*`, `bf16_*` | **none** (approximate); quality characterized in §5.4 | opt-in, owner decision | `test_low_precision_inputs`, `precision_study.py` |

Full suite: **74 tests OK** (`research/phase5/ci_runs/20261030_212013_precision/diagnostics/regression.log`). GPU tests skip
without CUDA, Triton or the checkpoint.

## 5. Results

### 5.1 Baselines and the exact path (EXPERIMENT_LOG V1-FULL-02, P5-CI-01, P5-INDUCTOR-01)
- V1 3-mode matrix re-run (`research/v1/runs/20261030_183607000`): rc=0, exact tokens, report gate FULL MATRIX.
- 6-round cross-process run (`20261030_184148`): A/A 2/2 PASS; adaptive vs full WIN 10/10 at capacity 80/128 (1.089–1.398);
  graph vs prefix-SDPA eager WIN 24/24 (1.45–3.72×). Steady peak allocation: eager 267, full 291, adaptive 461 MiB.
- Plain `torch.compile` of the step (`compiled_static`, no graph): graph modes beat it 24/24 (1.57–3.10×). Decode at this
  size is dispatch-bound, so the CUDA graph is the main win.

### 5.2 Kernel work (KERNEL-01, KERNEL-02)
- **Profile:** masked SDPA is 21–60% of step GPU time and scales with capacity, not valid length. B16 cap 256: 0.626 ms vs a
  theoretical 0.357 ms at 211 GB/s for 75.5 MB of K/V. B1: 0.200 ms for 4.7 MB (latency-bound, 6 CTAs on 24 SMs).
- **Triton split-K decode attention** (`research/phase5/decode_attention.py`): reads only slots ≤ position (NaN-poison test);
  error vs fp64 ≤ 2× masked SDPA's. Per layer call (CUDA-graph timed, hot): B1 cap 256 35 µs → 2.7–3.4 µs; B16 cap 256
  103–109 µs → 4.1–13.7 µs.

| In-process comparison (12 workloads) | Verdict | Range (95% CI at the ends) |
|---|---|---|
| adaptive_graph → triton_adaptive_graph | WIN 12/12 | 1.217 [1.184, 1.253] – 1.608 [1.567, 1.643] |
| compiled_adaptive_graph → triton_adaptive_graph | WIN 12/12 | 1.110 – 1.476 |
| triton_full → triton_adaptive | INCONCLUSIVE 10/10, A/A 2/2 | 0.999 – 1.039 |
| triton_full → compiled_triton_full | WIN 12/12 | 1.059 [1.051, 1.083] – 1.108 [1.093, 1.122] |
| adaptive_graph → compiled_triton_full | WIN 12/12 | 1.293 [1.218, 1.351] – 1.737 [1.669, 1.780] |

Finding: because the kernel reads only valid slots, **capacity buckets stop mattering**. The full-capacity Triton modes
match the adaptive ones with 2 graphs instead of 8.

- **What's left in the step** (B16 cap 256, compiled_triton 0.435 ms GPU): GEMMs 0.28 ms (65%), attention 0.10 ms, fused
  elementwise ≈ 0.05 ms. The GEMMs read 48.7 MB of fp32 weights per step, a theoretical 0.231 ms at 211 GB/s, so they
  already run at ≈ 78–83% of that floor.

### 5.3 Decoding features (EOS-01, SAMPLE-01, RAGGED-01)
- **EOS:** mask overhead −0.2% (noise). On real story prompts, 91–99% of greedy rows emit EOS within budget. The early-exit
  upper bound is 37–45% of decode steps at B=1 but ≤ 3% at B=16, so early exit is not implemented.
- **Sampling:** in V=8 tiny-model tests, the chi-square test against exactly enumerated probabilities passes for our sampler
  and for nanoGPT's `GPT.generate`. p-values are uniform over 20 seeds, and three transform mutants are rejected. Cost vs
  greedy (in-process): top-k +3–8%, top-p +6–10%, both +8–13%.
- **Ragged:** batch-16 ragged vs 16 sequential batch-1 calls: **7.96–7.97×**. Within ≈ 1% of a same-shape uniform batch.
  251/256 rows bit-exact to per-row decoding, 5 near-ties, 0 outside the rule.

### 5.4 Low-precision weights (PREC-01, approximate)

| | ΔNLL vs fp32, 262k val tokens | argmax agreement | free-running identical (128 × 224) | divergences that are near-ties | speed vs fp32 compiled_triton |
|---|---|---|---|---|---|
| fp32 compiled_triton (control) | — | — | 124/128 | 4/4 | — |
| **fp16** | −4.1e-6 [−1.2e-5, +3.3e-6] | 99.88% | 105/128 | 23/23 (46/46 incl. benchmark/random prompts) | **1.386–1.802×** |
| bf16 | +1.12e-4 [+5.2e-5, +1.7e-4] | 99.04% | 30/128 | 28/98 | 1.380–1.769× (≤ fp16) |

fp16 dominates bf16 on quality and speed. **Owner decision:** both stay opt-in approximate modes, not under the near-tie
contract and not default. fp16's largest margin at a divergence was 7.8e-3, close to the 1e-2 threshold, and overflow was
not observed but is not excluded.

## 6. Failures, negative results and fixes

| Issue | Evidence | Outcome |
|---|---|---|
| All EfficientGPT work was untracked in git; `origin` pointed at karpathy/nanoGPT | AUDIT-01 | Committed on local `efficientgpt/main`; `origin` → `upstream`; nothing pushed |
| Earlier V1 "full" run was interrupted (no CSV) | `research/v1/runs/20261029_214112150` | Re-run passes (V1-FULL-02); the old run is kept as evidence |
| First performance claims had no CIs; identical-work cells differed by up to 8% | V1-FULL-02 | Raw timings + bootstrap + A/A (P5-CI-01) |
| `torch.compile(mode="reduce-overhead")` skips cudagraphs ("mutated inputs") | P5-INDUCTOR-01 | Unsupported; manual capture used instead |
| Inductor results not bit-exact | 8/288 divergences, all at margins ≤ 9.7e-4 | Near-tie contract for compiled modes |
| Compiled-in-graph step 1.28–1.6× **slower** at B=16 | profile: `scatter_` → full-cache scatter + copy | `index_copy_` variant, bit-identical in eager (P2-CGRAPH-01) |
| Cross-process A/A failures (1.037, then 0.971) | AA-01 | Not a mode effect (in-process A/A ±0.4%); in-process benchmark built; residual ≈ 1.5–3% pair offsets **unexplained** |
| Microbenchmark timed host launch latency, not kernels ("hot" > "cold") | `…_triton_attn/diagnostics/microbench.log` | Discarded; re-measured from CUDA-graph replays |
| Cold-cache kernel times implied 302 GB/s (> 211 GB/s measured) | microbench_graph.log | Small cold values marked unreliable (flush subtraction noise ≈ ±5 µs) |
| Triton kernel failed under Inductor (fp64 scale promoted the softmax state) | KERNEL-02 | Cast scores to fp32 (no-op in eager) |
| Inductor mutation-analysis fallback on the combine kernel | KERNEL-02 | Verified no added copies (100 → 83 kernels/step) |
| Ragged prefill length mask was redundant (an equivalent mutant survived) | RAGGED-01 | Removed; causal attention suffices with right padding |
| Token-only pad-invariance test was too weak to catch a mask bug | RAGGED-01 | Test now compares K/V state; the mutant is killed |
| Short story openings repeat (11 distinct of 128 at p=8) | EOS-01 | p=8 results reported as low-diversity |
| bf16 weights: 70% of divergences beyond near-ties | PREC-01 | Not recommended over fp16 |

## 7. Caveats and limits

- One GPU, one small FP32-trained model, steady-state latency of known-budget requests. No serving queue, network or
  continuous batching. Benchmark prompts are synthetic (feature studies use real prompts).
- Capture/compile/validation costs aren't amortized into the latencies (e.g. compiled adaptive ≈ 4.3 s init incl. validation
  with a warm Inductor cache; cold compile ≈ 6.6 s in a smoke test).
- Memory for the Triton/compiled/fp16 modes wasn't measured (the in-process runs keep all modes resident).
- Not covered: compiled or Triton sampling, ragged with sampling/compiled/Triton, EOS early exit, multi-turn cache reuse,
  OOM recovery, power/thermal throttling decoding (one throttle bitmask was sampled idle, undecoded).
- ≈ 2.9 GB of VRAM was used by other processes during all runs.

## 8. Reproduce

From the repository root (`.venv` with torch 2.11.0+cu128 and triton-windows 3.7.1.post27; checkpoint
`out-tinystories-v2/baseline-2500.pt` and `data/tinystories_v2/` present):

```powershell
# full test suite (GPU tests skip without CUDA/Triton/checkpoint)
.\.venv\Scripts\python.exe -m unittest research.phase4.test_decoder research.phase4.test_compact research.phase5.test_pool research.phase5.test_prefix research.phase5.test_near_tie research.phase5.test_compiled_step research.phase5.test_analyze_ci research.phase5.test_inprocess research.phase5.test_eos research.phase5.test_sampling research.phase5.test_ragged research.phase5.test_graph_gpu research.v1.test_demo research.v1.test_report
# V1 exact matrix (cross-process) + report
.\.venv\Scripts\python.exe -m research.phase5.benchmark_review --all --rounds 2 --trials 5 --variants 3 --output <run>\benchmark.csv
.\.venv\Scripts\python.exe -m research.v1.report --results <run>\benchmark.csv --memory <run>\benchmark_memory.csv --output <run>\V1_REPORT.md --require-full
# in-process comparison of the main modes
.\.venv\Scripts\python.exe -m research.phase5.benchmark_inprocess --processes 3 --blocks 6 --requests 12 --modes adaptive_graph triton_full_graph compiled_triton_full_graph fp16_compiled_triton_full_graph --output <run>\benchmark_raw.csv
# feature and quality studies
.\.venv\Scripts\python.exe -m research.phase5.eos_study --output <run>\eos_study.json
.\.venv\Scripts\python.exe -m research.phase5.sampling_study --seed 1 --output <run>\run1.json
.\.venv\Scripts\python.exe -m research.phase5.ragged_study --seed 1 --output <run>\run1.json
.\.venv\Scripts\python.exe -m research.phase5.precision_study --output <run>\quality.json
```

Each run directory in `research/phase5/ci_runs/` keeps its raw CSV, logs, `environment_before/after.txt` and
`diagnostics/` scripts. `analyze_ci --raw <run>/benchmark_raw.csv --output <run>/CI_REPORT.md` re-derives any CI table.

## 9. Open items

1. GEMMs are the remaining step cost and near their fp32 bandwidth floor. Further gains need fewer weight bytes (the fp16
   approximate mode) or more rows per weight read (larger batches).
2. Triton attention for the ragged and sampled steps (the kernel already takes per-row positions).
3. The cause of the ≈ 1.5–3% cross-process pair offsets (AA-01) is unresolved. Decisions below 5% use the in-process benchmark.
