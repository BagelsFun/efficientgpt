# Phase 5 extension plan: sampling, EOS stopping, ragged batches (2026-10-30)

**Status: decisions accepted 2026-10-30 (all four recommendations). Section 1 option A implemented (EXPERIMENT_LOG EOS-01);
option B not implemented (measured upper bound ≤ 3% at B=16, up to ≈ 45% at B=1). Section 2 (sampling) implemented with option A for the exact graph modes (EXPERIMENT_LOG SAMPLE-01); compiled sampling
not implemented. Section 3 (ragged) implemented with option A for greedy exact graph modes (EXPERIMENT_LOG RAGGED-01).** Each feature changes `docs/PHASE5_CONTRACT.md`. The owner decides the options
marked **Decision** before any code is written. Greedy, same-length, run-to-budget decoding stays exactly as it is (default,
bit-exact). Every new behavior is an opt-in path with its own contract section and tests.

## Facts this plan rests on
- Tokenizer `data/tinystories_v2/tokenizer.json`: BPE, vocab 4096, special tokens `[EOS]` = 0 and `[UNK]` = 1.
  `prepare_v2.py` appends `[EOS]` after every story, and `meta.pkl` stores `eos_id: 0`. So EOS is a trained token. Because
  training text is concatenated stories, the model starts a new story after EOS.
- The only existing sampler is `model.py` `GPT.generate`: `logits / temperature` → optional top-k (others set to −inf) →
  softmax → `torch.multinomial`. It has no top-p, no repetition penalty and no EOS stop (`sample_v2.py` uses T = 0.8,
  k = 50 and cuts at `[EOS]` afterwards).
- The decode step (`StaticKVDecoder.step`, `step_index_copy`) already keeps position, token and history on the GPU, so
  per-step logic can stay inside a CUDA Graph if it's written with tensor ops and no host syncs.
- Graph modes are bit-exact with `generate_reference` at the same batch size. Compiled modes follow the near-tie rule.
  Logits computed at different batch sizes are **not** guaranteed bit-identical (different GEMM kernels).

## Recommended order
1. **EOS stopping (greedy)**: smallest change, fully exact, and needed for any realistic latency number.
2. **Sampling**: needs a new RNG contract and statistical tests.
3. **Ragged batches**: largest change (per-row positions in prefill and decode).

Each step lands with tests and an in-process benchmark (`benchmark_inprocess.py`) before the next one starts.

---

## 1. EOS stopping
**Semantics (proposed):** per row, the first generated EOS is kept in the output and marks the row finished. Later slots hold
`pad_id` (proposed: EOS itself, id 0). `generate` also returns `lengths[B]`, the number of generated tokens including EOS.
`max_new_tokens` stays a hard budget, so bucket choice is unchanged (capacity = prompt + budget).

| Option | How | Pros | Cons |
|---|---|---|---|
| **A. Fixed steps + device mask** | `finished[B]` lives on the GPU; each step does `token = where(finished, pad, token)` and `finished \|= token == eos`. Always replays budget − 1 steps. | Graph-friendly, no host sync, exact. Smallest diff. | Saves no compute when rows finish early. |
| B. A + early exit every k steps | Replay k steps, then one host read of `finished.all()`. Stop when every row is done. | Real latency savings when all rows finish early. | One sync per k steps (cost to measure). Overhead when rows don't finish. |
| C. Per-row eviction / continuous batching | Refill finished rows with new requests. | Serving-grade throughput. | Scheduler and cache management: out of Phase 5 scope. |

**Recommendation:** A first, then B with k chosen by measurement (try 4, 8, 16).
**Decision:** pad value (EOS vs a separate pad id) and whether B is wanted.

**Correctness tests:**
- The reference gets the same finished-mask semantics, and outputs must be exactly equal.
- To force early, simultaneous and staggered EOS on demand, tests pass an `eos_id` that the tiny or trained model emits
  often (e.g. the most frequent greedy token), since real EOS may not appear within short budgets.
- Cases: EOS at step 1, EOS on the last step, all rows at once, no row finishing, B = 1 and 16, reused graphs
  (long-then-short), and repeated calls with the finished state reset in `reset()`.
- Mutation check: drop the mask and expect the tests to fail.

**Benchmark:** latency vs real EOS position on the trained model, A vs B, and tokens/s counting only real tokens. Measured
story lengths including EOS (`data/tinystories_v2/*.bin`): val p10/p50/p90 = 140/184/249 tokens, train 145/195/360. Only
91.9% (val) and 79.7% (train) fit in the 256-token context, and about 5% end within 128. So EOS arrives late: option B only
pays off with short prompts and budgets close to 256 − p. Several current workloads (budgets 16–64) would rarely hit EOS,
so EOS workloads need long budgets (e.g. prompt 8–64, budget 192–248).

## 2. Sampling (temperature, top-k, top-p)
**Proposed transform order** (matches `GPT.generate` and adds top-p): `logits / T` → top-k → top-p → probabilities →
sample. `T = 0` means the existing greedy path (unchanged contract). Top-p keeps the smallest set of highest-probability
tokens whose cumulative probability is ≥ p and always keeps the top-1. Use `torch.sort(..., stable=True)` (the default is
unstable), so ties at the cutoff resolve by token id in both the reference and graph paths. No repetition penalty in the
first version.

**RNG contract: the key decision**

| Option | How | Testable guarantee | Cons |
|---|---|---|---|
| **A. Pre-drawn uniforms + inverse CDF** | Before replay, draw `u[B, budget]` from a seeded `torch.Generator` into a static buffer. Step t picks the first index where `cumsum(probs) > u[:, t]`. The RNG never runs inside the graph. | **Token-exact given u**: the reference consumes the same u, so graph modes must match bit-for-bit (they already have bit-identical logits). | Not stream-identical to `GPT.generate`'s `torch.multinomial` for the same seed; same distribution, different draws. Cumsum rounding bias ~1e-7 per token (documented). |
| B. `torch.multinomial` inside the graph | Uses PyTorch's graph-safe Philox seed/offset handling. | Distributional only; token equality with eager isn't guaranteed. | Weaker tests (statistics only). Harder to debug. |
| C. Gumbel-max with pre-drawn noise | `argmax(logits/T + g)`, g drawn up front. | Token-exact given g, like A. | Top-p still needs a sort. Gumbel noise is B × V per step (4096×). |

**Recommendation:** A. It keeps the strongest kind of test (exact equality) and needs only B floats per step.
**Decision:** accept that sampled text won't match `sample_v2.py` draw-for-draw for the same seed.

**Compiled modes:** a near-tie rule for sampling. A divergence is acceptable only where `u` lands within a tolerance of a CDF
boundary of the reference distribution (the analogue of the logit-margin rule), or where an earlier greedy-style near-tie
explains it.

**Correctness tests:**
1. **Exact given u** (graph vs reference): random prompts, B = 1/16, T ∈ {0.5, 0.8, 1.0, 1.5}, k ∈ {1, 50, none},
   p ∈ {0.5, 0.9, 1.0}, plus long-then-short graph reuse.
2. **Sampler implements the declared distribution:** a tiny model with vocab 8 and 2–3 steps, with every sequence's exact
   probability enumerated from the model's logits after the transforms. Sample N = 10⁵–10⁶ sequences from the graph path.
   Run a chi-square goodness-of-fit test and report total variation distance with a bootstrap CI, plus the sample size.
   Expected failure modes: wrong transform order, an off-by-one in top-p, or a temperature applied after top-k.
3. **Same distribution as nanoGPT semantics:** a two-sample test of our sampler vs `GPT.generate` (multinomial) on the same
   tiny setup with top-p disabled. This should be indistinguishable at the chosen N; report the p-value.
4. **Edge cases:** k = 1 equals greedy, p small enough to keep only the top-1 equals greedy, T → large approaches uniform
   over the kept set, an all-masked row is impossible by construction (top-1 always kept), u = 0 and u just below 1.
5. **With EOS:** sampled EOS stops a row, through the same tests as in section 1.

**Benchmark:** per-step overhead of each transform vs greedy, in-process (top-p adds a 4096-wide sort per row and step).

## 3. Ragged batches (different prompt lengths per row)
| Option | How | Pros | Cons |
|---|---|---|---|
| **A. Per-row positions** | `position[B]` instead of a scalar. Cache writes use per-row indices (the scatter / index_put path already takes an index tensor). Mask `locations <= position[b]` per row. Prefill per row (eager), or batched with a padding mask. | No pad tokens inside sequences. Absolute learned `wpe` maps naturally to per-row positions. | New step variant. Prefill becomes the bottleneck if it's done per row. |
| B. Left padding + attention mask | Pad short prompts on the left. Positions start at the first real token. Mask out pad slots. | Familiar (HF-style). | Pad slots waste cache. Position bookkeeping is error-prone. |
| C. Bucket by length outside the decoder | Group requests by prompt length and keep same-length batches. | Zero decoder change. | Not true ragged support. Batching efficiency depends on traffic. |

**Recommendation:** A, with capacity = max_i(p_i) + budget. Do C first as a cheap stop-gap if ragged traffic is only hypothetical.
**Decision:** whether ragged support is needed now at all.

**Correctness tests:**
- **Batched ragged reference** (same B, eager, per-row masks): exact equality with graph modes.
- **Per-row B = 1 reference:** a semantic check under the near-tie rule, because B = 1 and B = 16 GEMMs aren't bit-identical.
- Cases: lengths 1 … capacity − 1 mixed in one batch, one row at the capacity limit, all rows equal (must match the current
  path exactly), and combined with EOS and sampling.

## What stays the same
The default API (greedy, same-length, run to budget), its bit-exact contract and all existing tests and benchmarks. New paths
are opt-in flags on the decoders and `DecoderPool`, plus new `benchmark_review` / `benchmark_inprocess` workloads. Peak
memory is reported per mode in separate processes, as today.

## Decisions needed (summary)
1. **EOS:** pad value (EOS = 0 vs a separate id) and whether early exit (option B) is wanted after A.
2. **Sampling RNG:** pre-drawn uniforms + inverse CDF (recommended; exact given u, not draw-identical to `sample_v2.py`) vs
   in-graph multinomial (distributional tests only).
3. **Sampling scope:** temperature + top-k + top-p, no repetition penalty (recommended) vs more.
4. **Ragged:** per-row positions now (recommended if needed) vs length-bucketing only vs defer.
