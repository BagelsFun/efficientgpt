# Phase 5 contract (derived from code, 2026-10-30)

## Mechanism
Not speculative decoding, early exit or adaptive precision. Phase 5 is **CUDA Graph reuse by KV-cache capacity bucket**.

```
request (prompt [B,p], n new tokens)
  -> DecoderPool.get                  research/phase5/adaptive_pool.py
       full:     capacity = block_size (256)
       adaptive: capacity = smallest of (80,128,192,256) >= p+n
       key (B, capacity) -> cached GraphDecoder / CompactGraphDecoder (phase4)
  -> decoder.generate(prompt, n)
       reset: eager prefill (reference_step), copy K/V[:p], token, position=p
       replay captured StaticKVDecoder.step n-1 times
         (masked SDPA over the full fixed cache; position is a GPU tensor)
       finish: history[:, :p+n]
```
Baseline paths: `generate_reference` (concatenated-KV eager, phase4/decoder.py) and
`PrefixSDPADecoder` (sliced-prefix eager, phase5/prefix_eager.py).

## Guarantees
- **Exact mode only.** For greedy decoding, output tokens must equal `generate_reference` exactly (`torch.equal`)
  under the same matmul precision policy. No approximate mode exists.
- Greedy argmax only. There's no temperature, top-k/top-p, repetition penalty, EOS stop or sampling RNG.
- All rows in a batch share the same prompt length (no ragged batches). `p >= 1`, `n >= 1`, `p + n <= block_size`.
- Requests that exceed the capacity are rejected with `ValueError`, never silently truncated.
- Graphs and buffers stay resident for the pool's lifetime (memory is traded for capture reuse).
- Stale K/V from earlier, longer requests sits beyond `position` and is masked out. This is checked by
  `test_graph_gpu.py` and a mask-removal mutation check.

## Compiled baseline (`compiled_static`, opt-in)
`torch.compile` of the same `StaticKVDecoder.step` (fullgraph, static shapes). It is **not bit-exact**: Inductor reorders FP32
reductions (max |logit diff| ≈ 2e-3..6e-3). A sequence passes if it is identical, or its first divergence sits where the reference
top-1/top-2 logit margin is < 1e-2 (`benchmark_review.near_tie_match`). The near-tie count is reported. `reduce-overhead` is
unsupported for this step (cudagraphs skipped: mutated inputs).

## Compiled graph modes (`compiled_full_graph`, `compiled_adaptive_graph`, opt-in)
`CompiledGraphDecoder`: Inductor-compiled `step_index_copy` (bit-identical to `StaticKVDecoder.step` in eager; `index_copy_`
instead of `scatter_` so Inductor writes one cache row in place), captured in the manual CUDA Graph and pooled like
full/adaptive. Same near-tie rule as `compiled_static`, **not** the exact contract. Fastest measured mode
(see EXPERIMENT_LOG P2-CGRAPH-01), but it stays opt-in until the owner accepts the near-tie contract.

## EOS stopping (opt-in, `research/phase5/eos.py`)
`EOSDecoder(decoder, eos_id=0, pad_id=eos_id)` wraps any mode. Per row, the first **generated** `eos_id` is kept, later
generated slots are `pad_id`, and `lengths[b]` = generated tokens up to and including that EOS (`count` if none). Prompt
tokens are never inspected. Every decoder still runs its full budget (fixed steps); bucket choice is unchanged. Exactness
is the same as the wrapped mode (graph modes: bit-exact vs `generate_reference_fed_pad`, which feeds `pad_id` after EOS;
the post-mask is provably identical because batch rows never interact). Early exit is not implemented (EXPERIMENT_LOG EOS-01).

## Sampling (opt-in, `research/phase5/sampling.py`)
`SamplingConfig(temperature > 0, top_k, top_p)`, applied as `logits / T` → top-k (ties at the k-th value kept, as
`GPT.generate`) → softmax → top-p (smallest stable-sorted prefix with mass ≥ p; top-1 always kept) → renormalize. Tokens
are drawn by inverse CDF from caller-provided uniforms `u[B, count]` (`draw_uniforms`; one per row and generated token);
zero-mass tokens are never chosen. **Exact given u:** `SampledGraphDecoder` (full/adaptive pools via `sampled_factory`)
equals `generate_reference_sampled` token for token; its step logits are bit-identical to the incremental reference.
**Distribution:** same as `GPT.generate` for temperature/top-k (verified statistically), but not the same draws for a seed.
Temperature 0 means: use the greedy decoders. Composes with `EOSDecoder` (uniforms passed through). No compiled
(Inductor) sampling mode and no repetition penalty.

## Ragged batches (opt-in, greedy, `research/phase5/ragged.py`)
Right-padded prompts [B, Pmax] (pad values never read) + `lengths` [B] in [1, Pmax] → generated tokens [B, count].
Per-row positions; cache slot = absolute position. `RaggedGraphDecoder` (pool via `factory=RaggedGraphDecoder`, buckets on
Pmax + count) **equals** `RaggedEngine.generate_eager` exactly; with equal lengths it equals the uniform decoders exactly;
each row matches batch-1 `generate_reference` under the near-tie rule (cross-batch-size GEMMs aren't bit-identical;
measured 251/256 rows bit-exact). EOS: apply `mask_after_eos(generated, 0, eos_id)`. Not supported with sampling or
compiled modes.

## Triton decode attention (`triton_full_graph`, `triton_adaptive_graph`, opt-in)
`research/phase5/decode_attention.py`: split-K FP32 decode attention reading only slots ≤ position[b]; same semantics as
masked SDPA (scale 1/sqrt(D)) but a different summation order, so the **near-tie** token rule applies, as for compiled
modes. Kernel error vs fp64 must be ≤ 2× masked SDPA's (or 1e-6). FP32, power-of-two head dim, needs Triton. Fastest
measured modes (EXPERIMENT_LOG KERNEL-01). `compiled_triton_*_graph` additionally compiles the step with Inductor around
the Triton kernels (KERNEL-02); same near-tie rule. Exact modes remain the default.

## Triton attention in ragged and sampled steps (opt-in)
`attention=decode_attention` on `RaggedGraphDecoder` / `sampled_factory`. **Ragged:** each row within the near-tie rule vs
the exact ragged engine and vs batch-1 decoding; pad values never affect output. **Sampling:** judged by distribution, not
token agreement. (1) Chi-square vs exactly enumerated probabilities through the graph path (df ≥ 5). (2) Teacher-forced TV
distance to the exact sampled step no larger (median, p99) than TF32-vs-IEEE between two exact implementations.
**All decoders:** state is reset per request; a request after a longer one equals it on a fresh decoder, bit for bit.

## Low-precision weights (APPROXIMATE, opt-in: `fp16_*`, `bf16_*`)
Weights and KV cache in fp16/bf16, fp32 accumulation. **No token contract:** validated only for well-formed output; quality
is characterized by `research/phase5/precision_study.py` (EXPERIMENT_LOG PREC-01). fp16 stayed within the fp32 near-tie rule
on every observed divergence (46/46, max margin 7.8e-3), but that's empirical; bf16 doesn't. Never a default without owner
approval.

## Precision
Benchmarks set `torch.backends.cuda.matmul.fp32_precision = 'tf32'`. Measured TF32-graph vs IEEE-FP32-reference
agreement was 3072/3072 greedy tokens on 48 random sequences. That's an observation for this checkpoint, not a guarantee.

## Tests
| Test | Covers | Device |
|---|---|---|
| `research.phase5.test_pool` | bucket choice, reuse keys, rejection | CPU (fake decoder) |
| `research.phase5.test_prefix` | prefix-SDPA eager == reference | CPU |
| `research.phase5.test_graph_gpu` | graph pool == reference: random prompts, long-then-short reuse, bucket edges, n=1, B>1, tf32+ieee, tiny model + trained ckpt | CUDA (skips otherwise) |
| `research.phase5.test_near_tie` | near-tie rule logic | CPU |
| `research.phase5.test_eos` | EOS mask semantics; post-mask == feed-pad reference | CPU |
| `research.phase5.test_graph_gpu.TestGraphPoolEOS` | EOS-wrapped full/adaptive pools == feed-pad reference (real + forced EOS) | CUDA |
| `research.phase5.test_sampling` | transforms, inverse CDF, exact given u (eager), greedy limits, chi-square vs enumerated probabilities (ours + `GPT.generate`) | CPU |
| `research.phase5.test_graph_gpu.TestSampledGraphPool` | sampled full/adaptive pools == reference given u, with and without EOS | CUDA |
| `research.phase5.test_ragged` | ragged validation, prefill vs batch 1, pad invariance (tokens + K/V), uniform equivalence, per-row near-tie | CPU |
| `research.phase5.test_graph_gpu.TestRaggedGraphPool` | ragged pools == eager engine; uniform equivalence; per-row near-tie | CUDA |
| `research.phase5.test_graph_gpu.TestDecodeAttention` | Triton attention vs fp64 (≤ 2× SDPA error), never reads past position, Triton and compiled-Triton pools within near-tie rule | CUDA + Triton |
| `research.phase5.test_graph_gpu.TestDecodeAttention.test_low_precision_inputs` | Triton attention with bf16/fp16 inputs (fp32 math) vs fp64 | CUDA + Triton |
| `research.phase5.test_gpu_combinations` | Triton ragged, Triton sampling (chi-square + TV noise floor), KV state across lengths, ragged + EOS | CUDA + Triton |
| `research.phase5.test_compiled_step` | `step_index_copy` bit-identical to the scatter_ step | CPU |
| `research.phase5.test_graph_gpu.TestCompiledBaseline` | compiled_static and compiled graph pool reuse within the near-tie rule | CUDA + Triton |

## Not covered
Compiled sampling, EOS early exit, ragged + sampling, ragged + compiled, multi-turn cache reuse, and OOM recovery. Proposed design: `docs/PHASE5_SAMPLING_PLAN.md`.
