# EfficientGPT architecture: Qwen3-1.7B batch-1 greedy inference

Scope: how a request becomes tokens on the fast path (`assistant/backends.py` `HFBackend` with `quant='int8'`,
`fast_decode=True`), what each component guarantees, and how it was measured. Everything below is read from the code;
numbers are quoted only from `reports/EFFICIENTGPT_V3_RESEARCH_REPORT.md` ("V3 report") and are labelled. Target
hardware in the reports: one RTX 4060 (8 GB), Windows. Batch 1, greedy only on the fast path.

## 1. Overview

```
chat(messages, max_tokens, temperature)                       assistant/backends.py HFBackend.chat
  |
  v
tokenizer.apply_chat_template(add_generation_prompt, enable_thinking, tools) -> input_ids [1, p]
  |
  +-- temperature>0, or p+max_tokens > fast_cache_len (4096), or fast_decode off
  |       -> model.generate() + TextStreamer subclass (SyncStreamer)         (fallback, section 8)
  v
pick_bucket(p + max_tokens, GRAPH_BUCKETS=(256,384,512,1024,2048,4096), limit=fast_cache_len) -> capacity
  |
  v
_decoder[capacity] (LRU, max 3)  else GraphGreedyDecoder(model, capacity)  [StaticCache + pinned buffer]
  |      first use of a bucket: _capture() = 1-token prefill, 3 warm-ups on side stream, torch.cuda.graph
  v
cache.reset()                                    (in-place zeroing; tensor addresses unchanged)
  |
  v
PREFILL: eager HF forward over all p tokens, logits_to_keep=1, writes StaticCache
  |        int8 layers with >16 rows: int8 -> bf16 dequant + cuBLAS GEMM, scale in fp32
  |        argmax -> static `token` buffer ; `position` buffer := p
  v
DECODE LOOP (host)                                         DECODE STEP (device, one CUDA graph replay)
  copy token -> pinned host[i] (non_blocking)   ------>    compiled forward(token, position, StaticCache)
  record CUDA event                                          - int8 Triton GEMV per linear (custom op)
  graph.replay()  (not issued after the last token)          - attention over the WHOLE static cache
  emit tokens whose event.query() is true                    argmax(logits) -> token.copy_
  (block only if run-ahead > lookahead=8)                    position.add_(1)
  |
  v
on_token(tok) -> emit(): record perf_counter, drop EOS ids, tok.decode(all kept ids);
                 hold output if it ends in U+FFFD; send only the new suffix
  |
  v
EOS in set {generation_config.eos_token_id, tokenizer.eos_token_id, <|im_end|>} -> stop, discard run-ahead tokens
  |
  v
split_thinking_and_tools(raw) -> Reply(text, thinking, tool_calls, GenStats(ttft_s, total_s, decode_tok_s, token_times))
```

Layers of the repo that this document covers:

| Layer | Files |
|---|---|
| Serving backend | `assistant/backends.py` |
| Decode loop / graph | `research/realmodel/fast_decode.py` |
| int8 weights and kernel | `research/realmodel/int8_linear.py` |
| Benchmark methodology | `research/realmodel/bench_int8.py`, `analyze_int8.py` |
| Phase 5 (TinyStories-scale GPT) graph pools and attention kernels | `research/phase5/`, `docs/PHASE5_CONTRACT.md` |
| Kernel search | `research/kernelagent/` |

V3 applies Phase 5 ideas (CUDA-graph decode step, capacity buckets) to a real Hugging Face model; it does not reuse
the Phase 5 decoders themselves (the Phase 5 `DecoderPool` serves the repo's own small GPT, not Qwen3).

## 2. Model input processing

`HFBackend.chat` calls `tokenizer.apply_chat_template(messages, tools=..., add_generation_prompt=True,
enable_thinking=bool(think), return_tensors='pt', return_dict=True)` and moves tensors to the device. Batch size is 1
(`GraphGreedyDecoder.generate` raises `ValueError` otherwise, as it does for `max_new_tokens < 1` or
`p + max_new_tokens > max_cache_len`). The fast path is taken only when `fast_decode` is set, `temperature == 0`, and
`p + max_tokens <= fast_cache_len` (default 4096).

Post-processing of the generated text: `_chat_graph` strips `<|im_end|>` / `<|endoftext|>` and
`split_thinking_and_tools` separates `<think>...</think>` and `<tool_call>{json}</tool_call>` blocks (malformed JSON is
returned as a call named `__malformed__`, not dropped).

## 3. Prefill execution

Prefill is an ordinary eager forward of the same HF model (`self.model(input_ids=..., past_key_values=self.cache,
use_cache=True, logits_to_keep=1)`), not the compiled step, so it works for any prompt length that fits the bucket
and writes the first p cache slots. Only the last position's logits are computed (`logits_to_keep=1`) and fed to
`argmax`; the result seeds the static `token` buffer and `position` is filled with p.

For int8 weights the custom op `efficientgpt::int8_linear` branches on row count: more than `MAX_DECODE_ROWS = 16`
rows (prefill) dequantises `w` to bf16 and calls cuBLAS, `((x2 @ w.to(bf16).t()).float() * s).to(bf16)`; the scale is
applied in fp32 like the decode kernel. The V3 report states TTFT is worse than llama.cpp (54 vs 33 ms) and that
prefill "dequantises int8 weights; not optimised" (V3 report, sections 1 and 8).

## 4. KV-cache management

- `GraphGreedyDecoder.__init__` allocates `transformers.StaticCache(config, max_cache_len)`. Keys, values and the
  device-resident `cumulative_length` keep fixed addresses, which is what lets one captured graph serve every request.
  `generate()` calls `self.cache.reset()` (in-place zeroing) at the start of each request. No cache is carried across
  requests (no prefix reuse).
- Why capacity buckets: static-cache attention reads the whole allocated cache every step regardless of how many
  slots are filled. The `_chat_graph` comment records that a fixed 4096-slot cache made decode about 2x slower than a
  request-sized one (V3 bench; V3 report section 3: "fixed 4096-slot cache was 1.9× slower"; 55.2 vs 111.9 tok/s is 2.0×).
- `pick_bucket(need, buckets, limit)` returns the smallest bucket `<= limit` that is `>= need`, with `limit` itself
  as a final candidate, or `None`.
- `_chat_graph` keeps `self._decoder` as a dict `capacity -> GraphGreedyDecoder`, ordered by recency (pop and
  re-insert on use). `MAX_GRAPH_DECODERS = 3`: before creating a fourth, the oldest is deleted and
  `torch.cuda.empty_cache()` is called, because each bucket owns a KV cache and a graph memory pool. Capture happens
  lazily on a bucket's first request; `capture_s` is reported separately and excluded from TTFT.
- Stale slots from an earlier request are zeroed by `reset()`; the contract that stale K/V beyond `position` must be
  masked (Phase 5, `docs/PHASE5_CONTRACT.md`) applies to the Phase 5 decoders, which do not zero.
- Phase 5 analogue (`research/phase5/adaptive_pool.py`): `DecoderPool` with `choose_capacity` and capacities
  `(80, 128, 192, 256)`, keyed by `(batch, capacity)`, graphs resident for the pool's lifetime, requests that exceed
  capacity rejected with `ValueError`.

## 5. Quantised operations

`research/realmodel/int8_linear.py`.

- Weights: `quantize_weight` is per-output-channel symmetric: `s = max|w| / 127` (clamped at 1e-12),
  `q = round(w / s)` clamped to [-127, 127], stored as int8 `[N, K]` plus fp32 `s[N]`. It works in 8192-row chunks
  (bit-identical to whole-matrix, lower load peak; the code comment cites a 6.8 GB peak without chunking on the 8 GB
  GPU for the 151,936 x 2,048 lm_head).
- `quantize_model` replaces every `nn.Linear` in the decoder layers and the `lm_head` by `Int8Linear` (buffers
  `qweight`, `scale`; bias kept as a parameter), deleting each bf16 weight as it goes. Embeddings and norms stay bf16.
  A tied lm_head becomes an untied int8 copy and `config.tie_word_embeddings` is set to False.
- Decode kernel math: int8 is converted to bf16 in registers (exact: all of [-127, 127] is representable in bf16),
  multiplied on tensor cores with `tl.dot` (activation rows padded to M = 16), accumulated in fp32, and the per-channel
  scale is applied to the fp32 accumulator before the bf16 store.
- Kernel family, selected by `CONFIGS.get((k, n), DEFAULT)` as `(family, BLOCK_N, BLOCK_K, SPLITK, warps)`:
  `dot16` (one program per BLOCK_N output columns, loops over all of K) and `dot16_splitk` (`_part` writes fp32
  partials `[splits, 16, N]`, `_reduce` sums them and applies the scale). `DEFAULT = ('dot16', 32, 128, 0, 4)`.
  `CONFIGS` is empty, so every linear uses the default.
- Why the default is not split-K: the KernelAgent warm-L2 microbenchmark winner was split-K 8, but chosen end to end
  (`research/realmodel/sweep_configs.py`) `dot16` gave 9.51 vs 10.32 ms per decode step; the split-K reduce adds about
  197 extra kernels per token (comment in `int8_linear.py`; V3 report section 2-3).
- Integration: `@torch.library.custom_op('efficientgpt::int8_linear', mutates_args=())` with a `register_fake` that
  returns `x.new_empty(*x.shape[:-1], w.shape[0])`, so `torch.compile` and CUDA graph capture treat the op as one
  opaque node. `int8_matmul` raises `ValueError` above 16 rows (the kernels pad M to 16 and would leave rows
  uninitialised). The op requires bf16 activations.
- Other quant modes in `HFBackend.__init__`: `bf16`, `fp16`, `nf4` and `bnb8` (bitsandbytes), `torchao_int8`
  (torchao `Int8WeightOnlyConfig`, baseline, with the same untying), optional LoRA adapter via PEFT.

## 6. CUDA Graph capture and replay

`GraphGreedyDecoder` (`research/realmodel/fast_decode.py`).

What is captured: `_step()` compiled with `torch.compile(mode='default', fullgraph=True)` (Inductor without its own
cudagraphs) and captured once with `torch.cuda.graph`. The step is the model forward over `self.token` /
`self.position` / `self.cache` with `logits_to_keep=1`, then `self.token.copy_(argmax)` and `self.position.add_(1)`.
All state the step touches (token, position, cache tensors) lives in static buffers, so the replay needs no host
arguments.

`_capture()`: (1) a 1-token eager forward initialises the cache lazily; (2) three warm-up calls of the compiled step
on a side `torch.cuda.Stream` (compilation, autotune, allocator warm-up) with stream waits in both directions;
(3) capture; (4) `cache.reset()`. The warm-up writes into the cache, hence the reset.

Why the decode loop matters (profile in the V3 report section 2, refined by `research/realmodel/streamer_overhead.py`):
the int8 GEMV streamed weights at about 222 GB/s, at the DRAM limit, while the GPU was idle ≈ 2 ms per token (wall 10.95 vs 9.0 ms of kernels); ≈ 0.6 ms of it was host-side loop overhead in HF `generate()` (removed by the sync-free graph loop: 91.3 → 96.8 tok/s), the rest sat at kernel boundaries inside the step (reduced by removing ≈ 197 split-K reduce kernels: → 105.1 tok/s). The text streamer itself costs ≤ 0.07 ms/token (`runs/20261031_v4/streamer_overhead.json`). The decode loop therefore never reads a token before it is needed to be emitted.

Loop semantics in `generate()`:
- Each iteration enqueues a non-blocking copy of the current token into pinned `self.host[i]`, records a
  `torch.cuda.Event`, then calls `graph.replay()` (skipped once `launched == max_new_tokens`).
- Emission: while `emitted < launched` and (`events[emitted].query()` is true, or run-ahead exceeds `lookahead`, or all
  tokens are launched), the host synchronises that event, reads the token, calls `on_token`, and checks EOS.
  The host blocks only on tokens at least `lookahead` (default 8) steps old, never on the step it just launched.
- EOS: when an emitted token is in `eos_ids`, generation stops; up to `lookahead` further steps may already have been
  replayed, and their tokens are never emitted. The cache and buffers are dirty afterwards, which is why the next call
  starts with `cache.reset()`. The EOS token itself is passed to `on_token`; `_chat_graph.emit` drops it from text.
- Returned stats: `ttft_s` (to first emitted token, from before prefill), `total_s`, `tokens`, `capture_s`.
- `FastGreedyDecoder` in the same file is the earlier `mode='reduce-overhead'` variant that clones the token
  between replays; `HFBackend` uses `GraphGreedyDecoder`.

EOS set (`HFBackend._eos_ids`): `generation_config.eos_token_id` (int or list) union `tokenizer.eos_token_id` union
`convert_tokens_to_ids('<|im_end|>')`, so it matches what `generate()` stops on.

Streaming text (`_chat_graph.emit`): tokens are appended to `kept` and the whole list is re-decoded; if the text ends
in U+FFFD (a character split across byte-level tokens) nothing is emitted until it completes, otherwise only the new
suffix is passed to `on_token`. `emit` also appends `time.perf_counter()` per token into `GenStats.token_times`.

## 7. Triton kernels

int8 GEMV (`int8_linear.py`):
- `_dot16(X, W, S, O, M, N, K, ...)`: grid `(cdiv(N, BLOCK_N),)`; each program holds a `[16, BLOCK_N]` fp32
  accumulator, loops `k0` over K in `BLOCK_K` steps, loads x `[16, BLOCK_K]` (masked, other=0) and the transposed
  int8 tile of W, converts both to bf16, `acc += tl.dot(...)`, then multiplies by the loaded scale and stores bf16.
- `_part` / `_reduce`: split-K variant with `program_id(1)` selecting a K chunk of `KCHUNK`; `_reduce` sums
  `SPLITK` partials then scales. Selected only if a `CONFIGS` entry names a non-`dot16` family.

Decode attention (Phase 5, `research/phase5/decode_attention.py`; used by the Phase 5 TinyStories GPT decoders
`triton_*_graph` / `DispatchTritonGraphDecoder`, not by the Qwen3 path, which uses HF's attention in the compiled step):
- `_split_kernel` + `_combine_kernel` (`decode_attention`): split-K over keys. Each `(row, head, split)` program reads
  only slots `<= position[b]` (position read on device, so it is graph-capturable), keeps an fp32 online softmax and
  writes `(max, sum, weighted V)`; the combine kernel merges. `default_config`: `(64, 32, 4)` when `batch*heads >= 48`,
  else `(16, 16, 2)` as `(SPLIT, BLOCK_N, warps)`.
- `_single_pass_kernel` (`decode_attention_single_pass`, `block_n=64`, `warps=2`): one program per `(row, head)` walks
  slots `0..position[b]` with an fp32 online softmax, no partials and no combine. Found by KernelAgent-Ada.
- Dispatch `decode_attention_dispatch`: shape-only (positions are on device inside a graph). `use_single_pass` is true
  when `batch*heads >= SINGLE_PASS_MIN_PROGRAMS (48)` or `capacity <= SINGLE_PASS_MAX_CAPACITY_FEW_PROGRAMS (64)`;
  otherwise split-K. Thresholds come from `research/phase5/dispatch_study.py`.
- Contract: different fp32 summation order from masked SDPA, so near-tie token rule (section 11).

## 8. Output generation

- Fast path: greedy only (`argmax` inside the graph). No temperature, top-k, top-p or batching.
- Fallback in `chat`: `temperature > 0` or too-long prompts go to `model.generate()` with `SyncStreamer` (a
  `TextStreamer` subclass whose `put()` timestamps each generated token), `do_sample=temperature > 0`, and
  `top_p=0.95, top_k=20` for sampling. `cache_implementation='static'` is added if `static_cache=True`. The
  streamer runs on the calling thread on purpose (the code comment notes a worker thread would hang on exceptions and
  that Inductor CUDA-graph trees keep thread-local state).
- Both paths return `Reply` with `GenStats`; `decode_tok_s = (n-1) / (total - ttft)` on the graph path and
  `(n_new-1) / (t1 - first)` on the generate path (same quantity, as `total` and `t1` are both end of generation).
- Phase 5 also has opt-in exact sampling (`research/phase5/sampling.py`) and EOS masking (`eos.py`) for its own
  decoders; they are not wired into `HFBackend`.

## 9. Performance measurement

`research/realmodel/bench_int8.py` and `analyze_int8.py`.

- Fresh process per mode per round (`child()` writes JSON); rounds alternate the mode order; weights of two modes do
  not fit side by side on 8 GB and a fresh process removes cache/compile carry-over.
- Per child: load, `chat(max_tokens=64)` (first call, recorded as `first_call_s`), one `chat(256)` to settle
  compilation, then `--reps` timed calls of 256 new tokens. Each request has a unique `[request <uuid>]` prefix so
  there is no prefix reuse.
- Decode speed: `(n-1) / (total - TTFT)`, excluding the first token. TTFT is client wall clock to the first emitted
  token (prefill included, capture excluded on the graph path).
- Inter-token latency: `latency_fields` pools gaps between consecutive `GenStats.token_times` over all reps and reports
  `itl_ms_p50`, `itl_ms_p95`, `itl_samples`, plus `e2e_s_median`. Note for the graph path that `token_times` are host
  emission times, so gaps reflect run-ahead batching (several tokens can be emitted back to back), not per-token GPU
  step time.
- Memory fields: `weights_allocated_mib`, `load_peak_allocated_mib`, `torch_peak_allocated_mib` (steady-state, after
  `reset_peak_memory_stats`), `torch_peak_reserved_mib` (`max_memory_reserved`), `process_device_used_mib`
  (`mem_get_info` delta).
- Contention: `gpu_util()` (nvidia-smi) is recorded before each child; `contended = not util <= --busy-pct`.
- Statistics (`analyze_int8.py`): per-mode median of each round's median, ratio vs baseline as ratio of medians of
  round medians, 95% CI by bootstrap that resamples ROUNDS (5000 resamples, `random.Random(0)`; pooling reps as
  independent understates variance), plus paired per-round min-max. Quality: teacher-forced NLL on identical windows
  (through the prefill path, so it measures quantised weights, not the decode kernel), greedy agreement of the first 64
  tokens next to a bf16-vs-bf16 noise floor, and assistant-quality-v1.
- Measured (V3 report section 4; 256 new tokens, 4 fresh-process rounds, GPU 0-1%, RTX 4060): int8 + graph decoder
  111.9 tok/s, TTFT 54 ms; llama.cpp Q8_0 via Ollama 109.9 tok/s, TTFT 33 ms (ratio 1.018 [1.015, 1.051], reported as
  parity); HF `generate()` int8 99.7; torchao int8 84.4; HF bf16 + graph decoder 63.7; HF bf16 static/compile 59.3.
  Re-verification on final code (V3 report section 4): 1.029x [1.013, 1.076] vs llama.cpp over 3 rounds.
- Step ladder (V3 report section 3): 91.3 -> 96.8 (sync-free loop) -> 105.1 (no split-K) -> 111.9 tok/s (adaptive
  buckets).

## 10. KernelAgent-Ada pipeline

`research/kernelagent/`: template-based (no LLM) kernel search over small tasks drawn from measured hot paths.

```
tasks.py  TASKS[name]: seeded input generator + edge cases, eager reference, fp64 truth, bytes model,
   |      best_existing, dev shapes (search) and heldout shapes (final only)
   |      tasks: decode_attention, add_layernorm, temperature_softmax, topk_softmax, inverse_cdf_sample,
   |             argmax, int8_gemv
   v
templates.py  FAMILIES[task][family] = {space, default, body}; render() -> standalone Triton source with run(inp)
   |          (families e.g. int8_gemv dot16 / dot16_splitk, decode_attention split-K / single_pass)
   v
screen.py  AST policy screen: import allowlist (triton, triton.language, torch, math), allowlisted torch attributes,
   |       banned names (open, exec, eval, os, subprocess, F, nn ...), requires a @triton.jit kernel + run(inp)
   v
evaluate.py  fresh subprocess per candidate (--worker, JSON on stdin/stdout), wall-clock timeout (300 s),
   |         per-process GPU memory fraction cap
   |   correctness: fp64 truth contract
   |     float : max|cand - truth| <= 4.0 * max|eager_fp32 - truth| + floor, finite where truth is finite
   |     index : exact equality with reference      sample: chosen token's fp64 CDF interval contains u*total (1e-5)
   |     input-mutation guard: inputs cloned, candidate run on the clone, any change -> rejected
   |     seeds (0,1) plus per-task edge cases, on every dev shape
   |   timing: each callable in its own CUDA graph; trials interleaved in rotating order among
   |           candidate / eager / Inductor (torch.compile of reference) / best existing;
   |           bootstrap ratio CI over trials (research.phase5.analyze_ci.ratio_ci);
   |           GPU busy -> result marked `contended`, not to be reported
   v
search.py  strategies at equal budget: single_shot, random, local (neighbour moves from family defaults),
   |       local_meta (neighbour order guided by spills / roofline fraction). Score = geometric mean over dev shapes
   |       of median ms. Incorrect or screened candidates cost budget but never become incumbent.
   v
history.jsonl  (candidate id = hash of source; identical sources reused, not re-run)
   v
run_study.py  task x strategy x seed at equal budget, waits for an idle GPU, resumable; ONE held-out evaluation per
   |          task for the best dev candidate only
   v
end-to-end check (outside the package): research/realmodel/sweep_configs.py and bench_int8.py
```

Lesson recorded in the repo: a winner on warm-L2 isolated microbenchmarks must be confirmed in the real pipeline. The
split-K int8 config won the KernelAgent microbenchmark but lost end to end by 0.81 ms per decode step because of the
extra reduce kernels (`int8_linear.py` comment on `DEFAULT`; V3 report sections 2-3), so the shipped default is the
end-to-end winner.

## 11. Precision contracts

| Contract | Meaning | Where it applies |
|---|---|---|
| Exact | Tokens equal the reference exactly (`torch.equal`) | Phase 5 default eager/graph decoders vs `generate_reference` (greedy only; `docs/PHASE5_CONTRACT.md`); Phase 5 sampling given the same uniforms; `step_index_copy` vs `scatter_` step; `GraphGreedyDecoder` vs HF `generate()` on tiny random Qwen3 models, bf16 and int8 (`research/realmodel/test_fast_decode.py`, V3 report section 5) |
| Near-tie | Sequence passes if identical, or the first divergence sits where the reference top-1/top-2 logit margin is < 1e-2 (`benchmark_review.near_tie_match`) | Inductor-compiled Phase 5 modes, Triton decode attention (different fp32 summation order), ragged batch across batch sizes; `GraphGreedyDecoder` on real Qwen3-1.7B bf16 (V3 report: 10/16 identical 64-token outputs, mean common prefix 45.8 vs 42.9 between two HF bf16 paths, per `fast_decode.py` docstring) |
| Approximate | Weights changed; no token contract, quality measured | int8 (`int8_linear.py`): teacher-forced NLL -0.15% [-0.21%, -0.09%] vs bf16, quality-v1 39/52 equal to bf16, greedy common prefix 18.8 of 64 tokens vs bf16 (V3 report section 5); Phase 5 `fp16_*` / `bf16_*` low-precision modes (no token contract); bnb8 / nf4 / torchao baselines |

Kernel-level verification for the int8 GEMV (V3 report section 5): fp64-truth contract, mutation tests (missing scale,
skipped K block, unsigned reinterpretation rejected), held-out shapes.

## Limits (from the code and V3 report section 8)

One GPU and one model measured; batch 1; greedy only on the fast path; sampling and batching use `generate()`; prompts
above `fast_cache_len` (4096) use `generate()`; no prefix/KV reuse across requests; prefill is not optimised.
