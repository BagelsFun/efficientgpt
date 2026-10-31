# Upstream contribution candidates (none submitted; each needs the owner's approval)

Ranked by evidence strength. Each item links to the measurement in this repository. Drafts are written so that they could be
filed after review. Nothing here has been sent.

## 1. transformers: enable `DeferredStopCheck` on CUDA (strongest; measured)
**Finding.** transformers 5.21 already has `DeferredStopCheck` (`generation/utils.py`). It reads the stop flag one step
late through pinned memory and an event, so the host never waits on the device. It is enabled only for MPS:
`is_supported` returns False unless `device.type == "mps"`, with the comment *"we should enable it for cuda if we observe
perf gains"*.

**Measurement** (`research/realmodel/streamer_overhead.py`, `runs/20261031_v4/streamer_overhead.json`):
- Qwen3-1.7B bf16, static cache + torch.compile, greedy, 256 forced new tokens, RTX 4060 / Windows / torch 2.11 /
  transformers 5.21.0, one process, interleaved reps.

| generate() variant | ms/token (6 reps, median) | tok/s |
|---|---|---|
| default | 15.884 (15.87–15.90) | 63.0 |
| `DeferredStopCheck` forced on CUDA | **15.221** (15.20–15.25) | **65.7 (+4.3%)** |
| default + TextStreamer | 15.948 | 62.7 |
| deferred + TextStreamer | 15.232 | 65.7 |

Greedy outputs were token-identical between the default and deferred paths without a streamer (checked in
this run; the streamer variants were not compared and should be before filing). For reference, a fully graph-captured loop (EfficientGPT) reaches 15.159
ms/token on the same setup, so the deferred check recovers almost all of the host overhead.

**Draft PR description.**
> Enable `DeferredStopCheck` for CUDA devices. On an RTX 4060 (Qwen3-1.7B, bf16, static cache, torch.compile, greedy) this
> reduces per-token time from 15.88 to 15.22 ms (+4.3% decode throughput) with identical greedy outputs; the gain is the
> same with a TextStreamer. Repro script attached.
>
> Open questions for maintainers:
> - behaviour with `return_dict_in_generate` / cache rollback (already handled by `is_croppable`);
> - assisted decoding (excluded, as for MPS);
> - multi-GPU `synced_gpus`;
> - data-centre GPUs (only one consumer GPU measured here).

**Before submitting:**
- measure on Linux, a second GPU and dynamic cache;
- run the transformers generation test suite with the change;
- check transformers' contribution guidelines.

## 2. transformers documentation: static-cache attention cost grows with `max_cache_len`
**Finding.** With a static cache, every decode step attends over the whole `max_cache_len` (masked). In EfficientGPT a
fixed 4096-slot cache made batch-1 decoding of Qwen3-1.7B about 2× slower than a request-sized cache: 55.2 vs 111.9 tok/s
(`runs/20261031_v4/v3_bench/` vs `v3_final/`).

**Candidate:** a short documentation note (or an issue) recommending a `max_cache_len` close to prompt + new tokens, or
bucketed caches for servers, with the measurement.

**Before submitting:** reproduce with `generate(cache_implementation="static")` and explicit `max_cache_len` values on
the HF path (our measurement is from EfficientGPT's own loop).

## 3. torchao: int8 weight-only decode performance depends on Inductor max-autotune
**Finding.** `Int8WeightOnlyConfig` decode speed was +2–7% higher with Inductor max-autotune in separate runs (82.8 vs
84.4–90.5 tok/s). The reviewer noted that its weight-only mm fuses the dequantisation only under max-autotune.

**Status: weak evidence.** There is no same-run comparison. A useful contribution would be a same-process A/B and, if
confirmed, a documentation note.

## 4. Methodology write-up (not code): microbenchmark winners must be confirmed end to end
KernelAgent-Ada's warm-L2 winner for the int8 GEMV (split-K) was 8% slower inside the model, because of an extra reduce
kernel per linear (≈ 197 per token). A blog post or a talk proposal, rather than a PR.

## Not candidates (checked)
- **transformers TextStreamer overhead:** 0.01–0.06 ms/token over no streamer (0.04 ms of it the streamer hook itself), negligible.
- **triton-windows:** no reproducible bug was encountered during this work.
