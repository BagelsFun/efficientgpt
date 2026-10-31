# transformers: enable `DeferredStopCheck` on CUDA (prepared, not submitted)

**Status:** evidence and a minimal patch are ready. Nothing has been filed upstream.

## Read this before submitting
transformers' `CONTRIBUTING.md` (checked 2026-10-31) asks first-time contributors not to use code agents to create issues
or PRs, and says PRs that look agent-written will probably be closed. This material was prepared with an AI coding
assistant. Use it as evidence and notes only:
1. reproduce the measurement yourself;
2. understand the change well enough to defend it in review;
3. write the issue and the PR in your own words.

The drafts below are an outline of the facts, not text to paste.

## The finding
- **What upstream has.** `DeferredStopCheck` came in with PR huggingface/transformers#47975 (merged 2026-09-04). It reads the
  per-step "all sequences finished?" flag one step late, through pinned memory and an event, so the host never blocks on the
  device. The cost is one extra forward, which is rolled back.
- **Where it is enabled.** `DeferredStopCheck.is_supported` turns it on only for `device.type == "mps"`. The comment says
  *"we should enable it for cuda if we observe perf gains"*.
- **Why CUDA was left out.** The PR expected no CUDA gain, because the GPU step overlaps the host's work. Its description
  lists `cuda` and `mps`; the merged code enables MPS only.
- **What we measured.** On an RTX 4060 it is a consistent +5%, in both eager and compiled static-cache decoding, with
  identical outputs.

## Patch
[`deferred_stop_cuda.patch`](deferred_stop_cuda.patch) (6 lines) applies cleanly to transformers `main` at b41ca9c
(2026-10-26). It widens the device check to `("cuda", "mps")` and leaves every other condition unchanged:
- assistant models are still excluded;
- a returned cache must still be croppable.

## Evidence
Script: `research/realmodel/deferred_stop_cuda.py`. Result: `runs/20261031_v4/deferred_stop_cuda.json`. The script applies
exactly the patched `is_supported` by monkeypatching, and compares it with the unmodified library in one process.

Test setup:
- **Model and decoding:** Qwen3-1.7B, bf16, greedy, batch 1.
- **Workload:** 256 forced new tokens, 12 interleaved repetitions per variant with alternating order.
- **Machine:** RTX 4060 8 GB on Windows 11; torch 2.11.0+cu128, transformers 5.21.0.
- **GPU load:** 0% before and after each timing block.

| Regime | Default ms/token | Patched ms/token | Speed-up [95% bootstrap CI] |
|---|---|---|---|
| eager, `DynamicCache` (default `generate()`) | 36.83 | 35.01 | **1.052 [1.050, 1.058]** |
| `cache_implementation="static"` (compiled step) | 16.94 | 16.10 | **1.053 [1.048, 1.059]** |

In both regimes every one of the 12 patched repetitions was faster than every one of the 12 default ones. An earlier run
(`deferred_stop_cuda_run1_util_sampling_invalid.json`, 8 reps) gave the same ratios, 1.056 and 1.053. Its contention
readings are invalid because utilisation was sampled right after the script's own GPU work; the script was fixed.

**Correctness, in both regimes.** Each case below produced exactly the same tokens with and without the patch:
- 4 prompts: two stop at EOS (after 4 and 20 tokens) and two reach `max_new_tokens=160`;
- a `max_new_tokens` cut-off;
- a left-padded batch of 3 whose rows finish at different steps;
- a `TextStreamer`: the streamed text equals the returned text;
- `return_dict_in_generate` + `output_scores`: equal sequences and bitwise-equal scores.

The script also records whether the deferred path actually engaged. It did in every patched case except one: with a
returned `StaticCache`, which cannot be cropped, the gate correctly kept it off.

## Caveats
- **One consumer GPU on Windows.** Windows' WDDM driver model has higher sync and launch costs than Linux, so the gain may be
  smaller on Linux and data-centre GPUs. **Measure on Linux before filing.**
- **One model.** Qwen3-1.7B is the only model measured. Smaller models (more host-bound) and larger ones (more GPU-bound)
  should both be checked.
- **Not covered:** multi-GPU / `synced_gpus`; sampling (`do_sample=True`); the CUDA run of upstream's own test suite.

## Before filing (owner)
1. Fork transformers and branch from `main`, then `git apply deferred_stop_cuda.patch`.
2. Run `pytest tests/generation/test_utils.py -k "DeferredStopCheck"`. Upstream's `DeferredStopCheckIntegrationTest`
   already forces both code paths on any accelerator. Also run the generation test suite on a CUDA machine.
3. Re-run `python -m research.realmodel.deferred_stop_cuda` on Linux, ideally on a second GPU, with a small model (for
   example SmolLM2-135M) and with Qwen3-1.7B.
4. Open an **issue** first (a small, maintainer-friendly step): the measurement table, the script, and the question whether
   CUDA should be enabled. Offer the PR if maintainers agree.

## Outline for the issue (facts to cover, in your own words)
- What `DeferredStopCheck` does and that it is MPS-only, quoting the comment about CUDA.
- The table above, with hardware, OS, versions, and how it was measured (interleaved, 12 reps, CI).
- The correctness checks, plus the observation that the gain also appears in eager mode, contrary to the expectation in
  #47975.
- Caveats: one consumer GPU, Windows; Linux numbers if you have them.
- The one-line change, and an offer to send a PR.

## Outline for the PR (if maintainers agree)
- Title: `[generate] enable DeferredStopCheck on CUDA`.
- Body: link the issue; the measurement table; the tests run (upstream generation tests on CUDA, plus the script's
  correctness cases); the caveats.
- Diff: `deferred_stop_cuda.patch` only. No new tests are needed, because the existing integration test already covers
  both paths on any accelerator. Ask maintainers whether they want a CUDA-gated test.
