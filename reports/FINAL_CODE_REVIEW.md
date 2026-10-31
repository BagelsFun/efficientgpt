# Final code review (Phase F), 2026-10-31

Branch `showcase` (from tag `v3`). This file records what was reviewed and tested, what was found, and what remains open.
It does not claim the code is bug-free.

## Reviews performed
| Review | Scope | Findings | Outcome |
|---|---|---|---|
| Independent review, Phase F (separate agent, read-only) | all code changed since `v3`; every number in README, BENCHMARK_REPORT §1a and the eight new reports/docs checked against raw evidence | 10: 2 code issues (demo crash paths, a no-op mutation pattern), 1 misleading comment (Ollama chunks), 4 wording/sourcing issues (CI wording, torchao quality source, autotune gain, TTFT estimate shown as measurement), 1 unit error (GB = MiB/1000), 2 links to not-yet-created files | all fixed in <private-commit>, except the links: `FINAL_CODE_REVIEW.md` now exists; the demo-recording link points to the recording made below |
| Earlier independent reviews | V3 code (7), R2 fairness (6), R1 (9), V1 pre-release (10), KA-06 (10), KA-02 (8) | see `docs/EXPERIMENT_LOG.md` | fixed or disclosed |

The reviewer found these numbers correct against the evidence:
- throughput 111.9 / 109.9 / 99.7 / 84.4 / 63.7 / 59.3 tok/s and all their CIs;
- the v3_verify re-check (111.1 / 107.9, 1.029 [1.013, 1.076]);
- TTFTs, ΔNLL, quality-v1 and the greedy prefix;
- the optimisation ladder, the profile figures, the 149/149 test counts and the demo output.
No document claims EfficientGPT is faster than llama.cpp.

## Tests actually run
| Run | Result | Evidence |
|---|---|---|
| Clean `git worktree` of commit <private-commit> (all suites, hash-checked inputs) | **149/149 passed, 0 failed, 0 skipped** (Phase 5 86, KernelAgent-Ada + real model 30, assistant 33) | `runs/20261031_v4/showcase_clean_checkout.json` |
| Affected tests after the review fixes (backends, demo) | the fixes touched a comment, demo code and docs only; demos re-run below | — |
| Demo 1 (`research.realmodel.demo`) on the final code | runs end to end: streams 96 tokens, stats, bf16 agreement 46/96 (speed not quoted: another application was using the GPU) | `runs/20261031_v4/demo1_busy_gpu.txt` |
| Demo 3 (`research.showcase.demo_kernelagent`) on a busy GPU | steps 1–3 pass (screen rejects `import os`; correct kernel 16/16; mutated kernel rejected, error 5.8e5 > bound 179); step 4 stops cleanly because the evaluator refuses contended timings, which is the intended behaviour | `runs/20261031_v4/demo_kernelagent_busy_gpu.txt` |
| Reproduction commands | every `python -m` module and flag in the docs exists (reviewer); the clean checkout and demos were executed | above |

## Defects found and fixed during the showcase
- **Measurement:** the first showcase benchmark ran under an external GPU load (19–30% utilisation from other
  applications). It was detected by the harness and kept as `showcase_bench_CONTAMINATED_external_gpu_load/`; it is not used.
- **Documentation:**
  - memory quoted in "GB" that were MiB/1000;
  - "statistically on par" when the CI excludes 1;
  - torchao quality from default compile shown on the max-autotune row;
  - an estimate of the TTFT dequantisation cost presented as a profile.
- **Code:**
  - demo crash paths;
  - a no-op mutation pattern;
  - a misleading comment on Ollama streaming granularity.

## Idle-GPU items (completed after the owner closed other GPU applications)
| Item | Result | Evidence |
|---|---|---|
| Showcase latency run (4 rounds, all processes at 0–1%) | EfficientGPT 117.3 tok/s; parity with llama.cpp 1.011 [1.009, 1.013]; 1.30× torchao; 1.86× bf16; inter-token latency p50/p95 8.49/8.55 ms | `runs/20261031_v4/showcase_bench/REPORT.md` |
| Prefill profile | int8→bf16 copies 21.3 ms of GPU time; eager-prefill host overhead ≈ 21–25 ms in both precisions; corrected the earlier 40 ms arithmetic bound | `runs/20261031_v4/showcase_profile_prefill.json` |
| Timed Demo 3 | all six steps; held-out 20/20 correct | `runs/20261031_v4/demo_kernelagent_output.txt` |
| Demo 2 (1 round) | 117.3 / 115.6 / 63.0 tok/s, consistent with the 4-round run | `runs/20261031_v4/demo_compare_output.txt` |

## Final validation (final commit <private-commit>)
- Clean `git worktree` regression: **149/149 passed, 0 failed, 0 skipped**
  (`runs/20261031_v4/showcase_clean_checkout_final.json`).
- Checkpoints: 7/7 SHA-256 OK.
- Remote unchanged: `main` cc5d255, `efficientgpt-v2` <private-commit>, `v2` → <private-commit>, `v3` → <private-commit>.
- The `showcase` branch is local only. Git status is clean apart from this evidence file, which is committed with it.

## Remaining risks
- Benchmarks on this machine are sensitive to other GPU applications; the harness flags but cannot prevent this.
- `requirements-lock.txt` has not been installed into a fresh virtualenv.
- Ollama's inter-token latency is measured per streamed chunk (occasionally two tokens).
