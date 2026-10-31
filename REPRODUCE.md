# Reproducing EfficientGPT

Environment: `docs/ENVIRONMENT.md` (Windows 11, RTX 4060 8 GB, Python 3.11 `.venv`, torch 2.11.0+cu128,
triton-windows 3.7.1). All commands run from the repository root with `.\.venv\Scripts\python.exe` (written `python` below).
Before any timing: `nvidia-smi` should show ≈ 0% utilisation (close games/launchers; timings under load are invalid and
the harnesses flag them). Inputs that are not in git: the TinyStories checkpoint `out-tinystories-v2/baseline-2500.pt`
(SHA-256 in `docs/CHECKPOINT_MANIFEST.sha256`), tokenizer `data/tinystories_v2/tokenizer.json` and validation split
`data/tinystories_v2/val.bin` (used by `test_gpu_combinations`; hashes recorded by `research.release.clean_checkout`); for the assistant,
Ollama with `qwen3.5:4b` and `nomic-embed-text`, and HF `Qwen/Qwen3-1.7B` revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.

**The TinyStories checkpoint and data are not distributed.** To recreate them: `python prepare_v2.py` (downloads
`roneneldan/TinyStories`, trains the tokenizer, writes `data/tinystories_v2/`), then `python train.py config/tinystories_v2.py`
(2500 iterations, about 12M parameters). A retrained checkpoint will not match the SHA-256 in `docs/CHECKPOINT_MANIFEST.sha256`,
so Phase 5 absolute numbers can differ slightly; the Qwen3-1.7B results (V3) need none of this — see
`docs/REPRODUCE_ON_ANOTHER_MACHINE.md`.

## 1. Tests (correctness gates)
```
python -m unittest research.phase4.test_decoder research.phase4.test_compact research.phase5.test_pool research.phase5.test_prefix research.phase5.test_graph_gpu research.phase5.test_analyze_ci research.phase5.test_near_tie research.phase5.test_compiled_step research.phase5.test_inprocess research.phase5.test_eos research.phase5.test_sampling research.phase5.test_ragged research.phase5.test_gpu_combinations research.v1.test_demo research.v1.test_report
python -m unittest research.kernelagent.test_evaluate research.kernelagent.test_search research.realmodel.test_int8_linear research.realmodel.test_fast_decode
python -m unittest assistant.test_assistant assistant.bench.test_quality_suite assistant.train.test_train
python -m unittest research.release.test_release
```
All four suites (156 tests) from a pristine `git worktree`, with the hash-checked private inputs:
`python -m research.release.clean_checkout --ref HEAD --out runs/<stamp>/clean_checkout.json` (add `--no-inputs` to
simulate an external reproducer; tests needing the private TinyStories checkpoint then skip). Latest results (private inputs present):
`runs/20261031_v4/recognition_clean_checkout.json` (156/156); which tests need a GPU: `runs/20261031_v4/test_gpu_census.json`
(recorded at 154 tests; the two release tests added since are CPU-only).

## 2. Phase 5 decoder benchmark (12.19M TinyStories GPT)
```
python -m research.phase5.benchmark_inprocess --modes adaptive_graph compiled_triton_full_graph prefix_sdpa_eager --output research/phase5/ci_runs/<stamp>/benchmark_raw.csv
```
In-process, interleaved, 3 fresh processes × 6 blocks × 12 requests, 12 workloads; writes `CI_REPORT.md` with bootstrap CIs
(promotion rule: ratio ≥ 1.05 and CI excludes 1). Add `fp16_compiled_triton_full_graph` for the approximate fp16 mode,
`dispatch_triton_adaptive_graph` for the KernelAgent-Ada attention. Re-analyse: `python -m research.phase5.analyze_ci --raw <csv> --output <md>`.

## 3. KernelAgent-Ada
```
python -m research.kernelagent.evaluate --smoke --no-timing          # every template, correctness only
python -m research.kernelagent.run_study --out results/kernelagent/study_<stamp> --busy-pct 15
python -m research.phase5.dispatch_study --out runs/<stamp>/dispatch_grid.json    # shape rule evidence
python -m research.phase5.attention_error_audit --out runs/<stamp>/attention_error_audit.json
```
The study runs 4 strategies × 6 tasks at budget 12 and evaluates each task winner once on held-out shapes; it refuses
contended timings. Report: `reports/KERNELAGENT_ADA_REPORT.md`.

## 3b. Real model: Qwen3-1.7B weight-only int8 decode (R1)
```
python -m research.kernelagent.run_study --out runs/<stamp>/int8_study --tasks int8_gemv --busy-pct 15
python -m research.realmodel.bench_int8 --out runs/<stamp>/int8_e2e --rounds 4 --reps 5 --quality
python -m research.realmodel.analyze_int8 --dir runs/<stamp>/int8_e2e        # REPORT.md: round-level CIs, ΔNLL, agreement
python -m assistant.cli --backend hf:Qwen/Qwen3-1.7B --quant int8           # use it interactively (approximate mode)
```
Each mode runs in fresh processes (≈ 40–60 min for 5 modes × 2 rounds with quality; bitsandbytes modes are slow). The int8
kernel config shipped in `research/realmodel/int8_linear.py` (`DEFAULT`) is the study winner.

## 4. Assistant
```
python -m assistant.demo                                                   # 2 documented tasks, auto-graded, exit 0 = pass
python -m assistant.bench.run --backend ollama:qwen3.5:4b --skip-quality --repeats 5 --out runs/<stamp>/speed.json
python -m assistant.bench.static_cache_ab --skip-equivalence --out runs/<stamp>/static_cache_ab.json
python -m assistant.bench.run --backend ollama:qwen3.5:4b --skip-speed --out runs/<stamp>/quality.json   # quality v1.1, 52 items
python -m assistant.bench.agent_e2e --backend ollama:qwen3.5:4b --configs model_only rag_tools --system calibrated --out runs/<stamp>/agent.json
python -m assistant.bench.coding_agent --backend ollama:qwen3.5:4b --out runs/<stamp>/coding.json
python -m assistant.cli                                                    # interactive chat (tools need approval to execute)
```
Agent answers are re-gradable without the model: `python -m assistant.bench.regrade_agent <json> --labels <manual_labels.json>`.

## 5. Where each published number comes from
`BENCHMARK_REPORT.md` maps every headline number to its evidence directory; `docs/EXPERIMENT_LOG.md` has the full history,
including discarded runs (leaked, contaminated, launch-bound) and why.
