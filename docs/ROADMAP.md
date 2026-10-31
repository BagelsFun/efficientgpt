# Roadmap (living document)

Owner priorities, newest first. Each item lists its exit evidence. Status is factual: DONE means committed with evidence,
WIP means partially built, PLANNED means nothing has been executed.

## Mission (2026-10-30): strongest practical local assistant on RTX 4060 8 GB / 32 GB RAM
Inspiration: frontier assistants. **No frontier-level claims without independent benchmark evidence.** The TinyStories
decoder stays an independent inference-research project and is never presented as a general-purpose LLM.

| Phase | Item | Status | Exit evidence |
|---|---|---|---|
| A | Preserve research, checkpoints, results | DONE | `docs/CHECKPOINT_MANIFEST.sha256`; 80-test suite; committed run dirs |
| B | Local stack audit (Ollama, models, packages) | DONE | `docs/ASSISTANT.md` §Environment |
| B | Modular assistant: backends, context, retrieval, permissioned tools, CLI | DONE (prototype) | `assistant/`, `assistant.test_assistant` |
| B | Measured baseline: TTFT, tok/s, memory, quality (assistant-quality-v1) | DONE (Ollama cold TTFT + idle-GPU re-run open) | `results/assistant/bench/*` → `docs/ASSISTANT.md` §Results |
| B | Agent end-to-end on repo questions (model-only vs RAG vs RAG+tools) | DONE (v1.1, leak-free) | `assistant.bench.agent_e2e` |
| C | Specialisation data (Python / PyTorch / CUDA / GPU perf): licensed, validated, deduplicated | WIP (pipeline v2 + pilot; 18 examples; scale-up needs ≈ 8–15 GPU-hours) | dataset card + dedup report vs the held-out `test` split |
| C | Small QLoRA adapter on Qwen3-1.7B (nf4 base) | WIP (fit test passed: 3.4 GiB peak; no real adapter yet) | VRAM fit, train/val loss, held-out comparison vs base |
| C | Scale up only after a measured gain | PLANNED | CI-backed improvement on untouched test items |
| D | Tool-use planning, code testing, error recovery, structured verification | PLANNED | agent-v2 suite; model-only vs tool-assisted reported separately |
| D | RL (e.g. GRPO) only after SFT and agent improvements are evaluated | PLANNED | ablation vs equal-compute non-RL |
| E | Inference engineering for the assistant (reuse Phase 5 lessons, KernelAgent-Ada) | WIP (HF static cache ≈ 1.5× decode, indicative) | fair vs eager / torch.compile / llama.cpp-class engines |
| F | Portfolio: reports, benchmark suites, architecture docs | ONGOING | `reports/`, `docs/` |

## Returning to KernelAgent-Ada (parked 2026-10-30)
State (2026-10-31, KA-01): 6 tasks, 14 template families, AST screen, **evaluator (step 1) and search controller built;
smoke (step 2) passed for correctness (14/14)**; mutation tests 10/10, controller tests 8/8. Steps 3–4 done 2026-10-31 (KA-03,
`reports/KERNELAGENT_ADA_REPORT.md`). Remaining: step 5 (bridge) and end-to-end integration of the decode-attention winner. Original plan:
1. `research/kernelagent/evaluate.py`: isolated subprocess per candidate (timeout, `set_per_process_memory_fraction`),
   correctness vs fp64 with the tasks' dtype-aware rules, CUDA-graph replay timing interleaved with eager / Inductor / best
   existing, Triton metadata (registers, spills, shared memory), roofline fraction vs the measured 211 GB/s.
2. Smoke: every default template on every dev shape (expect and log compile errors).
3. Strategies at equal budget: single-shot, random, reflection without profiler, reflection with profiler (seeds ≥ 2).
4. Held-out evaluation once; ablation table; `reports/KERNELAGENT_ADA_REPORT.md`.
5. Bridge to Phase E: point the same harness at assistant-model hot ops (e.g. sampling, RMSNorm, decode attention for
   Qwen-shaped heads), where a measured bottleneck exists.

## EfficientGPT Phase 5 (inference research) — complete
See `reports/PHASE5_REPORT.md`. Open items: GEMM-bound step (fp16 approximate mode is the measured lever), cause of the
≈ 2.9% per-process GPU speed level (below application control; OFFSET-02), compiled sampling, ragged + sampling.
