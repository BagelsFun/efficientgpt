# EfficientGPT Assistant

A modular local assistant for an RTX 4060 8 GB / 32 GB RAM machine. It is independent of the TinyStories inference
research in `research/` (that 12M-parameter model is not a general-purpose LLM and is not used here).

## Architecture (`assistant/`)

```
cli.py ── Agent (agent.py) ──┬── Backend (backends.py)      OllamaBackend (HTTP, stdlib) | HFBackend (transformers; bf16/fp16/nf4)
                             ├── ContextManager (context.py) token-budgeted history, system prompt always kept
                             ├── Retriever (retrieval.py)    BM25 over repo .md/.py/.txt, optional nomic-embed-text, RRF fusion
                             └── Toolbox (tools.py)          read: read_file, list_dir, search_code, retrieve_docs
                                                             execute (approval per call): run_unittests (repo modules only)
bench/  quality_v1.py (52 graded items) · grading.py · run.py (speed/memory/quality) · agent_e2e.py (8 repo questions × 3 configs)
```

- **Swappable backends:** anything with `chat(messages, tools, max_tokens, temperature, think, on_token) -> Reply` works. A
  stronger local model is a new backend spec (`ollama:<tag>` or `hf:<repo>`), not a rewrite.
- **Safety model:** tools are confined to the workspace root. Checkpoints, `data/`, `results/`, `.git`, `.venv` and
  secret-like files are refused. There are no write or delete tools. Execute-level tools need an approval callback per call
  (the CLI asks y/N; benchmarks never approve). Model-generated code in the quality suite runs in `python -I` in an empty temp
  dir with a timeout, after a screen; that is a filter, not a sandbox.
- **Honest measurement:** TTFT is client-measured to the first streamed token; Ollama decode/prefill rates come from server
  token counts and durations; HF decode rate is (tokens − 1) / time after the first token. Greedy decoding, thinking off
  unless stated.

## Environment (checked 2026-10-30, before any install)
- Ollama 0.40.12 running, with `qwen3.5:4b` (4.2B, Q4_K_M, tools/thinking/vision, 262k context) and `nomic-embed-text` (137M,
  768-d) already pulled by the owner.
- `.venv`: torch 2.11.0+cu128, transformers 5.21.0, tokenizers, safetensors, datasets present.
- Installed with owner approval: `Qwen/Qwen3-1.7B` (Apache-2.0, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, 3.80 GiB,
  HF cache) and peft 0.21.2, accelerate 1.15.0, bitsandbytes 0.50.2, psutil 7.2.2 (dry run first; torch unchanged; freezes
  in `results/assistant/`).
- About 2.9 GB of the 8 GB of VRAM is used by other processes on this machine, so the assistant has about 5 GB.

## Usage
```powershell
.\.venv\Scripts\python.exe -m assistant.cli --backend ollama:qwen3.5:4b
.\.venv\Scripts\python.exe -m assistant.cli --backend hf:Qwen/Qwen3-1.7B --quant bf16 --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
.\.venv\Scripts\python.exe -m assistant.bench.run --backend hf:Qwen/Qwen3-1.7B --quant bf16 --out results\assistant\bench\x.json
.\.venv\Scripts\python.exe -m assistant.bench.agent_e2e --backend ollama:qwen3.5:4b --out results\assistant\bench\x_agent.json
.\.venv\Scripts\python.exe -m unittest assistant.test_assistant assistant.bench.test_quality_suite
```

## Benchmarks
`assistant-quality-v1` (`assistant/bench/quality_v1.py`): 16 Python functions with hidden tests, 16 integer-answer word
problems, 20 PyTorch/CUDA/GPU multiple-choice questions, all written for this project. Every category is split half
`dev` / half `test`. **The `test` half is held out for fine-tuning comparisons and must never be used for prompt tuning or
training.** Reference solutions are verified by `assistant.bench.test_quality_suite`. This is a small project benchmark: it
measures relative progress on this machine, not general capability, and 52 items give wide uncertainty (±1 item ≈ ±2%).

## Results (2026-10-30, greedy, thinking off; EXPERIMENT_LOG ASSIST-01)

| model / backend | quality v1.1 | held-out test | decode tok/s | memory |
|---|---|---|---|---|
| qwen3.5:4b Q4_K_M (Ollama) | **47/52** | **22/26** | **69.6–71.2** idle (≈ 50 under load) | 3.13 GiB VRAM (`ollama ps`) |
| Qwen3-1.7B bf16 (HF, static cache) | 39/52 | 18/26 | **57.7** idle | 3.35 GiB peak allocated |
| Qwen3-1.7B bf16 (HF, dynamic cache) | 39/52 | 18/26 | 26.8 idle (17–23 under load) | 4.3 GiB |
| Qwen3-1.7B nf4 (HF) | 36/52 | 17/26 | 20 | 2.3 GiB |

Agent end-to-end (assistant-agent-v1.1, 8 repo questions): qwen3.5:4b 0/8 model-only, 8/8 with retrieval, 8/8 with
retrieval + tools; Qwen3-1.7B 0/8, 6/8, 6/8 (EXPERIMENT_LOG ASSIST-02; an earlier run leaked answers and is discarded).

**Default for interactive use: `ollama:qwen3.5:4b`** (best measured quality and fastest on this GPU). **Base for adaptation
experiments: Qwen3-1.7B** (fits QLoRA on 8 GB; HF weights available).

Idle-GPU speed (2026-10-31, ASSIST-04, `runs/20261031_v4/assistant_speed_idle/`): Ollama cold-prefill TTFT (unique prompt
prefixes) 84 ms / 253 ms / 706 ms for 61 / 713 / 2,229 prompt tokens. HF static cache vs dynamic, 10 interleaved reps x 256
tokens: **2.15x decode [2.14, 2.16]** (57.7 vs 26.8 tok/s), TTFT 45 vs 43 ms; first call 6.2 s vs 2.8 s. 57.7 tok/s reads
≈ 196 GB/s of bf16 weights, ≈ 93% of the measured 211 GB/s DRAM bandwidth, so further batch-1 HF speed needs fewer weight
bytes (quantisation), not kernels. Static cache is now the CLI default for HF backends (`--no-static-cache` to opt out;
outputs are not bit-identical to the dynamic path: 40/52 identical, equal quality, ASSIST-03). Benchmarks keep the dynamic
cache so earlier scores stay reproducible. The ASSIST-01/03 speeds were measured under an external GPU load.
