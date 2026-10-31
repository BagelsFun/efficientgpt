# EfficientGPT demos

Four short demonstrations for a reviewer. All run on one CUDA GPU (developed on an RTX 4060 8 GB, Windows 11) from the
repository root with the project's virtual environment (`.\.venv\Scripts\python.exe`, written `python` below).
**Close other GPU applications first**: timings taken while another program uses the GPU are wrong, and the benchmark
harness flags them (`contended`).

Prerequisites: Hugging Face cache with `Qwen/Qwen3-1.7B` revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (downloaded
automatically by transformers on first use, ≈ 4 GB); for Demo 2, Ollama with `ollama pull qwen3:1.7b-q8_0`.

## Demo 1: real-model inference (≈ 1 min)
```
python -m research.realmodel.demo
```
Loads Qwen3-1.7B, quantises every linear layer to int8 (EfficientGPT Triton kernel), captures the decode step as one CUDA
graph, streams a greedy answer token by token, prints TTFT / decode tok/s / GPU memory, then reloads bf16 and reports how
many leading tokens agree. Expected output (abridged, idle-GPU recording `runs/20261031_v4/demo1_output.txt`):
```
[int8 + graph] 185 tokens, TTFT 57 ms, decode 108.3 tok/s, GPU memory 3.35 GiB peak (incl. load)
[bf16 reference] first 46 of 172 tokens identical to int8 (diverges later; int8 is approximate)
```
Decode speed depends on answer length and cache bucket: an earlier 160-token run gave 122.1 tok/s
(`runs/20261031_v4/v3_demo_run.txt`); the controlled 256-token benchmark gives 111.9 tok/s. Greedy answers vary in length
between versions, so token counts may differ.

## Demo 2: performance comparison (≈ 6–10 min)
```
python -m research.showcase.demo_compare            # EfficientGPT int8+graph, llama.cpp Q8_0 (Ollama), HF bf16 static
python -m research.showcase.demo_compare --with-torchao   # adds torchao int8 max-autotune (slow first compile)
```
Same harness as the published numbers (fresh process per mode, unique prompts, 256 new tokens), one round instead of four,
so it prints point estimates without a confidence interval. Published 4-round results: `BENCHMARK_REPORT.md` §1a.

## Demo 3: KernelAgent-Ada (≈ 2–4 min)
```
python -m research.showcase.demo_kernelagent
```
Shows, on the real-model task `int8_gemv`: the task and its dev / held-out shapes; an unsafe candidate rejected by the
static screen; a correct candidate passing and a mutated one (scale dropped) rejected by the fp64-truth contract in an
isolated subprocess; a small `local` search with CUDA-graph timing against eager / Inductor / dense bf16 cuBLAS, written
to a JSONL history; a single held-out evaluation of the winner; and the caveat that the shipped kernel config was chosen
end to end. Recorded run on an idle GPU: `runs/20261031_v4/demo_kernelagent_output.txt` (6/6 candidates correct; selected
split-K config 13.44 µs on dev shapes; held-out 20/20 correct, 1.8–4.3× dense bf16 cuBLAS on the small shapes), plus
`runs/20261031_v4/demo_kernelagent/` (history + JSON). Demo 2 recording: `runs/20261031_v4/demo_compare_output.txt`.

## Demo 4: reproduce the evidence
```
python -m research.release.clean_checkout --ref HEAD --out runs/<stamp>/clean_checkout.json   # all tests from a pristine worktree
python -m research.realmodel.bench_int8 --out runs/<stamp>/bench --modes int8_graph llamacpp_q8 bf16_static torchao_int8_static_at --rounds 4 --reps 5
python -m research.realmodel.analyze_int8 --dir runs/<stamp>/bench
```
Full instructions: `REPRODUCE.md`; benchmark verification: `reports/BENCHMARK_VERIFICATION.md`.

## Troubleshooting
| Symptom | Cause / fix |
|---|---|
| Numbers 20–30% lower than published, `contended: true` in the JSON | another program is using the GPU; close it and re-run |
| First request slow (10–60 s) | torch.compile + CUDA-graph capture on first use per cache bucket; excluded from TTFT and reported as `first_call_s` |
| `llamacpp_q8` fails | Ollama not running or model not pulled: `ollama pull qwen3:1.7b-q8_0` |
| `torchao_int8_static_at` very slow to start | Inductor max-autotune benchmarks GEMM variants (30–360 s per process) |
| CUDA out of memory | close other GPU programs; the int8 path needs ≈ 2.5 GiB, bf16 ≈ 3.4 GiB, load peak ≈ 3.4 GiB |
| `ModuleNotFoundError: assistant...` | run from the repository root with `python -m ...` |

## Recording script (≈ 3 minutes of screen time)
1. (0:00) `nvidia-smi` showing an idle RTX 4060; one sentence: "batch-1 decode is bound by weight bandwidth".
2. (0:15) Demo 1: let the answer stream; pause on the statistics line and the bf16 agreement line.
3. (1:00) Open `BENCHMARK_REPORT.md` §1a: the 4-round table (parity with llama.cpp, 1.33x torchao, 1.89x HF bf16), point at
   the CI column and the "parity, below 5%" wording.
4. (1:40) Demo 3: show the rejected unsafe candidate, the rejected mutated kernel, the search table, the held-out line.
5. (2:30) Open `reports/EFFICIENTGPT_INDUSTRY_RESEARCH_REPORT.md` §2 (profile: kernel at DRAM limit, host gap) and the
   limitations section (TTFT, batch 1). End on the reproduce commands.
