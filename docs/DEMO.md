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
Single rounds vary by a few percent: two runs on 2026-10-31 gave EfficientGPT / llama.cpp = 1.012 and 0.987
(`runs/demo_compare_20261031_203336/`, `runs/demo_compare_20261031_203623/`). That is why the claim is parity.

## Demo 3: KernelAgent-Ada (≈ 2–4 min)
```
python -m research.showcase.demo_kernelagent
```
Shows, on the real-model task `int8_gemv`: the task and its dev / held-out shapes; an unsafe candidate rejected by the
static screen; a correct candidate passing and a mutated one (scale dropped) rejected by the fp64-truth contract in an
isolated subprocess; a small `local` search with CUDA-graph timing against eager / Inductor / dense bf16 cuBLAS, written
to a JSONL history; a single held-out evaluation of the winner; and the caveat that the shipped kernel config was chosen
end to end. Recorded runs: `runs/20261031_v4/demo_kernelagent_output.txt` (6/6 candidates correct; selected split-K config
13.44 µs on dev shapes; held-out 20/20 correct, 1.8–4.3× dense bf16 cuBLAS on the small shapes) and
`runs/20261031_v4/demo3_output_release.txt` (13.80 µs; 20/20; 1.85–4.33×), plus
`runs/20261031_v4/demo_kernelagent/` (history + JSON). Demo 2 recordings: `runs/20261031_v4/demo_compare_output.txt` and `runs/20261031_v4/demo2_output_release.txt`.

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
| `UnicodeEncodeError: 'charmap'` when output is redirected (Windows) | fixed on 2026-10-31: the entry points switch redirected output to UTF-8 (`research/console.py`) |

## Recording script: 75 seconds

**Status:** no video has been recorded yet. The script is checked against fresh terminal runs from 2026-10-31:
`runs/20261031_v4/demo1_output_release.txt`, `demo3_output_release.txt` and `demo2_output_release.txt`.

**Setup.**
1. Close other GPU programs and run Demo 1 and Demo 3 once beforehand, so first-use compilation is cached.
2. Use a terminal at about 110 columns and a large font, with two terminal tabs ready.
3. Open `BENCHMARK_REPORT.md` §1a and `README.md` "Limitations" in an editor.

**Rules for an honest edit.**
- Demo 1 is shown in real time; the streaming speed *is* the result.
- Demo 3's search takes about 2 minutes. Cut it, and caption the cut "search: 6 candidates, ≈ 2 min, cut".
- Do not speed up Demo 1.
- The 4-engine comparison takes about 8 minutes per round. Show the recorded table and say so; don't imply it ran live.

| Time | Screen | Say (≈ 150 words in total) |
|---|---|---|
| 0:00–0:08 | `nvidia-smi`: RTX 4060, ≈ 0% utilisation | "One consumer GPU, idle. Batch-1 decoding is limited by how fast weights stream from memory." |
| 0:08–0:30 | `python -m research.realmodel.demo`: the answer streams, then the stats line `185 tokens, TTFT 59 ms, decode 108.6 tok/s` | "Qwen3-1.7B with EfficientGPT's int8 Triton kernel and a single CUDA graph per token, in plain PyTorch, about 108 tokens a second. The bf16 check below shows int8 is approximate: the first 46 tokens match, then it diverges." |
| 0:30–0:50 | `python -m research.showcase.demo_kernelagent`, steps 2–3: `unsafe candidate -> ['import os']`, then `mutated candidate (scale dropped): incorrect (error 5.77e+05 > bound 179)`; cut; step 5 `held-out 20/20 correct` | "KernelAgent-Ada searches Triton kernels, but it verifies first. A candidate that imports os is rejected before it runs. A kernel that silently drops the int8 scale is rejected against fp64 ground truth. Only correct candidates are timed, then checked once on held-out shapes." |
| 0:50–1:05 | `BENCHMARK_REPORT.md` §1a table; highlight 111.9 / 109.9 and the CI | "Four engines, four fresh-process rounds each. We match llama.cpp: 1.02×, inside our 5% bar, so we call it parity, not a win. We are 1.33× torchao int8 and 1.89× Hugging Face bf16." |
| 1:05–1:15 | README "Limitations" | "Limits: batch 1, greedy only, one GPU on Windows, and llama.cpp still has a faster first token. Every number links to raw JSON; reproduction reports from other GPUs are welcome." |

**Optional extra (+15 s).** Show `research/showcase/demo_kernelagent` step 6 ("split-K won here but was slower inside the
model"). It illustrates why microbenchmark wins need end-to-end confirmation.

**After recording.** Check every spoken number against the recorded terminal output. Re-record if a run printed different
numbers; do not edit the narration to fit.
