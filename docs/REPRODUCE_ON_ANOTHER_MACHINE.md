# Reproducing the Qwen3-1.7B benchmark on another GPU or on Linux

Goal: an independent engineer measures EfficientGPT against the same baselines on their own machine and sends back the
raw results. Expect different absolute tok/s on other GPUs. The **ratios** between engines on the same machine are what
should be compared with `BENCHMARK_REPORT.md` §1a. Nothing here requires the private TinyStories checkpoint.

## 1. Requirements
- NVIDIA GPU with ≥ 8 GB, compute capability ≥ 8.0 (the int8 kernel uses bf16 tensor-core `tl.dot`); driver supporting CUDA
  12.8. Developed on an RTX 4060 (sm_89).
- Python 3.11, git, about 10 GB of disk (model cache plus Ollama model).
- Idle GPU: no other GPU programs (browsers with hardware acceleration, games, overlays). The harness records utilisation
  before every process and flags anything above 15% as `contended`.

## 2. Environment
```bash
git clone <repository> efficientgpt && cd efficientgpt
python3.11 -m venv .venv && source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==5.21.0" "torchao==0.18.0" numpy tiktoken
```
- On Linux, `triton` is installed with torch. On Windows, install `triton-windows==3.7.1.post27`.
- `requirements-lock.txt` lists the exact versions of the reference machine (a Windows `pip freeze`); use it to pin
  further if results differ.
- Optional baseline, llama.cpp via Ollama: install Ollama (tested 0.40.12), then `ollama pull qwen3:1.7b-q8_0`; the
  reference digest is 29b6a7a420a4.

The model downloads on first use. Pin the revision: `Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (≈ 4 GB).

## 3. Sanity checks (≈ 5 min)
```bash
python -m unittest research.realmodel.test_int8_linear research.realmodel.test_fast_decode research.kernelagent.test_search
python -m research.realmodel.demo --tokens 128
```
Phase 5 tests that need the TinyStories checkpoint skip automatically when it is absent; the real-model path doesn't need
them.

## 4. Benchmark (≈ 50–70 min; leave the machine alone)
```bash
nvidia-smi                                    # utilisation ≈ 0%
python -m research.realmodel.bench_int8 --out runs/repro_<gpu>_<date> \
    --modes int8_graph llamacpp_q8 bf16_static torchao_int8_static_at --rounds 4 --reps 5
python -m research.realmodel.analyze_int8 --dir runs/repro_<gpu>_<date>
```
- Drop `llamacpp_q8` if Ollama isn't installed.
- Drop `torchao_int8_static_at` to save time; max-autotune compiles for 0.5–6 min per process.
- Add `--quality --quality-modes bf16_static int8_graph` for the quality rows (+15 min).

## 5. What to send back
- The whole `runs/repro_<gpu>_<date>/` directory: per-process JSON, logs and `REPORT.md`.
- `nvidia-smi -q | head -40`, `python -c "import torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__)"`,
  OS and CPU model.
- Any process flagged `contended: true`, and anything unusual (thermal throttling, other GPU users).

## 6. Interpreting results
| Expectation on a different GPU | Why |
|---|---|
| int8 / bf16 ratio ≈ 1.6–2.0 | batch-1 decode is weight-bandwidth bound; int8 halves the weight bytes |
| int8 vs llama.cpp Q8_0 ≈ parity, could go either way | both run near the bandwidth limit; llama.cpp's kernels and Q8_0 block format differ per GPU |
| TTFT: llama.cpp better | EfficientGPT's int8 prefill dequantises weights and runs eagerly (`reports/BENCHMARK_LIMITATIONS.md`) |
| Data-centre GPUs (H100 etc.) | larger L2 and bandwidth change the balance; the kernel config was chosen on an RTX 4060 and may not be optimal (re-run `python -m research.realmodel.sweep_configs --out runs/sweep_<gpu>`) |

A result that contradicts ours is useful: please report it with the raw files.
