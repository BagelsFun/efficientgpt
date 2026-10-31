# Reproducing the Qwen3-1.7B benchmark on another GPU or on Linux

Goal: an independent engineer measures EfficientGPT against the same baselines on their own machine and sends back the
raw results. Expect different absolute tok/s on other GPUs. The **ratios** between engines on the same machine are what
should be compared with `BENCHMARK_REPORT.md` §1a. Nothing here requires the private TinyStories checkpoint.

## One command (Linux / WSL2)
After the environment in §2:
```bash
bash research/release/reproduce.sh            # environment, tests, demo, full benchmark: about 1 hour
bash research/release/reproduce.sh --quick    # int8 vs bf16 only: about 20 minutes
bash research/release/reproduce.sh --dry-run  # show the commands without running them
```
It writes `runs/repro_<gpu>_<date>/` (environment record `env.json`, `nvidia-smi -q`, `pip freeze`, one log per step,
`STATUS.txt`, the benchmark JSON and `REPORT.md`) and packs it as `runs/repro_<gpu>_<date>.tar.gz`. A failing step is
recorded and the script carries on. Nothing is uploaded; read `env.json` and `pip_freeze.txt` before you share them.
Options: `--skip-tests`, `--no-llamacpp`, and `--deferred-stop` (adds the transformers `DeferredStopCheck` measurement on
Qwen3-1.7B and SmolLM2-135M; its run time has not been measured; see `docs/upstream/transformers-deferred-stop-cuda/`). On Windows, run the
§3 and §4 commands directly, or the script from Git Bash.

## 0. At a glance

| Experiment | Needs | Time | Command |
|---|---|---|---|
| CPU tests (109 of 156) | Python 3.11, torch, transformers, numpy, tiktoken; no GPU | minutes | §3 with `CUDA_VISIBLE_DEVICES=-1` (GPU tests skip) |
| GPU tests for the real-model path (int8 kernel, graph decoder, KernelAgent-Ada evaluator) | CUDA GPU (sm_80+) | ≈ 5 min (§3) | §3 |
| Demo 1: Qwen3-1.7B int8 + graph decoder | GPU ≥ 8 GB; Qwen3-1.7B from the Hugging Face Hub (≈ 4 GB, downloaded once) | ≈ 1 min | `python -m research.realmodel.demo` |
| Demo 3: KernelAgent-Ada | CUDA GPU | ≈ 3 min | `python -m research.showcase.demo_kernelagent` |
| Headline benchmark vs bf16 and torchao | GPU ≥ 8 GB, torchao 0.18 | ≈ 50–70 min with all baselines (§4) | §4 |
| ... plus the llama.cpp baseline | Ollama ≥ 0.40 and `qwen3:1.7b-q8_0` | included above | §4 |
| Phase 5 decoder (TinyStories) results and 8 of its tests | the TinyStories checkpoint, **not distributed**: run `python prepare_v2.py` and `python train.py config/tinystories_v2.py` (2500 iterations; training time was not recorded). A retrained model's numbers will differ slightly. | — | `REPRODUCE.md` §2 |
| Local assistant experiments | Ollama with `qwen3.5:4b` and `nomic-embed-text`; optional peft / bitsandbytes | varies | `docs/ASSISTANT.md` |

**Supported:** NVIDIA GPUs with compute capability ≥ 8.0 (Ampere, Ada, Hopper, Blackwell), ≥ 8 GB, with a driver that
supports CUDA 12.8. Only an RTX 4060 (sm_89) on Windows 11 has been tested.

**Untested; expected to work:** Linux, WSL2 and other sm_80+ GPUs. The kernels are plain Triton and PyTorch.

**Not supported:**
- **Turing and older (sm_75 and below: T4, RTX 20xx, GTX):** the int8 kernel uses bf16 `tl.dot`.
- **AMD ROCm and Apple MPS:** not tried.
- **CPU-only:** tests only, no inference demos.
- **Multi-GPU:** batch-1 decode on one GPU only.

**Environment drift:** results are pinned to torch 2.11.0, transformers 5.21.0, triton 3.7 and torchao 0.18.0. Other
versions may change both speed and the behaviour of the `generate()` baselines. `requirements-lock.txt` is a Windows
`pip freeze`, a reference only; it is not installable on Linux as is.

## 1. Requirements
- NVIDIA GPU with ≥ 8 GB, compute capability ≥ 8.0 (the int8 kernel uses bf16 tensor-core `tl.dot`); driver supporting CUDA
  12.8. Developed on an RTX 4060 (sm_89).
- Python 3.11, git, about 10 GB of disk (model cache plus Ollama model).
- Idle GPU: no other GPU programs (browsers with hardware acceleration, games, overlays). The harness records utilisation
  before every process and flags anything above 15% as `contended`.

## 2. Environment
```bash
git clone https://github.com/BagelsFun/efficientgpt && cd efficientgpt
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
Post in the repository's **Discussions → Hardware results** (the form asks for exactly the fields below), or open an issue
with the **Reproduction report** template, and attach `runs/repro_<gpu>_<date>.tar.gz`, or:
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

A result that contradicts ours is useful: please report it with the raw files. No independent reproduction has been
reported yet; this section will list them, with links, as they arrive.
