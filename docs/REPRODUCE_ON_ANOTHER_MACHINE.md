# Reproducing the Qwen3-1.7B benchmark on another GPU or on Linux

Goal: an independent engineer measures EfficientGPT against the same baselines on their own machine and sends back the
raw results. Expect different absolute tok/s on other GPUs. The **ratios** between engines on the same machine are what
should be compared with `BENCHMARK_REPORT.md` §1a. Nothing here requires the private TinyStories checkpoint.

## One command (Linux / WSL2)
After the environment and the checks in §2:
```bash
bash research/release/reproduce.sh            # environment, tests, demo, full benchmark: about 1-1.5 hours (estimate: §3 + §4)
bash research/release/reproduce.sh --quick    # int8 vs bf16 only: shorter (one informal run: about 14 min on a busy RTX 4060)
bash research/release/reproduce.sh --dry-run  # show the commands without running them
```
It writes `runs/repro_<gpu>_<date>/` (environment record `env.json`, `nvidia-smi -q`, `pip freeze`, one log per step,
`STATUS.txt`, the benchmark JSON and `REPORT.md`) and packs it as `runs/repro_<gpu>_<date>.tar.gz`. A failing step is
recorded, the script carries on, and it exits with status 1 at the end. Nothing is uploaded.

**Privacy before sharing:** the script removes GPU serial numbers and UUIDs, the list of running programs, local
`file://` paths in `pip freeze`, and paths under your home directory (shown as `~`). It cannot catch everything: open the
archive and look through every file, logs included, for user names, host names or paths (including WSL paths such
as `/mnt/c/Users/<name>`) before you attach it.

If something goes wrong, see [`docs/TROUBLESHOOTING.md`](TROUBLESHOOTING.md).
Options: `--skip-tests`, `--no-llamacpp`, and `--deferred-stop` (adds the transformers `DeferredStopCheck` measurement on
Qwen3-1.7B and SmolLM2-135M; its run time has not been measured; see `docs/upstream/transformers-deferred-stop-cuda/`). On Windows, run the
§3 and §4 commands directly, or the script from Git Bash.

**No Linux GPU at hand?** [`colab/EfficientGPT_Colab_Reproduction.ipynb`](../colab/EfficientGPT_Colab_Reproduction.ipynb)
runs the `--quick` path on a Google Colab L4 (run once by the author) or A100 (untested) runtime; which GPUs are
offered depends on the Colab plan, and the T4 is not supported. Its background-run and download steps (5–6) have not
yet been tested on Colab. llama.cpp is skipped there (no Ollama).

## 0. At a glance

| Experiment | Needs | Time | Command |
|---|---|---|---|
| CPU tests (109 of 156) | Python 3.11, torch, transformers, numpy, tiktoken; no GPU | minutes | the four `python -m unittest` commands in `REPRODUCE.md` §1, each prefixed with `CUDA_VISIBLE_DEVICES=-1` (reference environment: 156 run, the 47 GPU tests skip; a minimal environment may lack packages some assistant tests need) |
| GPU tests for the real-model path (int8 kernel, graph decoder, KernelAgent-Ada evaluator) | CUDA GPU (sm_80+) | ≈ 5 min (§3) | §3 |
| Demo 1: Qwen3-1.7B int8 + graph decoder | GPU ≥ 8 GB; Qwen3-1.7B from the Hugging Face Hub (≈ 4 GB, downloaded once) | ≈ 1 min | `python -m research.realmodel.demo` |
| Demo 3: KernelAgent-Ada | CUDA GPU | ≈ 3 min | `python -m research.showcase.demo_kernelagent` |
| Headline benchmark vs bf16 and torchao | GPU ≥ 8 GB, torchao 0.18 | ≈ 50–70 min with all baselines (§4) | §4 |
| ... plus the llama.cpp baseline | Ollama ≥ 0.40 and `qwen3:1.7b-q8_0` | included above | §4 |
| Phase 5 decoder (TinyStories) results and 8 of its tests | the TinyStories checkpoint, **not distributed**: run `python prepare_v2.py` and `python train.py config/tinystories_v2.py` (2500 iterations; training time was not recorded). A retrained model's numbers will differ slightly. | — | `REPRODUCE.md` §2 |
| Local assistant experiments | Ollama with `qwen3.5:4b` and `nomic-embed-text`; optional peft / bitsandbytes | varies | `docs/ASSISTANT.md` |

**Supported:** NVIDIA GPUs with compute capability ≥ 8.0 (Ampere, Ada, Hopper, Blackwell), ≥ 8 GB, with a driver that
supports CUDA 12.8 (Linux: driver R570 or newer; `nvidia-smi` shows "CUDA Version: 12.8" or higher). Headline
benchmarks: only an RTX 4060 (sm_89) on Windows 11.

**Checked once by the author:** Linux on an NVIDIA L4 (Google Colab, Python 3.13, Triton 3.6.0): GPU tests pass and
the `--quick` benchmark runs (`reports/L4_LINUX_VALIDATION_2026-11-01.md`). **Untested:** WSL2 and other sm_80+ GPUs;
llama.cpp and torchao on Linux. The kernels are plain Triton and PyTorch.

**Not supported:**
- **Turing and older (sm_75 and below: T4, RTX 20xx, GTX):** the int8 kernel uses bf16 `tl.dot`.
- **AMD ROCm and Apple MPS:** not tried.
- **CPU-only:** tests only, no inference demos.
- **Multi-GPU:** batch-1 decode on one GPU only.

**Environment drift:** results were measured with torch 2.11.0, transformers 5.21.0, torchao 0.18.0 and Triton 3.7.1
(`triton-windows`, the reference machine). On Linux, torch 2.11.0 installs Triton 3.6.0 instead; that combination is
untested, and Triton versions can change kernel speed, so please report the Triton version with your results. Other
versions may change both speed and the behaviour of the `generate()` baselines. `requirements-lock.txt` is a Windows
`pip freeze`, a reference only; it is not installable on Linux as is.

## 1. Requirements
- NVIDIA GPU with ≥ 8 GB, compute capability ≥ 8.0 (the int8 kernel uses bf16 tensor-core `tl.dot`); driver supporting CUDA
  12.8 (Linux: R570 or newer). Developed on an RTX 4060 (sm_89).
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
- On Linux, `triton` is installed with torch (3.6.0 for torch 2.11.0). On Windows, install `triton-windows==3.7.1.post27`.
- Python 3.11 is not packaged on every distribution (for example Ubuntu 24.04 ships 3.12). Use `uv venv --python 3.11`
  or the deadsnakes PPA; other Python versions are untested.
- `requirements-lock.txt` lists the exact versions of the reference machine (a Windows `pip freeze`); use it to pin
  further if results differ.
- Optional baseline, llama.cpp via Ollama: install Ollama (tested 0.40.12), then `ollama pull qwen3:1.7b-q8_0`; the
  reference digest is 29b6a7a420a4. While a model is loaded, `ollama ps` should show it at 100% GPU; a partly-CPU model
  makes the llama.cpp baseline meaningless.

The model (≈ 4 GB) downloads on first use. Nothing to do for the revision: the code pins
`Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.

Before an hour-long run, check that PyTorch sees the GPU:
```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv   # compute_cap ≥ 8.0, memory ≥ 8 GB
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
# expected: 2.11.0+cu128 True (8, x) or higher
bash research/release/reproduce.sh --dry-run
```

## 3. Sanity checks (≈ 5 min; problems: [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md))
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
- `nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv` (not the full `nvidia-smi -q`, which
  contains the GPU's serial number and UUID), `python -c "import torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__)"`,
  OS and CPU model.
- Check everything for user names, host names and paths before posting (see "Privacy before sharing" above).
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
