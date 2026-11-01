# Troubleshooting a reproduction

For `bash research/release/reproduce.sh` and the steps in [`REPRODUCE_ON_ANOTHER_MACHINE.md`](REPRODUCE_ON_ANOTHER_MACHINE.md).
Rows marked **(seen)** happened during development on the reference machine (RTX 4060, Windows 11); the others are
expected from the setup but have not been seen on Linux yet. If you hit something not listed, please report it: install
problems are useful findings too.

## Checks before a long run
| Check | Command | Expect |
|---|---|---|
| GPU, compute capability, memory | `nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv` | compute_cap ≥ 8.0, memory ≥ 8 GB |
| Driver supports CUDA 12.8 | `nvidia-smi` (header: "CUDA Version") | 12.8 or higher (Linux driver R570 or newer) |
| GPU idle | `nvidia-smi` | utilisation ≈ 0%, no other compute processes |
| Python version | `python --version` (inside the venv) | 3.11.x |
| PyTorch sees the GPU | `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"` | `2.11.0+cu128 True (8, x)` or higher |
| The script's plan | `bash research/release/reproduce.sh --dry-run` | the list of steps, no errors |

## Symptoms
| Symptom | Likely cause | What to do |
|---|---|---|
| `python3.11: command not found` | the distribution ships another Python (Ubuntu 24.04: 3.12) | `uv venv --python 3.11`, or the deadsnakes PPA |
| `torch.cuda.is_available()` prints False | driver too old for CUDA 12.8, or a CPU-only torch wheel | update the driver; reinstall torch with `--index-url https://download.pytorch.org/whl/cu128` |
| Triton compile error about bf16 / `tl.dot` on a T4, RTX 20xx or GTX card | compute capability below 8.0 | not supported: the int8 kernel uses bf16 tensor-core dot |
| CUDA out of memory | other programs on the GPU (desktop, browser), or a GPU under 8 GB | close them; on an 8 GB card a session without a desktop helps |
| Every round flagged `contended: true` **(seen)** | other GPU work during timing | close GPU programs and rerun; contended speeds are not comparable (correctness results still are) |
| The first run of each mode takes minutes **(seen)** | model download (≈ 4 GB), Triton / Inductor compilation, CUDA-graph capture | expected once; later runs reuse the caches |
| The torchao mode takes 0.5–6 min per process **(seen)** | max-autotune compilation | expected; or use `--quick`, which skips it |
| `llama.cpp baseline skipped: Ollama not found` | Ollama not installed | install it and `ollama pull qwen3:1.7b-q8_0`, or rerun with `--no-llamacpp` and send that |
| `llamacpp_q8 r<N>: FAILED rc=...` | the Ollama service isn't running, or the model is missing | start the service (`ollama serve`), pull the model, rerun |
| llama.cpp far slower than EfficientGPT | the model is partly on the CPU | `ollama ps` while it runs should show 100% GPU; report what it shows |
| The wrong GPU is used on a multi-GPU machine | device numbering | `CUDA_VISIBLE_DEVICES=<index as shown by nvidia-smi> bash research/release/reproduce.sh` (the script sets PCI bus order) |
| The archive is named `repro_nogpu_...` | `nvidia-smi` is not on PATH (some containers; WSL2 keeps it in `/usr/lib/wsl/lib`) | add it to PATH and rerun |
| `UnicodeEncodeError` in a log **(seen, Windows)** | console encoding | fixed in the current code; update your checkout |
| Hugging Face download errors | network or Hub rate limits | retry; logging in is optional (the model is public) |
| The demo's bf16 agreement line shows a difference | expected: int8 and bf16 outputs diverge after some tokens | not a failure; mention it under correctness findings |
| A unit test fails | a real problem or an environment difference | please report it with its log; correctness findings matter more than speed |
| The script exits with status 1 | at least one step failed (see `STATUS.txt`) | the archive is still written; attach it with a note |

## Before you share results
Open the archive and look through every file, logs included, for user names, host names and paths. The script removes
GPU serial numbers and UUIDs, the list of running programs, local `file://` paths in `pip freeze` and paths under your
home directory, but it cannot catch everything.

Post results in the repository's Discussions, category **Hardware results**, or with the **Reproduction report** issue
template.
