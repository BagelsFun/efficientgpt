# Industry showcase audit (Phase A), 2026-10-31

Scope: the `showcase` branch (created from tag `v3`, <private-commit>) of the private repository `<private-repository>`.
Every check below was executed in this session; commands are given so they can be re-run.

## Results by severity
| # | Severity | Finding | Status |
|---|---|---|---|
| A1 | **Medium** | 37 committed evidence logs (e.g. `research/phase5/ci_runs/*/diagnostics/`, `runs/20261031_v4/v*_final*/`) contain absolute local paths that include the owner's Windows user name. Harmless for a private repo; a privacy item before any public release. | Owner decision (redact in a release export, or accept). Evidence files are not rewritten here. |
| A2 | **Medium** | All 58 project commits carry the owner's personal e-mail address as author. History may not be rewritten (owner rule), so publishing the repo publishes the address. | Owner decision (publish as is, or publish a squashed export authored with a GitHub no-reply address). |
| A3 | **Low** | The repository's `LICENSE` is nanoGPT's MIT licence (© 2022 Andrej Karpathy). It does not state a licence or copyright line for the owner's added code. | Resolved 2026-10-31: owner chose MIT; `LICENSE` now carries a second copyright line for the additions. |
| A4 | **Low** | Benchmark baselines need things that aren't in git: Ollama 0.40.12 plus `qwen3:1.7b-q8_0` (digest 29b6a7a420a4), the HF Qwen3-1.7B cache (rev 70d244cc), and torchao 0.18.0 (installed `--no-deps`). These are documented, but they are runtime dependencies an external reviewer must install. | Documented in `REPRODUCE.md`, `requirements-lock.txt`, `reports/BENCHMARK_VERIFICATION.md`. |
| A5 | **Low** | `requirements-lock.txt` pins the dev venv but is a `pip freeze` of one Windows machine (torch `+cu128` wheel). A clean-venv install from it has not been executed (≈ 3 GB download). | Labelled UNVERIFIED in `BENCHMARK_VERIFICATION.md`. |
| A6 | **Informational** | Unsupported-claim review: the README, benchmark, research and career reports describe llama.cpp as **parity** (below the 5% bar), torchao as tuned with max-autotune, and quantisation as approximate. No report claims that EfficientGPT is faster than llama.cpp, production-ready, or a new algorithm. | Re-checked in Phase F against raw evidence. |

No Critical or High findings.

## Checks executed (all passed unless listed above)
| Check | Command / method | Result |
|---|---|---|
| Git object integrity | `git fsck --no-dangling` | clean |
| Tags | `git rev-parse v2^{commit} v3^{commit}` vs `git ls-remote origin` | v2 = <private-commit>, v3 = <private-commit>, identical locally and remotely |
| Protected branches | `git ls-remote origin` | remote `main` = cc5d255 (untouched), `efficientgpt-v2` = <private-commit>; work happens on local `showcase` |
| Checkpoint preservation | `sha256sum -c docs/CHECKPOINT_MANIFEST.sha256` | 7/7 OK (originals unmodified since 2026-10-30) |
| Credentials in history | regex for GitHub/OpenAI/HF/AWS/Slack tokens, private keys, password assignments over `git log -p upstream/master..showcase` | 0 matches |
| Model / dataset / binary files in history | `git log --name-only` filtered for .pt/.pth/.ckpt/.bin/.safetensors/.gguf/.pkl/.npy/.parquet/.onnx and `data/` binaries | none ever committed (gitignored: checkpoints, `data/`, `results/`, `.venv/`) |
| Largest blobs | `git rev-list --objects ... | git cat-file --batch-check` | 770 KB (a benchmark log); no oversized artifacts |
| Third-party code vendored | `git grep` for foreign copyright / SPDX headers in `*.py` | none; all third-party code is consumed as installed packages |
| Third-party model weights | repo content | none; Qwen3-1.7B is downloaded from Hugging Face at runtime (Apache-2.0), the Q8_0 GGUF through Ollama |
| Private training data | repo content | none; the distillation data (`results/assistant/data/`) is gitignored and was never committed |
| Hidden dependencies | clean `git worktree` run (`research.release.clean_checkout`) | V3: found and fixed one undeclared test input (`data/tinystories_v2/val.bin`); 149/149 after the fix. Re-run for this release in Phase F |

## Dependency licences (installed metadata / bundled licence files)
| Component | Version | Licence |
|---|---|---|
| nanoGPT (base of this repo) | upstream 3adf61e | MIT |
| PyTorch | 2.11.0+cu128 | BSD-3-Clause |
| triton-windows | 3.7.1.post27 | MIT |
| transformers / tokenizers / safetensors / huggingface-hub / datasets / peft / accelerate | 5.21.0 / 0.23.3 / 0.8.0 / 2.3.0 / 5.1.0 / 0.21.2 / 1.15.0 | Apache-2.0 |
| torchao | 0.18.0 | BSD-3-Clause (bundled licence file) |
| bitsandbytes | 0.50.2 | MIT |
| numpy | 2.4.6 | BSD-3-Clause (and bundled permissive licences) |
| tiktoken | 0.14.0 | MIT |
| Ollama | 0.40.12 | MIT (runtime tool, not redistributed) |
| Qwen3-1.7B weights | rev 70d244cc | Apache-2.0 (downloaded at runtime, not redistributed) |
| TinyStories dataset (Phase 5 training/eval data) | local copy | not redistributed; its licence must be checked by the owner before any data release (none planned) |
