# Linux validation on an NVIDIA L4 (Google Colab), 2026-11-01

A cross-platform check of EfficientGPT by its author on a second GPU and on Linux. **These are new measurements, kept
separate from the RTX 4060 / Windows results in `BENCHMARK_REPORT.md`, which this report does not change.**

**What this is not:**
- **Not an independent reproduction.** The author ran it, on the author's Colab account.
- **Not a test of the llama.cpp parity claim.** llama.cpp (Ollama) was not run on the L4, and neither was torchao.
- **Not the full benchmark.** This is `reproduce.sh --quick`: int8 graph decoder vs Hugging Face bf16 only.

## Setup
| Item | Value |
|---|---|
| GPU | NVIDIA L4, compute capability 8.9, 23,034 MiB (22.03 GiB visible to PyTorch), 58 SMs |
| Driver | 580.82.07 (power limit 72 W) |
| OS | Linux 6.6.159+ x86_64, glibc 2.39 (Google Colab runtime image) |
| CPU | Intel Xeon @ 2.20 GHz |
| Python | 3.13.16 |
| PyTorch / CUDA | 2.11.0+cu128 / 12.8 |
| Triton | 3.6.0 |
| Transformers / torchao | 5.21.0 / 0.18.0 |
| Run | 2026-11-01, 04:33–04:47 UTC |
| Code | commit bb1a5bf; notebook steps 1–4 (`colab/EfficientGPT_Colab_Reproduction.ipynb`), then `research/release/reproduce.sh --quick` started from the Colab terminal |
| Model | Qwen/Qwen3-1.7B at revision 70d244cc (pinned by the code) |

Differences from the RTX 4060 reference: Linux instead of Windows, Python 3.13 instead of 3.11, Triton 3.6.0 instead of
3.7.1 (`triton-windows`), and a 72 W data-centre GPU with 58 SMs instead of a desktop RTX 4060 (both are Ada
Lovelace, sm_89).

## Correctness
| Suite | Result |
|---|---|
| KernelAgent-Ada (`test_evaluate`, `test_search`) | 23 run, 23 passed, 0 skipped |
| Real model (`test_int8_linear`, `test_fast_decode`) | 7 run, 7 passed, 0 skipped |
| Demo, int8 vs the bf16 reference (128 tokens) | first 46 tokens identical, then diverges, as expected for int8 (approximate contract) |

20 of these 30 tests need a GPU and skip without one; none was skipped here, so all 20 GPU tests ran on the L4.

## Performance (Qwen3-1.7B, batch 1, greedy, 256 new tokens; 4 fresh-process rounds × 5 timed requests per engine)
| Round | Hugging Face bf16, static cache | EfficientGPT int8 + graph decoder |
|---|---|---|
| r0 | 60.1 tok/s (TTFT 46 ms) | 107.5 tok/s (TTFT 62 ms) |
| r1 | 60.0 (46) | 106.4 (61) |
| r2 | 59.8 (47) | 106.4 (61) |
| r3 | 59.5 (48) | 105.9 (61) |
| **Median of round medians** | **59.9 tok/s** | **106.4 tok/s** |

- **int8 / bf16: 1.777×, 95% CI [1.767, 1.797]** (bootstrap over rounds; paired per-round range 1.774–1.789). Recomputed
  from the 40 per-request speeds in the raw files with the same method, with a second random seed and with an exact
  enumeration of all 65,536 round resamples: the same ratio and interval each time. `bench/summary.json` shows 1.780
  [1.774, 1.784] because it pools all requests as independent, which understates the variation between processes; it is
  not the reported figure.
- All 40 timed requests generated the full 256 tokens.
- GPU utilisation was 0% before every one of the 8 processes; none was flagged as contended.
- Peak memory: 2,342–2,367 MiB for int8; 3,346–3,376 MiB for bf16.
- A single demo run (not part of the benchmark) decoded 128 tokens at 113.3 tok/s with a 63 ms first token.

## Compared with the published RTX 4060 results (different machines; not a combined result)
| | RTX 4060, Windows 11 (`BENCHMARK_REPORT.md` §1a; CI from `runs/20261031_v4/v3_final/REPORT.md`) | L4, Linux (this report) |
|---|---|---|
| EfficientGPT int8 + graph | 111.9 tok/s | 106.4 tok/s |
| HF bf16 + static cache | 59.3 tok/s | 59.9 tok/s |
| int8 / bf16 | 1.89 [1.885, 1.948] | 1.78 [1.767, 1.797] |
| TTFT, int8 / bf16 | 54 / 46 ms | 61 / 47 ms |
| llama.cpp Q8_0 | 109.9 tok/s (parity, 1.02×) | not run |
| torchao int8 | 84.4 tok/s | not run |

What the comparison supports:
- The code installs and passes its GPU tests on Linux, on a second GPU and on Python 3.13.
- The int8 path's advantage over bf16 holds on the L4, but is smaller than on the RTX 4060 (1.78× vs 1.89×): bf16 is
  about the same speed while int8 is about 5% slower.

What it does not support:
- Anything about llama.cpp or torchao on the L4.
- A cause for the smaller int8 gain. One untested possibility: the int8 kernel configuration was chosen on the RTX 4060
  (`docs/REPRODUCE_ON_ANOTHER_MACHINE.md` §6); re-running `research/realmodel/sweep_configs.py` on an L4 would test it.

## Run record
- A first attempt the same day was lost before producing results, when the controlling browser slept and Colab replaced
  the runtime. It also showed that Colab's preinstalled PyTorch (2.11.0+**cu130**) satisfied the notebook's plain
  `torch==2.11.0` pin. The notebook now pins `torch==2.11.0+cu128`, runs the benchmark in the background and offers
  progress snapshots, which is meant to keep an interrupted runtime from losing everything; these new steps (5–6) have
  been tested only outside Colab so far.
- In this run the benchmark was started from the Colab terminal in the background, equivalent to the notebook's new
  step 5. The code of the notebook's steps 1–4 (GPU check, install, checkout, PyTorch check) ran on the L4 as published;
  only the pinned commit has since moved to 718e3b3, which adds the privacy-filter fix below.
- Raw files: [`runs/20261101_l4_colab/`](../runs/20261101_l4_colab/README.md), the script's run directory
  `repro_NVIDIA_L4_20261101_043259` (environment record, `nvidia-smi -q`, `pip freeze`, per-step logs, per-process
  benchmark JSON, `analysis.json`, `REPORT.md`, `STATUS.txt`) plus its console output, archived after the script
  finished. Before publication every file was checked for user names, host names, paths outside `/content`, tokens and
  hardware identifiers. The one finding: the `GPU PDI`, a per-device identifier that the script's filter missed. It was
  removed from `nvidia_smi_q.txt` (with three empty ID fields), and the filter was fixed in commit 718e3b3 (no earlier
  published file contained one).
  The notebook pins that commit.
- Compute: Colab Pro, about 1.94 compute units per hour; both attempts together used about 1.4 units.
