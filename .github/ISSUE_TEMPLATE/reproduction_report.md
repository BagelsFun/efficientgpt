---
name: Reproduction report
about: Share results from running the EfficientGPT benchmark or demos on your own GPU (matching or contradicting ours)
title: "Reproduction: <GPU> / <OS>"
labels: reproduction
---

**Hardware and software**
- GPU (`nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv`; please don't paste the full
  `nvidia-smi -q`, which contains your GPU's serial number and UUID):
- OS and CPU:
- `python -c "import torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__)"`:
- triton / torchao versions:
- Ollama version (if the llama.cpp baseline was run):

**What you ran** (exact commands; see `docs/REPRODUCE_ON_ANOTHER_MACHINE.md`)

**Results**
- `REPORT.md` from `analyze_int8` (paste the table):
- Any process flagged `contended: true`?
- Test suite outcome (`python -m unittest ...`), if run:

**Raw files**: please attach or link the whole `runs/repro_<gpu>_<date>/` directory (or the `.tar.gz` from
`research/release/reproduce.sh`). Look through every file first, logs included, for user names, host names and paths.

**Anything unusual** (thermal throttling, other GPU programs, errors, deviations from the instructions):
