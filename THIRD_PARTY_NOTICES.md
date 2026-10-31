# Third-party notices

This repository is a fork of **nanoGPT** (https://github.com/karpathy/nanoGPT), © 2022 Andrej Karpathy, MIT licence (see
`LICENSE`); nanoGPT-derived files (`model.py`, `train.py`, `sample.py`, `configurator.py`, `bench.py`, `data/*/prepare.py`,
`config/`) remain under that licence. The additions made in this fork are also
released under the MIT licence, © 2026 BagelsFun (second copyright line in `LICENSE`).

No third-party source code is vendored. The following are used as installed dependencies or external tools and are not
redistributed:

| Component | Licence | Use |
|---|---|---|
| PyTorch 2.11 (incl. TorchInductor) | BSD-3-Clause | framework, compiler |
| Triton (triton-windows 3.7.1 on Windows) | MIT | GPU kernels |
| Hugging Face transformers, tokenizers, safetensors, huggingface-hub, datasets, peft, accelerate | Apache-2.0 | model loading, generation, LoRA |
| torchao 0.18.0 | BSD-3-Clause | baseline (int8 weight-only) |
| bitsandbytes 0.50.2 | MIT | baseline (LLM.int8, nf4) |
| numpy | BSD-3-Clause (with bundled permissive licences) | arrays |
| tiktoken | MIT | tokenisation (nanoGPT data prep) |
| Ollama 0.40.12 / llama.cpp | MIT | baseline engine (external tool) |
| Qwen3-1.7B model weights (Alibaba Qwen team) | Apache-2.0 (per its Hugging Face model card) | downloaded at runtime; not redistributed; the recorded Ollama modelfile in `runs/` reproduces the licence text as shipped |
| TinyStories dataset (roneneldan/TinyStories) | CDLA-Sharing-1.0 (per its Hugging Face dataset card) | downloaded by `prepare_v2.py` to train the Phase 5 model; the data is not redistributed (logs contain only model outputs) |

Names of companies and products (NVIDIA, Meta, Google, PyTorch, llama.cpp, Ollama, Qwen, …) are used only to identify
hardware and software; no affiliation or endorsement is implied.
