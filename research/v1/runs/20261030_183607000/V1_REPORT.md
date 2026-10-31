# EfficientGPT V1 — Reproducible benchmark report

**Status:** FULL PHASE 5 MATRIX VALIDATED ON THIS MACHINE

Source: locally generated benchmark CSV files. No independent reproduction.
Model: the owner's trained TinyStories GPT (not a general-purpose assistant).
Reference: PyTorch SDPA eager prefix KV reuse, not vLLM or TensorRT-LLM.
Measurements use synthetic prompts with fixed known maximum outputs; no EOS, serving or network overhead.

## Median p50 generation latency across recorded rounds

| Batch / prompt / generated | Eager prefix SDPA (ms) | Full graph (ms) | Adaptive graph (ms) | Adaptive vs eager throughput |
|---|---:|---:|---:|---:|
| 1 / 8 / 16 | 33.1829 | 12.2828 | 9.6651 | 3.433x |
| 1 / 64 / 16 | 33.1978 | 12.2062 | 10.1074 | 3.285x |
| 1 / 64 / 64 | 134.5369 | 42.9908 | 35.6099 | 3.778x |
| 1 / 128 / 32 | 66.1900 | 21.5787 | 20.6502 | 3.205x |
| 1 / 128 / 64 | 136.4038 | 40.2966 | 39.0532 | 3.493x |
| 1 / 192 / 32 | 66.1591 | 21.6694 | 22.0160 | 3.005x |
| 16 / 8 / 16 | 35.2991 | 20.3546 | 14.4919 | 2.436x |
| 16 / 64 / 16 | 34.3082 | 20.9449 | 14.9145 | 2.300x |
| 16 / 64 / 64 | 133.7722 | 78.5584 | 54.5071 | 2.454x |
| 16 / 128 / 32 | 67.6653 | 41.4643 | 36.9071 | 1.833x |
| 16 / 128 / 64 | 134.6028 | 79.7037 | 69.6974 | 1.931x |
| 16 / 192 / 32 | 68.4435 | 45.7643 | 44.5709 | 1.536x |

## Peak PyTorch GPU memory and graph cost per run

| Round | Mode | Graphs | Setup peak alloc (MiB) | Steady peak alloc (MiB) | Steady peak reserved (MiB) | Capture (ms) | Init + correctness (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | adaptive_graph | 8 | 461.643 | 460.698 | 648.0 | 28.356 | 4320.4339 |
| 1 | full_graph | 2 | 292.109 | 291.164 | 402.0 | 6.8686 | 4263.8897 |
| 1 | prefix_sdpa_eager | 0 | 267.698 | 266.753 | 336.0 | 0.0 | 5850.3353 |
| 2 | adaptive_graph | 8 | 461.643 | 460.698 | 648.0 | 30.374 | 4334.3062 |
| 2 | full_graph | 2 | 292.109 | 291.164 | 402.0 | 6.5321 | 4303.5858 |
| 2 | prefix_sdpa_eager | 0 | 267.698 | 266.753 | 336.0 | 0.0 | 5678.1218 |

## Acceptance gates

- Comparison modes present: 3/3
- Matched workloads: 12/12
- Required for full validation: exact reference-token match, all 12 workloads for all 3 modes in at least 2 process rounds, completed error-free.
- Graph capture, initialization and memory are not amortized into steady-state p50 times; performance may reverse in low-reuse deployments.
- No novel algorithm, production competitiveness, profitability or external hardware validation is demonstrated.
