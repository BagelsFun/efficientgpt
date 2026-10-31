# Environment (observed 2026-10-30)

| Item | Observed | How |
|---|---|---|
| OS | Windows 11 Pro 10.0.26200, native (no WSL) | `platform.platform()` |
| GPU | NVIDIA GeForce RTX 4060, CC 8.9, 24 SMs, 24 MiB L2, 8188 MiB | `nvidia-smi`, `torch.cuda.get_device_properties` |
| Driver | 617.14 | `nvidia-smi` |
| Clocks / temp at idle | SM 2820 MHz (max 3135), 44–50 °C, P0 | `nvidia-smi` |
| Background VRAM | ~2.9 GB used by other processes before runs | `nvidia-smi` |
| Python | 3.11.9 in `.venv` | `.venv\Scripts\python.exe` |
| PyTorch | 2.11.0+cu128, CUDA 12.8, `cuda.is_available()` True | probe |
| TF32 default | matmul `fp32_precision` off; benchmarks set `'tf32'` explicitly | probe, `benchmark_review.py` |
| Triton | `triton-windows==3.7.1.post27` installed 2026-10-30; Inductor GPU compile works (no MSVC on PATH needed) | `import triton`, `research/phase5/ci_runs/20261030_185752_inductor/diagnostics/smoke_compile.log` |
| nvcc / ncu / nsys | not on PATH | `Get-Command` |
| Host RAM / disk | 31.6 GB / 415 GB free on C: | CIM, `Get-PSDrive` |

Unknowns: power draw (`nvidia-smi` reports N/A), throttle reasons, Nsight counter access.
Model under test: `out-tinystories-v2/baseline-2500.pt`, 12.19M params, block_size 256, FP32 weights.
