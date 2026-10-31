# EfficientGPT — Phase 1 Research Results

## Hardware
NVIDIA RTX 4060, 8GB VRAM
PyTorch 2.11.0+cu128

## Models
Original: 12.19 million parameters
Small: 5.77 million parameters

## Validation quality
Original validation loss: 2.2069
Small validation loss: 2.2937

## Cached inference, batch size 16
Original: 8868.21 tokens/sec
Small: 8919.01 tokens/sec

## TF32, 128-token prefill, batch size 16
Original FP32: 8.6774 ms
Original TF32: 5.0593 ms
Small FP32: 4.3373 ms
Small TF32: 2.9355 ms

## Limitations
- Single GPU
- Small experimental language models
- Short context window
- Limited evaluation samples
- Different training budgets
- No comparison with production inference engines
- No original algorithmic breakthrough demonstrated

## Next research objectives
1. Identify a specific unresolved efficiency problem.
2. Review existing published methods.
3. Establish strong reference implementations.
4. Design and test a falsifiable hypothesis.
5. Reproduce findings across models and hardware.
