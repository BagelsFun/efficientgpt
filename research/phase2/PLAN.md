# EfficientGPT — Phase 2

## Research question
Can workload-aware precision selection improve
language-model inference efficiency?

## Baselines
1. FP32
2. TF32
3. Existing KV-cache implementation

## Experimental method
- Test both trained model sizes
- Use identical prompts and batch sizes
- Measure first-token latency
- Measure total generation latency
- Measure throughput and GPU memory
- Evaluate numerical and quality differences
- Include precision-switching overhead
- Repeat tests to estimate variability

## Success criteria
- Faster than the strongest fixed-precision baseline
- No meaningful quality degradation
- Improvements reproducible across workloads

## Limitations
Results on one RTX 4060 do not establish
performance on production GPUs.

## Next stage
Compare against modern optimized inference
implementations and review related research.
