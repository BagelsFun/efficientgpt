r"""EfficientGPT Phase 4: local CUDA inference memory profile.

Run from the original nanoGPT repository root, after copying this script into
research/phase4/memory_profile.py:
    .\.venv\Scripts\python.exe -m research.phase4.memory_profile

Three decoding paths use the SAME trained model, prompts and output lengths.
Reports PyTorch-allocator peaks (not whole-system or nvidia-smi GPU memory),
including the construction/capture of each decoder. Do not publish checkpoints.
"""

import argparse
import csv
import gc
import platform
from pathlib import Path

import torch
from tokenizers import Tokenizer
from .benchmark_decode import load_model, make_prompts
from .decoder import GraphDecoder, StaticKVDecoder, generate_reference

MIB = 1024 * 1024


def mib(value):
    return round(value / MIB, 3)


def cleanup_cuda():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def profile_mode(model, mode, prompts, generated, expected):
    cleanup_cuda()
    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()

    engine = None
    if mode == 'dynamic_kv_eager':
        run = lambda prompt: generate_reference(model, prompt, generated)
        capture_ms = 0.0
    elif mode == 'static_kv_eager':
        engine = StaticKVDecoder(model, prompts[0].size(0))
        run = lambda prompt: engine.generate_eager(prompt, generated)
        capture_ms = 0.0
    elif mode == 'static_kv_graph':
        engine = GraphDecoder(model, prompts[0])
        run = lambda prompt: engine.generate(prompt, generated)
        capture_ms = float(engine.capture_ms)
    else:
        raise ValueError(mode)

    with torch.inference_mode():
        for i in range(8):
            index = i % len(prompts)
            produced = run(prompts[index])
            torch.cuda.synchronize()
            if not torch.equal(produced.cpu(), expected[index]):
                raise AssertionError(f'Token mismatch: {mode}, prompt {index}')
            del produced

    torch.cuda.synchronize()
    # Peak measures include construction, graph capture, and all requests.
    row = {
        'mode': mode,
        'batch_size': prompts[0].size(0),
        'prompt_tokens': prompts[0].size(1),
        'generated_tokens': generated,
        'output_correct': True,
        'baseline_allocated_mib': mib(allocated_before),
        'baseline_reserved_mib': mib(reserved_before),
        'peak_allocated_mib': mib(torch.cuda.max_memory_allocated()),
        'peak_reserved_mib': mib(torch.cuda.max_memory_reserved()),
        'incremental_peak_allocated_mib': mib(max(0, torch.cuda.max_memory_allocated() - allocated_before)),
        'capture_ms': round(capture_ms, 4),
    }
    del run, engine
    cleanup_cuda()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('out-tinystories-v2/baseline-2500.pt'))
    parser.add_argument('--tokenizer', type=Path, default=Path('data/tinystories_v2/tokenizer.json'))
    parser.add_argument('--output', type=Path, default=Path('research/phase4/memory_results.csv'))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error('An NVIDIA CUDA GPU is required')
    if not args.tokenizer.exists():
        parser.error(f'Tokenizer missing: {args.tokenizer}')
    if not args.checkpoint.exists():
        parser.error(f'Checkpoint missing: {args.checkpoint}')

    print('LOCAL GPU MEMORY PROFILE — not production evidence', flush=True)
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    print('Python:', platform.python_version(), 'PyTorch:', torch.__version__, flush=True)
    print('IMPORTANT: load only checkpoints you created and trust.', flush=True)

    model = load_model(args.checkpoint)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    precision = torch.backends.cuda.matmul
    old_precision = precision.fp32_precision
    precision.fp32_precision = 'tf32'
    rows = []
    try:
        with torch.inference_mode():
            for batch, prompt_length, generated in [(1, 8, 64), (16, 128, 64)]:
                if prompt_length + generated > model.config.block_size:
                    raise ValueError('Requested workload exceeds model block_size')
                prompts = make_prompts(tokenizer, model.config.vocab_size, batch, prompt_length, count=4)
                # Store reference tokens on CPU so GPU tensor lifetimes do not
                # distort each method's peak allocated memory.
                expected = [generate_reference(model, p, generated).cpu() for p in prompts]
                cleanup_cuda()
                print(f'\nWORKLOAD: batch={batch}, prompt={prompt_length}, generate={generated}', flush=True)
                for mode in ('dynamic_kv_eager', 'static_kv_eager', 'static_kv_graph'):
                    row = profile_mode(model, mode, prompts, generated, expected)
                    rows.append(row)
                    print(f"{mode}: PASS  peak allocated={row['peak_allocated_mib']} MiB  "
                          f"peak reserved={row['peak_reserved_mib']} MiB  "
                          f"incremental peak allocated={row['incremental_peak_allocated_mib']} MiB  "
                          f"capture={row['capture_ms']} ms", flush=True)
                del prompts, expected
                cleanup_cuda()
    finally:
        precision.fp32_precision = old_precision

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print('\nMEMORY PROFILE COMPLETE', flush=True)
    print('CSV:', args.output.resolve(), flush=True)
    print('PyTorch allocator only; peak reserved may include graph-private pools.', flush=True)
    print('Does not include CUDA driver allocations, model loading, or other applications.', flush=True)


if __name__ == '__main__':
    main()
