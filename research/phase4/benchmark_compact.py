"""Isolated-process comparison of full-capacity vs compact CUDA Graph KV caches.

Run from the nanoGPT ROOT:
    python -m research.phase4.benchmark_compact --all

Requires existing Phase 4 decoder.py and benchmark_decode.py files, your trusted
local TinyStories checkpoint/tokenizer, and an NVIDIA CUDA GPU.

Each method runs in a separate process to avoid comparing different live graph
private pools. Reports PyTorch allocated/reserved peaks, not device-wide VRAM.
"""

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .benchmark_decode import load_model, make_prompts
from .compact_cache import CompactGraphDecoder
from .decoder import GraphDecoder, generate_reference

MIB = 1024 * 1024


def mib(x):
    return round(x / MIB, 3)


def p95(values):
    a = sorted(values)
    return a[min(len(a)-1, math.ceil(0.95 * len(a))-1)]


def run_one(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if args.prompt + args.generate > args.block_check:
        raise ValueError("Workload may exceed model context")
    model = load_model(args.checkpoint)
    if args.prompt + args.generate > model.config.block_size:
        raise ValueError("Workload exceeds actual model context")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    prompts = make_prompts(tokenizer, model.config.vocab_size, args.batch, args.prompt, args.requests)
    torch.backends.cuda.matmul.fp32_precision = "tf32"

    with torch.inference_mode():
        baseline_alloc = torch.cuda.memory_allocated()
        baseline_res = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        if args.mode == "full":
            decoder = GraphDecoder(model, prompts[0])
            capacity = model.config.block_size
        else:
            decoder = CompactGraphDecoder(model, prompts[0], args.generate)
            capacity = args.prompt + args.generate

        for prompt in prompts:
            expected = generate_reference(model, prompt, args.generate)
            actual = decoder.generate(prompt, args.generate)
            torch.cuda.synchronize()
            if not torch.equal(expected, actual):
                raise AssertionError("Generated tokens don't match dynamic KV reference")
            del expected, actual

        for _ in range(3):
            decoder.generate(prompts[0], args.generate)
        torch.cuda.synchronize()

        latencies = []
        for i in range(args.trials):
            order = list(range(len(prompts)))
            if i % 2:
                order.reverse()
            for j in order:
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = decoder.generate(prompts[j], args.generate)
                torch.cuda.synchronize()
                latencies.append((time.perf_counter()-start)*1000)
                del output

        row = dict(
            mode=args.mode, batch=args.batch, prompt_tokens=args.prompt,
            generated_tokens=args.generate, cache_capacity=capacity,
            trials=args.trials, request_variants=args.requests,
            p50_ms=round(statistics.median(latencies), 4),
            p95_ms=round(p95(latencies), 4),
            min_ms=round(min(latencies), 4), max_ms=round(max(latencies), 4),
            graph_capture_ms=round(decoder.capture_ms, 4),
            peak_allocated_mib=mib(torch.cuda.max_memory_allocated()),
            peak_reserved_mib=mib(torch.cuda.max_memory_reserved()),
            incremental_peak_allocated_mib=mib(max(0, torch.cuda.max_memory_allocated()-baseline_alloc)),
            incremental_peak_reserved_mib=mib(max(0, torch.cuda.max_memory_reserved()-baseline_res)),
            correctness="PASS",
        )
        return row


def run_all(args):
    all_rows = []
    output = Path(args.output)
    for batch, prompt, generated in ((1, 8, 64), (16, 128, 64)):
        for mode in ("full", "compact"):
            print(f"Testing {mode}: batch={batch}, prompt={prompt}, generated={generated}", flush=True)
            call = [
                sys.executable, "-m", "research.phase4.benchmark_compact",
                "--mode", mode, "--batch", str(batch), "--prompt", str(prompt),
                "--generate", str(generated), "--trials", str(args.trials),
                "--requests", str(args.requests), "--checkpoint", str(args.checkpoint),
                "--tokenizer", str(args.tokenizer), "--block-check", str(args.block_check),
            ]
            proc = subprocess.run(call, check=False, capture_output=True, text=True)
            if proc.returncode != 0:
                print(proc.stdout, flush=True)
                print(proc.stderr, file=sys.stderr, flush=True)
                raise RuntimeError(f"Benchmark failed for {mode}, batch={batch}")
            row = json.loads(proc.stdout.split("RESULT_JSON:")[-1].strip())
            all_rows.append(row)
            print(f"  {row['p50_ms']} ms median, {row['peak_allocated_mib']} MiB allocated peak, "
                  f"{row['peak_reserved_mib']} MiB reserved peak; correctness {row['correctness']}",flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print("\nCOMPARISONS (compact vs full CUDA Graph):", flush=True)
    for batch in (1, 16):
        full, compact = [r for r in all_rows if r["batch"] == batch]
        memory_change = 100*(compact["incremental_peak_allocated_mib"] / full["incremental_peak_allocated_mib"]-1) if full["incremental_peak_allocated_mib"] else float("nan")
        speed_change = 100*(full["p50_ms"] / compact["p50_ms"]-1)
        print(f"batch={batch}: allocated-memory change={memory_change:+.2f}%, "
              f"generation speed change={speed_change:+.2f}%, "
              f"outputs correct: {full['correctness'] == compact['correctness'] == 'PASS'}",flush=True)
    print(f"Saved experiment CSV: {output}", flush=True)
    print("Local experiment only. No novel invention or industrial baseline demonstrated.", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="Run isolated comparisons on two workloads")
    ap.add_argument("--mode", choices=("full", "compact"), default="full")
    ap.add_argument("--checkpoint", type=Path, default=Path("out-tinystories-v2/baseline-2500.pt"))
    ap.add_argument("--tokenizer", type=Path, default=Path("data/tinystories_v2/tokenizer.json"))
    ap.add_argument("--output", default="research/phase4/compact_cache_results.csv")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--prompt", type=int, default=8)
    ap.add_argument("--generate", type=int, default=64)
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--requests", type=int, default=4)
    ap.add_argument("--block-check", type=int, default=256)
    args = ap.parse_args()
    if args.trials < 3 or args.requests < 2 or min(args.batch, args.prompt, args.generate) <= 0:
        ap.error("Use >=3 trials, >=2 requests, and positive batch/prompt/generate")
    if args.all:
        run_all(args)
    else:
        print("RESULT_JSON:"+json.dumps(run_one(args)), flush=True)


if __name__ == "__main__":
    main()
