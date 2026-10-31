from pathlib import Path
import time
import statistics
import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
from tokenizers import Tokenizer

assert torch.cuda.is_available(), "CUDA GPU required"

ckpt = torch.load(
    "out-tinystories-v2/baseline-2500.pt",
    map_location="cpu",
    weights_only=False
)

model = GPT(GPTConfig(**ckpt["model_args"]))
weights = {
    k.removeprefix("_orig_mod."): v
    for k, v in ckpt["model"].items()
}
model.load_state_dict(weights)
model.eval().to("cuda")

tokenizer = Tokenizer.from_file(
    "data/tinystories_v2/tokenizer.json"
)

prompt = torch.tensor(
    [tokenizer.encode(
        "Once upon a time, a little fox"
    ).ids],
    dtype=torch.long,
    device="cuda"
)

@torch.inference_mode()
def cached_step(ids, cache=None):
    B, T = ids.shape
    offset = 0 if cache is None else cache[0][0].size(-2)
    assert offset + T <= model.config.block_size

    pos = torch.arange(
        offset, offset + T, device=ids.device
    )
    tr = model.transformer
    x = tr.drop(tr.wte(ids) + tr.wpe(pos))
    new_cache = []

    for i, block in enumerate(tr.h):
        a = block.attn
        h = block.ln_1(x)
        C = h.size(-1)
        H = a.n_head

        q, k, v = a.c_attn(h).split(C, dim=2)

        def shape(tensor):
            return tensor.view(
                B, T, H, C // H
            ).transpose(1, 2)

        q, k, v = shape(q), shape(k), shape(v)

        if cache is not None:
            k = torch.cat((cache[i][0], k), dim=-2)
            v = torch.cat((cache[i][1], v), dim=-2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0,
            is_causal=(cache is None)
        )

        y = y.transpose(1, 2).contiguous()
        y = y.view(B, T, C)

        x = x + a.resid_dropout(a.c_proj(y))
        x = x + block.mlp(block.ln_2(x))
        new_cache.append((k, v))

    logits = model.lm_head(tr.ln_f(x)[:, -1:, :])
    return logits, new_cache

@torch.inference_mode()
def generate_normal(n):
    ids = prompt.clone()
    for _ in range(n):
        logits, _ = model(ids)
        next_id = logits.argmax(dim=-1)
        ids = torch.cat((ids, next_id), dim=1)
    return ids

@torch.inference_mode()
def generate_cached(n):
    ids = prompt.clone()
    logits, cache = cached_step(ids)

    for i in range(n):
        next_id = logits.argmax(dim=-1)
        ids = torch.cat((ids, next_id), dim=1)

        if i + 1 < n:
            logits, cache = cached_step(next_id, cache)

    return ids

def benchmark(name, fn):
    fn(16)
    speeds = []

    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn(128)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        speeds.append(128 / elapsed)

    result = statistics.median(speeds)
    print(f"{name}: {result:.2f} tokens/sec")
    return result


import csv

small_ckpt = torch.load(
    "out-efficientgpt-small/ckpt.pt",
    map_location="cpu",
    weights_only=False
)
small = GPT(GPTConfig(**small_ckpt["model_args"]))
small.load_state_dict({
    k.removeprefix("_orig_mod."): v
    for k, v in small_ckpt["model"].items()
})
small.eval().to("cuda")

models = [
    ("Original 12M", model),
    ("Small 5.77M", small)
]

precision = torch.backends.cuda.matmul
original_mode = precision.fp32_precision
base_prompt = prompt.clone()
records = []

def measure(mode, x, trials=3, loops=15):
    precision.fp32_precision = mode
    speeds = []

    with torch.inference_mode():
        for _ in range(5):
            cached_step(x)
        torch.cuda.synchronize()

        for _ in range(trials):
            torch.cuda.synchronize()
            start = time.perf_counter()

            for _ in range(loops):
                cached_step(x)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            speeds.append(elapsed * 1000 / loops)

    return statistics.median(speeds)

def quality_check(x):
    with torch.inference_mode():
        precision.fp32_precision = "ieee"
        a, _ = cached_step(x)

        precision.fp32_precision = "tf32"
        b, _ = cached_step(x)

    agreement = (
        a.argmax(-1) == b.argmax(-1)
    ).float().mean().item()

    difference = (a - b).abs().max().item()
    return agreement, difference

print("EFFICIENTGPT ADAPTIVE PRECISION V1")

try:
    for name, current_model in models:
        model = current_model

        for length in (8, 64, 128):
            repeats = (
                length + base_prompt.size(1) - 1
            ) // base_prompt.size(1)

            for batch in (1, 16):
                x = base_prompt.repeat(
                    batch, repeats
                )[:, :length]

                # Calibration, not final test data
                cal_fp = measure("ieee", x)
                cal_tf = measure("tf32", x)
                agree, diff = quality_check(x)

                # Require a meaningful speed advantage
                # and matching next-token predictions.
                choice = (
                    "tf32"
                    if cal_tf < cal_fp * 0.97
                    and agree == 1.0
                    else "ieee"
                )

                # Independent timing trials
                fp = measure("ieee", x, trials=5)
                tf = measure("tf32", x, trials=5)
                adaptive = tf if choice == "tf32" else fp

                record = {
                    "model": name,
                    "prompt_tokens": length,
                    "batch": batch,
                    "selected": choice,
                    "fp32_ms": round(fp, 4),
                    "tf32_ms": round(tf, 4),
                    "adaptive_ms": round(adaptive, 4),
                    "agreement": agree,
                    "max_logit_diff": round(diff, 6)
                }
                records.append(record)

                print(
                    f"{name} | tokens={length} "
                    f"| batch={batch} | chosen={choice}"
                )
                print(
                    f"FP32={fp:.3f}ms "
                    f"TF32={tf:.3f}ms "
                    f"Adaptive={adaptive:.3f}ms"
                )

    output = Path(
        "research/phase2/adaptive_results.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(records[0])
        )
        writer.writeheader()
        writer.writerows(records)

    print("\nAGGREGATE RESULTS")
    for key in ("fp32_ms", "tf32_ms", "adaptive_ms"):
        print(
            key,
            round(sum(r[key] for r in records), 3),
            "ms"
        )

    print("\nADAPTIVE EXPERIMENT COMPLETE")
    print("Results:", output)

finally:
    precision.fp32_precision = original_mode
