import time
import statistics
import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
from tokenizers import Tokenizer

assert torch.cuda.is_available(), "CUDA GPU required"

ckpt = torch.load(
    "out-efficientgpt-small/ckpt.pt",
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


import math
import numpy as np

backend = torch.backends.cuda.matmul
original_precision = backend.fp32_precision

def set_precision(mode):
    backend.fp32_precision = mode

base_prompt = prompt.clone()
torch.cuda.synchronize()

try:
    for length in (8, 64, 128):
        repeats = math.ceil(length / base_prompt.size(1))
        x = base_prompt.repeat(16, repeats)[:, :length]

        print(f"\nPROMPT LENGTH: {length}")
        times = {"ieee": [], "tf32": []}

        for mode in times:
            set_precision(mode)
            with torch.inference_mode():
                for _ in range(10):
                    cached_step(x)
            torch.cuda.synchronize()

        for trial in range(12):
            order = ("ieee", "tf32")
            if trial % 2:
                order = order[::-1]

            for mode in order:
                set_precision(mode)
                torch.cuda.synchronize()
                start = time.perf_counter()

                with torch.inference_mode():
                    for _ in range(20):
                        cached_step(x)

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                times[mode].append(elapsed * 1000 / 20)

        fp = statistics.median(times["ieee"])
        tf = statistics.median(times["tf32"])

        print(f"FP32 prefill: {fp:.4f} ms")
        print(f"TF32 prefill: {tf:.4f} ms")
        print(f"Speedup: {fp/tf:.3f}x")
        print(f"Latency change: {(tf/fp-1)*100:+.2f}%")

        set_precision("ieee")
        with torch.inference_mode():
            reference, _ = cached_step(x)

        set_precision("tf32")
        with torch.inference_mode():
            candidate, _ = cached_step(x)

        diff = (reference - candidate).abs()
        agree = (
            reference.argmax(-1) == candidate.argmax(-1)
        ).float().mean().item()

        print(f"Maximum logit difference: {diff.max().item():.6f}")
        print(f"Next-token agreement: {agree*100:.1f}%")

    data = np.memmap(
        "data/tinystories_v2/val.bin",
        dtype=np.uint16,
        mode="r"
    )

    rng = np.random.default_rng(2026)
    block = model.config.block_size
    starts = rng.integers(
        0, len(data)-block-1, size=(10, 8)
    )

    batches = []
    for positions in starts:
        a = np.stack([
            data[j:j+block] for j in positions
        ]).astype(np.int64)
        b = np.stack([
            data[j+1:j+block+1] for j in positions
        ]).astype(np.int64)
        batches.append((
            torch.from_numpy(a).to("cuda"),
            torch.from_numpy(b).to("cuda")
        ))

    print("\nVALIDATION QUALITY")

    for mode in ("ieee", "tf32"):
        set_precision(mode)
        losses = []
        with torch.inference_mode():
            for x, y in batches:
                _, loss = model(x, y)
                losses.append(loss.item())
        print(
            f"{mode}: validation loss "
            f"{statistics.mean(losses):.6f}"
        )

finally:
    set_precision(original_precision)

print("\nTF32 EXPERIMENT COMPLETE")
