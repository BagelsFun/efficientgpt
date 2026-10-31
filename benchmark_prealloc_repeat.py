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


@torch.inference_mode()
def prealloc_step(ids, buffers, offset):
    B, T = ids.shape
    tr = model.transformer
    pos = torch.arange(offset, offset + T, device=ids.device)
    x = tr.drop(tr.wte(ids) + tr.wpe(pos))

    for i, block in enumerate(tr.h):
        a = block.attn
        h = block.ln_1(x)
        C = h.size(-1)
        H = a.n_head

        q, k, v = a.c_attn(h).split(C, dim=2)
        q, k, v = [
            t.reshape(B, T, H, C // H).transpose(1, 2)
            for t in (q, k, v)
        ]

        kb, vb = buffers[i]
        kb[:, :, offset:offset+T].copy_(k)
        vb[:, :, offset:offset+T].copy_(v)

        y = F.scaled_dot_product_attention(
            q,
            kb[:, :, :offset+T],
            vb[:, :, :offset+T],
            dropout_p=0.0,
            is_causal=(offset == 0)
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + a.resid_dropout(a.c_proj(y))
        x = x + block.mlp(block.ln_2(x))

    return model.lm_head(tr.ln_f(x)[:, -1:, :])

@torch.inference_mode()
def generate_preallocated(n):
    B = prompt.size(0)
    H = model.config.n_head
    C = model.config.n_embd
    S = model.config.block_size

    buffers = [
        (
            torch.empty(B, H, S, C // H, device="cuda"),
            torch.empty(B, H, S, C // H, device="cuda")
        )
        for _ in range(model.config.n_layer)
    ]

    ids = prompt.clone()
    logits = prealloc_step(ids, buffers, 0)

    for i in range(n):
        nxt = logits.argmax(dim=-1)
        ids = torch.cat((ids, nxt), dim=1)
        if i + 1 < n:
            logits = prealloc_step(nxt, buffers, ids.size(1) - 1)

    return ids


base_prompt = prompt.clone()

for batch in (1, 16):
    prompt = base_prompt.repeat(batch, 1)

    print("\nBATCH SIZE:", batch)

    assert torch.equal(
        generate_cached(128),
        generate_preallocated(128)
    ), "Output mismatch"

    methods = [
        ("Dynamic", generate_cached),
        ("Preallocated", generate_preallocated)
    ]

    speeds = {name: [] for name, _ in methods}
    memory = {name: [] for name, _ in methods}

    for name, fn in methods:
        fn(32)

    for trial in range(12):
        order = methods if trial % 2 == 0 else methods[::-1]

        for name, fn in order:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            start = time.perf_counter()
            fn(128)
            torch.cuda.synchronize()

            seconds = time.perf_counter() - start
            speeds[name].append(batch * 128 / seconds)
            memory[name].append(
                torch.cuda.max_memory_allocated() / 1048576
            )

    for name, _ in methods:
        print(
            name,
            "median throughput:",
            round(statistics.median(speeds[name]), 2),
            "tokens/sec"
        )
        print(
            name,
            "peak PyTorch allocation:",
            round(max(memory[name]), 2),
            "MB"
        )

    a = statistics.median(speeds["Dynamic"])
    b = statistics.median(speeds["Preallocated"])

    print("Speed change:", round((b / a - 1) * 100, 2), "%")
    print("Output correctness: PASS")

print("\nREPEATED BENCHMARK COMPLETE")
