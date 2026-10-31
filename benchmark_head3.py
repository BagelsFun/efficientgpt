import time
import statistics
import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
from tokenizers import Tokenizer

assert torch.cuda.is_available(), "CUDA GPU required"

ckpt = torch.load(
    "out-tinystories-head3/ckpt.pt",
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

print("Checking output correctness...")
normal = generate_normal(128)
cached = generate_cached(128)
same = torch.equal(normal, cached)
print("Identical output:", same)

print("Benchmarking...")
a = benchmark("Normal", generate_normal)
b = benchmark("KV cache", generate_cached)

print(f"Speed change: {(b/a - 1)*100:+.2f}%")
print("EXPERIMENT COMPLETE")
