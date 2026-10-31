import time
import statistics
import torch
from model import GPT, GPTConfig
from tokenizers import Tokenizer

assert torch.cuda.is_available(), "CUDA GPU required"

checkpoint = torch.load(
    "out-tinystories-v2/baseline-2500.pt",
    map_location="cpu",
    weights_only=False
)

model = GPT(GPTConfig(**checkpoint["model_args"]))

weights = {
    k.removeprefix("_orig_mod."): v
    for k, v in checkpoint["model"].items()
}

model.load_state_dict(weights)
model.eval().to("cuda")

tokenizer = Tokenizer.from_file(
    "data/tinystories_v2/tokenizer.json"
)

ids = tokenizer.encode(
    "Once upon a time, a little fox"
).ids

x = torch.tensor([ids], device="cuda")

def generate(n):
    return model.generate(
        x,
        max_new_tokens=n,
        temperature=0.8,
        top_k=50
    )

print("Benchmarking your AI...")
print("Parameters:", sum(p.numel() for p in model.parameters()))

with torch.inference_mode():
    generate(20)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    speeds = []

    for i in range(3):
        torch.manual_seed(42)
        torch.cuda.synchronize()
        start = time.perf_counter()

        generate(128)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        speed = 128 / elapsed
        speeds.append(speed)
        print(f"Run {i+1}: {speed:.2f} tokens/sec")

print("Average speed:", round(statistics.mean(speeds), 2), "tokens/sec")
print("Peak VRAM:", round(torch.cuda.max_memory_allocated()/1024**3, 2), "GB")
print("BENCHMARK COMPLETE")
