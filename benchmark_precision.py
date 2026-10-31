import torch
import time
import statistics
from model import GPT, GPTConfig
from tokenizers import Tokenizer

assert torch.cuda.is_available(), "CUDA required"

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

def benchmark(name, use_bf16):
    speeds = []
    peaks = []

    with torch.inference_mode():
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=use_bf16
        ):
            model.generate(x, 32, 0.8, 50)

        for i in range(5):
            torch.manual_seed(42)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            start = time.perf_counter()
            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=use_bf16
            ):
                model.generate(x, 128, 0.8, 50)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            speeds.append(128 / elapsed)
            peaks.append(
                torch.cuda.max_memory_allocated()
            )

    speed = statistics.median(speeds)
    memory = max(peaks) / 1024**3

    print(f"{name}: {speed:.2f} tokens/sec")
    print(f"Peak PyTorch allocation: {memory:.3f} GB")
    return speed

print("Starting precision comparison...")
fp32 = benchmark("FP32", False)
bf16 = benchmark("BF16", True)
fp32_check = benchmark("FP32 repeat", False)

baseline = (fp32 + fp32_check) / 2
change = (bf16 / baseline - 1) * 100

print(f"BF16 speed change: {change:+.2f}%")
print("EXPERIMENT COMPLETE")
