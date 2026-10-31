import time
import statistics
import torch
from tokenizers import Tokenizer
from model import GPT, GPTConfig

assert torch.cuda.is_available(), "CUDA required"

paths = {
    "6 heads": "out-tinystories-v2/baseline-2500.pt",
    "3 heads": "out-tinystories-head3/ckpt.pt"
}

models = {}

for name, path in paths.items():
    ckpt = torch.load(
        path, map_location="cpu", weights_only=False
    )
    m = GPT(GPTConfig(**ckpt["model_args"]))
    weights = {
        k.removeprefix("_orig_mod."): v
        for k, v in ckpt["model"].items()
    }
    m.load_state_dict(weights)
    m.eval().to("cuda")
    models[name] = m

tokenizer = Tokenizer.from_file(
    "data/tinystories_v2/tokenizer.json"
)

ids = tokenizer.encode(
    "Once upon a time, a little fox"
).ids

for batch in (1, 16):
    x = torch.tensor(
        [ids], dtype=torch.long, device="cuda"
    ).repeat(batch, 1)

    print(f"\nBATCH SIZE: {batch}")
    results = {name: [] for name in models}

    with torch.inference_mode():
        for m in models.values():
            m.generate(x, 16, temperature=0.8, top_k=50)
        torch.cuda.synchronize()

        for trial in range(10):
            order = list(models)
            if trial % 2:
                order.reverse()

            for name in order:
                torch.manual_seed(42)
                torch.cuda.synchronize()
                start = time.perf_counter()

                output = models[name].generate(
                    x, 128, temperature=0.8, top_k=50
                )

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                assert output.size(1) == x.size(1) + 128
                results[name].append(
                    batch * 128 / elapsed
                )

    medians = {}
    for name, speeds in results.items():
        medians[name] = statistics.median(speeds)
        print(
            f"{name}: {medians[name]:.2f} tokens/sec"
        )

    change = (
        medians["3 heads"] / medians["6 heads"] - 1
    ) * 100
    print(f"3-head speed difference: {change:+.2f}%")

print("\nARCHITECTURE COMPARISON COMPLETE")
