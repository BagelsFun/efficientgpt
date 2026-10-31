import time
import statistics
import torch
from model import GPT, GPTConfig
from tokenizers import Tokenizer

assert torch.cuda.is_available(), "CUDA GPU required"

ckpt = torch.load(
    "out-tinystories-v2/baseline-2500.pt",
    map_location="cpu",
    weights_only=False
)

model = GPT(GPTConfig(**ckpt["model_args"]))
model.load_state_dict({
    k.removeprefix("_orig_mod."): v
    for k, v in ckpt["model"].items()
})
model.eval().cuda()

tokenizer = Tokenizer.from_file(
    "data/tinystories_v2/tokenizer.json"
)
base = tokenizer.encode(
    "Once upon a time, a little fox"
).ids

precision = torch.backends.cuda.matmul
previous = precision.fp32_precision
precision.fp32_precision = "tf32"

try:
    for batch, length in (
        (1, 8), (1, 128),
        (16, 8), (16, 128)
    ):
        ids = (base * ((length + len(base)-1)//len(base)))[:length]
        x = torch.tensor(
            [ids] * batch, device="cuda", dtype=torch.long
        )

        print(f"\nBATCH={batch}, PROMPT={length}")

        with torch.inference_mode():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())

            with torch.cuda.stream(stream):
                for _ in range(5):
                    model(x)

            torch.cuda.current_stream().wait_stream(stream)

            static_x = x.clone()
            graph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(graph):
                graph_logits, _ = model(static_x)

            eager_logits, _ = model(x)
            graph.replay()
            torch.cuda.synchronize()

            error = (
                eager_logits - graph_logits
            ).abs().max().item()

            print("Maximum output difference:", error)

            methods = {
                "Eager TF32": lambda: model(x),
                "CUDA Graph": graph.replay
            }
            timings = {name: [] for name in methods}

            for trial in range(8):
                order = list(methods)
                if trial % 2:
                    order.reverse()

                for name in order:
                    torch.cuda.synchronize()
                    start = time.perf_counter()

                    for _ in range(50):
                        methods[name]()

                    torch.cuda.synchronize()
                    timings[name].append(
                        (time.perf_counter()-start)*1000/50
                    )

            med = {
                name: statistics.median(values)
                for name, values in timings.items()
            }

            for name, ms in med.items():
                print(f"{name}: {ms:.4f} ms/forward")

            print(
                "Speedup:",
                round(med["Eager TF32"]/med["CUDA Graph"], 3),
                "x"
            )

finally:
    precision.fp32_precision = previous

print("\nCUDA GRAPH BENCHMARK COMPLETE")
