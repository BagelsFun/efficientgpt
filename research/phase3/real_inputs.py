import time
import math
import statistics
import torch
from model import GPT, GPTConfig

assert torch.cuda.is_available()

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

backend = torch.backends.cuda.matmul
previous = backend.fp32_precision
backend.fp32_precision = "tf32"

try:
    with torch.inference_mode():
        for batch, length in [(1, 8), (16, 128)]:
            print(f"\nBATCH={batch}, LENGTH={length}")

            inputs = [
                torch.randint(
                    model.config.vocab_size,
                    (batch, length),
                    device="cuda"
                )
                for _ in range(16)
            ]

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(5):
                    model(inputs[0])

            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()

            static_x = inputs[0].clone()
            graph = torch.cuda.CUDAGraph()

            t0 = time.perf_counter()
            with torch.cuda.graph(graph):
                graph_output, _ = model(static_x)
            torch.cuda.synchronize()
            setup = (time.perf_counter() - t0) * 1000

            for x in inputs:
                expected, _ = model(x)
                static_x.copy_(x)
                graph.replay()
                torch.cuda.synchronize()

                assert torch.allclose(
                    expected, graph_output,
                    atol=1e-4, rtol=1e-4
                ), "Output mismatch!"

            print("16 different inputs: PASS")

            results = {"Eager": [], "Graph": []}

            for trial in range(8):
                order = ["Eager", "Graph"]
                if trial % 2:
                    order.reverse()

                for method in order:
                    torch.cuda.synchronize()
                    start = time.perf_counter()

                    for i in range(64):
                        x = inputs[i % len(inputs)]
                        if method == "Eager":
                            model(x)
                        else:
                            static_x.copy_(x)
                            graph.replay()

                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    results[method].append(elapsed * 1000 / 64)

            eager = statistics.median(results["Eager"])
            graphed = statistics.median(results["Graph"])

            print(f"Setup: {setup:.3f} ms")
            print(f"Eager: {eager:.4f} ms/pass")
            print(f"Graph + copy: {graphed:.4f} ms/pass")
            print(f"Speedup: {eager/graphed:.3f}x")

            if eager > graphed:
                needed = max(
                    1, math.ceil(setup / (eager - graphed))
                )
                print(f"Break-even: {needed} passes")
            else:
                print("No measured steady-state advantage")

finally:
    backend.fp32_precision = previous

print("\nREAL-INPUT TEST COMPLETE")
