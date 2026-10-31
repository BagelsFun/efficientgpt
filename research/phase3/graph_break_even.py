import time
import math
import statistics
import torch
from model import GPT, GPTConfig

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

precision = torch.backends.cuda.matmul
previous = precision.fp32_precision
precision.fp32_precision = "tf32"

try:
    for batch, length in [(1, 8), (16, 128)]:
        print(f"\nBATCH={batch}, PROMPT={length}")

        x = torch.arange(
            length, device="cuda", dtype=torch.long
        ).unsqueeze(0).repeat(batch, 1)

        x %= model.config.vocab_size

        with torch.inference_mode():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())

            with torch.cuda.stream(stream):
                for _ in range(5):
                    model(x)

            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()

            start = time.perf_counter()

            static_x = x.clone()
            graph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(graph):
                graph_logits, _ = model(static_x)

            torch.cuda.synchronize()

            setup_ms = (time.perf_counter() - start) * 1000

            reference, _ = model(x)
            graph.replay()
            torch.cuda.synchronize()

            assert torch.allclose(
                reference, graph_logits,
                rtol=1e-4, atol=1e-5
            ), "Graph output mismatch"

            methods = {
                "Eager": lambda: model(x),
                "Graph": graph.replay
            }
            results = {name: [] for name in methods}

            for trial in range(6):
                order = list(methods)
                if trial % 2:
                    order.reverse()

                for name in order:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()

                    for _ in range(50):
                        methods[name]()

                    torch.cuda.synchronize()

                    ms = (time.perf_counter()-t0)*1000/50
                    results[name].append(ms)

            eager = statistics.median(results["Eager"])
            replay = statistics.median(results["Graph"])

            print(f"Graph setup cost: {setup_ms:.3f} ms")
            print(f"Eager forward: {eager:.4f} ms")
            print(f"Graph replay: {replay:.4f} ms")

            if eager > replay:
                breakeven = math.ceil(
                    setup_ms / (eager - replay)
                )
                print(
                    "Conservative break-even:",
                    breakeven, "forward passes"
                )
            else:
                print("No speed benefit measured")

            for requests in (1, 10, 100, 1000):
                eager_total = requests * eager
                graph_total = setup_ms + requests * replay

                print(
                    f"{requests} passes: "
                    f"Eager={eager_total:.2f}ms "
                    f"Graph={graph_total:.2f}ms"
                )

finally:
    precision.fp32_precision = previous

print("\nBREAK-EVEN EXPERIMENT COMPLETE")
