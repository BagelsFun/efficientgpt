import math
import numpy as np
import torch
from model import GPT, GPTConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(
    "out-tinystories-head3/ckpt.pt",
    map_location="cpu",
    weights_only=False
)

model = GPT(GPTConfig(**checkpoint["model_args"]))

weights = {
    k.removeprefix("_orig_mod."): v
    for k, v in checkpoint["model"].items()
}

model.load_state_dict(weights)
model.eval().to(device)

data = np.memmap(
    "data/tinystories_v2/val.bin",
    dtype=np.uint16,
    mode="r"
)

block = model.config.block_size
rng = np.random.default_rng(2026)
losses = []

print("Evaluating AI V2 on validation data...")

with torch.inference_mode():
    for i in range(30):
        starts = rng.integers(
            0, len(data) - block - 1, size=8
        )

        x = torch.tensor(
            np.stack([data[j:j+block] for j in starts]).astype(np.int64),
            device=device
        )
        y = torch.tensor(
            np.stack([data[j+1:j+block+1] for j in starts]).astype(np.int64),
            device=device
        )

        _, loss = model(x, y)
        losses.append(loss.item())

avg_loss = sum(losses) / len(losses)

print(f"Validation loss: {avg_loss:.4f}")
print(f"Perplexity: {math.exp(avg_loss):.2f}")
print(f"Token predictions evaluated: {30*8*block:,}")
print("EVALUATION COMPLETE")
