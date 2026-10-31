import torch
from tokenizers import Tokenizer
from model import GPT, GPTConfig

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading your AI V2...")

checkpoint = torch.load(
    "out-tinystories-v2/ckpt.pt",
    map_location="cpu",
    weights_only=False
)

model = GPT(GPTConfig(**checkpoint["model_args"]))

weights = {
    k.removeprefix("_orig_mod."): v
    for k, v in checkpoint["model"].items()
}

model.load_state_dict(weights)
model.eval()
model.to(device)

tokenizer = Tokenizer.from_file(
    "data/tinystories_v2/tokenizer.json"
)

prompt = "Once upon a time, a little fox"

ids = tokenizer.encode(prompt).ids

x = torch.tensor(
    [ids],
    dtype=torch.long,
    device=device
)

print("\nGenerating your AI's story...\n")

with torch.inference_mode():
    output = model.generate(
        x,
        max_new_tokens=180,
        temperature=0.8,
        top_k=50
    )

text = tokenizer.decode(
    output[0].tolist(),
    skip_special_tokens=False
)

print(text.split("[EOS]")[0])
