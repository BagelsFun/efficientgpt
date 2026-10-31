from pathlib import Path
from itertools import islice
import pickle
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

folder = Path("data/tinystories_v2")
folder.mkdir(parents=True, exist_ok=True)

def get_stories(split, count):
    print(f"Loading {count} {split} stories...", flush=True)
    ds = load_dataset(
        "roneneldan/TinyStories",
        split=split,
        streaming=True
    )
    return [
        row["text"].strip()
        for row in islice(ds, count)
        if row.get("text", "").strip()
    ]

train = get_stories("train", 30000)
val = get_stories("validation", 1500)

print("Building your own tokenizer...", flush=True)

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=False
)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=4096,
    min_frequency=2,
    special_tokens=["[EOS]", "[UNK]"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)

tokenizer.train_from_iterator(train, trainer=trainer)
tokenizer.save(str(folder / "tokenizer.json"))

eos = tokenizer.token_to_id("[EOS]")

for name, stories in [("train", train), ("val", val)]:
    print(f"Encoding {name} data...", flush=True)
    with (folder / f"{name}.bin").open("wb") as f:
        for story in stories:
            ids = tokenizer.encode(story).ids + [eos]
            np.asarray(ids, dtype=np.uint16).tofile(f)

with (folder / "meta.pkl").open("wb") as f:
    pickle.dump({
        "vocab_size": tokenizer.get_vocab_size(),
        "eos_id": eos
    }, f)

print("SUCCESS! AI V2 training data is ready.")
