"""EfficientGPT V1 — run your trained TinyStories GPT with a trusted checkpoint.

From the nanoGPT project root:
    python -m research.v1.demo --prompt "Once upon a time" --tokens 32
    python -m research.v1.demo --mode compact-graph --prompt "Once upon a time" --tokens 32

Greedy, deterministic text completion; not a general-purpose chatbot.
For security, load only model checkpoints created by you or trusted sources.
"""
import argparse
import time
from pathlib import Path


def load_trusted_model(checkpoint, device):
    import torch
    from model import GPT, GPTConfig
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint}")
    # IMPORTANT: `weights_only=False` can execute code from untrusted pickles.
    # This is intended only for the owner's own trusted nanoGPT checkpoint.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = GPTConfig(**ckpt["model_args"])
    model = GPT(config)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def generate(model, prompt_ids, token_count, mode="prefix"):
    import torch
    from ..phase4.decoder import _check_request, generate_reference, GraphDecoder
    from ..phase4.compact_cache import CompactGraphDecoder
    from ..phase5.prefix_eager import PrefixSDPADecoder
    _check_request(model, prompt_ids, token_count)
    if mode == "prefix":
        return PrefixSDPADecoder(model, prompt_ids.shape[0]).generate(prompt_ids, token_count)
    if mode == "reference":
        return generate_reference(model, prompt_ids, token_count)
    if prompt_ids.device.type != "cuda":
        raise ValueError("CUDA Graph mode needs an NVIDIA GPU; use --mode prefix on CPU")
    if mode == "full-graph":
        return GraphDecoder(model, prompt_ids).generate(prompt_ids, token_count)
    if mode == "compact-graph":
        return CompactGraphDecoder(model, prompt_ids, token_count).generate(prompt_ids, token_count)
    raise ValueError(f"Unknown decoder mode: {mode}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", default="Once upon a time, a little fox")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--mode", choices=("prefix", "reference", "full-graph", "compact-graph"), default="prefix")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--checkpoint", type=Path, default=Path('out-tinystories-v2/baseline-2500.pt'))
    ap.add_argument("--tokenizer", type=Path, default=Path('data/tinystories_v2/tokenizer.json'))
    ap.add_argument("--verify", action="store_true", help="Compare token IDs with the reference implementation")
    args = ap.parse_args()
    if args.tokens < 1:
        ap.error("--tokens must be positive")
    if not args.tokenizer.is_file():
        ap.error(f"Tokenizer missing: {args.tokenizer}")
    import torch
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit("Install the existing nanoGPT tokenizers dependency in your venv") from exc
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA unavailable; use --device cpu and --mode prefix")
    if device == "cpu" and args.mode.endswith("graph"):
        ap.error("CUDA Graph mode requires CUDA; use --mode prefix")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    ids = tokenizer.encode(args.prompt).ids
    if not ids:
        ap.error("Prompt must tokenize to at least one token")
    with torch.inference_mode():
        model = load_trusted_model(args.checkpoint, device)
        if len(ids) + args.tokens > model.config.block_size:
            ap.error(f"Prompt ({len(ids)} tokens) + output ({args.tokens}) exceed model context ({model.config.block_size})")
        prompt_ids = torch.tensor([ids], dtype=torch.long, device=device)
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = generate(model, prompt_ids, args.tokens, args.mode)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if args.verify:
            from ..phase4.decoder import generate_reference
            expected = generate_reference(model, prompt_ids, args.tokens)
            if not torch.equal(result, expected):
                raise AssertionError("Correctness FAIL: generated tokens differ from reference")
            print("Correctness: PASS (exact greedy token match)")
        print("\n=== EfficientGPT V1 TinyStories completion ===")
        print(tokenizer.decode(result[0].tolist(), skip_special_tokens=False))
        print("\nDevice:", device, "| Mode:", args.mode, "| Total one-shot run:", round(elapsed_ms, 2), "ms")
        print("One-shot graph runs include graph construction and are not a steady-state benchmark.")


if __name__ == "__main__":
    main()
