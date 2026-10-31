"""EOS stopping for Phase 5 decoders (fixed steps + mask; opt-in).

Contract (docs/PHASE5_CONTRACT.md, "EOS stopping"): per row the first
generated `eos_id` is kept, later generated slots are `pad_id` (default: the
EOS id itself, 0 for the TinyStories tokenizer), and `lengths[b]` counts the
generated tokens up to and including that EOS (`count` if none). The prompt is
never inspected. Every decoder still runs its full `count` budget.

Masking after generation is exactly equivalent to feeding `pad_id` back after
a row finishes: batch rows never interact, and tokens after a row's EOS only
influence that row's later (masked) outputs. `generate_reference_fed_pad`
implements the feed-pad semantics step by step so tests can check this.
"""

import torch

from ..phase4.decoder import _check_request, reference_step


def mask_after_eos(tokens, prompt_len, eos_id, pad_id=None):
    """Return (masked tokens, lengths) for [B, prompt_len + count] greedy/sampled output."""
    pad_id = eos_id if pad_id is None else pad_id
    generated = tokens[:, prompt_len:]
    is_eos = generated == eos_id
    seen_before = is_eos.long().cumsum(1) - is_eos.long()
    out = tokens.clone()
    out[:, prompt_len:] = generated.masked_fill(seen_before > 0, pad_id)
    first = is_eos.float().argmax(1)  # first EOS index; 0 when none
    lengths = torch.where(is_eos.any(1), first + 1, torch.full_like(first, generated.size(1)))
    return out, lengths


class EOSDecoder:
    """Wrap any Phase 5 decoder (graph, compiled, eager) with EOS stopping."""

    def __init__(self, decoder, eos_id=0, pad_id=None):
        self.decoder = decoder
        self.eos_id = eos_id
        self.pad_id = eos_id if pad_id is None else pad_id

    def generate(self, prompt, count, *args):
        """Extra args (e.g. sampling uniforms) are passed through to the wrapped decoder."""
        tokens = self.decoder.generate(prompt, count, *args)
        return mask_after_eos(tokens, prompt.size(1), self.eos_id, self.pad_id)


@torch.inference_mode()
def generate_reference_fed_pad(model, prompt, count, eos_id, pad_id=None):
    """Slow reference: once a row emits EOS, feed pad_id as its next input token."""
    _check_request(model, prompt, count)
    pad_id = eos_id if pad_id is None else pad_id
    logits, cache = reference_step(model, prompt)
    finished = torch.zeros(prompt.size(0), dtype=torch.bool, device=prompt.device)
    lengths = torch.full((prompt.size(0),), count, dtype=torch.long, device=prompt.device)
    pieces = [prompt]
    for i in range(count):
        token = logits.argmax(-1).squeeze(1)
        token = torch.where(finished, torch.full_like(token, pad_id), token)
        newly = ~finished & (token == eos_id)
        lengths = torch.where(newly, torch.full_like(lengths, i + 1), lengths)
        finished = finished | newly
        pieces.append(token.unsqueeze(1))
        if i != count - 1:
            logits, cache = reference_step(model, token.unsqueeze(1), cache)
    return torch.cat(pieces, dim=1), lengths
