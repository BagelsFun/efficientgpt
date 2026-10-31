"""Ragged batches for Phase 5 (opt-in, greedy): per-row prompt lengths, per-row positions.

Input: right-padded prompts [B, Pmax] (pad values are ignored) plus lengths
[B] in [1, Pmax]. Output: the generated tokens [B, count] only. Cache slot ==
absolute position per row: row b writes generated token t at slot
lengths[b] + t, overwriting its pad slots, and attends to slots <= its own
position, so stale or pad K/V is never read.

Contract (docs/PHASE5_CONTRACT.md, "Ragged batches"): RaggedGraphDecoder ==
RaggedEngine.generate_eager exactly (same kernels); equal lengths reproduce
the existing uniform decoders exactly; every row matches batch-1
generate_reference under the near-tie rule (GEMMs at different batch sizes
are not bit-identical). Capacity = Pmax + count (DecoderPool buckets on Pmax).
"""

import time

import torch
import torch.nn.functional as F

from ..phase4.decoder import _check_request, generate_reference


def check_ragged(model, prompts, lengths, count):
    _check_request(model, prompts, count)
    if lengths.shape != (prompts.size(0),) or lengths.dtype != torch.long or lengths.device != prompts.device:
        raise ValueError('lengths must be int64 [batch] on the prompt device')
    if int(lengths.min()) < 1 or int(lengths.max()) > prompts.size(1):
        raise ValueError('lengths must be in [1, prompts.size(1)]')


@torch.inference_mode()
def ragged_prefill(model, prompts, lengths):
    """Batched causal prompt forward over right-padded prompts.

    With right padding, causal attention alone keeps every real token from
    seeing pads (they all come later), so no length mask is needed; pad
    positions produce K/V that decode masks out and overwrites. Returns logits
    at each row's last real token [B, 1, V] and per-layer (k, v) [B, H, Pmax, D].
    Same kernels as reference_step, so equal lengths reproduce it exactly.
    """
    tr = model.transformer
    b, p = prompts.shape
    x = tr.drop(tr.wte(prompts) + tr.wpe(torch.arange(p, device=prompts.device)))
    cache = []
    for block in tr.h:
        attn = block.attn
        hidden = block.ln_1(x)
        c = hidden.size(-1)
        h = attn.n_head
        q, k, v = attn.c_attn(hidden).split(c, dim=2)
        q, k, v = (t.view(b, p, h, c // h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        x = x + attn.resid_dropout(attn.c_proj(y.transpose(1, 2).contiguous().view(b, p, c)))
        x = x + block.mlp(block.ln_2(x))
        cache.append((k, v))
    last = x.gather(1, (lengths - 1).view(b, 1, 1).expand(b, 1, x.size(-1)))
    return model.lm_head(tr.ln_f(last)), cache


@torch.inference_mode()
def generate_reference_ragged(model, prompts, lengths, count):
    """Gold semantics: each row decoded alone (batch 1) by generate_reference."""
    check_ragged(model, prompts, lengths, count)
    rows = []
    for b, n in enumerate(lengths.tolist()):
        rows.append(generate_reference(model, prompts[b:b + 1, :n], count)[:, n:])
    return torch.cat(rows, 0)


class RaggedEngine:
    """Fixed-capacity cache with per-row positions; step() is graph-capturable.

    `attention(q, keys, values, position[B])` replaces the masked SDPA when given
    (e.g. decode_attention.decode_attention: near-tie contract instead of exact).
    """

    def __init__(self, model, batch, capacity, attention=None):
        if model.training:
            raise ValueError('Call model.eval() before constructing decoder')
        c = model.config
        if batch < 1 or not 2 <= capacity <= c.block_size:
            raise ValueError('Invalid batch or cache capacity')
        self.model, self.batch, self.capacity = model, batch, capacity
        self.attention = attention
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.heads, self.depth = c.n_head, c.n_embd // c.n_head
        shape = (batch, self.heads, capacity, self.depth)
        self.keys = [torch.zeros(shape, dtype=self.dtype, device=self.device) for _ in range(c.n_layer)]
        self.values = [torch.zeros(shape, dtype=self.dtype, device=self.device) for _ in range(c.n_layer)]
        self.position = torch.zeros(batch, dtype=torch.long, device=self.device)
        self.lengths = torch.ones(batch, dtype=torch.long, device=self.device)
        self.token = torch.zeros((batch, 1), dtype=torch.long, device=self.device)
        self.history = torch.zeros((batch, capacity), dtype=torch.long, device=self.device)
        self.locations = torch.arange(capacity, device=self.device).view(1, 1, 1, capacity)

    @torch.inference_mode()
    def reset(self, prompts, lengths):
        if prompts.shape[0] != self.batch or prompts.size(1) >= self.capacity:
            raise ValueError('Batch mismatch or no room for a recurrent step')
        check_ragged(self.model, prompts, lengths, 1)
        logits, prefill = ragged_prefill(self.model, prompts, lengths)
        p = prompts.size(1)
        for i, (k, v) in enumerate(prefill):
            self.keys[i][:, :, :p, :].copy_(k)
            self.values[i][:, :, :p, :].copy_(v)
        self.token.copy_(logits.argmax(-1))
        self.position.copy_(lengths)
        self.lengths.copy_(lengths)
        self.history[:, :p].copy_(prompts)
        return logits

    @torch.inference_mode()
    def step(self):
        """GPU-only recurrent step with per-row positions (no host syncs)."""
        tr = self.model.transformer
        b = self.batch
        x = tr.drop(tr.wte(self.token) + tr.wpe(self.position).unsqueeze(1))
        idx = self.position.view(b, 1, 1, 1).expand(b, self.heads, 1, self.depth)
        allowed = self.locations <= self.position.view(b, 1, 1, 1)
        for i, block in enumerate(tr.h):
            a = block.attn
            hidden = block.ln_1(x)
            c = hidden.size(-1)
            q, k, v = a.c_attn(hidden).split(c, dim=2)
            q, k, v = (t.view(b, 1, self.heads, self.depth).transpose(1, 2) for t in (q, k, v))
            self.keys[i].scatter_(2, idx, k)
            self.values[i].scatter_(2, idx, v)
            if self.attention is None:
                y = F.scaled_dot_product_attention(q, self.keys[i], self.values[i], attn_mask=allowed,
                                                   dropout_p=0.0, is_causal=False)
            else:
                y = self.attention(q, self.keys[i], self.values[i], self.position)
            x = x + a.resid_dropout(a.c_proj(y.transpose(1, 2).contiguous().view(b, 1, c)))
            x = x + block.mlp(block.ln_2(x))
        logits = self.model.lm_head(tr.ln_f(x))
        self.history.scatter_(1, self.position.view(b, 1), self.token)
        self.token.copy_(logits.argmax(-1))
        self.position.add_(1)

    @torch.inference_mode()
    def finish(self, count):
        """Generated tokens [B, count]: history slots lengths[b] .. lengths[b] + count - 1."""
        if int((self.lengths + count).max()) > self.capacity:
            raise ValueError('Exceeded cache capacity')
        self.history.scatter_(1, self.position.view(self.batch, 1), self.token)
        idx = self.lengths.view(-1, 1) + torch.arange(count, device=self.device).view(1, -1)
        return self.history.gather(1, idx)

    @torch.inference_mode()
    def generate_eager(self, prompts, count, lengths):
        check_ragged(self.model, prompts, lengths, count)
        if prompts.size(1) + count > self.capacity:
            raise ValueError('Request exceeds cache capacity')
        self.reset(prompts, lengths)
        for _ in range(count - 1):
            self.step()
        return self.finish(count)


class RaggedGraphDecoder:
    """CUDA Graph of RaggedEngine.step; pool with DecoderPool(..., factory=RaggedGraphDecoder).

    The graph is specialised by (batch, capacity), not by lengths, so one
    graph serves any length mix whose Pmax + count fits the capacity.
    """

    def __init__(self, model, prompt, capacity, attention=None):
        if prompt.device.type != 'cuda':
            raise ValueError('A CUDA GPU is required for CUDA Graph capture')
        _check_request(model, prompt, capacity - prompt.size(1))
        self.capacity = capacity
        self.engine = RaggedEngine(model, prompt.size(0), capacity, attention)
        full = torch.full((prompt.size(0),), prompt.size(1), dtype=torch.long, device=prompt.device)
        stream = torch.cuda.Stream(device=prompt.device)
        stream.wait_stream(torch.cuda.current_stream(prompt.device))
        with torch.inference_mode(), torch.cuda.stream(stream):
            for _ in range(3):  # warm up eager kernels on a side stream before capture
                self.engine.reset(prompt, full)
                self.engine.step()
        torch.cuda.current_stream(prompt.device).wait_stream(stream)
        with torch.inference_mode():
            self.engine.reset(prompt, full)
            torch.cuda.synchronize(prompt.device)
            t0 = time.perf_counter()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.engine.step()
            torch.cuda.synchronize(prompt.device)
        self.capture_ms = 1000 * (time.perf_counter() - t0)

    @torch.inference_mode()
    def generate(self, prompts, count, lengths):
        check_ragged(self.engine.model, prompts, lengths, count)
        if prompts.size(1) + count > self.capacity:
            raise ValueError('Request exceeds captured cache capacity')
        self.engine.reset(prompts, lengths)
        for _ in range(count - 1):
            self.graph.replay()
        return self.engine.finish(count)
