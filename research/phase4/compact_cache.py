"""Conservative experiment: capacity-limited KV cache for the Phase 4 decoder.

This reuses the existing, tested StaticKVDecoder.step and GraphDecoder._capture
functions without modifying the original decoder.py or model.py.

Capacity = prompt tokens + maximum generated tokens (rather than entire
model block_size). This is fixed for a graph's lifetime. This may save
cache/mask memory but is NOT a new inference algorithm.
"""

import torch

from .decoder import GraphDecoder, StaticKVDecoder, _check_request


class CompactStaticKVDecoder(StaticKVDecoder):
    """Fixed-capacity cache using the same recurrent operation as StaticKVDecoder."""

    def __init__(self, model, batch, capacity):
        if model.training:
            raise ValueError("Call model.eval() before constructing decoder")
        if batch < 1 or not 2 <= capacity <= model.config.block_size:
            raise ValueError("Invalid batch or cache capacity")
        if model.config.n_embd % model.config.n_head:
            raise ValueError("Head dimension must divide embedding size")

        self.model = model
        self.device = next(model.parameters()).device
        self.batch = batch
        c = model.config
        self.limit = c.block_size  # model's absolute position limit
        self.capacity = capacity   # smaller amount of cache storage
        self.heads = c.n_head
        self.depth = c.n_embd // c.n_head
        self.dtype = next(model.parameters()).dtype
        shape = (batch, self.heads, capacity, self.depth)
        self.keys = [torch.zeros(shape, dtype=self.dtype, device=self.device) for _ in range(c.n_layer)]
        self.values = [torch.zeros(shape, dtype=self.dtype, device=self.device) for _ in range(c.n_layer)]
        self.position = torch.zeros(1, device=self.device, dtype=torch.long)
        self.token = torch.zeros((batch, 1), device=self.device, dtype=torch.long)
        self.history = torch.zeros((batch, capacity), device=self.device, dtype=torch.long)
        self.locations = torch.arange(capacity, device=self.device).view(1, 1, 1, capacity)
        self.prompt_len = None

    @torch.inference_mode()
    def reset(self, prompt):
        if prompt.ndim != 2 or prompt.size(1) >= self.capacity:
            raise ValueError("Prompt must leave room in cache for a recurrent step")
        return super().reset(prompt)

    @torch.inference_mode()
    def generate_eager(self, prompt, count):
        _check_request(self.model, prompt, count)
        if prompt.size(1) + count > self.capacity:
            raise ValueError("Request exceeds compact cache capacity")
        return super().generate_eager(prompt, count)


class CompactGraphDecoder(GraphDecoder):
    """A CUDA Graph decoder specialised for a maximum total request length.

    Instantiate one graph per batch/shape class. If a request would exceed the
    chosen capacity, reject it rather than silently overflowing graph buffers.
    """

    def __init__(self, model, prompt, max_generated_tokens):
        if prompt.device.type != "cuda":
            raise ValueError("A CUDA GPU is required for CUDA Graph capture")
        _check_request(model, prompt, max_generated_tokens)
        self.capacity = prompt.size(1) + max_generated_tokens
        self.engine = CompactStaticKVDecoder(model, prompt.size(0), self.capacity)
        self.graph = None
        self.capture_ms = None
        self._capture(prompt)

    @torch.inference_mode()
    def generate(self, prompt, count):
        if prompt.size(1) + count > self.capacity:
            raise ValueError("Request exceeds captured compact cache capacity")
        return super().generate(prompt, count)
