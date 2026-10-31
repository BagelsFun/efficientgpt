"""Experimental fixed-size CUDA graph pool for known maximum response lengths.

This is an engineering experiment, NOT a new inference method. Graphs are
specialised by (batch size, total KV capacity). This does NOT implement
unbounded streaming, EOS detection, batching/scheduling or cache eviction.
"""

from ._imports import graph_classes


def choose_capacity(prompt_tokens, max_generated_tokens, capacities, context_limit):
    """Select smallest allowed bucket that fits prompt + requested output budget."""
    if min(prompt_tokens, max_generated_tokens) <= 0:
        raise ValueError("Prompt and generation budget must be positive")
    total = prompt_tokens + max_generated_tokens
    if total > context_limit:
        raise ValueError("Request exceeds model context window")
    choices = sorted({int(c) for c in capacities if 2 <= int(c) <= context_limit})
    for cap in choices:
        if total <= cap:
            return cap
    raise ValueError("No capacity bucket large enough for request")


class DecoderPool:
    """Same-model graph reuse for multiple prompt lengths at fixed batch/capacity.

    Graphs and buffers remain resident until the pool is discarded. That can
    increase GPU memory usage; this is intentional for measuring the tradeoff.
    `factory` may be provided for non-GPU policy unit tests.
    """

    def __init__(self, model, mode, capacities=(80, 128, 192, 256), factory=None):
        if mode not in ("full", "adaptive"):
            raise ValueError("mode must be 'full' or 'adaptive'")
        self.model = model
        self.mode = mode
        self.capacities = capacities
        self.factory = factory
        self.graphs = {}
        self.total_capture_ms = 0.0
        self.total_initialization_ms = 0.0

    def get(self, prompt, generated):
        import time
        if prompt.ndim != 2:
            raise ValueError("Prompt should be shape [batch, tokens]")
        batch, p = prompt.shape
        if self.mode == "full":
            cap = choose_capacity(p, generated, (self.model.config.block_size,),
                                  self.model.config.block_size)
        else:
            cap = choose_capacity(p, generated, self.capacities,
                                  self.model.config.block_size)
        key = (batch, cap)
        if key not in self.graphs:
            if self.factory is None:
                GraphDecoder, CompactGraphDecoder = graph_classes()
                if self.mode == "full":
                    create = lambda: GraphDecoder(self.model, prompt)
                else:
                    create = lambda: CompactGraphDecoder(
                        self.model, prompt, max_generated_tokens=cap-p)
            else:
                create = lambda: self.factory(self.model, prompt, cap)
            t0 = time.perf_counter()
            decoder = create()
            self.total_initialization_ms += (time.perf_counter()-t0)*1000
            self.total_capture_ms += float(getattr(decoder, "capture_ms", 0) or 0)
            self.graphs[key] = decoder
        return self.graphs[key], cap

    @property
    def graph_count(self):
        return len(self.graphs)
