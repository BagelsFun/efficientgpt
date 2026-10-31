"""Sync-free greedy decoding for Hugging Face decoder models (V3; EfficientGPT Phase 5 ideas applied to Qwen3-1.7B).

Profiling (runs/20261031_v4/v3_profile/) showed the int8 GEMV already streams weights at ≈ 222 GB/s (the DRAM limit), but
the GPU idles ≈ 2 ms per token (18% of a 10.95 ms token). Refined later (streamer_overhead.py): ≈ 0.6 ms is HF `generate()`'s
per-step host loop (the streamer itself ≤ 0.07 ms), the rest kernel boundaries inside the step. Originally attributed to:
HF `generate()` and its streamer synchronise with
the GPU every token (`.tolist()` / stopping checks), so the CPU only prepares step t+1 after step t has finished.

This loop keeps everything on the device: the same compiled static-cache forward (torch.compile, CUDA graphs), argmax on
the GPU, the next token fed back as a device tensor. Tokens still stream one by one: each step enqueues a non-blocking copy
of its token into pinned host memory and records a CUDA event; the host emits every token whose event has completed, never
blocking on the step it just launched. Run-ahead is bounded (`lookahead` steps), so EOS stops generation at most
`lookahead` steps late and the extra tokens are discarded (never emitted). Greedy only.

Contract (near-tie, measured): greedy tokens equal HF generate's on the tiny-model tests (bf16 and int8); on Qwen3-1.7B
bf16 they diverge at near-ties of differently compiled kernels: 10/16 identical 64-token outputs, mean common prefix
45.8 tokens vs a 42.9-token floor between two HF bf16 paths (runs/20261031_v4/v3_final).
"""

import time

import torch


class GraphGreedyDecoder:
    """One manual CUDA graph per decode step (EfficientGPT Phase 5 GraphDecoder, applied to an HF model).

    The step = compiled model forward (Inductor, no Inductor cudagraphs) + argmax + in-place update of the static token and
    position buffers, captured once with torch.cuda.graph. Per token the host does one `replay()` plus a non-blocking copy
    of the token into pinned memory and an event, so Python, Dynamo guards and cudagraph-tree bookkeeping leave the token
    loop. The StaticCache tensors (keys, values, cumulative_length) keep fixed addresses; `reset()` zeroes them in place.
    """

    def __init__(self, model, max_cache_len, lookahead=8):
        from transformers import StaticCache
        self.model, self.max_cache_len, self.lookahead = model, max_cache_len, lookahead
        self.device = next(model.parameters()).device
        self.cache = StaticCache(config=model.config, max_cache_len=max_cache_len)
        self.token = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self.position = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self.host = torch.empty(max_cache_len, dtype=torch.long, pin_memory=True)
        self._compiled = torch.compile(self._step, mode='default', fullgraph=True)
        self.graph = None

    def _step(self):
        logits = self.model(input_ids=self.token, position_ids=self.position, past_key_values=self.cache, use_cache=True,
                            logits_to_keep=1).logits
        self.token.copy_(logits[:, -1].argmax(-1, keepdim=True))
        self.position.add_(1)

    @torch.inference_mode()
    def _capture(self):
        # initialise the cache lazily with a 1-token prefill, warm up (compiles, autotunes) on a side stream, capture
        self.model(input_ids=self.token, past_key_values=self.cache, use_cache=True, logits_to_keep=1)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._compiled()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._compiled()
        self.cache.reset()

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens, eos_ids=(), on_token=None):
        p = input_ids.shape[1]
        if input_ids.shape[0] != 1:  # token / position buffers are [1, 1] (V3 audit)
            raise ValueError('GraphGreedyDecoder supports batch size 1')
        if max_new_tokens < 1:
            raise ValueError('max_new_tokens must be >= 1')
        if p + max_new_tokens > self.max_cache_len:
            raise ValueError('prompt + max_new_tokens exceeds max_cache_len')
        capture_s = 0.0
        if self.graph is None:
            t = time.perf_counter()
            self._capture()
            capture_s = time.perf_counter() - t  # reported separately, not inside TTFT
        self.cache.reset()
        eos = set(eos_ids)
        t0 = time.perf_counter()
        logits = self.model(input_ids=input_ids, past_key_values=self.cache, use_cache=True, logits_to_keep=1).logits
        self.token.copy_(logits[:, -1].argmax(-1, keepdim=True))
        self.position.fill_(p)
        events, out, emitted, first, launched, done = [], [], 0, None, 0, False
        while not done:
            self.host[launched:launched + 1].copy_(self.token.view(-1), non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
            events.append(ev)
            launched += 1
            if launched < max_new_tokens:
                self.graph.replay()  # computes the next token into self.token
            while emitted < launched and (events[emitted].query() or launched - emitted > self.lookahead
                                          or launched >= max_new_tokens):
                events[emitted].synchronize()
                tok = int(self.host[emitted])
                if first is None:
                    first = time.perf_counter()
                out.append(tok)
                if on_token:
                    on_token(tok)
                emitted += 1
                if tok in eos:
                    done = True
                    break
            if launched >= max_new_tokens and emitted >= launched:
                done = True
        torch.cuda.synchronize()
        return out, {'ttft_s': first - t0, 'total_s': time.perf_counter() - t0, 'tokens': len(out), 'capture_s': capture_s}


class FastGreedyDecoder:
    def __init__(self, model, max_cache_len, lookahead=8):
        from transformers import StaticCache
        self.model, self.max_cache_len, self.lookahead = model, max_cache_len, lookahead
        self.cache = StaticCache(config=model.config, max_cache_len=max_cache_len)
        self.device = next(model.parameters()).device
        self.step = torch.compile(self._step, mode='reduce-overhead', fullgraph=True)
        self.host = torch.empty(max_cache_len, dtype=torch.long, pin_memory=True)

    def _step(self, token, position):
        out = self.model(input_ids=token, position_ids=position, past_key_values=self.cache, use_cache=True,
                         logits_to_keep=1)
        return out.logits[:, -1].argmax(-1, keepdim=True)

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens, eos_ids=(), on_token=None):
        """input_ids [1, p] on the model's device. Returns (new token ids, stats dict with ttft_s / total_s)."""
        p = input_ids.shape[1]
        if p + max_new_tokens > self.max_cache_len:
            raise ValueError('prompt + max_new_tokens exceeds max_cache_len')
        self.cache.reset()
        eos = set(eos_ids)
        t0 = time.perf_counter()
        logits = self.model(input_ids=input_ids, past_key_values=self.cache, use_cache=True, logits_to_keep=1).logits
        token = logits[:, -1].argmax(-1, keepdim=True)
        position = torch.full((1, 1), p, dtype=torch.long, device=self.device)
        events, out, emitted, first = [], [], 0, None
        launched = 0
        stop_at = None
        while True:
            # enqueue the copy of the newest token (step `launched`) and an event marking it
            self.host[launched:launched + 1].copy_(token.view(-1), non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
            events.append(ev)
            launched += 1
            # emit every finished token without blocking on the newest step
            while emitted < launched and (events[emitted].query() or launched - emitted > self.lookahead):
                events[emitted].synchronize()
                tok = int(self.host[emitted])
                if first is None:
                    first = time.perf_counter()
                out.append(tok)
                if on_token:
                    on_token(tok)
                emitted += 1
                if tok in eos:
                    stop_at = emitted
                    break
            if stop_at is not None or launched >= max_new_tokens:
                break
            torch.compiler.cudagraph_mark_step_begin()
            # the graph's output buffer is reused by the next replay: clone the 1-element token before feeding it back
            token = self.step(token, position).clone()
            position = position + 1
        if stop_at is None:  # drain the tail
            for i in range(emitted, launched):
                events[i].synchronize()
                tok = int(self.host[i])
                if first is None:
                    first = time.perf_counter()
                out.append(tok)
                if on_token:
                    on_token(tok)
                if tok in eos:
                    break
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        return out, {'ttft_s': first - t0, 'total_s': t1 - t0, 'tokens': len(out)}
