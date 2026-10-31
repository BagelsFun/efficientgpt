"""Independent, non-invasive decoder for the supplied nanoGPT GPT class.

No model.py edits. The preallocated KV path uses a fixed-size attention mask;
its CUDA graph captures one recurrent greedy decoding step with all state on GPU.
"""

import torch
import torch.nn.functional as F


@torch.inference_mode()
def reference_step(model, ids, cache=None):
    """Reproduce benchmark_kv_cache.py cached_step, but without global state."""
    batch, steps = ids.shape
    offset = 0 if cache is None else cache[0][0].size(-2)
    if offset + steps > model.config.block_size:
        raise ValueError('Prompt + output exceeds model context window')
    tr = model.transformer
    pos = torch.arange(offset, offset + steps, device=ids.device)
    x = tr.drop(tr.wte(ids) + tr.wpe(pos))
    updated = []
    for i, block in enumerate(tr.h):
        attn = block.attn
        hidden = block.ln_1(x)
        b, t, c = hidden.shape
        h = attn.n_head
        q, k, v = attn.c_attn(hidden).split(c, dim=2)
        q = q.view(b, t, h, c // h).transpose(1, 2)
        k = k.view(b, t, h, c // h).transpose(1, 2)
        v = v.view(b, t, h, c // h).transpose(1, 2)
        if cache is not None:
            k = torch.cat((cache[i][0], k), dim=-2)
            v = torch.cat((cache[i][1], v), dim=-2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=(cache is None)
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        x = x + attn.resid_dropout(attn.c_proj(y))
        x = x + block.mlp(block.ln_2(x))
        updated.append((k, v))
    logits = model.lm_head(tr.ln_f(x)[:, -1:, :])
    return logits, updated


@torch.inference_mode()
def generate_reference(model, prompt, count):
    """Greedy generation with dynamically concatenated past K/V."""
    _check_request(model, prompt, count)
    logits, cache = reference_step(model, prompt)
    pieces = [prompt]
    for i in range(count):
        token = logits.argmax(-1)
        pieces.append(token)
        if i != count - 1:
            logits, cache = reference_step(model, token, cache)
    return torch.cat(pieces, dim=1)


def _check_request(model, prompt, count):
    if count < 1 or prompt.ndim != 2 or prompt.size(1) < 1:
        raise ValueError('Need nonempty rank-2 prompt and >=1 output token')
    if prompt.size(1) + count > model.config.block_size:
        raise ValueError('Prompt + output exceeds context window; sliding window is not supported')
    if prompt.dtype != torch.long or prompt.device != next(model.parameters()).device:
        raise ValueError('Prompt must use int64 indices on model device')
    if model.training:
        raise ValueError('Model must be in eval() mode')


class StaticKVDecoder:
    """One fixed-batch decoder. K/V and token/position/history live on device.

    The recurrent step is graph-capturable on supported CUDA/PyTorch setups.
    Attention reads a full fixed-sized cache but masks unfilled slots. This is
    correct for inference but may be slower than efficient variable-length SDPA.
    """

    def __init__(self, model, batch):
        if model.training:
            raise ValueError('Call model.eval() before constructing decoder')
        self.model = model
        self.device = next(model.parameters()).device
        self.batch = batch
        c = model.config
        self.limit = c.block_size
        self.heads = c.n_head
        self.depth = c.n_embd // c.n_head
        self.dtype = next(model.parameters()).dtype
        shape = (batch, self.heads, self.limit, self.depth)
        self.keys = [torch.zeros(shape, dtype=self.dtype, device=self.device) for _ in range(c.n_layer)]
        self.values = [torch.zeros(shape, dtype=self.dtype, device=self.device) for _ in range(c.n_layer)]
        self.position = torch.zeros(1, device=self.device, dtype=torch.long)
        self.token = torch.zeros((batch, 1), device=self.device, dtype=torch.long)
        self.history = torch.zeros((batch, self.limit), device=self.device, dtype=torch.long)
        self.locations = torch.arange(self.limit, device=self.device).view(1, 1, 1, self.limit)
        self.prompt_len = None

    @torch.inference_mode()
    def reset(self, prompt):
        if prompt.shape[0] != self.batch:
            raise ValueError('Decoder batch size is fixed')
        if prompt.size(1) >= self.limit:
            raise ValueError('Need room for at least one output token')
        _check_request(self.model, prompt, 1)
        logits, prefill = reference_step(self.model, prompt)
        p = prompt.size(1)
        for i, (key, value) in enumerate(prefill):
            self.keys[i][:, :, :p, :].copy_(key)
            self.values[i][:, :, :p, :].copy_(value)
        self.token.copy_(logits.argmax(-1))
        self.position.fill_(p)
        self.history[:, :p].copy_(prompt)
        self.prompt_len = p
        return logits

    @torch.inference_mode()
    def step(self):
        """GPU-only recurrent step; no Python data-dependent control flow."""
        tr = self.model.transformer
        # The position is a one-element GPU tensor, not a Python index.
        x = tr.drop(tr.wte(self.token) + tr.wpe(self.position))
        idx = self.position.view(1, 1, 1, 1).expand(self.batch, self.heads, 1, self.depth)
        allowed = self.locations <= self.position.view(1, 1, 1, 1)
        for i, block in enumerate(tr.h):
            a = block.attn
            h = block.ln_1(x)
            b, t, c = h.shape
            q, k, v = a.c_attn(h).split(c, dim=2)
            q = q.view(b, t, self.heads, self.depth).transpose(1, 2)
            k = k.view(b, t, self.heads, self.depth).transpose(1, 2)
            v = v.view(b, t, self.heads, self.depth).transpose(1, 2)
            self.keys[i].scatter_(2, idx, k)
            self.values[i].scatter_(2, idx, v)
            y = F.scaled_dot_product_attention(
                q, self.keys[i], self.values[i], attn_mask=allowed,
                dropout_p=0.0, is_causal=False,
            )
            y = y.transpose(1, 2).contiguous().view(b, t, c)
            x = x + a.resid_dropout(a.c_proj(y))
            x = x + block.mlp(block.ln_2(x))
        logits = self.model.lm_head(tr.ln_f(x)[:, -1:, :])
        self.history.scatter_(1, self.position.view(1, 1).expand(self.batch, 1), self.token)
        self.token.copy_(logits.argmax(-1))
        self.position.add_(1)

    @torch.inference_mode()
    def finish(self, count):
        if self.prompt_len is None:
            raise ValueError('Call reset first')
        p = self.prompt_len
        if p + count > self.limit:
            raise ValueError('Exceeded context window')
        self.history[:, p + count - 1:p + count].copy_(self.token)
        return self.history[:, :p + count].clone()

    @torch.inference_mode()
    def generate_eager(self, prompt, count):
        _check_request(self.model, prompt, count)
        self.reset(prompt)
        for _ in range(count - 1):
            self.step()
        return self.finish(count)


class GraphDecoder:
    """Single captured recurring step for a fixed batch + context capacity.

    Graph capture mutates static K/V and token state. Reset overwrites the
    live prefix before *every* request, then graph.replay advances it on GPU.
    """

    def __init__(self, model, prompt):
        if prompt.device.type != 'cuda':
            raise ValueError('CUDA Graphs require a CUDA device')
        self.engine = StaticKVDecoder(model, prompt.size(0))
        self.graph = None
        self.capture_ms = None
        self._capture(prompt)

    @torch.inference_mode()
    def _capture(self, prompt):
        import time
        engine = self.engine
        # Warm up eager ops on a side stream (as required by PyTorch).
        s = torch.cuda.Stream(device=prompt.device)
        s.wait_stream(torch.cuda.current_stream(prompt.device))
        with torch.cuda.stream(s):
            for _ in range(3):
                engine.reset(prompt)
                engine.step()
        torch.cuda.current_stream(prompt.device).wait_stream(s)
        engine.reset(prompt)
        torch.cuda.synchronize(prompt.device)
        t0 = time.perf_counter()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            engine.step()
        torch.cuda.synchronize(prompt.device)
        self.capture_ms = 1000 * (time.perf_counter() - t0)
        engine.reset(prompt)

    @torch.inference_mode()
    def generate(self, prompt, count):
        _check_request(self.engine.model, prompt, count)
        self.engine.reset(prompt)
        for _ in range(count - 1):
            self.graph.replay()
        return self.engine.finish(count)
