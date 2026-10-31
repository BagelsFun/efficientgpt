"""torch.compile (Inductor) baselines for the same recurrent decode step.

Compiles phase4 StaticKVDecoder.step (fixed full-context cache, masked SDPA,
GPU-resident position) with static shapes, so it does exactly the work of the
full-graph mode. Prefill stays eager, as in every other mode. Modes:
  'default'          Inductor/Triton kernels, one Python-dispatched call per step
  'reduce-overhead'  Inductor plus its own CUDA Graphs (cudagraph trees)
Compilation and warmup happen in the constructor and are reported separately.
"""

import time

import torch
import torch.nn.functional as F

from ..phase4.compact_cache import CompactStaticKVDecoder
from ..phase4.decoder import GraphDecoder, StaticKVDecoder, _check_request

# One static-shape specialization per (batch, capacity); never fall back to
# eager silently once the default limit of 8 is reached.
torch._dynamo.config.recompile_limit = 64
if hasattr(torch._dynamo.config, 'fail_on_recompile_limit_hit'):
    torch._dynamo.config.fail_on_recompile_limit_hit = True


def decode_logits(self, attention=None):
    """Transformer body of one recurrent step: writes K/V at `position`, returns logits [B, 1, V].

    Uses index_copy_ for the cache writes. Leaves token, position and history
    untouched. `self` is a StaticKVDecoder (or compact subclass). `attention`
    (q, keys, values, position) -> y replaces the masked SDPA when given.
    """
    tr = self.model.transformer
    x = tr.drop(tr.wte(self.token) + tr.wpe(self.position))
    allowed = self.locations <= self.position.view(1, 1, 1, 1)
    for i, block in enumerate(tr.h):
        a = block.attn
        h = block.ln_1(x)
        b, t, c = h.shape
        q, k, v = a.c_attn(h).split(c, dim=2)
        q = q.view(b, t, self.heads, self.depth).transpose(1, 2)
        k = k.view(b, t, self.heads, self.depth).transpose(1, 2)
        v = v.view(b, t, self.heads, self.depth).transpose(1, 2)
        self.keys[i].index_copy_(2, self.position, k)
        self.values[i].index_copy_(2, self.position, v)
        if attention is None:
            y = F.scaled_dot_product_attention(
                q, self.keys[i], self.values[i], attn_mask=allowed,
                dropout_p=0.0, is_causal=False,
            )
        else:
            y = attention(q, self.keys[i], self.values[i], self.position)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        x = x + a.resid_dropout(a.c_proj(y))
        x = x + block.mlp(block.ln_2(x))
    return self.model.lm_head(tr.ln_f(x)[:, -1:, :])


def step_index_copy(self):
    """StaticKVDecoder.step with index_copy_ in place of scatter_ for cache writes.

    Bit-identical in eager. Under Inductor, scatter_ into the cache is
    functionalized into a full-cache scatter plus copy-back every step
    (0.62 of 1.79 ms GPU time at B=16); index_copy_ is reinplaced into a
    one-row write. `self` is a StaticKVDecoder (or compact subclass).
    """
    logits = decode_logits(self)
    self.history.index_copy_(1, self.position, self.token)
    self.token.copy_(logits.argmax(-1))
    self.position.add_(1)


class CompiledStaticDecoder:
    def __init__(self, model, prompt, mode='default', warmup=3):
        if prompt.device.type != 'cuda':
            raise ValueError('Compiled baseline is measured on CUDA only')
        self.engine = StaticKVDecoder(model, prompt.size(0))
        # Compile the undecorated step; callers already run under inference_mode.
        self.step = torch.compile(StaticKVDecoder.step.__wrapped__, mode=mode,
                                  fullgraph=True, dynamic=False)
        self.mode = mode
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(warmup):
                self.engine.reset(prompt)
                for _ in range(3):
                    self.step(self.engine)
        torch.cuda.synchronize()
        self.compile_ms = 1000 * (time.perf_counter() - t0)

    @torch.inference_mode()
    def generate(self, prompt, count):
        _check_request(self.engine.model, prompt, count)
        self.engine.reset(prompt)
        for _ in range(count - 1):
            self.step(self.engine)
        return self.engine.finish(count)


class CompiledGraphDecoder(GraphDecoder):
    """Inductor-compiled step_index_copy replayed from a manually captured CUDA Graph.

    Same fixed-capacity cache as CompactGraphDecoder (capacity == block_size
    gives the full-graph shapes). Compilation happens during the side-stream
    warmup inside GraphDecoder._capture, so the graph records only the
    generated kernels. Pass as DecoderPool(..., factory=...).
    """

    def __init__(self, model, prompt, capacity):
        if prompt.device.type != 'cuda':
            raise ValueError('A CUDA GPU is required for CUDA Graph capture')
        _check_request(model, prompt, capacity - prompt.size(1))
        self.capacity = capacity
        self.engine = CompactStaticKVDecoder(model, prompt.size(0), capacity)
        compiled = torch.compile(step_index_copy, fullgraph=True, dynamic=False)
        self.engine.step = lambda: compiled(self.engine)
        self.graph = None
        self.capture_ms = None
        self._capture(prompt)

    @torch.inference_mode()
    def generate(self, prompt, count):
        if prompt.size(1) + count > self.capacity:
            raise ValueError('Request exceeds captured cache capacity')
        return super().generate(prompt, count)
