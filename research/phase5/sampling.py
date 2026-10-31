"""Stochastic decoding for Phase 5 (opt-in): temperature -> top-k -> top-p.

Contract (docs/PHASE5_CONTRACT.md, "Sampling"): tokens are drawn by inverse
CDF from uniforms the caller pre-draws (`draw_uniforms`), one per row and
generated token. Given the same uniforms, `SampledGraphDecoder` must match
`generate_reference_sampled` token for token: both use `filtered_probs` and
`sample_from`, and the graph step's logits are bit-identical to the
incremental reference (EXPERIMENT_LOG SAMPLE-01). Same distribution as
`GPT.generate` (temperature, top-k), but NOT the same draws for a given seed.
Greedy (temperature 0) stays on the existing decoders.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from ..phase4.compact_cache import CompactStaticKVDecoder
from ..phase4.decoder import GraphDecoder, _check_request, reference_step
from .compiled_baseline import decode_logits


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None

    def __post_init__(self):
        if not self.temperature > 0:
            raise ValueError('temperature must be > 0; use the greedy decoders for temperature 0')
        if self.top_k is not None and self.top_k < 1:
            raise ValueError('top_k must be >= 1')
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError('top_p must be in (0, 1]')


def filtered_probs(logits, cfg):
    """[B, V] logits -> [B, V] probabilities after temperature, top-k, then top-p.

    Top-k matches GPT.generate: logits below the k-th largest become -inf (ties
    at the k-th value are kept). Top-p keeps the smallest prefix of the
    stable descending sort whose mass reaches top_p; the top token is always kept.
    """
    x = logits.float() / cfg.temperature
    if cfg.top_k is not None and cfg.top_k < x.size(-1):
        kth = torch.topk(x, cfg.top_k, dim=-1).values[:, -1:]
        x = x.masked_fill(x < kth, float('-inf'))
    probs = F.softmax(x, dim=-1)
    if cfg.top_p is not None and cfg.top_p < 1:
        ordered, order = torch.sort(probs, dim=-1, descending=True, stable=True)
        drop_sorted = (ordered.cumsum(-1) - ordered) >= cfg.top_p
        drop = torch.zeros_like(drop_sorted).scatter(-1, order, drop_sorted)
        probs = probs.masked_fill(drop, 0.0)
        probs = probs / probs.sum(-1, keepdim=True)
    return probs


def sample_from(probs, u):
    """Inverse CDF: token j with cdf[j-1] <= u * total < cdf[j]; never a zero-probability token.

    probs [B, V], u [B] in [0, 1) -> tokens [B, 1] (int64).
    """
    cdf = probs.cumsum(-1)
    target = (u * cdf[:, -1]).unsqueeze(-1)
    token = torch.searchsorted(cdf, target, right=True)
    # Guard the u * total == total rounding edge: fall back to the last token with mass.
    ids = torch.arange(probs.size(-1), device=probs.device)
    last = torch.where(probs > 0, ids, torch.zeros_like(ids)).amax(-1, keepdim=True)
    return torch.minimum(token, last)


def draw_uniforms(batch, count, generator, device='cuda'):
    """One U[0, 1) per row and generated token; seed `generator` for reproducibility."""
    return torch.rand((batch, count), generator=generator, device=device, dtype=torch.float32)


def _check_uniforms(prompt, count, uniforms):
    if uniforms.shape[0] != prompt.size(0) or uniforms.shape[1] < count or uniforms.dtype != torch.float32:
        raise ValueError('uniforms must be float32 [batch, >= count]')


@torch.inference_mode()
def generate_reference_sampled(model, prompt, count, cfg, uniforms):
    """Concatenated-KV reference: same transforms and draws as the graph decoder."""
    _check_request(model, prompt, count)
    _check_uniforms(prompt, count, uniforms)
    logits, cache = reference_step(model, prompt)
    pieces = [prompt]
    for i in range(count):
        token = sample_from(filtered_probs(logits[:, -1, :], cfg), uniforms[:, i])
        pieces.append(token)
        if i != count - 1:
            logits, cache = reference_step(model, token, cache)
    return torch.cat(pieces, dim=1)


class SampledEngine(CompactStaticKVDecoder):
    """Fixed-capacity decoder whose recurrent step samples with on-device uniforms."""

    def __init__(self, model, batch, capacity, cfg, attention=None):
        super().__init__(model, batch, capacity)
        self.cfg = cfg
        self.attention = attention  # None: masked SDPA (exact given u); else e.g. Triton decode attention
        self.uniforms = torch.zeros((batch, capacity), device=self.device, dtype=torch.float32)
        self.step_index = torch.zeros(1, dtype=torch.long, device=self.device)

    @torch.inference_mode()
    def reset(self, prompt):
        logits = super().reset(prompt)  # eager prefill, as in the reference
        self.token.copy_(sample_from(filtered_probs(logits[:, -1, :], self.cfg), self.uniforms[:, 0]))
        self.step_index.fill_(1)
        return logits

    @torch.inference_mode()
    def step(self):
        """GPU-only sampled step; graph-capturable (no host syncs)."""
        logits = decode_logits(self, attention=self.attention)
        u = self.uniforms.gather(1, self.step_index.view(1, 1).expand(self.batch, 1)).squeeze(1)
        self.history.index_copy_(1, self.position, self.token)
        self.token.copy_(sample_from(filtered_probs(logits[:, -1, :], self.cfg), u))
        self.position.add_(1)
        self.step_index.add_(1)

    @torch.inference_mode()
    def generate_eager(self, prompt, count, uniforms):
        _check_uniforms(prompt, count, uniforms)
        self.uniforms[:, :count].copy_(uniforms[:, :count])
        return super().generate_eager(prompt, count)


class SampledGraphDecoder(GraphDecoder):
    """CUDA Graph of the sampled step; pool with DecoderPool(..., factory=sampled_factory(cfg))."""

    def __init__(self, model, prompt, capacity, cfg, attention=None):
        if prompt.device.type != 'cuda':
            raise ValueError('A CUDA GPU is required for CUDA Graph capture')
        _check_request(model, prompt, capacity - prompt.size(1))
        self.capacity = capacity
        self.engine = SampledEngine(model, prompt.size(0), capacity, cfg, attention)
        self.graph = None
        self.capture_ms = None
        self._capture(prompt)

    @torch.inference_mode()
    def generate(self, prompt, count, uniforms):
        if prompt.size(1) + count > self.capacity:
            raise ValueError('Request exceeds captured cache capacity')
        _check_uniforms(prompt, count, uniforms)
        self.engine.uniforms[:, :count].copy_(uniforms[:, :count])
        return super().generate(prompt, count)


def sampled_factory(cfg, attention=None):
    """DecoderPool factory: one graph per (batch, capacity) for this sampling config (and attention)."""
    return lambda model, prompt, capacity: SampledGraphDecoder(model, prompt, capacity, cfg, attention)
