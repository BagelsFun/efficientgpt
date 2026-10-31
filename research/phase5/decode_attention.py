"""Triton split-K decode attention (one query per row and head) for Phase 5.

Replaces masked SDPA over the whole fixed cache in the decode step. Each
(row, head) splits its keys across NSPLIT programs that read only slots
<= position[b] (slots past it are never loaded), keep an online softmax in
FP32, and write (max, sum, weighted V) partials; a combine kernel merges them.
Same semantics as SDPA with mask `slot <= position[b]` and scale 1/sqrt(D),
but a different FP32 summation order: NOT bit-exact, so modes built on it use
the near-tie token rule (docs/PHASE5_CONTRACT.md, "Triton decode attention").
Graph-capturable: grid depends only on shapes, lengths are read on device.
"""

import torch
import triton
import triton.language as tl

from ..phase4.compact_cache import CompactStaticKVDecoder
from ..phase4.decoder import GraphDecoder, _check_request
from .compiled_baseline import decode_logits


@triton.jit
def _split_kernel(Q, K, V, POS, M_OUT, L_OUT, ACC_OUT,
                  sq_b, sq_h, sq_d, sk_b, sk_h, sk_n, sk_d, sv_b, sv_h, sv_n, sv_d, s_pos,
                  H, scale,
                  D: tl.constexpr, BLOCK_N: tl.constexpr, SPLIT: tl.constexpr, NSPLIT: tl.constexpr):
    bh = tl.program_id(0)
    s = tl.program_id(1)
    b = bh // H
    h = bh % H
    length = tl.load(POS + b * s_pos) + 1
    start = s * SPLIT
    end = tl.minimum(start + SPLIT, length)
    d = tl.arange(0, D)
    q = tl.load(Q + b * sq_b + h * sq_h + d * sq_d).to(tl.float32)
    m_i = tl.zeros([1], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([D], dtype=tl.float32)
    for n0 in range(start, end, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        valid = n < end
        k = tl.load(K + b * sk_b + h * sk_h + n[:, None] * sk_n + d[None, :] * sk_d, mask=valid[:, None], other=0.0).to(tl.float32)
        # Inductor passes `scale` as fp64; keep the loop-carried softmax state fp32 (no-op in eager).
        scores = tl.where(valid, (tl.sum(k * q[None, :], 1) * scale).to(tl.float32), float('-inf'))
        m_new = tl.maximum(m_i, tl.max(scores, 0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        v = tl.load(V + b * sv_b + h * sv_h + n[:, None] * sv_n + d[None, :] * sv_d, mask=valid[:, None], other=0.0).to(tl.float32)
        l_i = l_i * alpha + tl.sum(p, 0)
        acc = acc * alpha + tl.sum(p[:, None] * v, 0)
        m_i = m_new
    out = bh * NSPLIT + s
    one = tl.arange(0, 1)
    tl.store(M_OUT + out + one, m_i)
    tl.store(L_OUT + out + one, l_i)
    tl.store(ACC_OUT + out * D + d, acc)


@triton.jit
def _combine_kernel(M_IN, L_IN, ACC_IN, OUT, so_b, so_h, so_d, H,
                    D: tl.constexpr, NSPLIT: tl.constexpr):
    bh = tl.program_id(0)
    b = bh // H
    h = bh % H
    s = tl.arange(0, NSPLIT)
    d = tl.arange(0, D)
    m = tl.load(M_IN + bh * NSPLIT + s)
    l = tl.load(L_IN + bh * NSPLIT + s)
    acc = tl.load(ACC_IN + (bh * NSPLIT + s)[:, None] * D + d[None, :])
    w = tl.exp(m - tl.max(m, 0))  # empty splits have m = -inf -> weight 0; split 0 is never empty
    tl.store(OUT + b * so_b + h * so_h + d * so_d, tl.sum(acc * w[:, None], 0) / tl.sum(l * w, 0))


def default_config(batch, heads, capacity):
    """(SPLIT, BLOCK_N, num_warps): more splits when few (row, head) pairs exist."""
    if batch * heads >= 48:
        return 64, 32, 4
    return 16, 16, 2


def decode_attention(q, k, v, position, config=None):
    """q [B, H, 1, D], k/v [B, H, cap, D] (fp32/bf16/fp16, fp32 math); position int64 [1] or [B]: keys at slots <= position[b]."""
    b, h, one, d = q.shape
    cap = k.size(2)
    if one != 1 or k.shape != (b, h, cap, d) or v.shape != k.shape:
        raise ValueError('decode_attention expects q [B,H,1,D] and k/v [B,H,cap,D]')
    if q.dtype not in (torch.float32, torch.bfloat16, torch.float16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError('q, k, v must share one dtype: float32, bfloat16 or float16')
    if position.dtype != torch.long or position.numel() not in (1, b):
        raise ValueError('position must be int64 with 1 or B elements')
    split, block_n, warps = config or default_config(b, h, cap)
    nsplit = triton.next_power_of_2(triton.cdiv(cap, split))
    m = torch.empty((b * h * nsplit,), device=q.device, dtype=torch.float32)
    l = torch.empty_like(m)
    acc = torch.empty((b * h * nsplit, d), device=q.device, dtype=torch.float32)
    out = torch.empty((b, h, 1, d), device=q.device, dtype=q.dtype)  # fp32 math, stored in the input dtype
    s_pos = 0 if position.numel() == 1 else position.stride(0)
    _split_kernel[(b * h, nsplit)](q, k, v, position, m, l, acc,
                                   q.stride(0), q.stride(1), q.stride(3), *k.stride(), *v.stride(), s_pos,
                                   h, d ** -0.5, D=d, BLOCK_N=block_n, SPLIT=split, NSPLIT=nsplit, num_warps=warps)
    _combine_kernel[(b * h,)](m, l, acc, out, out.stride(0), out.stride(1), out.stride(3), h,
                              D=d, NSPLIT=nsplit, num_warps=1 if nsplit <= 8 else 2)
    return out


@triton.jit
def _single_pass_kernel(Q, K, V, POS, OUT,
                        sq_b, sq_h, sq_d, sk_b, sk_h, sk_n, sk_d, sv_b, sv_h, sv_n, sv_d, so_b, so_h, so_d, s_pos,
                        H, scale, D: tl.constexpr, BLOCK_N: tl.constexpr):
    """One program per (row, head) walks slots 0..position[b] with an FP32 online softmax: no partials, no combine.
    Found by KernelAgent-Ada (family single_pass, BLOCK_N 64, 2 warps; reports/KERNELAGENT_ADA_REPORT.md)."""
    bh = tl.program_id(0)
    b = bh // H
    h = bh % H
    length = tl.load(POS + b * s_pos) + 1
    d = tl.arange(0, D)
    q = tl.load(Q + b * sq_b + h * sq_h + d * sq_d).to(tl.float32)
    m_i = tl.zeros([1], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([D], dtype=tl.float32)
    for n0 in range(0, length, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        valid = n < length
        k = tl.load(K + b * sk_b + h * sk_h + n[:, None] * sk_n + d[None, :] * sk_d, mask=valid[:, None], other=0.0).to(tl.float32)
        scores = tl.where(valid, (tl.sum(k * q[None, :], 1) * scale).to(tl.float32), float('-inf'))
        m_new = tl.maximum(m_i, tl.max(scores, 0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        v = tl.load(V + b * sv_b + h * sv_h + n[:, None] * sv_n + d[None, :] * sv_d, mask=valid[:, None], other=0.0).to(tl.float32)
        l_i = l_i * alpha + tl.sum(p, 0)
        acc = acc * alpha + tl.sum(p[:, None] * v, 0)
        m_i = m_new
    tl.store(OUT + b * so_b + h * so_h + d * so_d, acc / l_i)


def decode_attention_single_pass(q, k, v, position, block_n=64, warps=2):
    """Same contract as decode_attention (fp32 math, near-tie rule), single kernel launch."""
    b, h, one, d = q.shape
    cap = k.size(2)
    if one != 1 or k.shape != (b, h, cap, d) or v.shape != k.shape:
        raise ValueError('decode_attention expects q [B,H,1,D] and k/v [B,H,cap,D]')
    if q.dtype not in (torch.float32, torch.bfloat16, torch.float16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError('q, k, v must share one dtype: float32, bfloat16 or float16')
    if position.dtype != torch.long or position.numel() not in (1, b):
        raise ValueError('position must be int64 with 1 or B elements')
    out = torch.empty((b, h, 1, d), device=q.device, dtype=q.dtype)
    s_pos = 0 if position.numel() == 1 else position.stride(0)
    _single_pass_kernel[(b * h,)](q, k, v, position, out, q.stride(0), q.stride(1), q.stride(3), *k.stride(), *v.stride(),
                                  out.stride(0), out.stride(1), out.stride(3), s_pos, h, d ** -0.5,
                                  D=d, BLOCK_N=block_n, num_warps=warps)
    return out


# Shape-only dispatch (positions live on the device inside a CUDA graph): single_pass when there are enough
# (row, head) programs to fill the GPU or the cache is short; split-K otherwise. Thresholds from the dispatch grid
# in research/phase5/dispatch_study.py (runs/20261031_v4/dispatch_grid.json), validated end to end separately.
SINGLE_PASS_MIN_PROGRAMS = 48
SINGLE_PASS_MAX_CAPACITY_FEW_PROGRAMS = 64


def use_single_pass(batch, heads, capacity):
    return batch * heads >= SINGLE_PASS_MIN_PROGRAMS or capacity <= SINGLE_PASS_MAX_CAPACITY_FEW_PROGRAMS


def decode_attention_dispatch(q, k, v, position):
    b, h, _, _ = q.shape
    if use_single_pass(b, h, k.size(2)):
        return decode_attention_single_pass(q, k, v, position)
    return decode_attention(q, k, v, position)


def triton_attention_step(self, attention=None):
    """step_index_copy with Triton decode attention; `self` is a StaticKVDecoder (or compact subclass)."""
    logits = decode_logits(self, attention=attention or decode_attention)
    self.history.index_copy_(1, self.position, self.token)
    self.token.copy_(logits.argmax(-1))
    self.position.add_(1)


class CompiledTritonGraphDecoder(GraphDecoder):
    """Inductor-compiled triton_attention_step (user Triton kernels inside the graph), manually captured.

    Inductor fuses the remaining elementwise ops; the two attention kernels are
    launched as-is. Pool with DecoderPool(..., factory=CompiledTritonGraphDecoder).
    """

    def __init__(self, model, prompt, capacity):
        # The compiled_baseline import at module top already raised Dynamo's recompile limit.
        if prompt.device.type != 'cuda':
            raise ValueError('A CUDA GPU is required for CUDA Graph capture')
        _check_request(model, prompt, capacity - prompt.size(1))
        self.capacity = capacity
        self.engine = CompactStaticKVDecoder(model, prompt.size(0), capacity)
        compiled = torch.compile(triton_attention_step, fullgraph=True, dynamic=False)
        self.engine.step = lambda: compiled(self.engine)
        self.graph = None
        self.capture_ms = None
        self._capture(prompt)

    @torch.inference_mode()
    def generate(self, prompt, count):
        if prompt.size(1) + count > self.capacity:
            raise ValueError('Request exceeds captured cache capacity')
        return super().generate(prompt, count)


class TritonGraphDecoder(GraphDecoder):
    """Manual CUDA Graph of the Triton-attention step; pool with DecoderPool(..., factory=TritonGraphDecoder)."""

    attention = None  # None: split-K decode_attention; subclasses swap the kernel

    def __init__(self, model, prompt, capacity):
        if prompt.device.type != 'cuda':
            raise ValueError('A CUDA GPU is required for CUDA Graph capture')
        _check_request(model, prompt, capacity - prompt.size(1))
        self.capacity = capacity
        self.engine = CompactStaticKVDecoder(model, prompt.size(0), capacity)
        attention = type(self).attention
        self.engine.step = lambda: triton_attention_step(self.engine, attention)
        self.graph = None
        self.capture_ms = None
        self._capture(prompt)  # side-stream warmup JIT-compiles the kernels before capture

    @torch.inference_mode()
    def generate(self, prompt, count):
        if prompt.size(1) + count > self.capacity:
            raise ValueError('Request exceeds captured cache capacity')
        return super().generate(prompt, count)


class DispatchTritonGraphDecoder(TritonGraphDecoder):
    """TritonGraphDecoder with the shape-dispatched attention (single_pass or split-K); near-tie contract."""
    attention = staticmethod(decode_attention_dispatch)
