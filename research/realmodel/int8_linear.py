"""Weight-only int8 linear layers for Hugging Face decoder models (R1; Qwen3-1.7B).

Batch-1 decode of a 1.7B bf16 model streams ≈ 3.4 GB of weights per token (≈ 94% of the step at the measured 211 GB/s),
so halving weight bytes is the lever. Weights are quantised per output channel, symmetric: s = max|w| / 127,
q = round(w / s) in [-127, 127]. Decode (≤ 16 rows) runs a Triton kernel that converts int8 to bf16 in registers (exact),
multiplies on tensor cores with fp32 accumulation and applies s to the accumulator (KernelAgent-Ada `int8_gemv`, family
dot16 / dot16_splitk). Prefill (> 16 rows) dequantises to bf16 and uses cuBLAS. The op is a torch.library custom op with a
fake implementation, so torch.compile / CUDA graphs (HF static cache) treat it as one opaque node.

APPROXIMATE contract: quantisation changes the model; quality is measured separately (bench_int8.py), never assumed.
"""

import torch
import triton
import triton.language as tl
from torch import nn

MAX_DECODE_ROWS = 16
# Launch configs per (K, N) chosen by the KernelAgent-Ada study (runs/20261031_v4/int8_study/); DEFAULT until then.
CONFIGS = {}
# (family, BLOCK_N, BLOCK_K, SPLITK, warps). V3: chosen END TO END (research/realmodel/sweep_configs.py: 9.51 vs 10.32 ms per
# decode step). The KA-07 warm-L2 microbenchmark winner, split-K 8, costs ~197 extra reduce kernels per token end to end.
DEFAULT = ('dot16', 32, 128, 0, 4)


@triton.jit
def _dot16(X, W, S, O, M, N, K, sxm, swn, som, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = tl.arange(0, 16)
    acc = tl.zeros([16, BLOCK_N], tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(X + rm[:, None] * sxm + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        w = tl.load(W + rn[None, :] * swn + rk[:, None], mask=(rn[None, :] < N) & (rk[:, None] < K), other=0)
        acc += tl.dot(x.to(tl.bfloat16), w.to(tl.bfloat16))
    s = tl.load(S + rn, mask=rn < N, other=0.0)
    tl.store(O + rm[:, None] * som + rn[None, :], (acc * s[None, :]).to(tl.bfloat16),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _part(X, W, P, M, N, K, sxm, swn, KCHUNK, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    sk = tl.program_id(1)
    rm = tl.arange(0, 16)
    acc = tl.zeros([16, BLOCK_N], tl.float32)
    k_end = tl.minimum(sk * KCHUNK + KCHUNK, K)
    for k0 in range(sk * KCHUNK, k_end, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(X + rm[:, None] * sxm + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < k_end), other=0.0)
        w = tl.load(W + rn[None, :] * swn + rk[:, None], mask=(rn[None, :] < N) & (rk[:, None] < k_end), other=0)
        acc += tl.dot(x.to(tl.bfloat16), w.to(tl.bfloat16))
    tl.store(P + sk * 16 * N + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _reduce(P, S, O, M, N, som, SPLITK: tl.constexpr, BLOCK: tl.constexpr):
    rn = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = tl.program_id(1)
    acc = tl.zeros([BLOCK], tl.float32)
    for sk in range(0, SPLITK):
        acc += tl.load(P + sk * 16 * N + m * N + rn, mask=rn < N, other=0.0)
    s = tl.load(S + rn, mask=rn < N, other=0.0)
    tl.store(O + m * som + rn, (acc * s).to(tl.bfloat16), mask=rn < N)


def int8_matmul(x2, w, s):
    """x2 bf16 [M <= 16, K] contiguous rows; w int8 [N, K]; s fp32 [N] -> bf16 [M, N]."""
    m, k = x2.shape
    n = w.shape[0]
    if m > MAX_DECODE_ROWS:  # the kernels pad M to 16; more rows would leave output rows uninitialised (review V1 #7)
        raise ValueError(f'int8_matmul handles at most {MAX_DECODE_ROWS} rows, got {m}')
    family, bn, bk, splitk, warps = CONFIGS.get((k, n), DEFAULT)
    out = torch.empty((m, n), device=x2.device, dtype=torch.bfloat16)
    if family == 'dot16':
        _dot16[(triton.cdiv(n, bn),)](x2, w, s, out, m, n, k, x2.stride(0), w.stride(0), out.stride(0),
                                      BLOCK_N=bn, BLOCK_K=bk, num_warps=warps)
        return out
    kchunk = triton.cdiv(triton.cdiv(k, splitk), bk) * bk
    splits = triton.cdiv(k, kchunk)
    part = torch.empty((splits, 16, n), device=x2.device, dtype=torch.float32)
    _part[(triton.cdiv(n, bn), splits)](x2, w, part, m, n, k, x2.stride(0), w.stride(0), kchunk,
                                        BLOCK_N=bn, BLOCK_K=bk, num_warps=warps)
    _reduce[(triton.cdiv(n, 256), m)](part, s, out, m, n, out.stride(0), SPLITK=splits, BLOCK=256, num_warps=4)
    return out


@torch.library.custom_op('efficientgpt::int8_linear', mutates_args=())
def int8_linear(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    if x2.dtype != torch.bfloat16:
        raise ValueError('int8_linear expects bf16 activations')
    if x2.shape[0] <= MAX_DECODE_ROWS:
        y = int8_matmul(x2.contiguous(), w, s)
    else:  # prefill: cuBLAS GEMM on the exact bf16 image of q, scale applied in fp32 like the decode kernel (review V1 #8)
        y = ((x2 @ w.to(torch.bfloat16).t()).float() * s).to(torch.bfloat16)
    return y.reshape(*x.shape[:-1], w.shape[0])


@int8_linear.register_fake
def _(x, w, s):
    return x.new_empty(*x.shape[:-1], w.shape[0])


def quantize_weight(weight):
    """Per-output-channel symmetric int8: returns (q int8 [N, K], s fp32 [N]) with weight ≈ q * s[:, None]."""
    # Row chunks: the whole-matrix float copies of the 151,936 x 2,048 lm_head pushed the load peak to 6.8 GB on an 8 GB
    # GPU (R1 benchmark). Elementwise per row, so chunking gives bit-identical q and s.
    w = weight.detach()
    q = torch.empty(w.shape, dtype=torch.int8, device=w.device)
    s = torch.empty(w.shape[0], dtype=torch.float32, device=w.device)
    for i in range(0, w.shape[0], 8192):
        blk = w[i:i + 8192].float()
        s[i:i + 8192] = (blk.abs().amax(1) / 127.0).clamp_min(1e-12)
        q[i:i + 8192] = torch.round(blk / s[i:i + 8192, None]).clamp_(-127, 127).to(torch.int8)
    return q, s


class Int8Linear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        q, s = quantize_weight(linear.weight)
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.register_buffer('qweight', q)
        self.register_buffer('scale', s)
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

    def forward(self, x):
        y = int8_linear(x, self.qweight, self.scale)
        return y if self.bias is None else y + self.bias

    def extra_repr(self):
        return f'in={self.in_features}, out={self.out_features}, int8 per-channel'


@torch.no_grad()
def quantize_model(model, include_lm_head=True):
    """Replace every nn.Linear in the decoder layers (and the lm_head) by Int8Linear, in place. Returns the count.
    Embeddings and norms stay bf16. A tied lm_head becomes an untied int8 copy (the embedding keeps its bf16 weight)."""
    # Walk by name and drop each bf16 Linear as soon as it is replaced: holding a list of modules kept every bf16 weight
    # alive until the end (load peak 5.1 GB instead of ≈ bf16 model + one layer).
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and (include_lm_head or n != 'lm_head')]
    count = 0
    for full in names:
        parent_name, _, child_name = full.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        child = getattr(parent, child_name)
        setattr(parent, child_name, Int8Linear(child))
        del child
        count += 1
        if count % 16 == 0:
            torch.cuda.empty_cache()
    if hasattr(model, 'config') and include_lm_head:
        model.config.tie_word_embeddings = False
    torch.cuda.empty_cache()
    return count
