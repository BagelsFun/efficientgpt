"""Pinned KernelAgent-Ada tasks drawn from measured EfficientGPT hot paths (task protocol v1).

Each task: input generator (seeded, plus edge cases), eager reference, an fp64
"truth" used to set dtype-aware tolerances, a bytes model for the roofline,
dev shapes (search) and held-out shapes (final evaluation only).
"""

import math

import torch
import torch.nn.functional as F

PROTOCOL = 'kernelagent-ada/tasks-v1'
VOCAB = 4096
HEADS, HEAD_DIM, EMBED = 6, 64, 384


def _decode_attention_inputs(shape, seed, edge=None):
    b, cap = shape
    g = torch.Generator(device='cuda').manual_seed(seed)
    scale = 8.0 if edge == 'large' else 1.0
    q = torch.randn(b, HEADS, 1, HEAD_DIM, generator=g, device='cuda') * scale
    k = torch.randn(b, HEADS, cap, HEAD_DIM, generator=g, device='cuda') * scale
    v = torch.randn(b, HEADS, cap, HEAD_DIM, generator=g, device='cuda')
    if edge == 'len1':
        pos = torch.zeros(b, dtype=torch.long, device='cuda')
    elif edge == 'full':
        pos = torch.full((b,), cap - 1, dtype=torch.long, device='cuda')
    else:
        pos = torch.randint(0, cap, (b,), generator=g, device='cuda')
    return {'q': q, 'k': k, 'v': v, 'position': pos}


def _decode_attention_ref(inp, dtype=None):
    q, k, v, pos = inp['q'], inp['k'], inp['v'], inp['position']
    if dtype is not None:
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    cap = k.size(2)
    allowed = torch.arange(cap, device=q.device).view(1, 1, 1, cap) <= pos.view(-1, 1, 1, 1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)


def _decode_attention_bytes(inp):
    valid = int((inp['position'] + 1).sum())
    return 4 * (2 * valid * HEADS * HEAD_DIM + 2 * inp['q'].numel())


def _decode_attention_best():
    from ..phase5.decode_attention import decode_attention
    return lambda inp: decode_attention(inp['q'], inp['k'], inp['v'], inp['position'])


def _rows(shape, seed, edge, scale=1.0):
    b, n = shape
    g = torch.Generator(device='cuda').manual_seed(seed)
    x = torch.randn(b, n, generator=g, device='cuda') * scale
    if edge == 'zeros':
        x.zero_()
    elif edge == 'large':
        x *= 50.0
    elif edge == 'dupmax':
        x[:, 7] = x.max(1).values + 1.0
        x[:, n - 3] = x[:, 7]  # duplicate maximum: first index must win
    return x


def _ln_inputs(shape, seed, edge=None):
    b, d = shape
    g = torch.Generator(device='cuda').manual_seed(seed + 1)
    return {'x': _rows(shape, seed, edge), 'residual': torch.randn(b, d, generator=g, device='cuda'),
            'weight': 1 + 0.1 * torch.randn(d, generator=g, device='cuda'), 'bias': 0.1 * torch.randn(d, generator=g, device='cuda')}


def _ln_ref(inp, dtype=None):
    x, r, w, b = (inp[k] if dtype is None else inp[k].to(dtype) for k in ('x', 'residual', 'weight', 'bias'))
    s = x + r
    return torch.cat([s, F.layer_norm(s, (s.size(-1),), w, b, 1e-5)], 1)


def _softmax_inputs(shape, seed, edge=None):
    return {'logits': _rows(shape, seed, edge, 3.0), 'temperature': 0.8}


def _softmax_ref(inp, dtype=None):
    x = inp['logits'] if dtype is None else inp['logits'].to(dtype)
    return torch.softmax(x / inp['temperature'], -1)


def _topk_inputs(shape, seed, edge=None):
    return {'logits': _rows(shape, seed, edge, 3.0), 'temperature': 0.8, 'k': 50}


def _topk_ref(inp, dtype=None):
    x = inp['logits'] if dtype is None else inp['logits'].to(dtype)
    x = x / inp['temperature']
    kth = torch.topk(x, inp['k'], -1).values[:, -1:]
    return torch.softmax(x.masked_fill(x < kth, float('-inf')), -1)


def _cdf_inputs(shape, seed, edge=None):
    b, n = shape
    g = torch.Generator(device='cuda').manual_seed(seed)
    probs = torch.softmax(_rows(shape, seed, None, 3.0), -1)
    if edge == 'sparse':
        probs[:, ::3] = 0
        probs = probs / probs.sum(-1, keepdim=True)
    u = torch.rand(b, generator=g, device='cuda')
    if edge == 'extreme_u':
        u[0], u[-1] = 0.0, 1 - 2 ** -24
    return {'probs': probs, 'u': u}


def _cdf_ref(inp, dtype=None):
    probs = inp['probs'] if dtype is None else inp['probs'].to(dtype)
    cdf = probs.cumsum(-1)
    target = (inp['u'].to(probs.dtype) * cdf[:, -1]).unsqueeze(-1)
    tok = torch.searchsorted(cdf, target, right=True)
    ids = torch.arange(probs.size(-1), device=probs.device)
    last = torch.where(probs > 0, ids, torch.zeros_like(ids)).amax(-1, keepdim=True)
    return torch.minimum(tok, last)


def _argmax_inputs(shape, seed, edge=None):
    return {'logits': _rows(shape, seed, edge, 3.0)}


def _argmax_ref(inp, dtype=None):
    x = inp['logits'] if dtype is None else inp['logits'].to(dtype)
    return x.argmax(-1, keepdim=True)


def _int8_inputs(shape, seed, edge=None):
    """Weight-only int8 linear y = x @ (W * s[:, None]).T for Qwen3-1.7B shapes: x bf16 [M, K], W int8 [N, K] (nn.Linear
    layout), per-output-channel fp32 scales s [N]."""
    m, k, n = shape
    g = torch.Generator(device='cuda').manual_seed(seed)
    x = torch.randn(m, k, generator=g, device='cuda')
    if edge == 'large':
        x = x * 30.0
    # generated directly as int8: an int32/int64 intermediate for the 151,936 x 2,048 lm_head needed 2.3 GB and hit the
    # evaluator's memory cap (held-out crash, KA-07)
    w = torch.randint(-127, 128, (n, k), generator=g, device='cuda', dtype=torch.int8)
    if edge == 'extreme_w':
        w = torch.where(w >= 0, torch.tensor(127, dtype=torch.int8, device='cuda'),
                        torch.tensor(-127, dtype=torch.int8, device='cuda'))
    s = (torch.rand(n, generator=g, device='cuda') * 0.02 + 1e-3)
    return {'x': x.to(torch.bfloat16), 'w': w, 'scale': s}


def _int8_ref(inp, dtype=None):
    """Eager reference (dtype None): dequantise to bf16, cuBLAS bf16 matmul, bf16 output, as a naive implementation would.
    fp64 truth: exact dequantisation and fp64 accumulation, chunked over N (a dense fp64 lm_head would need 2.5 GB)."""
    x, w, s = inp['x'], inp['w'], inp['scale']
    if dtype is None:
        return x @ (w.to(torch.bfloat16) * s.to(torch.bfloat16)[:, None]).t()
    x = x.to(dtype)
    return torch.cat([x @ (w[i:i + 16384].to(dtype) * s[i:i + 16384].to(dtype)[:, None]).t()
                      for i in range(0, w.size(0), 16384)], 1)


def _int8_best():
    """The baseline to beat: the dense bf16 weights the HF model uses today (2 bytes per weight), cuBLAS bf16 GEMM/GEMV."""
    cache = {}

    def fn(inp):
        key = inp['w'].data_ptr()
        if key not in cache:
            cache.clear()
            cache[key] = (inp['w'].to(torch.bfloat16) * inp['scale'].to(torch.bfloat16)[:, None]).contiguous()
        return inp['x'] @ cache[key].t()
    return fn


def _rowbytes(inp, key, outs=1):
    return inp[key].numel() * 4 * (1 + outs)


TASKS = {
    'decode_attention': dict(
        inputs=_decode_attention_inputs, reference=_decode_attention_ref, kind='float', floor=1e-6,
        bytes=_decode_attention_bytes, best_existing=_decode_attention_best,
        edges=('len1', 'full', 'large'),
        dev=[(16, 256), (1, 256), (16, 128)], heldout=[(8, 192), (4, 256), (2, 80), (16, 64)]),
    'add_layernorm': dict(
        inputs=_ln_inputs, reference=_ln_ref, kind='float', floor=1e-5,
        bytes=lambda inp: inp['x'].numel() * 4 * 4, best_existing=None, edges=('zeros', 'large'),
        dev=[(1, EMBED), (16, EMBED)], heldout=[(4, EMBED), (64, EMBED), (256, EMBED)]),
    'temperature_softmax': dict(
        inputs=_softmax_inputs, reference=_softmax_ref, kind='float', floor=1e-7,
        bytes=lambda inp: _rowbytes(inp, 'logits'), best_existing=None, edges=('zeros', 'large'),
        dev=[(1, VOCAB), (16, VOCAB)], heldout=[(4, VOCAB), (64, VOCAB), (16, 8192)]),
    'topk_softmax': dict(
        inputs=_topk_inputs, reference=_topk_ref, kind='float', floor=1e-7,
        bytes=lambda inp: _rowbytes(inp, 'logits'), best_existing=None, edges=('large', 'dupmax'),
        dev=[(1, VOCAB), (16, VOCAB)], heldout=[(4, VOCAB), (64, VOCAB), (16, 8192)]),
    'inverse_cdf_sample': dict(
        inputs=_cdf_inputs, reference=_cdf_ref, kind='sample', floor=0,
        bytes=lambda inp: inp['probs'].numel() * 4, best_existing=None, edges=('sparse', 'extreme_u'),
        dev=[(1, VOCAB), (16, VOCAB)], heldout=[(4, VOCAB), (64, VOCAB), (16, 8192)]),
    'argmax': dict(
        inputs=_argmax_inputs, reference=_argmax_ref, kind='index', floor=0,
        bytes=lambda inp: inp['logits'].numel() * 4, best_existing=None, edges=('dupmax', 'large'),
        dev=[(1, VOCAB), (16, VOCAB)], heldout=[(4, VOCAB), (64, VOCAB), (16, 8192)]),
    # Real-model target (R1): every Qwen3-1.7B linear at decode. Shapes (M, K, N). Dev: gate/up, down, k/v projections and a
    # batch-4 gate/up; held-out: q/o projection, the 151,936-wide lm_head, and other batch sizes.
    'int8_gemv': dict(
        inputs=_int8_inputs, reference=_int8_ref, kind='float', floor=1e-3,
        bytes=lambda inp: inp['w'].numel() + 4 * inp['scale'].numel() + 2 * inp['x'].numel()
        + 2 * inp['x'].size(0) * inp['w'].size(0),
        best_existing=_int8_best, edges=('large', 'extreme_w'),
        dev=[(1, 2048, 6144), (1, 6144, 2048), (1, 2048, 1024), (4, 2048, 6144)],
        heldout=[(1, 2048, 2048), (1, 2048, 151936), (4, 6144, 2048), (2, 2048, 1024), (8, 2048, 6144)]),
}


def task(name):
    t = dict(TASKS[name])
    t['name'] = name
    return t


def roofline_fraction(bytes_moved, seconds, bandwidth_gbps=211.0):
    """Fraction of the measured achievable DRAM bandwidth (EXPERIMENT_LOG KERNEL-01). Values > 1 mean the working set
    was served from L2 (timing reuses the same inputs), so it is an upper-bound indicator, not a DRAM utilisation."""
    return bytes_moved / seconds / (bandwidth_gbps * 1e9) if seconds > 0 else math.nan
