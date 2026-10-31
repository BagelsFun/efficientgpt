"""Candidate generator: template families that render Triton source for each task.

A candidate = (task, family, params). `render` returns standalone source with
@triton.jit kernels and `run(inp) -> output`; compiled-kernel handles are kept
in LAUNCHES so the evaluator can read registers/spills/shared memory. This is a
template-based generator (no LLM): families are algorithmic alternatives, and
params are launch/tiling choices.
"""

import itertools
import string

HEADER = '''import triton
import triton.language as tl
import torch

LAUNCHES = []


def _keep(handle):
    if len(LAUNCHES) < 4:
        LAUNCHES.append(handle)
'''

FAMILIES = {}


def family(task, name, space, default, body):
    FAMILIES.setdefault(task, {})[name] = {'space': space, 'default': default, 'body': body}


def render(task, fam, params):
    spec = FAMILIES[task][fam]
    return HEADER + string.Template(spec['body']).substitute({k: str(v) for k, v in params.items()})


def valid(task, fam, params):
    p = params
    if task == 'decode_attention' and fam == 'split_k':
        return p['BLOCK_N'] <= p['SPLIT']
    return True


def space_points(task, fam):
    spec = FAMILIES[task][fam]
    keys = sorted(spec['space'])
    for values in itertools.product(*(spec['space'][k] for k in keys)):
        p = dict(zip(keys, values))
        if valid(task, fam, p):
            yield p


# ---------------------------------------------------------------- argmax
family('argmax', 'row_whole', {'WARPS': [1, 2, 4, 8, 16]}, {'WARPS': 4}, '''
@triton.jit
def _k(X, O, N, sx, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    x = tl.load(X + r * sx + c, mask=c < N, other=float('-inf'))
    tl.store(O + r, tl.argmax(x, 0).to(tl.int64))


def run(inp):
    x = inp['logits']
    b, n = x.shape
    out = torch.empty((b, 1), device=x.device, dtype=torch.int64)
    _keep(_k[(b,)](x, out, n, x.stride(0), BLOCK=triton.next_power_of_2(n), num_warps=$WARPS))
    return out
''')

family('argmax', 'row_chunked', {'BLOCK': [256, 512, 1024, 2048], 'WARPS': [1, 2, 4, 8]}, {'BLOCK': 1024, 'WARPS': 4}, '''
@triton.jit
def _k(X, O, N, sx, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    best = tl.full([1], float('-inf'), tl.float32)
    best_i = tl.zeros([1], tl.int64)
    for s in range(0, N, BLOCK):
        c = s + tl.arange(0, BLOCK)
        x = tl.load(X + r * sx + c, mask=c < N, other=float('-inf'))
        m = tl.max(x, 0)
        i = (tl.argmax(x, 0) + s).to(tl.int64)
        take = m > best  # strict: keeps the earliest index on ties
        best_i = tl.where(take, i, best_i)
        best = tl.where(take, m, best)
    tl.store(O + r + tl.arange(0, 1), best_i)


def run(inp):
    x = inp['logits']
    b, n = x.shape
    out = torch.empty((b, 1), device=x.device, dtype=torch.int64)
    _keep(_k[(b,)](x, out, n, x.stride(0), BLOCK=$BLOCK, num_warps=$WARPS))
    return out
''')

family('argmax', 'split_two_stage', {'SPLITS': [2, 4, 8, 16], 'WARPS': [1, 2, 4]}, {'SPLITS': 8, 'WARPS': 2}, '''
@triton.jit
def _part(X, PM, PI, N, sx, SEG, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    s = tl.program_id(1)
    c = s * SEG + tl.arange(0, BLOCK)
    ok = (c < N) & (tl.arange(0, BLOCK) < SEG)
    x = tl.load(X + r * sx + c, mask=ok, other=float('-inf'))
    tl.store(PM + r * SPLITS + s, tl.max(x, 0))
    tl.store(PI + r * SPLITS + s, (tl.argmax(x, 0) + s * SEG).to(tl.int64))


@triton.jit
def _comb(PM, PI, O, SPLITS: tl.constexpr):
    r = tl.program_id(0)
    j = tl.arange(0, SPLITS)
    m = tl.load(PM + r * SPLITS + j)
    i = tl.load(PI + r * SPLITS + j)
    best = tl.max(m, 0)
    cand = tl.where(m == best, i, 2 ** 62)
    tl.store(O + r, tl.min(cand, 0))


def run(inp):
    x = inp['logits']
    b, n = x.shape
    seg = triton.cdiv(n, $SPLITS)
    pm = torch.empty((b * $SPLITS,), device=x.device, dtype=torch.float32)
    pi = torch.empty((b * $SPLITS,), device=x.device, dtype=torch.int64)
    out = torch.empty((b, 1), device=x.device, dtype=torch.int64)
    _keep(_part[(b, $SPLITS)](x, pm, pi, n, x.stride(0), seg, SPLITS=$SPLITS, BLOCK=triton.next_power_of_2(seg), num_warps=$WARPS))
    _comb[(b,)](pm, pi, out, SPLITS=$SPLITS, num_warps=1)
    return out
''')

# ---------------------------------------------------------------- temperature softmax
family('temperature_softmax', 'row_whole', {'WARPS': [1, 2, 4, 8, 16]}, {'WARPS': 4}, '''
@triton.jit
def _k(X, O, N, sx, inv_t, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    x = tl.load(X + r * sx + c, mask=c < N, other=float('-inf')) * inv_t
    e = tl.exp(x - tl.max(x, 0))
    tl.store(O + r * sx + c, e / tl.sum(e, 0), mask=c < N)


def run(inp):
    x = inp['logits']
    b, n = x.shape
    out = torch.empty_like(x)
    _keep(_k[(b,)](x, out, n, x.stride(0), 1.0 / inp['temperature'], BLOCK=triton.next_power_of_2(n), num_warps=$WARPS))
    return out
''')

family('temperature_softmax', 'online_chunked', {'BLOCK': [256, 512, 1024, 2048], 'WARPS': [1, 2, 4, 8]}, {'BLOCK': 1024, 'WARPS': 4}, '''
@triton.jit
def _k(X, O, N, sx, inv_t, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    m = tl.full([1], float('-inf'), tl.float32)
    l = tl.zeros([1], tl.float32)
    for s in range(0, N, BLOCK):
        c = s + tl.arange(0, BLOCK)
        x = tl.load(X + r * sx + c, mask=c < N, other=float('-inf')) * inv_t
        mn = tl.maximum(m, tl.max(x, 0))
        l = l * tl.exp(m - mn) + tl.sum(tl.exp(x - mn), 0)
        m = mn
    for s in range(0, N, BLOCK):
        c = s + tl.arange(0, BLOCK)
        x = tl.load(X + r * sx + c, mask=c < N, other=float('-inf')) * inv_t
        tl.store(O + r * sx + c, tl.exp(x - m) / l, mask=c < N)


def run(inp):
    x = inp['logits']
    b, n = x.shape
    out = torch.empty_like(x)
    _keep(_k[(b,)](x, out, n, x.stride(0), 1.0 / inp['temperature'], BLOCK=$BLOCK, num_warps=$WARPS))
    return out
''')

family('temperature_softmax', 'split_two_stage', {'SPLITS': [2, 4, 8, 16], 'WARPS': [1, 2, 4]}, {'SPLITS': 8, 'WARPS': 2}, '''
@triton.jit
def _part(X, PM, PL, N, sx, inv_t, SEG, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    s = tl.program_id(1)
    c = s * SEG + tl.arange(0, BLOCK)
    ok = (c < N) & (tl.arange(0, BLOCK) < SEG)
    x = tl.load(X + r * sx + c, mask=ok, other=float('-inf')) * inv_t
    m = tl.max(x, 0)
    tl.store(PM + r * SPLITS + s, m)
    tl.store(PL + r * SPLITS + s, tl.sum(tl.exp(x - m), 0))


@triton.jit
def _write(X, O, PM, PL, N, sx, inv_t, SEG, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    s = tl.program_id(1)
    j = tl.arange(0, SPLITS)
    pm = tl.load(PM + r * SPLITS + j)
    pl = tl.load(PL + r * SPLITS + j)
    m = tl.max(pm, 0)
    total = tl.sum(pl * tl.exp(pm - m), 0)
    c = s * SEG + tl.arange(0, BLOCK)
    ok = (c < N) & (tl.arange(0, BLOCK) < SEG)
    x = tl.load(X + r * sx + c, mask=ok, other=float('-inf')) * inv_t
    tl.store(O + r * sx + c, tl.exp(x - m) / total, mask=ok)


def run(inp):
    x = inp['logits']
    b, n = x.shape
    seg = triton.cdiv(n, $SPLITS)
    blk = triton.next_power_of_2(seg)
    pm = torch.empty((b * $SPLITS,), device=x.device, dtype=torch.float32)
    pl = torch.empty_like(pm)
    out = torch.empty_like(x)
    inv = 1.0 / inp['temperature']
    _keep(_part[(b, $SPLITS)](x, pm, pl, n, x.stride(0), inv, seg, SPLITS=$SPLITS, BLOCK=blk, num_warps=$WARPS))
    _write[(b, $SPLITS)](x, out, pm, pl, n, x.stride(0), inv, seg, SPLITS=$SPLITS, BLOCK=blk, num_warps=$WARPS)
    return out
''')

# ---------------------------------------------------------------- top-k (ties at the k-th value kept) + softmax
_TOPK_KEY = '''
@triton.jit
def _key(x):
    bits = x.to(tl.int32, bitcast=True)
    return tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)  # order-preserving int view of float32
'''

family('topk_softmax', 'bisect_whole_row', {'WARPS': [1, 2, 4, 8, 16]}, {'WARPS': 4}, _TOPK_KEY + '''
@triton.jit
def _k(X, O, N, sx, inv_t, K, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    x = tl.load(X + r * sx + c, mask=c < N, other=float('-inf')) * inv_t
    key = _key(x).to(tl.int64)
    lo = tl.full([1], -2147483648, tl.int64)  # count(key >= lo) = N >= K
    hi = tl.full([1], 2147483648, tl.int64)   # count(key >= hi) = 0 < K
    for _ in range(33):
        mid = (lo + hi) // 2
        ok = tl.sum((key >= mid).to(tl.int32), 0) >= K
        lo = tl.where(ok, mid, lo)
        hi = tl.where(ok, hi, mid)
    xk = tl.where(key >= lo, x, float('-inf'))  # lo = exact k-th largest key
    e = tl.exp(xk - tl.max(xk, 0))
    tl.store(O + r * sx + c, e / tl.sum(e, 0), mask=c < N)


def run(inp):
    x = inp['logits']
    b, n = x.shape
    out = torch.empty_like(x)
    _keep(_k[(b,)](x, out, n, x.stride(0), 1.0 / inp['temperature'], inp['k'], BLOCK=triton.next_power_of_2(n), num_warps=$WARPS))
    return out
''')

family('topk_softmax', 'bisect_chunked', {'BLOCK': [512, 1024, 2048], 'WARPS': [2, 4, 8]}, {'BLOCK': 1024, 'WARPS': 4}, _TOPK_KEY + '''
@triton.jit
def _k(X, O, N, sx, inv_t, K, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    lo = tl.full([1], -2147483648, tl.int64)
    hi = tl.full([1], 2147483648, tl.int64)
    for _ in range(33):
        mid = (lo + hi) // 2
        cnt = tl.zeros([1], tl.int32)
        for s in range(0, N, BLOCK):
            c = s + tl.arange(0, BLOCK)
            x = tl.load(X + r * sx + c, mask=c < N, other=float('-inf')) * inv_t
            cnt += tl.sum((_key(x).to(tl.int64) >= mid).to(tl.int32), 0)
        ok = cnt >= K
        lo = tl.where(ok, mid, lo)
        hi = tl.where(ok, hi, mid)
    m = tl.full([1], float('-inf'), tl.float32)
    l = tl.zeros([1], tl.float32)
    for s in range(0, N, BLOCK):
        c = s + tl.arange(0, BLOCK)
        x = tl.load(X + r * sx + c, mask=c < N, other=float('-inf')) * inv_t
        x = tl.where(_key(x).to(tl.int64) >= lo, x, float('-inf'))
        mn = tl.maximum(m, tl.max(x, 0))
        l = l * tl.exp(m - mn) + tl.sum(tl.exp(x - mn), 0)
        m = mn
    for s in range(0, N, BLOCK):
        c = s + tl.arange(0, BLOCK)
        x = tl.load(X + r * sx + c, mask=c < N, other=float('-inf')) * inv_t
        x = tl.where(_key(x).to(tl.int64) >= lo, x, float('-inf'))
        tl.store(O + r * sx + c, tl.exp(x - m) / l, mask=c < N)


def run(inp):
    x = inp['logits']
    b, n = x.shape
    out = torch.empty_like(x)
    _keep(_k[(b,)](x, out, n, x.stride(0), 1.0 / inp['temperature'], inp['k'], BLOCK=$BLOCK, num_warps=$WARPS))
    return out
''')

# ---------------------------------------------------------------- inverse-CDF sampling
family('inverse_cdf_sample', 'scan_whole_row', {'WARPS': [1, 2, 4, 8, 16]}, {'WARPS': 4}, '''
@triton.jit
def _k(P, U, O, N, sp, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    p = tl.load(P + r * sp + c, mask=c < N, other=0.0)
    cdf = tl.cumsum(p, 0)
    target = tl.load(U + r) * tl.sum(p, 0)
    tok = tl.sum((cdf <= target).to(tl.int64), 0)
    last = tl.max(tl.where(p > 0, c, 0), 0).to(tl.int64)
    tl.store(O + r, tl.minimum(tok, last))


def run(inp):
    p = inp['probs']
    b, n = p.shape
    out = torch.empty((b, 1), device=p.device, dtype=torch.int64)
    _keep(_k[(b,)](p, inp['u'], out, n, p.stride(0), BLOCK=triton.next_power_of_2(n), num_warps=$WARPS))
    return out
''')

family('inverse_cdf_sample', 'scan_chunked', {'BLOCK': [256, 512, 1024, 2048], 'WARPS': [1, 2, 4, 8]}, {'BLOCK': 1024, 'WARPS': 4}, '''
@triton.jit
def _k(P, U, O, N, sp, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    total = tl.zeros([1], tl.float32)
    for s in range(0, N, BLOCK):
        c = s + tl.arange(0, BLOCK)
        total += tl.sum(tl.load(P + r * sp + c, mask=c < N, other=0.0), 0)
    target = tl.load(U + r + tl.arange(0, 1)) * total
    carry = tl.zeros([1], tl.float32)
    tok = tl.zeros([1], tl.int64)
    last = tl.zeros([1], tl.int64)
    for s in range(0, N, BLOCK):
        c = s + tl.arange(0, BLOCK)
        p = tl.load(P + r * sp + c, mask=c < N, other=0.0)
        cdf = tl.cumsum(p, 0) + carry
        tok += tl.sum((cdf <= target).to(tl.int64), 0)
        last = tl.maximum(last, tl.max(tl.where(p > 0, c, 0), 0).to(tl.int64))
        carry += tl.sum(p, 0)
    tl.store(O + r + tl.arange(0, 1), tl.minimum(tok, last))


def run(inp):
    p = inp['probs']
    b, n = p.shape
    out = torch.empty((b, 1), device=p.device, dtype=torch.int64)
    _keep(_k[(b,)](p, inp['u'], out, n, p.stride(0), BLOCK=$BLOCK, num_warps=$WARPS))
    return out
''')

# ---------------------------------------------------------------- residual add + LayerNorm
family('add_layernorm', 'row_per_program', {'WARPS': [1, 2, 4, 8]}, {'WARPS': 4}, '''
@triton.jit
def _k(X, R, W, Bb, O, D, sx, so, eps, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    ok = c < D
    s = tl.load(X + r * sx + c, mask=ok, other=0.0) + tl.load(R + r * sx + c, mask=ok, other=0.0)
    tl.store(O + r * so + c, s, mask=ok)
    mean = tl.sum(s, 0) / D
    d = tl.where(ok, s - mean, 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(d * d, 0) / D + eps)
    y = d * rstd * tl.load(W + c, mask=ok, other=0.0) + tl.load(Bb + c, mask=ok, other=0.0)
    tl.store(O + r * so + D + c, y, mask=ok)


def run(inp):
    x = inp['x']
    b, d = x.shape
    out = torch.empty((b, 2 * d), device=x.device, dtype=x.dtype)
    _keep(_k[(b,)](x, inp['residual'], inp['weight'], inp['bias'], out, d, x.stride(0), out.stride(0), 1e-5,
                   BLOCK=triton.next_power_of_2(d), num_warps=$WARPS))
    return out
''')

family('add_layernorm', 'rows_per_program', {'ROWS': [2, 4, 8, 16], 'WARPS': [1, 2, 4, 8]}, {'ROWS': 4, 'WARPS': 4}, '''
@triton.jit
def _k(X, R, W, Bb, O, B, D, sx, so, eps, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    c = tl.arange(0, BLOCK)
    ok = (rows[:, None] < B) & (c[None, :] < D)
    s = tl.load(X + rows[:, None] * sx + c[None, :], mask=ok, other=0.0) + tl.load(R + rows[:, None] * sx + c[None, :], mask=ok, other=0.0)
    tl.store(O + rows[:, None] * so + c[None, :], s, mask=ok)
    mean = tl.sum(s, 1) / D
    d = tl.where(ok, s - mean[:, None], 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(d * d, 1) / D + eps)
    w = tl.load(W + c, mask=c < D, other=0.0)
    bb = tl.load(Bb + c, mask=c < D, other=0.0)
    tl.store(O + rows[:, None] * so + D + c[None, :], d * rstd[:, None] * w[None, :] + bb[None, :], mask=ok)


def run(inp):
    x = inp['x']
    b, d = x.shape
    out = torch.empty((b, 2 * d), device=x.device, dtype=x.dtype)
    _keep(_k[(triton.cdiv(b, $ROWS),)](x, inp['residual'], inp['weight'], inp['bias'], out, b, d, x.stride(0), out.stride(0), 1e-5,
                                       ROWS=$ROWS, BLOCK=triton.next_power_of_2(d), num_warps=$WARPS))
    return out
''')

# ---------------------------------------------------------------- decode attention
family('decode_attention', 'split_k', {'SPLIT': [8, 16, 32, 64, 128], 'BLOCK_N': [16, 32, 64], 'WARPS': [1, 2, 4, 8]},
       {'SPLIT': 32, 'BLOCK_N': 16, 'WARPS': 2}, '''
@triton.jit
def _split(Q, K, V, POS, MO, LO, AO, sqb, sqh, skb, skh, skn, svb, svh, svn, s_pos, H, scale,
           D: tl.constexpr, BLOCK_N: tl.constexpr, SPLIT: tl.constexpr, NSPLIT: tl.constexpr):
    bh = tl.program_id(0)
    s = tl.program_id(1)
    b = bh // H
    h = bh % H
    length = tl.load(POS + b * s_pos) + 1
    start = s * SPLIT
    end = tl.minimum(start + SPLIT, length)
    d = tl.arange(0, D)
    q = tl.load(Q + b * sqb + h * sqh + d)
    m = tl.zeros([1], tl.float32) - float('inf')
    l = tl.zeros([1], tl.float32)
    acc = tl.zeros([D], tl.float32)
    for n0 in range(start, end, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        ok = n < end
        k = tl.load(K + b * skb + h * skh + n[:, None] * skn + d[None, :], mask=ok[:, None], other=0.0)
        sc = tl.where(ok, tl.sum(k * q[None, :], 1) * scale, float('-inf'))
        mn = tl.maximum(m, tl.max(sc, 0))
        alpha = tl.exp(m - mn)
        p = tl.exp(sc - mn)
        v = tl.load(V + b * svb + h * svh + n[:, None] * svn + d[None, :], mask=ok[:, None], other=0.0)
        l = l * alpha + tl.sum(p, 0)
        acc = acc * alpha + tl.sum(p[:, None] * v, 0)
        m = mn
    o = bh * NSPLIT + s
    tl.store(MO + o + tl.arange(0, 1), m)
    tl.store(LO + o + tl.arange(0, 1), l)
    tl.store(AO + o * D + d, acc)


@triton.jit
def _combine(MI, LI, AI, OUT, sob, soh, H, D: tl.constexpr, NSPLIT: tl.constexpr):
    bh = tl.program_id(0)
    j = tl.arange(0, NSPLIT)
    d = tl.arange(0, D)
    m = tl.load(MI + bh * NSPLIT + j)
    w = tl.exp(m - tl.max(m, 0))
    acc = tl.load(AI + (bh * NSPLIT + j)[:, None] * D + d[None, :])
    l = tl.load(LI + bh * NSPLIT + j)
    tl.store(OUT + (bh // H) * sob + (bh % H) * soh + d, tl.sum(acc * w[:, None], 0) / tl.sum(l * w, 0))


def run(inp):
    q, k, v, pos = inp['q'], inp['k'], inp['v'], inp['position']
    b, h, _, d = q.shape
    cap = k.size(2)
    nsplit = triton.next_power_of_2(triton.cdiv(cap, $SPLIT))
    m = torch.empty((b * h * nsplit,), device=q.device, dtype=torch.float32)
    l = torch.empty_like(m)
    acc = torch.empty((b * h * nsplit, d), device=q.device, dtype=torch.float32)
    out = torch.empty((b, h, 1, d), device=q.device, dtype=torch.float32)
    s_pos = 0 if pos.numel() == 1 else pos.stride(0)
    _keep(_split[(b * h, nsplit)](q, k, v, pos, m, l, acc, q.stride(0), q.stride(1), k.stride(0), k.stride(1), k.stride(2),
                                  v.stride(0), v.stride(1), v.stride(2), s_pos, h, d ** -0.5,
                                  D=d, BLOCK_N=$BLOCK_N, SPLIT=$SPLIT, NSPLIT=nsplit, num_warps=$WARPS))
    _combine[(b * h,)](m, l, acc, out, out.stride(0), out.stride(1), h, D=d, NSPLIT=nsplit, num_warps=1 if nsplit <= 8 else 2)
    return out
''')

family('decode_attention', 'single_pass', {'BLOCK_N': [16, 32, 64, 128], 'WARPS': [1, 2, 4, 8]}, {'BLOCK_N': 32, 'WARPS': 4}, '''
@triton.jit
def _k(Q, K, V, POS, OUT, sqb, sqh, skb, skh, skn, svb, svh, svn, sob, soh, s_pos, H, scale,
       D: tl.constexpr, BLOCK_N: tl.constexpr):
    bh = tl.program_id(0)
    b = bh // H
    h = bh % H
    length = tl.load(POS + b * s_pos) + 1
    d = tl.arange(0, D)
    q = tl.load(Q + b * sqb + h * sqh + d)
    m = tl.zeros([1], tl.float32) - float('inf')
    l = tl.zeros([1], tl.float32)
    acc = tl.zeros([D], tl.float32)
    for n0 in range(0, length, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        ok = n < length
        k = tl.load(K + b * skb + h * skh + n[:, None] * skn + d[None, :], mask=ok[:, None], other=0.0)
        sc = tl.where(ok, tl.sum(k * q[None, :], 1) * scale, float('-inf'))
        mn = tl.maximum(m, tl.max(sc, 0))
        alpha = tl.exp(m - mn)
        p = tl.exp(sc - mn)
        v = tl.load(V + b * svb + h * svh + n[:, None] * svn + d[None, :], mask=ok[:, None], other=0.0)
        l = l * alpha + tl.sum(p, 0)
        acc = acc * alpha + tl.sum(p[:, None] * v, 0)
        m = mn
    tl.store(OUT + b * sob + h * soh + d, acc / l)


def run(inp):
    q, k, v, pos = inp['q'], inp['k'], inp['v'], inp['position']
    b, h, _, d = q.shape
    out = torch.empty((b, h, 1, d), device=q.device, dtype=torch.float32)
    s_pos = 0 if pos.numel() == 1 else pos.stride(0)
    _keep(_k[(b * h,)](q, k, v, pos, out, q.stride(0), q.stride(1), k.stride(0), k.stride(1), k.stride(2),
                       v.stride(0), v.stride(1), v.stride(2), out.stride(0), out.stride(1), s_pos, h, d ** -0.5,
                       D=d, BLOCK_N=$BLOCK_N, num_warps=$WARPS))
    return out
''')

# ---------------------------------------------------------------- int8_gemv (weight-only int8 linear, real-model R1)
# W int8 [N, K] (nn.Linear layout) is converted to bf16 in registers (exact: |w| <= 127) and multiplied on tensor cores
# with fp32 accumulation; the per-channel scale is applied once to the fp32 accumulator. M (rows) is padded to 16.
family('int8_gemv', 'dot16', {'BLOCK_N': [32, 64, 128], 'BLOCK_K': [64, 128, 256], 'WARPS': [2, 4, 8]},
       {'BLOCK_N': 64, 'BLOCK_K': 128, 'WARPS': 4}, '''
@triton.jit
def _k(X, W, S, O, M, N, K, sxm, swn, som, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = tl.arange(0, 16)
    acc = tl.zeros([16, BLOCK_N], tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(X + rm[:, None] * sxm + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        w = tl.load(W + rn[None, :] * swn + rk[:, None], mask=(rn[None, :] < N) & (rk[:, None] < K), other=0)
        acc += tl.dot(x.to(tl.bfloat16), w.to(tl.bfloat16))
    s = tl.load(S + rn, mask=rn < N, other=0.0)
    out = acc * s[None, :]
    tl.store(O + rm[:, None] * som + rn[None, :], out.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


def run(inp):
    x, w, s = inp['x'], inp['w'], inp['scale']
    m, k = x.shape
    n = w.shape[0]
    if m > 16:  # kernels pad M to 16 (review V1 #7)
        raise ValueError('at most 16 rows')
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    _keep(_k[(triton.cdiv(n, $BLOCK_N),)](x, w, s, out, m, n, k, x.stride(0), w.stride(0), out.stride(0),
                                          BLOCK_N=$BLOCK_N, BLOCK_K=$BLOCK_K, num_warps=$WARPS))
    return out
''')

# Split-K: few output blocks (k/v projections: N = 1024) under-fill 24 SMs, so K is split across programs; fp32 partials
# are reduced, scaled and cast by a second kernel (deterministic: no atomics).
family('int8_gemv', 'dot16_splitk', {'BLOCK_N': [32, 64, 128], 'BLOCK_K': [64, 128, 256], 'SPLITK': [2, 4, 8],
                                     'WARPS': [4]},
       {'BLOCK_N': 64, 'BLOCK_K': 128, 'SPLITK': 4, 'WARPS': 4}, '''
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


def run(inp):
    x, w, s = inp['x'], inp['w'], inp['scale']
    m, k = x.shape
    n = w.shape[0]
    if m > 16:  # kernels pad M to 16 (review V1 #7)
        raise ValueError('at most 16 rows')
    kchunk = triton.cdiv(triton.cdiv(k, $SPLITK), $BLOCK_K) * $BLOCK_K
    splits = triton.cdiv(k, kchunk)
    part = torch.empty((splits, 16, n), device=x.device, dtype=torch.float32)
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    _keep(_part[(triton.cdiv(n, $BLOCK_N), splits)](x, w, part, m, n, k, x.stride(0), w.stride(0), kchunk,
                                                    BLOCK_N=$BLOCK_N, BLOCK_K=$BLOCK_K, num_warps=$WARPS))
    _reduce[(triton.cdiv(n, 256), m)](part, s, out, m, n, out.stride(0), SPLITK=splits, BLOCK=256, num_warps=4)
    return out
''')
