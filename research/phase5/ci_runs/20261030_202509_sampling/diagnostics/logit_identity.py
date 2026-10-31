"""Are decode-step logits bit-identical: fixed-cache masked step vs incremental (concat-KV) reference_step?"""
import sys, torch
sys.path.insert(0, '.')
from pathlib import Path
from research.phase4.benchmark_decode import load_model
from research.phase4.decoder import reference_step
from research.phase4.compact_cache import CompactStaticKVDecoder
model = load_model(Path('out-tinystories-v2/baseline-2500.pt'))
tr = model.transformer
def step_logits(e):  # StaticKVDecoder.step body, returning logits (no state advance)
    import torch.nn.functional as F
    x = tr.drop(tr.wte(e.token) + tr.wpe(e.position))
    idx = e.position.view(1, 1, 1, 1).expand(e.batch, e.heads, 1, e.depth)
    allowed = e.locations <= e.position.view(1, 1, 1, 1)
    for i, block in enumerate(tr.h):
        a = block.attn; h = block.ln_1(x); b, t, c = h.shape
        q, k, v = a.c_attn(h).split(c, dim=2)
        q = q.view(b, t, e.heads, e.depth).transpose(1, 2); k = k.view(b, t, e.heads, e.depth).transpose(1, 2); v = v.view(b, t, e.heads, e.depth).transpose(1, 2)
        e.keys[i].scatter_(2, idx, k); e.values[i].scatter_(2, idx, v)
        y = F.scaled_dot_product_attention(q, e.keys[i], e.values[i], attn_mask=allowed, dropout_p=0.0, is_causal=False)
        x = x + a.resid_dropout(a.c_proj(y.transpose(1, 2).contiguous().view(b, t, c))); x = x + block.mlp(block.ln_2(x))
    return model.lm_head(tr.ln_f(x)[:, -1:, :])
for policy in ('tf32', 'ieee'):
    torch.backends.cuda.matmul.fp32_precision = policy
    g = torch.Generator().manual_seed(1)
    worst, ident, total = 0.0, 0, 0
    with torch.inference_mode():
        for B, p, cap in [(1, 8, 80), (16, 8, 80), (1, 100, 128), (16, 100, 256), (16, 200, 256)]:
            x = torch.randint(0, 4096, (B, p), generator=g).cuda()
            e = CompactStaticKVDecoder(model, B, cap); e.reset(x)
            _, cache = reference_step(model, x)
            tok = e.token.clone()
            for t in range(min(20, cap - p - 1)):
                got = step_logits(e)
                ref, cache = reference_step(model, tok, cache)
                d = float((got - ref).abs().max()); worst = max(worst, d); ident += int(d == 0.0); total += 1
                nxt = ref.argmax(-1); e.token.copy_(nxt); e.position.add_(1); tok = nxt
    print(f'{policy}: bit-identical steps {ident}/{total}, max |logit diff| {worst:.3e}')
