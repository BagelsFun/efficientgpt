"""Step variants for Inductor: does index_copy_ avoid the full-cache scatter copy?"""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, '.')
from pathlib import Path
from torch.profiler import profile, ProfilerActivity
from research.phase4.benchmark_decode import load_model
from research.phase4.decoder import StaticKVDecoder

def step_index_copy(self):
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
        y = F.scaled_dot_product_attention(q, self.keys[i], self.values[i], attn_mask=allowed,
                                           dropout_p=0.0, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        x = x + a.resid_dropout(a.c_proj(y))
        x = x + block.mlp(block.ln_2(x))
    logits = self.model.lm_head(tr.ln_f(x)[:, -1:, :])
    self.history.index_copy_(1, self.position, self.token)
    self.token.copy_(logits.argmax(-1))
    self.position.add_(1)

torch.backends.cuda.matmul.fp32_precision = 'tf32'
model = load_model(Path('out-tinystories-v2/baseline-2500.pt'))
B = 16
x = torch.randint(0, model.config.vocab_size, (B, 64)).cuda()
with torch.inference_mode():
    # exactness of the variant itself in eager (must be bit-identical to the scatter_ step)
    e1, e2 = StaticKVDecoder(model, B), StaticKVDecoder(model, B)
    e1.reset(x); e2.reset(x)
    for _ in range(40): StaticKVDecoder.step.__wrapped__(e1); step_index_copy(e2)
    same = all(torch.equal(a, b) for a, b in zip(e1.keys + e1.values + [e1.history, e1.token], e2.keys + e2.values + [e2.history, e2.token]))
    print('eager index_copy_ variant bit-identical to scatter_ step:', same)
    eng = StaticKVDecoder(model, B)
    comp = torch.compile(step_index_copy, fullgraph=True, dynamic=False)
    eng.reset(x)
    for _ in range(5): comp(eng)
    eng.reset(x); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(20): comp(eng)
        torch.cuda.synchronize()
    ev = [e for e in prof.key_averages() if e.device_type.name == 'CUDA']
    print(f'B={B} compiled index_copy_: GPU kernel time per step {sum(e.self_device_time_total for e in ev)/20/1000:.3f} ms')
    for e in sorted(ev, key=lambda e: -e.self_device_time_total)[:6]:
        print(f'  {e.self_device_time_total/20/1000:7.3f} ms  x{e.count//20:<3} {e.key[:90]}')
