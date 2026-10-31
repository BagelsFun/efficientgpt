"""Numerics of decode_attention vs masked SDPA and an fp64 reference; edge lengths, ragged positions."""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, '.')
from research.phase5.decode_attention import decode_attention
torch.backends.cuda.matmul.fp32_precision = 'ieee'
g = torch.Generator(device='cuda').manual_seed(0)
def ref64(q, k, v, pos):
    cap = k.size(2); allowed = torch.arange(cap, device=q.device).view(1, 1, 1, cap) <= pos.view(-1, 1, 1, 1)
    return F.scaled_dot_product_attention(q.double(), k.double(), v.double(), attn_mask=allowed)
worst = []
for B, cap in [(1, 80), (1, 256), (16, 128), (16, 256), (3, 37)]:
    for scale in (1.0, 8.0):
        q = torch.randn(B, 6, 1, 64, generator=g, device='cuda') * scale
        k = torch.randn(B, 6, cap, 64, generator=g, device='cuda') * scale
        v = torch.randn(B, 6, cap, 64, generator=g, device='cuda')
        for pos in [torch.tensor([0]), torch.tensor([cap - 1]), torch.tensor([cap // 2 + 3]),
                    torch.randint(0, cap, (B,), generator=torch.Generator().manual_seed(B))]:
            pos = pos.cuda()
            allowed = torch.arange(cap, device='cuda').view(1, 1, 1, cap) <= pos.view(-1, 1, 1, 1)
            sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
            ours = decode_attention(q, k, v, pos)
            r = ref64(q, k, v, pos)
            e_ours, e_sdpa = float((ours.double() - r).abs().max()), float((sdpa.double() - r).abs().max())
            worst.append((e_ours, e_sdpa, B, cap, scale, pos.tolist()[:3]))
print('max |err| vs fp64 (ours, sdpa):', max(w[0] for w in worst), max(w[1] for w in worst))
print('cases:', len(worst), ' ours<=2*sdpa+1e-6 in', sum(w[0] <= 2 * w[1] + 1e-6 for w in worst))
print('worst ours case:', max(worst))
# non-contiguous q from the decode step layout: view(b,1,h,d).transpose(1,2)
x = torch.randn(16, 1, 384, device='cuda').view(16, 1, 6, 64).transpose(1, 2)
k = torch.randn(16, 6, 256, 64, device='cuda'); v = torch.randn_like(k); pos = torch.tensor([200], device='cuda')
print('non-contiguous q max err vs fp64:', float((decode_attention(x, k, v, pos).double() - ref64(x, k, v, pos)).abs().max()))
