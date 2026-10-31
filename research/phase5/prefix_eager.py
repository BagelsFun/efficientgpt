"""PyTorch eager SDPA baseline with an allocated, unpadded KV prefix.

Research baseline, NOT a claim to match production engines or PyTorch compile.
Run as part of the existing nanoGPT research namespace; no changes to model.py.
"""

import torch
import torch.nn.functional as F
from ..phase4.decoder import _check_request, reference_step


class PrefixSDPADecoder:
    """One reusable decoder with a fixed batch and dynamically sliced KV prefix.

    Uses PyTorch's SDPA backend selection automatically. Unlike the graph path,
    attention sees only valid keys, not the entire padded cache. With Python
    indices and variable-length views this path is NOT CUDA-graph capturable.
    """

    def __init__(self, model, batch):
        if model.training:
            raise ValueError('Call model.eval() first')
        if batch < 1:
            raise ValueError('Batch must be positive')
        self.model = model
        self.batch = batch
        c = model.config
        if c.n_embd % c.n_head:
            raise ValueError('Embedding/head mismatch')
        self.heads = c.n_head
        self.dim = c.n_embd // c.n_head
        self.limit = c.block_size
        parameter = next(model.parameters())
        self.device = parameter.device
        shape = (batch, c.n_head, c.block_size, self.dim)
        self.keys = [torch.empty(shape, device=self.device, dtype=parameter.dtype)
                     for _ in range(c.n_layer)]
        self.values = [torch.empty(shape, device=self.device, dtype=parameter.dtype)
                       for _ in range(c.n_layer)]
        self.position_ids = torch.arange(c.block_size, dtype=torch.long,
                                         device=self.device)

    @torch.inference_mode()
    def generate(self, prompt, count):
        _check_request(self.model, prompt, count)
        if prompt.shape[0] != self.batch:
            raise ValueError('Batch mismatch')
        tr = self.model.transformer
        logits, prefill_cache = reference_step(self.model, prompt)
        p = prompt.size(1)
        for layer, (k, v) in enumerate(prefill_cache):
            self.keys[layer][:, :, :p, :].copy_(k)
            self.values[layer][:, :, :p, :].copy_(v)
        token = logits.argmax(-1)
        pieces = [prompt, token]
        for position in range(p, p + count - 1):
            x = tr.drop(tr.wte(token) + tr.wpe(self.position_ids[position:position + 1]))
            for layer, block in enumerate(tr.h):
                attn = block.attn
                hidden = block.ln_1(x)
                b, t, channels = hidden.shape
                q, k, v = attn.c_attn(hidden).split(channels, dim=2)
                q = q.view(b, t, self.heads, self.dim).transpose(1, 2)
                k = k.view(b, t, self.heads, self.dim).transpose(1, 2)
                v = v.view(b, t, self.heads, self.dim).transpose(1, 2)
                self.keys[layer][:, :, position:position + 1, :].copy_(k)
                self.values[layer][:, :, position:position + 1, :].copy_(v)
                # Attends to real prefix only, no mask or all-context padding.
                context_k = self.keys[layer][:, :, :position + 1, :]
                context_v = self.values[layer][:, :, :position + 1, :]
                y = F.scaled_dot_product_attention(
                    q, context_k, context_v, dropout_p=0.0, is_causal=False)
                y = y.transpose(1, 2).contiguous().view(b, t, channels)
                x = x + attn.resid_dropout(attn.c_proj(y))
                x = x + block.mlp(block.ln_2(x))
            token = self.model.lm_head(tr.ln_f(x)[:, -1:, :]).argmax(-1)
            pieces.append(token)
        return torch.cat(pieces, dim=1)
