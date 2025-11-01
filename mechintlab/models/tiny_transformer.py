
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MHA(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.d, self.h = d, n_heads
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, "b t (h d) -> b h t d", h=self.h)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.h)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.h)
        att = (q @ k.transpose(-1, -2)) / (k.size(-1) ** 0.5)
        att = att.softmax(dim=-1)
        y = att @ v
        y = rearrange(y, "b h t d -> b t (h d)")
        return self.o(y)

class Block(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = MHA(d, n_heads)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.ReLU(), nn.Linear(4*d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class TinyTransformer(nn.Module):
    def __init__(self, d=128, n_layers=2, n_heads=4, vocab=256):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, 512, d) * 0.01)
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        h = self.tok(x) + self.pos[:, : x.size(1), :]
        for b in self.blocks:
            h = b(h)
        return self.out(self.ln(h))

    def loss_on_batch(self, batch):
        x, y = batch["input_ids"], batch["labels"]
        logits = self.forward(x)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
