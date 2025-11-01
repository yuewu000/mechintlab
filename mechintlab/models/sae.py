
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, l1: float = 1e-3):
        super().__init__()
        self.enc = nn.Linear(d_in, d_hidden, bias=False)
        self.dec = nn.Linear(d_hidden, d_in, bias=False)
        self.l1 = l1

    def forward(self, x):
        z = F.relu(self.enc(x))
        xhat = self.dec(z)
        return xhat, z

    def loss(self, x):
        xhat, z = self.forward(x)
        recon = F.mse_loss(xhat, x)
        sparsity = self.l1 * z.abs().mean()
        return recon + sparsity, {"recon": recon.item(), "l1": sparsity.item()}
