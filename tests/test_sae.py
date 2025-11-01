
import torch
from mechintlab.models.sae import SparseAutoencoder

def test_sae_shapes():
    m = SparseAutoencoder(64, 16)
    x = torch.randn(8, 64)
    xhat, z = m(x)
    assert xhat.shape == x.shape
    assert z.shape == (8, 16)
