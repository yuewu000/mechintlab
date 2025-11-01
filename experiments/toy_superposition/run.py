
import torch, os
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange
from mechintlab.models.sae import SparseAutoencoder
from mechintlab.utils.seed import set_seed
from mechintlab.utils.io import ensure_dir, log_jsonl, timestamp

def make_data(n=4096, d_in=256, k=8, noise=0.05, seed=42):
    torch.manual_seed(seed)
    # synthetic sparse features -> linear mix -> observations
    features = torch.zeros(n, k)
    idx = torch.randint(0, k, (n,))
    features[torch.arange(n), idx] = 1.0
    W = torch.randn(k, d_in) / (d_in ** 0.5)
    X = features @ W + noise * torch.randn(n, d_in)
    return X

def main():
    set_seed(42)
    X = make_data()
    ds = TensorDataset(X, X)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    model = SparseAutoencoder(d_in=X.size(1), d_hidden=64, l1=2e-3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    run_dir = os.path.join("runs", "toy_superposition_" + timestamp())
    ensure_dir(run_dir)
    log_path = os.path.join(run_dir, "metrics.jsonl")

    for step in trange(1000):
        for xb, _ in dl:
            loss, stats = model.loss(xb)
            opt.zero_grad(); loss.backward(); opt.step()
        stats.update(step=step, total=loss.item())
        log_jsonl(stats, log_path)

    # Save tiny artifact
    torch.save(model.state_dict(), os.path.join(run_dir, "sae.pt"))
    print("Finished. Logs in:", log_path)

if __name__ == "__main__":
    main()
