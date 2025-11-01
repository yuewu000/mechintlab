
import torch, os
from torch.utils.data import DataLoader, Dataset
from tqdm import trange
from mechintlab.models.tiny_transformer import TinyTransformer
from mechintlab.utils.seed import set_seed
from mechintlab.utils.io import ensure_dir, log_jsonl, timestamp

class CopyTask(Dataset):
    def __init__(self, n=4096, T=32, vocab=128):
        torch.manual_seed(0)
        self.X = torch.randint(1, vocab, (n, T))
        self.Y = self.X.clone()
    def __len__(self): return self.X.size(0)
    def __getitem__(self, idx):
        x = self.X[idx]; y = self.Y[idx]
        return {"input_ids": x, "labels": y}

def main():
    set_seed(1)
    ds = CopyTask()
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    model = TinyTransformer(d=128, n_layers=2, n_heads=4, vocab=128)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    run_dir = os.path.join("runs", "induction_" + timestamp())
    ensure_dir(run_dir)
    log_path = os.path.join(run_dir, "metrics.jsonl")

    for step in trange(1500):
        for batch in dl:
            loss = model.loss_on_batch(batch)
            opt.zero_grad(); loss.backward(); opt.step()
        log_jsonl({"step": step, "loss": float(loss.item())}, log_path)

    torch.save(model.state_dict(), os.path.join(run_dir, "tt.pt"))
    print("Finished. Logs in:", log_path)

if __name__ == "__main__":
    main()
