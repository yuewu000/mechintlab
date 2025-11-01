# mechint-lab (Starter)

A minimal, reproducible **mechanistic interpretability** sandbox:
- **Toy superposition** with a Sparse Autoencoder (SAE)
- **Tiny Transformer** with **activation patching** (induction-head flavored)
- **Streamlit** dashboards (optional)

> Goal: show you can do *research + engineering*, run clean experiments, and communicate results.

## Quickstart
```bash
conda env create -f environment.yml && conda activate mechint
pip install -e .
# Run toy superposition (will generate a small synthetic dataset and train an SAE)
make toy-superposition
# Launch minimal dashboard (optional)
make viz
```

## Experiments
- `experiments/toy_superposition/run.py` – trains an SAE on synthetic features, logs sparsity & reconstruction.
- `experiments/induction_heads/run.py` – trains a tiny transformer on a copy task and enables activation patching.

## Results
Artifacts in `runs/` as JSONL + PNGs.

## License
MIT
