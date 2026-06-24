# Histopathology Out-of-Distribution Classification

Binary classification of histopathology image patches (**tumor** vs. **no tumor**) under
domain shift. Training patches come from three hospitals ("centers"), while validation and
test patches each come from a *different, unseen* center. Because staining, equipment, and
acquisition conditions differ between hospitals, the visual appearance shifts across
centers. The goal is a classifier whose accuracy degrades as little as possible under this
out-of-distribution (OOD) shift.


## Data

The data is **not** included in this repository (the `Data/` folder is git-ignored). It is
expected as three HDF5 files:

```
Data/
├── train.h5   # centers 0, 3, 4 — with labels
├── val.h5     # center 1        — with labels (held-out center)
└── test.h5    # another center  — no labels (submission target)
```

Each entry in an `.h5` file holds:

- `img` — the RGB patch (channel-first array)
- `label` — `0` (no tumor) or `1` (tumor)
- `metadata` — the first element is the center ID

## Notebooks

| Notebook | Description |
| --- | --- |
| [`getting_started.ipynb`](getting_started.ipynb) | Original baseline: frozen **DINOv2 ViT-S/14** features + a linear probe. Precomputes embeddings once, trains a logistic head, writes `baseline.csv`. |
| [`getting_started_ood_improved.ipynb`](getting_started_ood_improved.ipynb) | Stronger OOD pipeline (see below). Writes `submission_dinov3_ood.csv`. |

### Improved OOD pipeline

The improved notebook targets the center shift directly:

1. **DINOv3 ViT-S/16** backbone (`facebook/dinov3-vits16-pretrain-lvd1689m`), fine-tuned
   end-to-end (small enough for a 16 GB GPU with mixed precision).
2. **On-the-fly color / stain augmentation** to reduce reliance on center-specific staining.
3. **Center-balanced sampling** over `(center, label)` pairs so the largest source center
   does not dominate.
4. **Domain-adversarial training** — a gradient-reversal domain head pushes features to be
   center-invariant.
5. **Threshold search on validation**, since the challenge metric is accuracy (not BCE).
6. **Test-time augmentation** (flips) at submission time.

The classifier head consumes a concatenation of the CLS token and mean patch token
(averaged over the last four transformer layers).

## Setup

```bash
pip install torch torchvision torchmetrics transformers h5py numpy pandas matplotlib tqdm
```

A CUDA GPU is recommended (the improved notebook uses automatic mixed precision when a GPU
is available, and falls back to CPU otherwise).

## Usage

1. Place `train.h5`, `val.h5`, and `test.h5` under `Data/`.
2. Open either notebook and run all cells.
3. The final cells load the best checkpoint and write a submission CSV.

## Submission format

A CSV with two columns — the predicted class must be thresholded to `0` or `1`:

```
ID,Pred
0,1
1,0
...
```

## Artifacts

| File | Produced by | Notes |
| --- | --- | --- |
| `best_model.pth` | baseline notebook | linear-probe head weights |
| `best_dinov3_ood.pth` | improved notebook | full fine-tuned checkpoint (model + best threshold) |
| `baseline.csv` | baseline notebook | baseline submission |
| `submission_dinov3_ood.csv` | improved notebook | improved submission |
