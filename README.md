# Few-Shot Radio Map Reconstruction — Supplementary Code

This repository contains the core evaluation code supporting our two-stage few-shot radio map reconstruction framework:

1. **Phase 1** — Mask-guided latent diffusion sampling (hard anchor injection during reverse diffusion)
2. **Phase 2** — 3D inverse distance weighting (IDW) residual calibration in physical space

It also includes ablation and baseline implementations used in the paper:

- **Pure DiT + IDW** — Standard diffusion without mask injection, followed by Phase 2 calibration
- **Pure IDW** — Traditional interpolation from sparse measurements only (no generative prior)

---

## Repository Structure

```
supplementary_code/
├── metrics/                  # Evaluation metrics (RMSE, NMSE, MAE, PSNR, SSIM)
│   └── metrics.py
├── phase1/                   # Phase 1 diffusion samplers
│   └── sampler.py            # MaskedDiffusionSampler, PureDiffusionSampler
├── phase2/                   # Phase 2 IDW calibration
│   └── idw.py
├── pure_idw/                 # Pure IDW baseline
│   └── idw.py
├── data/                     # NPZ test dataset loader
│   └── dataset.py
├── utils/                    # Evaluation logger
│   └── logger.py
├── scripts/                  # End-to-end evaluation entry points
│   ├── evaluate_few_shot.py
│   ├── evaluate_abl_idw.py
│   └── evaluate_pure_idw.py
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Additional dependency:** The diffusion-based scripts require pretrained **RadioDiT** and **VAE** model definitions and checkpoints from the main project (`src/models/dit_3d.py`, `src/models/vae_3d.py`). Place this repository inside the main project tree, or add the project root to `PYTHONPATH`.

---

## Data Format

Test data should be stored as `.npz` files with the following keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `volume` or `latent` | `(N, C, D, H, W)` or `(C, D, H, W)` | Ground-truth radio map volume or latent |
| `condition` | `(N, 3, H, W)` or `(3, H, W)` | Conditioning input (required for DiT scripts) |

Supported modalities (channel index): `PathLoss` (0), `DoA_Azi` (1), `DoA_Ele` (2), `Delay` (3).

---

## Core Modules

### Metrics (`metrics/metrics.py`)

```python
from metrics import MetricsCalculator

calc = MetricsCalculator()
rmse = calc.compute_rmse(gt, pred)
nmse = calc.compute_nmse(gt, pred)
mae  = calc.compute_mae(gt, pred)
psnr = calc.compute_psnr(gt, pred, data_range=1.0)
ssim = calc.compute_ssim(gt, pred, data_range=1.0)
```

### Phase 1 — Masked Diffusion (`phase1/sampler.py`)

```python
from phase1 import MaskedDiffusionSampler

sampler = MaskedDiffusionSampler(num_timesteps=1000, device="cuda")
pred_latents = sampler.sample(
    model, shape, condition,
    gt_latent=gt_latents,
    sparse_mask=sparse_mask,
)
```

### Phase 1 Ablation — Pure Diffusion (`phase1/sampler.py`)

```python
from phase1 import PureDiffusionSampler

sampler = PureDiffusionSampler(num_timesteps=1000, device="cuda")
pred_latents = sampler.sample(model, shape, condition)
```

### Phase 2 — IDW Calibration (`phase2/idw.py`)

```python
from phase2 import apply_phase2_idw_calibration

calibrated = apply_phase2_idw_calibration(
    pred_volume=pred,       # (D, H, W)
    gt_volume=gt,           # (D, H, W)
    sampling_ratio=0.01,    # 1% sparse anchors
    power=2.0,
    k_neighbors=15,
)
```

### Pure IDW Baseline (`pure_idw/idw.py`)

```python
from pure_idw import apply_pure_idw_calibration

interpolated = apply_pure_idw_calibration(
    gt_volume=gt,
    sampling_ratio=0.01,
    power=2.0,
    k_neighbors=15,
)
```

> **Reproducibility:** Both IDW modules use a fixed random seed (`seed=42`) for anchor placement so that all experiments share identical sparse sample locations.

---

## Running Evaluations

Run scripts from the `supplementary_code` directory (or set `PYTHONPATH` accordingly).

### Full Method (Phase 1 + Phase 2)

```bash
python scripts/evaluate_few_shot.py \
    --dit_ckpt /path/to/dit_best.pt \
    --vae_ckpt /path/to/vae_best.pt \
    --data_path /path/to/latents \
    --split test1 \
    --sampling_ratio 0.01 \
    --modality PathLoss \
    --device_dit cuda:0 \
    --save_result_csv results/few_shot_evaluation.csv \
    --save_log_dir results
```

### Ablation: Pure DiT + IDW

```bash
python scripts/evaluate_abl_idw.py \
    --dit_ckpt /path/to/dit_best.pt \
    --vae_ckpt /path/to/vae_best.pt \
    --data_path /path/to/latents \
    --split test1 \
    --sampling_ratio 0.01 \
    --modality PathLoss \
    --device_dit cuda:0 \
    --save_result_csv results/abl_pure_dit_plus_idw.csv
```

### Baseline: Pure IDW

```bash
python scripts/evaluate_pure_idw.py \
    --data_path /path/to/latents \
    --split test1 \
    --sampling_ratio 0.01 \
    --modality PathLoss \
    --vae_ckpt /path/to/vae_best.pt   # only needed for latent-format data
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--sampling_ratio` | `0.01` | Fraction of voxels used as sparse anchors (e.g. `0.01` = 1%) |
| `--idw_power` | `2.0` | IDW distance decay exponent |
| `--idw_neighbors` | `15` | Number of nearest anchors per target voxel |
| `--steps` | `1000` | DDPM reverse diffusion steps |
| `--modality` | `PathLoss` | Radio map channel to evaluate |
| `--batch_size` | `64` | Evaluation batch size |

---

## Output

Each script writes:

- A **per-sample CSV** with metrics for every test sample
- A **timestamped log file** (`.txt`) with run configuration and summary statistics

---

## Citation

If you use this code, please cite our paper:

```bibtex
@article{your_paper_2026,
  title   = {Your Paper Title},
  author  = {Your Name et al.},
  journal = {Journal Name},
  year    = {2026}
}
```

*(Replace with the actual BibTeX entry after publication.)*

---

## License

See the main project repository for license information.
