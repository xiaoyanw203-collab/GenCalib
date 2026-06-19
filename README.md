# Few-Shot Radio Map Reconstruction — Supplementary Code

This repository contains the core evaluation code supporting our two-stage few-shot radio map reconstruction framework:

1. **Phase 1** — Mask-guided latent diffusion sampling (hard anchor injection during reverse diffusion)
2. **Phase 2** — 3D inverse distance weighting (IDW) residual calibration in physical space


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
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```


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


### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--sampling_ratio` | `0.01` | Fraction of voxels used as sparse anchors (e.g. `0.01` = 1%) |
| `--idw_power` | `2.0` | IDW distance decay exponent |
| `--idw_neighbors` | `15` | Number of nearest anchors per target voxel |
| `--steps` | `1000` | DDPM reverse diffusion steps |
| `--modality` | `PathLoss` | Radio map channel to evaluate |
| `--batch_size` | `64` | Evaluation batch size |

### Pretrained Weights

Due to file size and distribution constraints, the pretrained RadioDiT weights are not directly included in this repository. 
They are available from the authors upon reasonable request.

Please contact: wangxiaoyan241@mails.ucas.ac.cn
