"""Phase 2: 3D inverse distance weighting (IDW) residual calibration in physical space."""

import numpy as np
from scipy.spatial import cKDTree


def apply_phase2_idw_calibration(
    pred_volume,
    gt_volume,
    sampling_ratio=0.05,
    power=2.0,
    k_neighbors=15,
    seed=42,
):
    """
    Calibrate a predicted 3D volume using sparse anchor residuals and IDW interpolation.

    Args:
        pred_volume: Predicted volume of shape (D, H, W).
        gt_volume: Ground-truth volume of the same shape.
        sampling_ratio: Fraction of voxels used as sparse anchors.
        power: IDW distance decay exponent.
        k_neighbors: Number of nearest anchors per target voxel.
        seed: Random seed for reproducible anchor placement.

    Returns:
        Calibrated volume clipped to [0, 1].
    """
    d, h, w = pred_volume.shape

    rng = np.random.RandomState(seed)
    mask = rng.rand(d, h, w) < sampling_ratio
    anchor_coords = np.argwhere(mask)

    if len(anchor_coords) == 0:
        return pred_volume

    residuals = gt_volume[mask] - pred_volume[mask]
    grid_z, grid_y, grid_x = np.mgrid[0:d, 0:h, 0:w]
    target_coords = np.vstack((grid_z.flatten(), grid_y.flatten(), grid_x.flatten())).T

    tree = cKDTree(anchor_coords)
    distances, indices = tree.query(target_coords, k=min(k_neighbors, len(anchor_coords)))

    epsilon = 1e-10
    weights = 1.0 / (distances ** power + epsilon)
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    normalized_weights = weights / weights_sum

    neighbor_residuals = residuals[indices]
    interpolated_residuals = np.sum(normalized_weights * neighbor_residuals, axis=1)
    residual_field = interpolated_residuals.reshape(d, h, w)

    residual_field[mask] = residuals

    calibrated_volume = pred_volume + residual_field
    return np.clip(calibrated_volume, 0.0, 1.0)
