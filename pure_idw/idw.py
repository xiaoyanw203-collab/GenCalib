"""Pure IDW baseline: interpolate sparse measurements without a generative prior."""

import numpy as np
from scipy.spatial import cKDTree


def apply_pure_idw_calibration(
    gt_volume,
    sampling_ratio=0.05,
    power=2.0,
    k_neighbors=15,
    seed=42,
):
    """
    Traditional IDW interpolation from sparse anchors onto a zero-initialized volume.

    This baseline has no diffusion model prior: only the sparse ground-truth samples
    are used to fill the full 3D grid via inverse distance weighting.

    Args:
        gt_volume: Ground-truth volume of shape (D, H, W).
        sampling_ratio: Fraction of voxels used as sparse anchors.
        power: IDW distance decay exponent.
        k_neighbors: Number of nearest anchors per target voxel.
        seed: Random seed for reproducible anchor placement.

    Returns:
        Interpolated volume clipped to [0, 1].
    """
    d, h, w = gt_volume.shape
    blank_volume = np.zeros_like(gt_volume)

    rng = np.random.RandomState(seed)
    mask = rng.rand(d, h, w) < sampling_ratio
    anchor_coords = np.argwhere(mask)

    if len(anchor_coords) == 0:
        return blank_volume

    anchor_values = gt_volume[mask]
    grid_z, grid_y, grid_x = np.mgrid[0:d, 0:h, 0:w]
    target_coords = np.vstack((grid_z.flatten(), grid_y.flatten(), grid_x.flatten())).T

    tree = cKDTree(anchor_coords)
    distances, indices = tree.query(target_coords, k=min(k_neighbors, len(anchor_coords)))

    epsilon = 1e-10
    weights = 1.0 / (distances ** power + epsilon)
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    normalized_weights = weights / weights_sum

    neighbor_values = anchor_values[indices]
    interpolated_values = np.sum(normalized_weights * neighbor_values, axis=1)
    final_field = interpolated_values.reshape(d, h, w)

    final_field[mask] = anchor_values

    return np.clip(final_field, 0.0, 1.0)
