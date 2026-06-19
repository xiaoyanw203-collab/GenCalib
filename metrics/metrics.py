import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


class MetricsCalculator:
    """Compute RMSE, NMSE, MAE, PSNR, and SSIM between ground truth and predictions."""

    @staticmethod
    def compute_rmse(gt, pred):
        return np.sqrt(np.mean((gt - pred) ** 2))

    @staticmethod
    def compute_nmse(gt, pred):
        mse = np.mean((gt - pred) ** 2)
        norm = np.mean(gt ** 2)
        return 10 * np.log10((mse / (norm + 1e-10)))

    @staticmethod
    def compute_mae(gt, pred):
        return np.mean(np.abs(gt - pred))

    @staticmethod
    def compute_psnr(gt, pred, data_range=1.0):
        return psnr(gt, pred, data_range=data_range)

    @staticmethod
    def compute_ssim(gt, pred, data_range=1.0):
        if gt.ndim == 3:
            gt_flat = gt.reshape(-1, gt.shape[1], gt.shape[2])
            pred_flat = pred.reshape(-1, pred.shape[1], pred.shape[2])
        else:
            gt_flat = gt.reshape(-1, gt.shape[2], gt.shape[3])
            pred_flat = pred.reshape(-1, pred.shape[2], pred.shape[3])

        ssim_val = 0.0
        for i in range(gt_flat.shape[0]):
            ssim_val += ssim(gt_flat[i], pred_flat[i], data_range=data_range)
        return ssim_val / gt_flat.shape[0]
