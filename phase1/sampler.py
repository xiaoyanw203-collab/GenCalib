"""Phase 1: latent-space diffusion sampling with optional mask-guided anchoring."""

import torch
from tqdm import tqdm


class _DiffusionSchedule:
    """Shared DDPM noise schedule and posterior variance."""

    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cuda"):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (
            1.0 - torch.cat([torch.tensor([1.0]).to(device), self.alphas_cumprod[:-1]])
        ) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: add noise to clean latents at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def _ddpm_step(self, x, predicted_noise, t_index):
        coeff1 = self.sqrt_recip_alphas[t_index]
        coeff2 = (1 - self.alphas[t_index]) / self.sqrt_one_minus_alphas_cumprod[t_index]
        model_mean = coeff1 * (x - coeff2 * predicted_noise)

        if t_index == 0:
            return model_mean

        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(self.posterior_variance[t_index]) * noise


class MaskedDiffusionSampler(_DiffusionSchedule):
    """
    Phase 1 (full method): masked diffusion sampling in latent space.

    Known anchor locations are hard-constrained at each reverse step by injecting
    the forward-noised ground-truth latent values under the sparse mask.
    """

    def p_sample_masked(self, model, x, t, t_index, condition, gt_latent=None, sparse_mask=None):
        """Single reverse step with optional mask injection."""
        with torch.no_grad():
            predicted_noise = model(x, t, condition)

        x_pred_prev = self._ddpm_step(x, predicted_noise, t_index)

        if sparse_mask is not None and gt_latent is not None:
            if t_index == 0:
                x_known = gt_latent
            else:
                t_prev = torch.full((x.shape[0],), t_index - 1, device=self.device, dtype=torch.long)
                x_known = self.q_sample(gt_latent, t_prev)
            x_pred_prev = sparse_mask * x_known + (1.0 - sparse_mask) * x_pred_prev

        return x_pred_prev

    @torch.no_grad()
    def sample(self, model, shape, condition, gt_latent=None, sparse_mask=None, verbose=True):
        img = torch.randn(shape, device=self.device)
        iterable = reversed(range(0, self.num_timesteps))
        if verbose:
            iterable = tqdm(iterable, desc="Phase 1: Masked Sampling", total=self.num_timesteps, leave=False)

        for i in iterable:
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            img = self.p_sample_masked(model, img, t, i, condition, gt_latent, sparse_mask)
        return img


class PureDiffusionSampler(_DiffusionSchedule):
    """
    Phase 1 ablation: standard DDPM sampling without mask injection.

    Used as a zero-shot baseline before Phase 2 IDW calibration.
    """

    @torch.no_grad()
    def sample(self, model, shape, condition, verbose=True):
        img = torch.randn(shape, device=self.device)
        iterable = reversed(range(0, self.num_timesteps))
        if verbose:
            iterable = tqdm(iterable, desc="Pure DiT Sampling", total=self.num_timesteps, leave=False)

        for i in iterable:
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            predicted_noise = model(img, t, condition)
            img = self._ddpm_step(img, predicted_noise, i)
        return img
