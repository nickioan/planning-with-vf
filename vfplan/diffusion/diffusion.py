# Adapted from Diffuser (Janner et al. 2022, https://github.com/jannerm/diffuser, MIT License).
# See THIRD_PARTY_NOTICES.md.
import numpy as np
import torch
from torch import nn

from vfplan.diffusion.helpers import Losses, cosine_beta_schedule, extract


class ConditionalGaussianDiffusion(nn.Module):
    """DDPM over 4-footstep plans, conditioned on heightfield, character state and waypoint.

    Args:
        model: noise-prediction network, called as `model(x, embed, t)`.
        feature_extractor: encodes `(cond, init_st, wpt)` into `embed`.
        horizon: number of footsteps per plan.
        observation_dim: per-footstep dimension.
    """

    def __init__(
        self,
        model,
        feature_extractor,
        horizon,
        observation_dim,
        n_timesteps=20,
        loss_type="l2",
        clip_denoised=False,
        predict_epsilon=True,
        loss_discount=1.0,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.transition_dim = observation_dim
        self.model = model
        self.feature_extractor = feature_extractor

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # posterior q(x_{t-1} | x_t, x_0); the log is clipped because the posterior
        # variance is 0 at the beginning of the diffusion chain
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        self.loss_fn = Losses[loss_type](self.get_loss_weights(loss_discount))

    def get_loss_weights(self, discount):
        """Per-(footstep, dimension) loss weights; footstep t is weighted by discount**t."""
        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        return torch.einsum("h,t->ht", discounts, dim_weights)

    # ------------------------------------------------------------------ sampling

    def predict_start_from_noise(self, x_t, t, noise):
        """If predict_epsilon the model output is noise, otherwise it is x0."""
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, embed, t):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, embed, t))
        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        return self.q_posterior(x_start=x_recon, x_t=x, t=t)

    @torch.no_grad()
    def p_sample(self, x, embed, t):
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, embed=embed, t=t)
        model_std = torch.exp(0.5 * model_log_variance)
        noise = torch.randn_like(x)
        noise[t == 0] = 0  # no noise at the last denoising step
        return model_mean + model_std * noise

    @torch.no_grad()
    def conditional_sample(self, cond, init_st, wpt, n_samples=1):
        """Draws `n_samples` plans for a single conditioning input (batch of 1)."""
        device = self.betas.device
        shape = (n_samples, self.horizon, self.transition_dim)
        x = torch.randn(shape, device=device)

        embed = self.feature_extractor(cond, init_st, wpt).repeat(n_samples, 1)
        for i in reversed(range(0, self.n_timesteps)):
            t = torch.full((n_samples,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, embed, t)
        return x

    def forward(self, cond, init_st, wpt, n_samples=1):
        return self.conditional_sample(cond, init_st, wpt, n_samples=n_samples)

    # ------------------------------------------------------------------ training

    def q_sample(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, cond, init_st, wpt, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        embed = self.feature_extractor(cond, init_st, wpt)
        x_recon = self.model(x_noisy, embed, t)
        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            return self.loss_fn(x_recon, noise)
        return self.loss_fn(x_recon, x_start)

    def loss(self, x, cond, init_st, wpt):
        t = torch.randint(0, self.n_timesteps, (len(x),), device=x.device).long()
        return self.p_losses(x, cond, init_st, wpt, t)
