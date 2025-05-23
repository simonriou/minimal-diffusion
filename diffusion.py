import torch
from config import T, DEVICE
from utils import extract

# ======================
# Noise schedule
# ======================
beta = torch.linspace(1e-4, 0.02, T).to(DEVICE)
alpha = 1. - beta
alpha_cumprod = torch.cumprod(alpha, dim=0)
alpha_cumprod_prev = torch.cat([torch.tensor([1.0], device=DEVICE), alpha_cumprod[:-1]])

# Precomputed constants
sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
sqrt_one_minus_alpha_cumprod = torch.sqrt(1. - alpha_cumprod)
sqrt_recip_alpha = torch.sqrt(1.0 / alpha)
posterior_variance = beta * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)

# ======================
# Forward diffusion q(x_t | x_0)
# ======================
def q_sample(x_0, t, noise=None):
    """
    Sample x_t given x_0 and timestep t using the forward process.
    """
    if noise is None:
        noise = torch.randn_like(x_0)
    sqrt_alpha_cumprod_t = extract(sqrt_alpha_cumprod, t, x_0.shape)
    sqrt_one_minus_alpha_cumprod_t = extract(sqrt_one_minus_alpha_cumprod, t, x_0.shape)
    return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

# ======================
# Reverse process p(x_{t-1} | x_t)
# ======================
@torch.no_grad()
def p_sample(model, x_t, t):
    """
    One step of the reverse diffusion process.
    """
    t_tensor = torch.full((x_t.shape[0],), t, device=DEVICE, dtype=torch.long)

    # Predict noise
    predicted_noise = model(x_t, t_tensor)

    # Retrieve constants
    beta_t = extract(beta, t_tensor, x_t.shape)
    sqrt_one_minus_alpha_cumprod_t = extract(sqrt_one_minus_alpha_cumprod, t_tensor, x_t.shape)
    sqrt_recip_alpha_t = extract(sqrt_recip_alpha, t_tensor, x_t.shape)
    posterior_var_t = extract(posterior_variance, t_tensor, x_t.shape)

    # Compute model mean (µ_t)
    model_mean = sqrt_recip_alpha_t * (
        x_t - (beta_t / sqrt_one_minus_alpha_cumprod_t) * predicted_noise
    )

    # If t == 0, return mean directly (no noise), else sample from normal distribution
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_var_t) * noise

# ======================
# Full sampling loop
# ======================
@torch.no_grad()
def sample(model, image_size=32, n_samples=16):
    """
    Sample from the model starting from pure noise.
    """
    model.eval()
    x_t = torch.randn(n_samples, 3, image_size, image_size).to(DEVICE)
    for t_ in reversed(range(T)):
        x_t = p_sample(model, x_t, t_)
    return x_t