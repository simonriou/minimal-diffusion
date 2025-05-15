import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

class MinimalDiffusion(nn.Module):
    def __init__(self, img_channels=3, time_emb_dim=64):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.conv1 = nn.Conv2d(img_channels + time_emb_dim, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, img_channels, 3, padding=1)

    def forward(self, x, t):
        """
        x: images tensor, shape (batch, channels, height, width)
        t: timesteps tensor, shape (batch,)
        """
        # Compute sinusoidal time embedding
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)  # project

        # Reshape to (batch, time_emb_dim, 1, 1) and repeat spatially
        t_emb = t_emb.view(t.size(0), -1, 1, 1).repeat(1, 1, x.size(2), x.size(3))

        # Concatenate time embedding to input
        x = torch.cat([x, t_emb], dim=1)

        # Pass through conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)