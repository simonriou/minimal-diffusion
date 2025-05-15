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

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

        # FiLM style conditioning on time embedding: produce scale and shift
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels * 2),
            nn.ReLU()
        )

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        # Time conditioning
        gamma_beta = self.time_mlp(t_emb)  # (batch, out_channels*2)
        gamma, beta = gamma_beta.chunk(2, dim=1)  # each (batch, out_channels)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        h = h * (1 + gamma) + beta
        h = F.relu(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = h * (1 + gamma) + beta
        h = F.relu(h)
        return h

class MinimalUNet(nn.Module):
    def __init__(self, img_channels=3, time_emb_dim=128, base_channels=64):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Encoder
        self.conv1 = ConvBlock(img_channels, base_channels, time_emb_dim)
        self.conv2 = ConvBlock(base_channels, base_channels * 2, time_emb_dim)
        self.conv3 = ConvBlock(base_channels * 2, base_channels * 4, time_emb_dim)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8, time_emb_dim)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.conv_up3 = ConvBlock(base_channels * 8, base_channels * 4, time_emb_dim)

        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.conv_up2 = ConvBlock(base_channels * 4, base_channels * 2, time_emb_dim)

        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.conv_up1 = ConvBlock(base_channels * 2, base_channels, time_emb_dim)

        # Final output
        self.final_conv = nn.Conv2d(base_channels, img_channels, 1)

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)

        # Encoder
        x1 = self.conv1(x, t_emb)       # (B, base, H, W)
        x2 = self.conv2(self.pool(x1), t_emb)  # (B, base*2, H/2, W/2)
        x3 = self.conv3(self.pool(x2), t_emb)  # (B, base*4, H/4, W/4)

        # Bottleneck
        b = self.bottleneck(self.pool(x3), t_emb)  # (B, base*8, H/8, W/8)

        # Decoder
        d3 = self.upconv3(b)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.conv_up3(d3, t_emb)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.conv_up2(d2, t_emb)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.conv_up1(d1, t_emb)

        return self.final_conv(d1)