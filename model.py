import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import get_num_groups

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

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

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(get_num_groups(in_channels), in_channels)
        self.act1 = SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(get_num_groups(out_channels), out_channels)
        self.act2 = SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2)
        )

        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, t_emb):
        h = self.conv1(self.act1(self.norm1(x)))
        gamma, beta = self.time_mlp(t_emb).chunk(2, dim=1)
        gamma, beta = gamma[..., None, None], beta[..., None, None]
        h = h * (1 + gamma) + beta

        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.res_conv(x)

class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(get_num_groups(channels), channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        q = self.q(x_norm)
        k = self.k(x_norm)
        v = self.v(x_norm)

        attn = (q * k).sum(dim=1, keepdim=True) / math.sqrt(C)
        attn = attn.softmax(dim=-1)

        out = self.proj(v * attn)
        return x + out

class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        return self.conv(self.upsample(x))

class ImprovedUNetV2(nn.Module):
    def __init__(self, img_channels=3, time_emb_dim=128, base_channels=64):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Encoder
        self.conv1 = ResidualConvBlock(img_channels, base_channels, time_emb_dim)
        self.conv2 = ResidualConvBlock(base_channels, base_channels * 2, time_emb_dim)
        self.conv3 = ResidualConvBlock(base_channels * 2, base_channels * 4, time_emb_dim)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck1 = ResidualConvBlock(base_channels * 4, base_channels * 8, time_emb_dim)
        self.attn = SpatialAttention(base_channels * 8)
        self.bottleneck2 = ResidualConvBlock(base_channels * 8, base_channels * 8, time_emb_dim)

        # Decoder
        self.upconv3 = UpsampleConv(base_channels * 8, base_channels * 4)
        self.conv_up3 = ResidualConvBlock(base_channels * 8, base_channels * 4, time_emb_dim)

        self.upconv2 = UpsampleConv(base_channels * 4, base_channels * 2)
        self.conv_up2 = ResidualConvBlock(base_channels * 4, base_channels * 2, time_emb_dim)

        self.upconv1 = UpsampleConv(base_channels * 2, base_channels)
        self.conv_up1 = ResidualConvBlock(base_channels * 2, base_channels, time_emb_dim)

        # Final output
        self.final_norm = nn.GroupNorm(get_num_groups(base_channels), base_channels)
        self.final_act = SiLU()
        self.final_conv = nn.Conv2d(base_channels, img_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)

        x1 = self.conv1(x, t_emb)
        x2 = self.conv2(self.pool(x1), t_emb)
        x3 = self.conv3(self.pool(x2), t_emb)

        b = self.bottleneck1(self.pool(x3), t_emb)
        b = self.attn(b)
        b = self.bottleneck2(b, t_emb)

        d3 = self.upconv3(b)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.conv_up3(d3, t_emb)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.conv_up2(d2, t_emb)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.conv_up1(d1, t_emb)

        out = self.final_act(self.final_norm(d1))
        return self.final_conv(out)