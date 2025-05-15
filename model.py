import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = SiLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = SiLU()

        # Time embedding for FiLM conditioning (scale and shift)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2)
        )

        # If channel dimension changes, use a 1x1 conv for residual connection
        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.res_conv = nn.Identity()

        # Initialize second conv weights to zero for stable residual training
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)

        gamma_beta = self.time_mlp(t_emb)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        h = h * (1 + gamma) + beta

        h = self.act1(h)

        h = self.conv2(h)
        h = self.norm2(h)
        # Reuse the same gamma, beta for second conv (could also use another MLP for more expressiveness)
        h = h * (1 + gamma) + beta
        h = self.act2(h)

        return h + self.res_conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W)  # (B, C, N)
        qkv = self.qkv(h)  # (B, 3C, N)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)

        attn = torch.einsum('bhcn,bhcm->bhnm', q * self.scale, k * self.scale)
        attn = torch.softmax(attn, dim=-1)

        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.reshape(B, C, H * W)

        out = self.proj(out)
        out = out.view(B, C, H, W)

        return x + out

class ImprovedUNet(nn.Module):
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

        # Bottleneck with attention
        self.bottleneck = ResidualConvBlock(base_channels * 4, base_channels * 8, time_emb_dim)
        self.attn = AttentionBlock(base_channels * 8)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.conv_up3 = ResidualConvBlock(base_channels * 8, base_channels * 4, time_emb_dim)

        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.conv_up2 = ResidualConvBlock(base_channels * 4, base_channels * 2, time_emb_dim)

        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.conv_up1 = ResidualConvBlock(base_channels * 2, base_channels, time_emb_dim)

        # Final output conv
        self.final_conv = nn.Conv2d(base_channels, img_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)

        x1 = self.conv1(x, t_emb)
        x2 = self.conv2(self.pool(x1), t_emb)
        x3 = self.conv3(self.pool(x2), t_emb)

        b = self.bottleneck(self.pool(x3), t_emb)
        b = self.attn(b)

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