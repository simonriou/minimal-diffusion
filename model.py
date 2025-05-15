import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalDiffusion(nn.Module):
    def __init__(self, img_channels=3, time_emb_dim=256):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.net = nn.Sequential(
            nn.Conv2d(img_channels + 1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, img_channels, 3, padding=1),
        )

    def forward(self, x, t):
        t = t.unsqueeze(-1).float() / 1000  # scale by T
        t_embed = self.time_embed(t).view(t.size(0), -1, 1, 1).repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, t_embed[:, :1]], dim=1)
        return self.net(x)