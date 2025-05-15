import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalDiffusion(nn.Module):
    def __init__(self, img_channels=3, time_emb_dim=64):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.conv1 = nn.Conv2d(img_channels + time_emb_dim, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, img_channels, 3, padding=1)

    def forward(self, x, t):
        t = t.unsqueeze(-1).float() / 1000
        t_embed = self.time_embed(t).view(t.size(0), -1, 1, 1).repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, t_embed], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)