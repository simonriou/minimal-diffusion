from model import MinimalUNet
from diffusion import q_sample, beta, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod
from data import get_dataloader
from config import *
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

def train():
    model = MinimalUNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    dataloader = get_dataloader(BATCH_SIZE)

    for epoch in range(EPOCHS):
        for x_0, _ in tqdm(dataloader):
            x_0 = x_0.to(DEVICE)
            t = torch.randint(0, T, (x_0.size(0),), device=DEVICE)
            noise = torch.randn_like(x_0)
            x_t = q_sample(x_0, t, noise)
            predicted = model(x_t, t)
            loss = F.mse_loss(predicted, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/model_epoch{epoch}.pth")

if __name__ == "__main__":
    train()