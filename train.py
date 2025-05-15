import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy  # ← required!
from model import ImprovedUNetV2
from diffusion import q_sample
from data import get_dataloaders
from config import DEVICE, LR, EPOCHS, T, BATCH_SIZE

def train():
    model = ImprovedUNetV2().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # EMA model setup
    ema_model = deepcopy(model)
    ema_model.eval()  # ← Prevents training mode
    for param in ema_model.parameters():
        param.requires_grad_(False)  # ← Ensures no gradients are tracked
    ema_decay = 0.999

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    train_loader, val_loader = get_dataloaders(BATCH_SIZE)

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []

        for x_0, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x_0 = x_0.to(DEVICE)
            t = torch.randint(0, T, (x_0.size(0),), device=DEVICE)
            noise = torch.randn_like(x_0)
            x_t = q_sample(x_0, t, noise)
            predicted = model(x_t, t)
            loss = F.mse_loss(predicted, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update
            with torch.no_grad():
                for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                    ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_0, _ in val_loader:
                x_0 = x_0.to(DEVICE)
                t = torch.randint(0, T, (x_0.size(0),), device=DEVICE)
                noise = torch.randn_like(x_0)
                x_t = q_sample(x_0, t, noise)
                predicted = model(x_t, t)
                val_loss = F.mse_loss(predicted, noise)
                val_losses.append(val_loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f} | LR = {current_lr:.6f}")

        # Save models
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/model_epoch{epoch}.pth")
        torch.save(ema_model.state_dict(), f"checkpoints/ema_model_epoch{epoch}.pth")

if __name__ == "__main__":
    train()