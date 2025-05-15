from model import MinimalUNet
from diffusion import sample
from config import *
import matplotlib.pyplot as plt
import torchvision
import torch

import matplotlib
print(matplotlib.get_backend())

def main():
    print("Loading model...")
    model = MinimalUNet().to(DEVICE)
    model.load_state_dict(torch.load(f"checkpoints/model_epoch{EPOCHS-1}.pth", map_location=DEVICE), strict=False)
    model.eval()

    print("Sampling...")
    samples = sample(model, image_size=IMG_SIZE, n_samples=16)
    samples = (samples + 1) / 2  # scale to [0, 1]
    samples = torch.clamp(samples, 0.0, 1.0)  # ensure valid range

    grid = torchvision.utils.make_grid(samples, nrow=4)
    plt.imsave("samples.png", grid.permute(1, 2, 0).cpu().numpy())
    print("Saved image to samples.png")
    plt.show()

if __name__ == '__main__':
    main()