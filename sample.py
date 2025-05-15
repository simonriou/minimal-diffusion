from model import MinimalDiffusion
from diffusion import sample
from config import *
import matplotlib.pyplot as plt
import torchvision
import torch

import matplotlib
print(matplotlib.get_backend())

def main():
    print("Loading model...")
    model = MinimalDiffusion().to(DEVICE)
    model.load_state_dict(torch.load('model_epoch20.pth', map_location=DEVICE))
    model.eval()

    print("Sampling...")
    samples = sample(model, image_size=IMG_SIZE, n_samples=16)
    samples = (samples + 1) / 2  # rescale to [0, 1]

    grid = torchvision.utils.make_grid(samples, nrow=4)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.imsave("samples.png", grid.permute(1, 2, 0).cpu().numpy())
    print("Saved image to samples.png")

if __name__ == '__main__':
    main()