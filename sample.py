from model import SimpleDiffusionModel
from diffusion import sample
from config import *
import matplotlib.pyplot as plt
import torchvision

model = SimpleDiffusionModel().to(DEVICE)
model.load_state_dict(torch.load('model_epoch19.pt'))
model.eval()

samples = sample(model, image_size=IMG_SIZE, n_samples=16)
samples = (samples + 1) / 2  # to [0, 1]
grid = torchvision.utils.make_grid(samples, nrow=4)
plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
plt.axis('off')
plt.show()