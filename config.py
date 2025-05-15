import torch

T = 1000
BATCH_SIZE = 128
IMG_SIZE = 32
LR = 1e-4
EPOCHS = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'