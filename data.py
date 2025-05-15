from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def normalize(img):
    return img * 2. - 1.

def get_dataloader(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(normalize)
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)