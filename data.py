from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def normalize(img):
    return img * 2. - 1.

def get_dataloaders(batch_size, val_split=0.1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(normalize)
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader