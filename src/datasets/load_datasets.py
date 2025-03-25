import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset


def load_cifar10_datasets():
    transform = transforms.Compose([
        transforms.Resize(224),  # Resize CIFAR images to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                             std=[0.229, 0.224, 0.225])   # ImageNet std
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True,
                                     download=True, transform=transform)
    train_dataset = Subset(train_dataset, list(range(500)))
    test_dataset = datasets.CIFAR10(root="./data", train=False,
                                    download=True, transform=transform)
    test_dataset = Subset(test_dataset, list(range(500)))

    return train_dataset, test_dataset