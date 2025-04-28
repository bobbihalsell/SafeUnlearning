"""
Returns standard benchmark torchvision transformation pipeline for
CIFAR100.
"""

from torchvision import transforms

CIFAR100_IMAGE_SIZE = 32
CIFAR100_PADDING = 4
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def get_cifar100_train_transform():
    """
    Returns the standard preprocessing transform for Cifar100 training images
    """
    transform = transforms.Compose(
        [
            transforms.RandomCrop(CIFAR100_IMAGE_SIZE, padding=CIFAR100_PADDING),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
        ]
    )
    return transform


def get_cifar100_test_transform():
    """
    Returns the standard preprocessing transform for Cifar100 testing images
    """
    transform = transforms.Compose(
        [
            transforms.Resize(size=(CIFAR100_IMAGE_SIZE, CIFAR100_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
        ]
    )
    return transform
