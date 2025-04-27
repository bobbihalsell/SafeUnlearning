from .cifar10 import *
from .cifar100 import *
from .imagenet import *

DATASETS_TO_TRAIN_TRANSFORM = {
    "cifar5": get_cifar10_train_transform,
    "cifar10": get_cifar10_train_transform,
    "cifar100": get_cifar100_train_transform,
    "imagenet": get_imagenet_train_transform,
}


DATASETS_TO_TRANSFORM = {
    "cifar5": get_cifar10_test_transform,
    "cifar10": get_cifar10_test_transform,
    "cifar100": get_cifar100_test_transform,
    "imagenet": get_imagenet_test_transform,
}


DATASETS_TO_PARAMS = {
    "cifar5": (CIFAR10_MEAN, CIFAR10_STD, CIFAR10_IMAGE_SIZE),
    "cifar10": (CIFAR10_MEAN, CIFAR10_STD, CIFAR10_IMAGE_SIZE),
    "cifar100": (CIFAR100_MEAN, CIFAR100_STD, CIFAR100_IMAGE_SIZE),
    "imagenet": (IMAGENET_MEAN, IMAGENET_STD, IMAGENET_IMAGE_SIZE),
}
