from torchvision import transforms

IMAGENET_IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_imagenet_train_transform():
    transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGENET_IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform


def get_imagenet_test_transform():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGENET_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform
