from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_imagenet_test_transform():
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize to 256 pixels on the smaller side
        transforms.CenterCrop(224),  # Crop 224x224 center region
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform
