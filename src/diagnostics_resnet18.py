import torchvision
import timm


def init_resnet18(pretrained: bool):
    model = torchvision.models.get_model(
        'mobilenet_v2',
        weights="DEFAULT" if pretrained else None,
        )

    return model


if __name__ == '__main__':
    resnet18 = init_resnet18(pretrained=False)
