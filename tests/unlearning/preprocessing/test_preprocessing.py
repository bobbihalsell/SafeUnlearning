from torchvision import datasets, transforms
import torch
from src.unlearning.preprocessing import remove_samples_by_indices, remove_classes
import pytest

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR dataset
transform = transforms.Compose([transforms.ToTensor()])
cifar10_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)


@pytest.mark.unit
def test_remove_samples_by_indices():
    # Test that we can remove samples by indices from the dataset.
    forget_indices = [1]
    retain_set, forget_set = remove_samples_by_indices(cifar10_train, forget_indices, 
                                        return_forget=True, verbose=False)

    assert forget_set[0][1] == cifar10_train[1][1]  # Labels match
    assert torch.equal(forget_set[0][0], cifar10_train[1][0]) # Feature match

    forget_indices = []
    retain_set, forget_set = remove_samples_by_indices(cifar10_train, forget_indices, 
                                        return_forget=True, verbose=False)
    assert len(forget_set) == 0
    assert len(retain_set) == len(cifar10_train)

    forget_indices = list(range(10))
    retain_set, forget_set = remove_samples_by_indices(cifar10_train, forget_indices, 
                                        return_forget=True, verbose=False)
    for i in range(len(forget_set)):
        assert torch.equal(forget_set[i][0], cifar10_train[i][0])
        assert forget_set[i][1] == cifar10_train[i][1]
        assert torch.equal(retain_set[i][0], cifar10_train[10+i][0])
        assert retain_set[i][1] == cifar10_train[10+i][1]


@pytest.mark.unit
def test_remove_classes():
    retain_set, forget_set = remove_classes(cifar10_train, [1], 
                                            return_forget=True, verbose=False)

    assert all([forget_set[i][1] == 1 for i in range(len(forget_set))])
    assert not any([retain_set[i][1] == 1 for i in range(len(retain_set))])
