import torch
from torchvision.models import resnet18
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from src.unlearning.preprocessing import remove_classes
from src.unlearning.neggrad import NegGrad, NegGradPlus
from src.unlearning.eval import ClassificationEvaluator
from src.unlearning.utils import UnsupportedModelError
from sklearn.linear_model import LogisticRegression
import os
import pytest


def load_model(filepath: str):
    weights = torch.load(filepath, weights_only=True)
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, out_features=10)  # For CIFAR10
    model.load_state_dict(weights)

    return model


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


@pytest.fixture(scope='module')
def model():
    model_path = os.path.join(os.path.dirname(__file__), 'resnet18_ft_cifar10.pth')
    model = load_model(model_path)
    yield model  # Provide the model to the test
    del model


@pytest.mark.integration
def test_neggrad(model):
    """ Test base NegGrad on CIFAR10 works acceptably."""
    train_dataset, test_dataset = load_cifar10_datasets()
    # Pull out the forget set from the training data (Forget an entire class)
    forget_set_label = 0
    _, forget_set = remove_classes(dataset=train_dataset,
                                   forget_labels=[forget_set_label],
                                   return_forget=True)

    # Create dataloader, loss function and unlearner
    forget_dataloader = DataLoader(forget_set, batch_size=64, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    unlearner = NegGrad(original_model=model,
                        forget_dataloader=forget_dataloader)

    # Retrieve the unlearned model to save the model
    unlearned_model = unlearner.unlearn(loss_fn=loss_fn,
                                        num_epochs=3,
                                        lr=1e-3,
                                        weight_decay=0)

    # Filter the test data to compare original and unlearned performance on retain/forget set
    retain_test_set, forget_test_set = remove_classes(test_dataset,
                                                      forget_labels=[forget_set_label],
                                                      return_forget=True)

    retain_dataloader = DataLoader(retain_test_set, batch_size=32, shuffle=False)
    forget_dataloader = DataLoader(forget_test_set, batch_size=32, shuffle=False)

    # For now we can only report accuracy, other metrics to be implemented.
    evaluator = ClassificationEvaluator(original_model=model,
                                        unlearned_model=unlearned_model)

    results = evaluator.compare_accuracy(retain_dataloader=retain_dataloader,
                                         forget_dataloader=forget_dataloader,
                                         verbose=True)

    assert isinstance(results, dict)
    # Values provided by initial tests
    assert results['original_forget_acc'] > 0.7
    assert results['unlearned_forget_acc'] < 0.4
    assert results['original_retain_acc'] > 0.8
    assert results['unlearned_retain_acc'] > 0.7


@pytest.mark.integration
def test_neggradplus(model):
    """ Test NegGrad+ on CIFAR10 works acceptably."""
    train_dataset, test_dataset = load_cifar10_datasets()
    # Pull out the forget set from the training data (Forget an entire class)
    forget_set_label = 0
    retain_set, forget_set = remove_classes(dataset=train_dataset,
                                            forget_labels=[forget_set_label],
                                            return_forget=True)

    # Create dataloader, loss function and unlearner
    retain_dataloader = DataLoader(retain_set, batch_size=32, shuffle=True)
    forget_dataloader = DataLoader(forget_set, batch_size=32, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    unlearner = NegGradPlus(original_model=model,
                            forget_dataloader=forget_dataloader,
                            retain_dataloader=retain_dataloader)

    # Using similar hyperparameters as the NegGrad paper, more aggressive beta
    unlearned_model = unlearner.unlearn(loss_fn=loss_fn,
                                        num_epochs=2,
                                        lr=0.005,
                                        weight_decay=0.1,
                                        beta=0.99,
                                        use_l2_penalty=True)

    # Filter the test data to compare original and unlearned performance on retain/forget set
    retain_test_set, forget_test_set = remove_classes(test_dataset,
                                                      forget_labels=[forget_set_label],
                                                      return_forget=True)

    retain_dataloader = DataLoader(retain_test_set, batch_size=32, shuffle=False)
    forget_dataloader = DataLoader(forget_test_set, batch_size=32, shuffle=False)

    # For now we can only report accuracy, other metrics to be implemented.
    evaluator = ClassificationEvaluator(original_model=model,
                                        unlearned_model=unlearned_model)

    results = evaluator.compare_accuracy(retain_dataloader=retain_dataloader,
                                         forget_dataloader=forget_dataloader,
                                         verbose=True)

    assert isinstance(results, dict)
    assert results['original_forget_acc'] > 0.7
    assert results['unlearned_forget_acc'] < 0.2
    assert results['original_retain_acc'] > 0.8
    assert results['unlearned_retain_acc'] > 0.7


@pytest.mark.unit
@pytest.mark.parametrize("model", [
    None,  # Test a None type
    LogisticRegression(),  # Test a sklearn model
    1.3  # Test a float
])
def test_catch_bad_model(model):
    """ Test that unsupported models are appropriately handled."""
    train_dataset, _ = load_cifar10_datasets()

    forget_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    with pytest.raises(UnsupportedModelError) as exc_info:
        NegGrad(original_model=model,
                forget_dataloader=forget_dataloader)

    assert isinstance(exc_info.value, UnsupportedModelError)


@pytest.mark.unit
def test_neggradplus_fails_bad_retain(model):
    """ Test NegGrad throws error if beta > 0 (user intends to perform NegGrad+) without the retain set."""
    train_dataset, _ = load_cifar10_datasets()

    forget_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    with pytest.raises(TypeError) as exc_info:
        NegGradPlus(original_model=model,
                    forget_dataloader=forget_dataloader,
                    retain_dataloader=234)
    assert isinstance(exc_info.value, TypeError)


@pytest.mark.unit
@pytest.mark.parametrize("beta", [
    None,
    1.00001,
    -1
])
def test_neggradplus_fails_bad_beta(model, beta):
    train_dataset, _ = load_cifar10_datasets()
    forget_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    unlearner = NegGradPlus(original_model=model,
                            forget_dataloader=forget_dataloader,
                            retain_dataloader=forget_dataloader)

    with pytest.raises(ValueError) as exc_info:
        unlearner.unlearn(loss_fn=loss_fn,
                          num_epochs=5,
                          lr=1e-4,
                          beta=beta)

    assert isinstance(exc_info.value, ValueError)
