import torch
from torchvision.models import resnet18
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from src.unlearning.utils import remove_classes
from src.unlearning.gradient_ascent import GradientAscentUnlearner
from src.unlearning.eval import ClassificationEvaluator
import os


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
    train_dataset = Subset(train_dataset, list(range(1000)))
    test_dataset = datasets.CIFAR10(root="./data", train=False,
                                    download=True, transform=transform)
    test_dataset = Subset(test_dataset, list(range(1000)))

    return train_dataset, test_dataset

def test_gradient_ascent_cifar10():
    # A typical pipeline for a machine unlearning practitioner
    model_path = os.path.join(os.path.dirname(__file__), 'resnet18_ft_cifar10.pth')
    print(model_path)
    model = load_model(model_path)
    train_dataset, test_dataset = load_cifar10_datasets()
    # Pull out the forget set from the training data (Forget an entire class)
    forget_set_label = 0
    _, forget_set = remove_classes(dataset=train_dataset,
                                   forget_labels=[forget_set_label],
                                   return_forget=True)

    # Create dataloader, loss function and unlearner
    forget_dataloader = DataLoader(forget_set, batch_size=64, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    # Example where we perform gradient ascent unlearning
    unlearner = GradientAscentUnlearner(original_model=model)

    # Retrieve the unlearned model to save the model
    unlearned_model = unlearner.unlearn(forget_dataloader=forget_dataloader,
                                        loss_fn=loss_fn,
                                        num_epochs=5,
                                        lr=1e-4)

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
    # Check that GA unlearning working as expected
    # Values provided by initial tests
    assert results['unlearned_forget_acc'] < 0.3
    assert results['unlearned_retain_acc'] > 0.7
    # Baseline performance working properly
    assert results['original_retain_acc'] > 0.8
    assert results['original_forget_acc'] > 0.7
