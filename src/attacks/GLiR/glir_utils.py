from torch.utils.data import Subset
import numpy as np
from sklearn.metrics import (precision_score, recall_score, 
                             f1_score, accuracy_score)
import torch 


class ConfigError(Exception):
    def __init__(self, message='Configuration .YAML specification error.'):
        super().__init__(message)


def compute_accuracy(model, dataloader, device):
    # device = model.device
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def create_subset(dataset, fraction, return_indices=False):
    """
    Creates a subloader with the given fraction of data from the original 
        loader.

    :param loader: The original DataLoader
    :param fraction: Fraction of the dataset to keep (e.g., 0.1 for 10%)
    :return: New DataLoader with a subset of data
    """
    total_size = len(dataset)
    subset_size = int(total_size * fraction)

    indices = np.random.choice(total_size, subset_size, replace=False)
    subset = Subset(dataset, indices)
    if return_indices:
        return subset, indices
    return subset


def create_matched_subset(dataset, target_size, exempt_points=[], test_size= None):
    """
    Creates a subset from the loader with exactly `target_size` samples.
    
    Args:
        loader: Original DataLoader.
        target_size: Desired number of samples 
            (must be <= len(loader.dataset)).
        return_loader: If True, returns a DataLoader; else returns a Subset.
    """
    available_test_indices = list(
                                set(range(len(dataset))) - set(exempt_points)
                                )
    # Ensure target_size does not exceed the number of available indices
    if test_size is None:
         target_size = min(target_size, len(available_test_indices))
    else: 
        target_size = test_size
   

    indices = np.random.choice(available_test_indices, 
                               target_size, 
                               replace=False)
    subset = Subset(dataset, indices)

    return subset    


def calculate_metrics(y_true, y_pred):
    # Ensure the inputs are tensors
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)

    # Calculate TP, TN, FP, FN
    tp = torch.sum((y_true == 1) & (y_pred == 1))
    tn = torch.sum((y_true == 0) & (y_pred == 0))
    fp = torch.sum((y_true == 0) & (y_pred == 1))
    fn = torch.sum((y_true == 1) & (y_pred == 0))

    # Calculate accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(y_true.numpy(), y_pred.numpy())
    precision = precision_score(y_true.numpy(), y_pred.numpy())
    recall = recall_score(y_true.numpy(), y_pred.numpy())
    f1 = f1_score(y_true.numpy(), y_pred.numpy())

    # Return the summary of metrics
    return {
        "TP": tp.item(),
        "TN": tn.item(),
        "FP": fp.item(),
        "FN": fn.item(),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }
