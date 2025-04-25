from torch.utils.data import Subset
import numpy as np
from sklearn.metrics import (precision_score, recall_score, 
                             f1_score, accuracy_score)
import torch 


def compute_accuracy(model, dataloader, device):
    """
    Computes the classification accuracy of a model on a given dataset.

    Args:
        model (torch.nn.Module): Trained model to evaluate.
        dataloader (DataLoader): DataLoader providing the evaluation data.
        device (torch.device): Device on which to perform computations.

    Returns:
        float: Accuracy score (correct / total).
    """
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
    Creates a subset of a dataset by randomly selecting a fraction of samples.

    Args:
        dataset (Dataset): The original dataset.
        fraction (float): Fraction of the dataset to include (e.g., 0.1 for 10%).
        return_indices (bool): If True, also return the indices used.

    Returns:
        Subset: A PyTorch Subset of the original dataset.
        (optional) list of int: Indices of selected samples if return_indices is True.
    """
    total_size = len(dataset)
    subset_size = int(total_size * fraction)

    indices = np.random.choice(total_size, subset_size, replace=False)
    subset = Subset(dataset, indices)
    if return_indices:
        return subset, indices
    return subset


def create_matched_subset(dataset, target_size, 
                          exempt_points=[], test_size=None):
    """
    Creates a subset of specified size, excluding certain indices.

    Args:
        dataset (Dataset): The original dataset.
        target_size (int): Number of samples to select.
        exempt_points (list, optional): Indices to exclude from selection.
        test_size (float, optional): If provided, overrides target_size with a fraction.

    Returns:
        Subset: A PyTorch Subset with selected samples.
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
    """
    Calculates classification performance metrics.

    Args:
        y_true (array-like): Ground truth labels (binary).
        y_pred (array-like): Predicted labels (binary).

    Returns:
        dict: Dictionary containing:
            - "TP": True Positives
            - "TN": True Negatives
            - "FP": False Positives
            - "FN": False Negatives
            - "Accuracy": Accuracy score
            - "Precision": Precision score
            - "Recall": Recall score
            - "F1": F1 score
    """
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
