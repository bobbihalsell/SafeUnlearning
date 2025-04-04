import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def plot(losses):
    """
    Plots loss curves for different dataset subsets over epochs.
    
    Args:
        losses (dict): Dictionary where keys are dataset subset names (e.g., 'retain_losses', 'forget_losses')
                        and values are lists of losses over epochs.
    """

    plt.figure(figsize=(10, 6))
    
    for subset, loss_values in losses.items():
        loss_values = [loss.detach().cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in loss_values]
        plt.plot(range(1, len(loss_values) + 1), loss_values, label=subset)
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curves for Different Data Subsets")
    plt.legend()
    plt.grid()
    plt.show()
