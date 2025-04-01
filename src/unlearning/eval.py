import torch
import torch.nn as nn
from unlearning.utils import setup_device, UnsupportedModelError
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

class ClassificationEvaluator:
    """An evaluator that returns accuracy for multiple models on various datasets."""

    def __init__(self, **models):
        """Initialize the evaluator with any number of models."""
        self.device = setup_device()
        self.models = {}

        for name, model in models.items():
            if not isinstance(model, nn.Module):
                raise UnsupportedModelError(f'Model "{name}" is not a valid PyTorch model.')
            self.models[name] = model.to(self.device)

    def get_model_accuracy(self, model: nn.Module, dataloader: torch.utils.data.DataLoader):
        """Compute model accuracy over a dataset.

        Args:
            model (torch.nn.Module): A classification PyTorch model.
            dataloader (torch.utils.data.DataLoader): A dataloader for evaluation.

        Returns:
            float: Accuracy of the model on the provided dataset.
        """
        total, correct = 0, 0
        model.eval()

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predictions = outputs.max(1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        return correct / total if total > 0 else 0.0

    def compare_accuracy(self, verbose=False, **dataloaders):
        """Compare model accuracy across multiple datasets.

        Args:
            verbose (bool, optional): Whether to print the accuracy results.
            **dataloaders: Named dataloaders to evaluate.

        Returns:
            dict: A dictionary with accuracy values for each dataset and model.
        """
        if not self.models:
            raise ValueError("No models have been provided for evaluation.")

        results = {}

        for model_name, model in self.models.items():
            model.to(self.device)

            for dataset_name, dataloader in dataloaders.items():
                acc = self.get_model_accuracy(model, dataloader)
                results[f'{model_name}_{dataset_name}_acc'] = acc

                if verbose:
                    print(f'{model_name} accuracy on {dataset_name}: {acc:.4f}')

        return results
    
