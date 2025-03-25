import torch
import torch.nn as nn
from src.unlearning.utils import setup_device, UnsupportedModelError

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