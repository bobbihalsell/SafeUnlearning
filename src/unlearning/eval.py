import torch
import torch.nn as nn
from src.unlearning.utils import setup_device, UnsupportedModelError


class ClassificationEvaluator:
    """ An evaluator that returns before/after accuracy for original and unlearned models."""
    def __init__(self, original_model: nn.Module, unlearned_model: nn.Module):
        if not isinstance(original_model, nn.Module) or not isinstance(unlearned_model, nn.Module):
            raise UnsupportedModelError('A provided model is not a Pytorch model.')

        self.device = setup_device()
        self.original_model = original_model
        self.unlearned_model = unlearned_model

    def get_model_accuracy(self, model: nn.Module, dataloader: torch.utils.data.DataLoader):
        """ Get model accuracy over a torch Dataloader.

        Args:
            model (torch.nn.Module): A classification Pytorch model.
            dataloader (torch.utils.data.Dataloader): A Dataloader for the dataset you want to compute accuracy on.

        Returns:
            accuracy (float): The model accuracy over the provided dataset.
        """
        total = 0
        correct = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predictions = outputs.max(1)
                correct += (predictions == labels).sum().item()
                total += len(labels)

        accuracy = correct / total if total > 0 else 0.0

        return accuracy

    def compare_accuracy(self,
                         retain_dataloader: torch.utils.data.DataLoader,
                         forget_dataloader: torch.utils.data.DataLoader, 
                         verbose: bool = False):
        """ Compare the before/after accuracy on retain and forget set for a classifier.

        Args:
            retain_dataloader (torch.utils.data.DataLoader): The retain set dataloader.
            forget_dataloader (torch.utils.data.DataLoader): The forget set dataloader.
            verbose (bool, optional): Whether to print change in accuray

        Returns:
            results (dict): A dictionary with 4 keys showing before/after results for 
            the classifier on the retain and forget set.
        """
        if self.unlearned_model is not None and self.original_model is not None:
            results = {}
            self.original_model.to(self.device)
            self.unlearned_model.to(self.device)
            # Gauge original model performance on test set without forget labels
            org_retain_accuracy = self.get_model_accuracy(self.original_model,
                                                          retain_dataloader)
            results['original_retain_acc'] = org_retain_accuracy
            # Gauge original model performance on test forget set
            org_forget_accuracy = self.get_model_accuracy(self.original_model,
                                                          forget_dataloader)
            results['original_forget_acc'] = org_forget_accuracy
            # Gauge unlearned model performance on test set without forget labels
            un_retain_accuracy = self.get_model_accuracy(self.unlearned_model,
                                                         retain_dataloader)
            results['unlearned_retain_acc'] = un_retain_accuracy
            # Gauge unlearned model performance on test forget set
            un_forget_accuracy = self.get_model_accuracy(self.unlearned_model,
                                                         forget_dataloader)
            results['unlearned_forget_acc'] = un_forget_accuracy
            if verbose:
                print(f'Original model retain accuracy: {org_retain_accuracy}')
                print(f'Original model forget accuracy: {org_forget_accuracy}')
                print(f'Unlearned model retain accuracy: {un_retain_accuracy}')
                print(f'Unlearned model forget accuracy: {un_forget_accuracy}')

            return results

        else:
            raise Exception('Unlearned model has either not been trained or not been provided. Please provide one first.')