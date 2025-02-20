import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.unlearning.base import BaseUnlearner
import copy


class GradientAscentUnlearner(nn.Module, BaseUnlearner):
    def __init__(self, original_model, unlearned_model=None, learning_rate=1e-4):
        nn.Module.__init__(self)
        BaseUnlearner.__init__(self)
        self.original_model = original_model
        self.unlearned_model = unlearned_model
        self.learning_rate = learning_rate
        if self.unlearned_model is None:
            self.original_model.train()

    def loss_steps(self, forget_dataloader, loss_fn, num_epochs):
        """
        Perform gradient ascent on model parameters using forget set.

        Will not work if self.unlearned_model is already defined.

        Args:
            forget_dataloader (torch.utils.data.DataLoader): The dataloader for the forget set.
            loss_fn (torch.nn loss function): Loss function that uses logits (e.g. nn.CrossEntropyLoss)
            num_epochs (int): Number of passes over the forget set for unlearning.

        Returns:
            loss (float): The computed loss.
        """
        if self.unlearned_model is None:
            self.unlearned_model = copy.deepcopy(self.original_model)
            self.unlearned_model.to(self.device)
            for _ in range(num_epochs):
                for images, labels in forget_dataloader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    # Zero the gradients in the model's parameters
                    self.unlearned_model.zero_grad()
                    # Forward pass through the model
                    output = self.unlearned_model(images)
                    loss = loss_fn(output, labels)
                    loss.backward()

                    # Perform gradient ascent
                    with torch.no_grad():
                        for param in self.unlearned_model.parameters():
                            if param.grad is not None:
                                param += self.learning_rate * param.grad

            # Return the model
            return self.unlearned_model

        else:
            raise Exception("""Trying to perform unlearning when an unlearning model has already been provided!
                            Instantiate this object without passing in unlearned_model if you wish to perform unlearning.""")

    def get_unlearning_performance(self, retain_dataloader, forget_dataloader, verbose=False):
        if self.unlearned_model is not None:
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
                print(f'Original model retain accuracy: {org_retain_accuracy}%')
                print(f'Original model forget accuracy: {org_forget_accuracy}%')
                print(f'Unlearned model retain accuracy: {un_retain_accuracy}%')
                print(f'Unlearned model forget accuracy: {un_forget_accuracy}%')

            return results

        else:
            raise Exception('Unlearned model has either not been trained or not been provided. Please provide one first.')
