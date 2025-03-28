# Used only if the model has not been pretrained, so we need to train to get our original model.
import torch.nn as nn
import torch
from unlearning.utils import setup_device

class Trainer:
    """ Train a model on a dataset from randomly initialized weights."""
    def __init__(self, model: nn.Module):
        self.device = setup_device()
        if not isinstance(model, nn.Module):
            raise TypeError('Provided model must be a nn.Module.')
        self.model = model
        self.model.to(self.device)

    def initialize_training_params(self, train_cfg):
        """Extract training parameters and initialize optimizer.

        Args:
            train_cfg (dict): A dictionary provided from the YAML file.

        Returns:
            epochs (int): Number of training epochs.
            criterion (nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Initialized optimizer.
        """
        self.epochs = train_cfg.get('epochs')
        # Loss function
        criterion = train_cfg.get('loss_fn')
        if criterion == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError('Unsupported loss function. Only cross_entropy is supported.')
        # Optimizer selection
        optimizer_type = train_cfg.get('optimizer')
        if optimizer_type == 'sgd':
            valid_sgd_keys = {'lr', 'momentum', 'weight_decay', 'dampening', 'nesterov'}
            sgd_params = {k: v for k, v in train_cfg.items() if k in valid_sgd_keys}

            self.optimizer = torch.optim.SGD(self.model.parameters(), **sgd_params)

        elif optimizer_type == 'adam':
            valid_adam_keys = {'lr', 'betas', 'eps', 'weight_decay', 'amsgrad'}
            adam_params = {k: v for k, v in train_cfg.items() if k in valid_adam_keys}

            self.optimizer = torch.optim.Adam(self.model.parameters(), **adam_params)

        else:
            raise ValueError(f"Unsupported optimizer '{optimizer_type}'. Only 'sgd' and 'adam' are supported.")

    def train_model(self, train_cfg, train_loader, val_loader=None):
        """Train the model using training configurations and data loaders."""
        # Always initialize the training parameters first
        self.initialize_training_params(train_cfg)
        
        self.model.train()
        # Training loop
        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            # Print epoch statistics
            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct / total
            print(f'Epoch [{epoch+1}/{self.epochs}] - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')

            # Validate if validation loader is provided
            if val_loader:
                val_loss, val_acc = self.evaluate_model(self.model, val_loader, self.criterion)
                print(f'Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
                self.model.train()

        return self.model

    def evaluate_model(self, test_loader):
        """Evaluate the model on the test set."""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_loss /= len(test_loader)
        accuracy = 100. * correct / total

        return test_loss, accuracy
