# Used only if the model has not been pretrained, so we need to train to get our original model.
import torch.nn as nn
import torch
from unlearning.utils import setup_device

class Trainer:
    """ Train a model on a dataset from randomly initialized weights."""
    def __init__(self):
        self.device = setup_device()

    def train_model(self, model, epochs, criterion, 
                    optimizer, train_loader, val_loader=None):
        """Train the model using provided train_loader and validation loader."""
        # Move model to device
        model = model.to(self.device)
        model.train()

        # Training loop
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            # Print epoch statistics
            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct / total
            print(f'Epoch [{epoch+1}/{epochs}] - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')

            # Validate if validation loader is provided
            if val_loader:
                val_loss, val_acc = self.evaluate_model(model, val_loader, criterion)
                print(f'Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
                model.train()

        return model

    def evaluate_model(self, model, test_loader, criterion=None):
        """Evaluate the model on the test set."""
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        model.eval()
        model = model.to(self.device)
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
  
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_loss = test_loss / len(test_loader)
        accuracy = 100. * correct / total

        return test_loss, accuracy
