import argparse
import yaml
import torchvision
import torch
from torch.utils.data import DataLoader
import timm
from datasets import load_datasets as src_datasets
from datasets import preprocessing
from train.utils import setup_device, set_seed
import wandb
import os


class TrainApp:
    """ Perform pretraining, or model loading and saving, for a model."""
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Path to .yaml file with configurations for training."
            )
        parser.add_argument("--config_path",
                            type=str,
                            required=True,
                            help="Path to .yaml file for training configs.")
        args = parser.parse_args()

        # Load in the instructions for the unlearning run from a filepath
        with open(args.config_path, 'r') as f:
            try:
                config = yaml.safe_load(f)
            except FileNotFoundError:
                raise FileNotFoundError('.yaml file not found.')

        self.seed = config['seed']
        set_seed(self.seed)

        model_cfg = config['model']
        self.model_name = model_cfg['name']
        self.pretrained = model_cfg['pretrained']
        self.num_classes = model_cfg['num_classes']
        self.model_save_dir = model_cfg['save_dir']
        assert self.model_save_dir is not None

        dataset_cfg = config['dataset']
        self.dataset_name = dataset_cfg['name']
        self.dataset_save_dir = dataset_cfg['load_dir']

        self.batch_sizes = dataset_cfg['batch_sizes']

        self.train_cfg = model_cfg['train_cfg']

    def initialize_model(self):
        """Initialize the model based on model name from user configuration."""
        if hasattr(torchvision.models, self.model_name):
            model = torchvision.models.get_model(
                self.model_name,
                weights="DEFAULT" if self.pretrained else None,
            )

            # Adjust the last layer based on model type
            if hasattr(model, "fc"):  # ResNet-style
                model.fc = torch.nn.Linear(
                    model.fc.in_features,
                    self.num_classes
                )
            elif hasattr(model, "classifier"):
                # MobileNet, EfficientNet, VGG, DenseNet
                if isinstance(model.classifier, torch.nn.Sequential):
                    # Handle cases where classifier is Sequential
                    last_layer_idx = len(model.classifier) - 1
                    model.classifier[last_layer_idx] = torch.nn.Linear(
                        model.classifier[last_layer_idx].in_features,
                        self.num_classes
                    )
                else:
                    model.classifier = torch.nn.Linear(
                        model.classifier.in_features,
                        self.num_classes
                    )
            else:
                raise AttributeError("Unknown classification layer "
                                     f'for {self.model_name}')

        else:
            print(f'Could not find {self.model_name} in torchvision.'
                  ' Looking in timm.')
            try:
                model = timm.create_model(
                    self.model_name,
                    pretrained=self.pretrained,
                    num_classes=self.num_classes,
                )
            except Exception:
                raise AttributeError(f"{self.model_name} not found.")

        return model

    def initialize_datasets(self):
        """ Load in the dataset from the folder"""
        if 

    def convert_to_dataloaders(self,
                               train_dataset,
                               val_dataset,
                               test_dataset):
        """ Convert the datasets to dataloaders before training."""
        train_batch_size = self.batch_sizes['train']
        val_batch_size = self.batch_sizes['val']
        test_batch_size = self.batch_sizes['test']

        train_dl = DataLoader(train_dataset,
                              batch_size=train_batch_size,
                              shuffle=True)
        val_dl = DataLoader(val_dataset,
                            batch_size=val_batch_size,
                            shuffle=False)
        test_dl = DataLoader(test_dataset,
                             batch_size=test_batch_size,
                             shuffle=False)

        return train_dl, val_dl, test_dl

    def pretrain(self):
        """ Perform pretraining of a model on a dataset."""
        lr = self.train_cfg['lr']
        num_epochs = self.train_cfg['epochs']
        weight_decay = self.train_cfg['weight_decay']

        wandb.init(
            project="TEST",
            config={
                "epochs": num_epochs,
                "batch_size": self.batch_sizes['train'],
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "model_name": self.model_name,
                "num_classes": self.num_classes,
            },
        )
        # Step 1: Initialize the model
        model = self.initialize_model()
        device = setup_device()
        model.to(device)

        # Step 2: Load datasets and dataloaders
        train_dataset, val_dataset, test_dataset = self.initialize_datasets()
        train_dl, val_dl, _ = self.convert_to_dataloaders(train_dataset,
                                                          val_dataset,
                                                          test_dataset)

        # Step 3: Set up loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)

        # Training configuration
        best_val_loss = float("inf")
        # Create the directory if it does not exist
        os.makedirs(self.model_save_dir, exist_ok=True)
        save_path = self.model_save_dir + f'/{self.model_name}_{self.seed}_original.pt'

        print(f"Starting training with device {device}...")
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            correct, total = 0, 0

            for batch in train_dl:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_loss /= len(train_dl.dataset)
            train_acc = 100.0 * correct / total

            wandb.log({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "epoch": epoch + 1
            })

            # Validation phase
            val_loss, val_acc = self.eval_model(criterion, model, val_dl)

            wandb.log({
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "epoch": epoch + 1
            })

            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                print(f"New best model saved at {save_path}")

        print("Training complete.")
        wandb.finish()

    def eval_model(self, criterion, model, val_dl):
        model.eval()
        device = setup_device()
        val_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for batch in val_dl:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_dl.dataset)
        val_acc = 100.0 * correct / total

        return val_loss, val_acc


if __name__ == '__main__':
    trainer = TrainApp()
    trainer.pretrain()
