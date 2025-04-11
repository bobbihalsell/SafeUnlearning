import torchvision
from train.image_loading import RobustImageFolder
import torch
from torch.utils.data import DataLoader
import timm
from datasets.cifar10 import get_cifar10_test_transform
from datasets.cifar100 import get_cifar100_test_transform
from datasets.imagenet import get_imagenet_test_transform
from unlearning.utils import ConfigError
from train.utils import setup_device, set_seed
import wandb
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue


class TrainApp:
    """ Perform pretraining, or model loading and saving, for a model."""
    def __init__(self, config: DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
        self.seed = config['seed']
        set_seed(self.seed)

        model_cfg = config['model']
        self.model_name = model_cfg['name']
        self.pretrained = model_cfg['pretrained']
        self.num_classes = model_cfg['num_classes']
        self.model_save_dir = model_cfg['save_dir']
        self.freeze_all_except_last = model_cfg['freeze_all_except_classifier']
        assert self.model_save_dir is not None

        dataset_cfg = config['dataset']
        self.dataset_name = dataset_cfg['name']
        self.dataset_save_dir = dataset_cfg['load_dir']

        self.batch_sizes = dataset_cfg['batch_sizes']
        self.num_workers = dataset_cfg.get('num_workers', 1)

        self.train_cfg = model_cfg['train_cfg']
        self.run_id = config['run_id']

        self.checkpoint_path = model_cfg.get('checkpoint_path', None)
        self.from_checkpoint = False  # Flag to determine whether to train from checkpoint
        if self.checkpoint_path is not None:
            self.from_checkpoint = True

    def initialize_model(self):
        """Initialize the model based on model name from user configuration."""
        if hasattr(torchvision.models, self.model_name):
            model = torchvision.models.get_model(
                self.model_name,
                weights="DEFAULT" if self.pretrained else None,
            )

            if self.dataset_name != 'imagenet':  # Imagenet-1k
                print('Replacing default classification head...')
                # Adjust the last layer to match number of classes
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

    def freeze_all_except_classifier(self, model):
        """Unfreeze only the last (classifier) layer."""
        for name, param in model.named_parameters():
            if "classifier" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        return model

    def get_transform(self):
        if self.dataset_name == 'cifar10':
            return get_cifar10_test_transform()
        elif self.dataset_name == 'cifar100':
            return get_cifar100_test_transform()
        elif self.dataset_name == 'imagenet':
            return get_imagenet_test_transform()
        else:
            raise ConfigError(f'dataset_name {self.dataset_name} '
                              ' not supported.')

    def initialize_dataloaders(self):
        """ Initialize dataloaders from the ImageNet dataset folder."""
        transform = self.get_transform()

        # Load datasets for each split
        splits = ['train', 'val']
        dataloaders = {}

        for split in splits:
            batch_size = self.batch_sizes[split]
            split_dir = os.path.join(self.dataset_save_dir, split)
            if not os.path.exists(split_dir):
                raise Exception(f'{split_dir} does not exist. Is the dataset '
                                'in ImageFolder format?')

            dataset = RobustImageFolder(root=split_dir, transform=transform)
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),  # Only shuffle train set
                num_workers=self.num_workers,
                pin_memory=True
            )

        return dataloaders

    def reinitialize_checkpoints(self, model, optimizer):
        """ Load in model and optimizer state dict from a checkpoint."""
        if self.from_checkpoint:
            # Load the checkpoint state_dict
            checkpoint = torch.load(self.checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])

            # Optionally, load optimizer state_dict if you want to resume training
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print(f"Loaded model from checkpoint: {self.checkpoint_path}")

            epoch = checkpoint.get('epoch', 0)

        else:
            raise Exception('reinitialize_checkpoint_states should not be '
                            'called if the user does not want to train '
                            'from checkpoint.')

        return model, optimizer, epoch

    def pretrain(self):
        """ Perform pretraining of a model on a dataset."""
        lr = self.train_cfg['lr']
        num_epochs = self.train_cfg['epochs']
        weight_decay = self.train_cfg['weight_decay']

        # Step 1: Initialize the model
        model = self.initialize_model()
        device = setup_device()
        model.to(device)

        # Step 2: Load datasets and dataloaders
        dataloaders = self.initialize_dataloaders()
        train_dl = dataloaders['train']
        val_dl = dataloaders['val']

        # Step 3: Set up loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)

        # Create the save directory
        os.makedirs(self.model_save_dir, exist_ok=True)
        save_path = self.model_save_dir + f'/{self.model_name}_{self.seed}_original.pt'

        if num_epochs == 0:
            # Handle case where user just wants to download pretrained weights
            checkpoint = {
                'model_state_dict': model.state_dict(),
            }
            torch.save(checkpoint, save_path)
            print(f'Model downloaded without training at {save_path}.')
            return None

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
            id=str(self.run_id),
            resume="allow"
        )
        start_epoch = 0
        if self.from_checkpoint:
            model, optimizer, start_epoch = self.reinitialize_checkpoints(
                model=model,
                optimizer=optimizer)

        if self.freeze_all_except_last:
            model = self.freeze_all_except_classifier(model)

        # Training configuration
        best_val_loss = float("inf")

        print(f"Starting training with device {device}...")
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            correct, total = 0, 0

            for i, batch in enumerate(train_dl):
                print(f'Training Batch {i}...')
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
                "epoch": start_epoch + epoch + 1
            })

            # Validation phase
            val_loss, val_acc = self.eval_model(criterion, model, val_dl)

            wandb.log({
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "epoch": start_epoch + epoch + 1
            })

            print(f"Epoch {start_epoch+epoch+1}/{start_epoch+num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Create a checkpoint dictionary
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': start_epoch + epoch + 1,
                    'best_val_loss': best_val_loss,
                }

                # Save the checkpoint
                torch.save(checkpoint, save_path)
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


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print('============ Run Configuration ============')
    print(OmegaConf.to_yaml(cfg))
    print('============================================')
    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise MissingMandatoryValue(
            'Missing the following required arguments in the configuration: '
            f'{missing_keys}. \n'
            'Hint: python file.py key=value sets the appropriate value.')

    trainer = TrainApp(config=cfg)
    trainer.pretrain()


if __name__ == "__main__":
    main()
