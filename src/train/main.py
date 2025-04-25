import torch
import wandb
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue
from importmodel import ImportModel
from utils import setup_device, set_seed
from utils import initialize_dataloaders as init_dataloaders
from train.config_validation import TrainValidator
import time


class TrainApp(TrainValidator):
    """ Perform pretraining, or model loading and saving, for a model."""
    def __init__(self, config: DictConfig):
        super().__init__(config)
        config = OmegaConf.to_container(config, resolve=True)
        self.config = config
        self.seed = config['seed']
        self.verbose = config['verbose']
        set_seed(self.seed)

    def load_model(self):
        """Initialize the model based on model name from user configuration."""
        importer = ImportModel(
            load_method=self.load_method,
            model_name=self.model_name,
            num_classes=self.num_classes,
            init_path=self.init_path,
            model_ckpt_path=self.original_model_ckpt_path,
            model_kwargs=self.model_kwargs,
            from_pretrained=self.pretrained,
            )
        model = importer.model
        return model

    def initialize_dataloaders(self):
        """ 
        Initialize dataloaders from the dataset folder.

        Returns: 
            dict: Dictionary with 'train' and 'val' DataLoader objects.
        """
        dataloaders = init_dataloaders(splits=['train', 'val'],
                                       batch_sizes=self.batch_sizes,
                                       num_workers=self.num_workers,
                                       dataset_name=self.dataset_name,
                                       dataset_save_dir=self.dataset_save_dir)
        return dataloaders

    def reinitialize_checkpoints(self, model, optimizer):
        """ 
        Load in model and optimizer state dict from a checkpoint.

        Args:
            model (torch.nn.Module): The model to load weights into.
            optimizer (torch.optim.Optimizer): The optimizer to load state into.

        Returns:
            model, optimizer, start_epoch with loaded weights and epoch count
        
        """
        if self.from_checkpoint:
            # Load the checkpoint state_dict
            checkpoint = torch.load(self.checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])

            # Optionally, load optimizer state_dict to resume training
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
        # Step 1: Initialize the model and WandB configs
        model = self.load_model()
        device = setup_device()
        model.to(device)
        if self.wandb_enabled:
            wandb.init(
                project=self.project_name,
                id=self.run_id,
                config=self.config,
                resume='allow'  # Allow to resume training from checkpoint
            )

        # Step 2: Load datasets and dataloaders
        dataloaders = self.initialize_dataloaders()
        train_dl = dataloaders['train']
        val_dl = dataloaders['val']

        # Step 3: Set up loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        # Create the save directory
        os.makedirs(self.model_save_dir, exist_ok=True)
        save_path = (self.model_save_dir +
                     f'/{self.model_name}_{self.seed}_original.pt')

        if self.epochs == 0:
            # Handle case where user just wants to download pretrained weights
            checkpoint = {
                'model_state_dict': model.state_dict(),
            }
            torch.save(checkpoint, save_path)
            print(f'Model downloaded without training at {save_path}.')
            return None

        start_epoch = 0
        if self.from_checkpoint:
            model, optimizer, start_epoch = self.reinitialize_checkpoints(
                model=model,
                optimizer=optimizer)

        # Evaluate initial model performance
        self._eval_initial_model(
            criterion,
            model,
            train_dl,
            val_dl
        )

        # Training configuration
        best_val_loss = float("inf")

        print(f"Starting training with device {device}...")
        for epoch in range(self.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            correct, total = 0, 0

            start_time = time.time()
            for i, batch in enumerate(train_dl):
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
            train_time = time.time() - start_time

            if self.wandb_enabled:
                wandb.log({
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "train_time": train_time,
                    "epoch": start_epoch + epoch + 1
                })
            # Validation phase
            val_loss, val_acc, eval_time = self.eval_model(criterion,
                                                           model,
                                                           val_dl)
            if self.wandb_enabled:
                wandb.log({
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "eval_time": eval_time,
                    "epoch": start_epoch + epoch + 1
                })
            if self.verbose:
                print(f"Epoch {start_epoch+epoch+1}/{start_epoch+self.epochs} ")
                print(f'Train Loss: {train_loss:.4f} '
                      f'Acc: {train_acc:.2f}%. '
                      f'Time: {train_time:.1f} s', end=' || ')
                print(f'Val Loss: {val_loss:.4f} '
                      f'Acc: {val_acc:.2f}%. '
                      f'Time: {eval_time:.1f} s', end=' || ')            

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

    def eval_model(self, criterion, model, val_dl):

        """
        Evaluates the model on the validation set.

        Args:
            criterion: The loss function (e.g., CrossEntropyLoss).
            model (torch.nn.Module): The model to evaluate.
            val_dl (DataLoader): DataLoader for the validation set.

        Returns:
            validation_loss, validation_accuracy
        """
        model.eval()
        device = setup_device()
        val_loss = 0.0
        correct, total = 0, 0

        start_time = time.time()
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
        time_taken = time.time() - start_time

        return val_loss, val_acc, time_taken

    def _eval_initial_model(self, criterion, model, train_dl, val_dl):
        initial_train_loss, initial_train_acc, time_train = self.eval_model(
            criterion,
            model,
            train_dl)
        initial_val_loss, initial_val_acc, time_val = self.eval_model(
            criterion,
            model,
            val_dl)
        if self.verbose:
            print(f'Initial Train Loss: '
                  f'{initial_train_loss:.4f}. '
                  f'Acc: {initial_train_acc:.2f}%. '
                  f'Time: {time_train:2f}s.', end=' || ')
            print(f'Initial Val Loss: '
                  f'{initial_val_loss:.4f}. '
                  f'Acc: {initial_val_acc:.2f}%. '
                  f'Time: {time_val:2f}s ', end=' || ')
        if self.wandb_enabled:
            wandb.log({
                'Initial Train Loss': initial_train_loss,
                'Initial Train Acc': initial_train_acc
            })
            wandb.log({
                'Initial Val Loss': initial_val_loss,
                'Initial Val Acc': initial_val_acc
            })


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
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
    if trainer.wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
