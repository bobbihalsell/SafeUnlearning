import argparse
import yaml
import timm
import torch
import torch.nn as nn
import torchvision
from unlearning.utils import save_model, set_seed, setup_device
import os
from datasets.preprocessing import get_all_loaders, save_loaders, load_loaders
from unlearning.config_validation import InputValidator
from unlearning.finetune import FinetuneUnlearner
from unlearning.scrub import SCRUB
from unlearning.kunlearn import KUnlearn
from unlearning.neggrad import NegGrad, NegGradPlus
from datasets import load_datasets as src_datasets
from train.main import TrainApp


class UnlearnApp(InputValidator):
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Specify path to unlearning .yaml file.")
        parser.add_argument("--config_path",
                            type=str,
                            required=True,
                            help="Path to .yaml file")
        args = parser.parse_args()

        # Load in the instructions for the unlearning run from a filepath
        with open(args.config_path, 'r') as f:
            try:
                config = yaml.safe_load(f)
            except FileNotFoundError:
                print('.yaml file not found.')

        # Perform input validation first
        super().__init__(config)

        self.device = setup_device()
        print(f'Using device: {self.device}')
        self.seed = config['seed']
        set_seed(self.seed)

        self.unlearn_params['loss_fn'] = nn.CrossEntropyLoss()
        # Output directory
        self.output_dir = config.get('output_dir', 'artifacts/')
        os.makedirs(self.output_dir, exist_ok=True)

    def initialize_model(self):
        """Initialize the model based on model name from user configuration."""
        if hasattr(torchvision.models, self.model_name):
            model = torchvision.models.get_model(
                self.model_name,
                weights=None,
            )

            # Adjust the last layer based on model type
            if hasattr(model, "fc"):  # ResNet-style
                model.fc = torch.nn.Linear(
                    model.fc.in_features, self.num_classes
                )
            elif hasattr(model, "classifier"):  # MobileNet, EfficientNet, VGG, DenseNet
                if isinstance(model.classifier, torch.nn.Sequential):
                    # Handle cases like MobileNet where classifier is Sequential
                    last_layer_idx = len(model.classifier) - 1
                    model.classifier[last_layer_idx] = torch.nn.Linear(
                        model.classifier[last_layer_idx].in_features,
                        self.num_classes
                    )
                else:
                    model.classifier = torch.nn.Linear(
                        model.classifier.in_features, self.num_classes
                    )
            else:
                raise AttributeError(
                    f"Unknown classification layer for {self.model_name}"
                )

        else:
            print(f"Couldn't find {self.model_name} in torchvision. "
                  "Looking in timm.")
            try:
                model = timm.create_model(
                    self.model_name,
                    num_classes=self.num_classes,
                )
            except Exception:
                raise AttributeError(
                    f"{self.model_name} not found in torchvision or timm."
                )

        return model

    def initialize_unlearner(self):
        """ Initialize the correct unlearner from user specification."""
        if self.unlearner_name == 'finetune':
            unlearner = FinetuneUnlearner(
                self.device,
            )
        elif self.unlearner_name == 'neggrad':
            unlearner = NegGrad(
                self.device,
            )
        elif self.unlearner_name == 'neggradplus':
            unlearner = NegGradPlus(
                self.device,
            )
        elif self.unlearner_name == 'scrub':
            unlearner = SCRUB(
                self.device,
            )
        elif self.unlearner_name == 'euk':
            unlearner = KUnlearn(
                k=self.unlearn_params['k'],
                method=self.unlearner_name,
                device=self.device,
                )
        elif self.unlearner_name == 'cfk':
            unlearner = KUnlearn(
                k=self.unlearn_params['k'],
                method=self.unlearner_name,
                device=self.device,
                )
        else:
            raise ValueError(f'unlearner_name {self.unlearner_name}'
                             ' not supported.')
        self.unlearner = unlearner
        return unlearner

    def initialize_dataset(self):
        """ Initialize the entire dataset based on dataset name."""
        if self.dataset_name == 'imagenet' and self.url is not None:
            # Download and save ImageNet
            src_datasets.download_imagenet_dataset_from_web(
                url=self.url,
                dataset_save_path=self.save_path)

        train_dataset, test_dataset = src_datasets.load_dataset(
            dataset_name=self.dataset_name,
            proportion=self.proportion,
            dataset_save_path=self.save_path)

        return train_dataset, test_dataset

    def save_loaders_to_disk(self, loaders_dict):
        """Save all dataloaders to disk."""
        loaders_dir = os.path.join(self.output_dir, 'loaders')
        os.makedirs(loaders_dir, exist_ok=True)
        save_loaders(path=loaders_dir, **loaders_dict)
        print(f"Saved loaders to {loaders_dir}")

    def load_loaders_from_disk(self):
        """Load dataloaders from disk if available."""
        loaders_dir = os.path.join(self.output_dir, 'loaders')
        try:
            loaders = load_loaders(loaders_dir)
            print(f"Loaded loaders from {loaders_dir}")
            return loaders
        except Exception as e:
            raise Exception(f"Error loading loaders: {e}")

    def load_model_from_disk(self, model_path):
        """Load model from disk."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")

        try:
            # Initialize appropriate model architecture
            model = self.initialize_model()
            # Load state dict
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint)
            model = model.to(self.device)
            print(f"Loaded model from {model_path}")
            return model

        except Exception as e:
            raise Exception(f'Error loading model: {e}')

    def run(self):
        print('running...')
        # Step 1: Initialize and prepare datasets
        train_dataset, test_dataset = self.initialize_dataset()
        print('datasets initialized')

        # Step 2: Check if loaders are already saved and should be loaded
        data_dict = None
        if self.save_loaders:
            data_dict = self.load_loaders_from_disk()
            print('loaders loaded')

        # If loaders weren't loaded, create them
        if data_dict is None:
            print('loaders not loaded... creating loaders')
            # Extract loader configurations
            batch_sizes = self.dataset_cfg['batch_sizes']
            shuffle_settings = self.dataset_cfg['shuffle_settings']

            # Create dataloaders
            forget_loader, retain_loader, train_loader, val_loader, test_loader = get_all_loaders(
                train_dataset,
                test_dataset,
                method=self.forget_method,
                batch_sizes=batch_sizes,
                shuffle_settings=shuffle_settings,
                val_ratio=self.val_ratio,
                **self.forget_params
            )
            data_dict = {
                'forget': forget_loader,
                'retain': retain_loader,
                'train': train_loader,
                'val': val_loader,
                'test': test_loader
                }

            print(data_dict.keys())
            # Save loaders if configured to do so
            if self.save_loaders:
                self.save_loaders_to_disk(data_dict)
                print('loaders saved')

        # Step 3: Initialize the pretrained model, and evaluate
        original_model = self.load_model_from_disk(self.model_ckpt_path)
        if isinstance(data_dict.get('val', None), torch.utils.data.DataLoader):
            val_loss, val_accuracy = TrainApp().eval_model(
                criterion=self.unlearn_params['loss_fn'],
                model=original_model,
                val_dl=data_dict['val'])
            print(f"Pre-unlearning Validation Loss: {val_loss:.4f}")
            print(f"Pre-unlearning Validation Accuracy: {val_accuracy:.2f}%\n")
        retain_loss, retain_accuracy = TrainApp().eval_model(
                criterion=self.unlearn_params['loss_fn'],
                model=original_model,
                val_dl=data_dict['retain'])
        forget_loss, forget_accuracy = TrainApp().eval_model(
                criterion=self.unlearn_params['loss_fn'],
                model=original_model,
                val_dl=data_dict['forget'])
        print(f"Pre-unlearning Retain Loss: {retain_loss:.4f}")
        print(f"Pre-unlearning Retain Accuracy: {retain_accuracy:.2f}%\n")
        print(f"Pre-unlearning Forget Loss: {forget_loss:.4f}")
        print(f"Pre-unlearning Forget Accuracy: {forget_accuracy:.2f}%\n")

        save_model(original_model,
                   output_dir=self.output_dir,
                   unlearning_algorithm=self.unlearner_name,
                   model_name=self.model_name,
                   seed=self.seed,
                   model_type='original')

        # Step 4: Unlearning
        unlearner = self.initialize_unlearner()
        print('unlearner initialized')
        unlearned_model, losses = unlearner.unlearn(original_model,
                                                    data_dict,
                                                    **self.unlearn_params)
        print('model unlearned')

        # Step 5: Save the unlearned model
        save_model(unlearned_model,
                   output_dir=self.output_dir,
                   unlearning_algorithm=self.unlearner_name,
                   model_name=self.model_name,
                   seed=self.seed,
                   model_type='unlearned',
                   payload=losses)

        # Step 6: Evaluate the model after unlearning
        # TODO handle case where model has not been trained but evaluate required nonetheless
        if isinstance(data_dict.get('val', None), torch.utils.data.DataLoader):
            val_loss, val_accuracy = TrainApp().eval_model(
                criterion=self.unlearn_params['loss_fn'],
                model=unlearned_model,
                val_dl=data_dict['val'])
            print(f"Post-unlearning Validation Loss: {val_loss:.4f}")
            print(f"Post-unlearning Validation Accuracy: {val_accuracy:.2f}%\n")
        retain_loss, retain_accuracy = TrainApp().eval_model(
                criterion=self.unlearn_params['loss_fn'],
                model=unlearned_model,
                val_dl=data_dict['retain'])
        forget_loss, forget_accuracy = TrainApp().eval_model(
                criterion=self.unlearn_params['loss_fn'],
                model=unlearned_model,
                val_dl=data_dict['forget'])
        print(f"Pre-unlearning Retain Loss: {retain_loss:.4f}")
        print(f"Pre-unlearning Retain Accuracy: {retain_accuracy:.2f}%\n")
        print(f"Pre-unlearning Forget Loss: {forget_loss:.4f}")
        print(f"Pre-unlearning Forget Accuracy: {forget_accuracy:.2f}%\n")

        return unlearned_model


if __name__ == '__main__':
    app = UnlearnApp()
    app.run()
    # Run pip install -e .
    # Run python src/unlearning/main.py --config_path src/unlearning/experiments/unlearn_simple.yaml
