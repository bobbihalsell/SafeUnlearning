import argparse
import yaml
import timm
import torch
import torch.nn as nn
import torchvision
from unlearning.utils import save_model, set_seed, setup_device
import os
from datasets.preprocessing import get_all_loaders, save_loaders, load_loaders
from unlearning.finetune import FinetuneUnlearner
from unlearning.scrub import SCRUB
from unlearning.kunlearn import KUnlearn
from unlearning.neggrad import NegGrad, NegGradPlus
from unlearning.trainer import Trainer
from datasets import load_datasets as src_datasets
from unlearning.eval import plot

DEFAULT_SEED = 42


class UnlearnApp:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Specify path to .yaml file with configurations to run unlearning process.")
        parser.add_argument("--config_path", type=str, required=True, help="The path to the .yaml file for unlearning configurations.")
        args = parser.parse_args()

        # Load in the instructions for the unlearning run from a filepath
        with open(args.config_path, 'r') as f:
            try:
                config = yaml.safe_load(f)
            except FileNotFoundError:
                print('.yaml file not found.')

        # Get data from the config
        self.device='cpu'
        # self.device = setup_device()
        self.seed = config.get('seed', DEFAULT_SEED)
        set_seed(self.seed)

        model_config= config['model']
        self.model_name = model_config['name']
        self.model_ckpt_path = model_config.get('model_ckpt_path', None)
        # Check whether to use built-in pretrained weights
        self.pretrained = not bool(self.model_ckpt_path)
        self.num_classes = model_config['num_classes']
        # Get the training configuration if provided
        self.train_cfg = model_config.get('train_cfg', None)

        # Process unlearner-specific parameters
        self.unlearner_name = config['unlearner']['name']
        self.evaluate = config['unlearner']['evaluate']
        self.verbose = config['unlearner']['verbose']
        self.unlearn_params = config['unlearner']['cfg']

        # Process dataset parameters
        self.dataset_name = config['dataset']['name']
        self.from_web = config['dataset']['from_web']
        self.url = config['dataset']['url']
        self.val_ratio = config['dataset']['val_ratio']
        self.save_loaders = config['dataset']['save_loaders']
        self.dataset_cfg = config['dataset']['cfg']

        # Process forget method parameters
        self.forget_method = config['forget_method']['name']
        self.forget_params = config['forget_method']['parameters']
        assert self.forget_params is not None  # TODO add YAML validation.

        # Output directory
        self.output_dir = config.get('output_dir', 'artifacts/')
        os.makedirs(self.output_dir, exist_ok=True)

    def _extract_unlearner_params(self):
        """
        Extract and process parameters specific to each unlearner type.
        Raises exception if required parameters are missing.
        """
        # common parameters for unlearners
        required_params = ['epochs', 'lr', 'weight_decay', 'use_l2_penalty', 'loss_fn']

        missing_params = []
        for param in required_params:
            if param not in self.unlearn_params:
                if self.unlearner_name == 'scrub' and param == 'epochs':
                    continue
                missing_params.append(param)
            if param == 'loss_fn':
                if self.unlearn_params['loss_fn'] == 'cross_entropy':
                    self.unlearn_params['loss_fn'] = nn.CrossEntropyLoss()
                else:
                    raise ValueError(f'Only cross_entropy loss_fn is allowed, received '
                                     f'{self.unlearn_params['loss_fn']}')
                
        if missing_params:
            raise ValueError(f"Missing required parameters for unlearning: {', '.join(missing_params)}")

        # Add specific parameters based on unlearner type
        if self.unlearner_name == 'neggradplus':
            try:
                self.unlearn_params['beta']
            except KeyError:
                raise ValueError("Missing required parameter 'beta' for NegGradPlus unlearner")

        elif self.unlearner_name == 'scrub':
            # Check for required SCRUB-specific parameters
            required_scrub_params = ['min_epochs', 'max_epochs', 'alpha', 'gamma']
            missing_scrub_params = []
            for param in required_scrub_params:
                if param not in self.unlearn_params:
                    missing_scrub_params.append(param)
            if missing_scrub_params:
                raise ValueError(f"Missing required parameters for SCRUB unlearner: {', '.join(missing_scrub_params)}")

        elif self.unlearner_name in ['euk', 'cfk']:
            # Check for k parameter
            try:
                self.unlearn_params['k']
            except KeyError:
                raise ValueError(f"Missing required parameter 'k' for {self.unlearner_name.upper()} unlearner")
            if self.unlearner_name == 'euk':
                # Check for EUk-specific parameters
                if 'reinit_method' not in self.unlearn_params:
                    raise ValueError("Missing required parameter 'reinit_method' for EUk unlearner")

    def initialize_model(self):
        """Initialize the model based on model name from user configuration."""
        if hasattr(torchvision.models, self.model_name):
            model = torchvision.models.get_model(
                self.model_name,
                weights="DEFAULT" if self.pretrained else None,
            )

            # Adjust the last layer based on model type
            if hasattr(model, "fc"):  # ResNet-style
                model.fc = torch.nn.Linear(model.fc.in_features, self.num_classes)
            elif hasattr(model, "classifier"):  # MobileNet, EfficientNet, VGG, DenseNet
                if isinstance(model.classifier, torch.nn.Sequential):  
                    # Handle cases like MobileNet where classifier is Sequential
                    last_layer_idx = len(model.classifier) - 1
                    model.classifier[last_layer_idx] = torch.nn.Linear(
                        model.classifier[last_layer_idx].in_features, self.num_classes
                    )
                else:
                    model.classifier = torch.nn.Linear(model.classifier.in_features, self.num_classes)
            else:
                raise AttributeError(f"Unknown classification layer for {self.model_name}")

        else:
            print(f"Couldn't find {self.model_name} in torchvision. Looking in timm")
            try:
                model = timm.create_model(
                    self.model_name,
                    pretrained=self.pretrained,
                    num_classes=self.num_classes,
                )
            except Exception:
                raise AttributeError(f"{self.model_name} not found in torchvision or timm.")

        if self.model_ckpt_path:
            checkpoint = torch.load(self.model_ckpt_path, map_location="cpu")
            checkpoint = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            model.load_state_dict(checkpoint)

        return model

    def initialize_unlearner(self):
        """ Initialize the correct unlearner from user specification."""
        if self.unlearner_name == 'finetune':
            unlearner = FinetuneUnlearner(self.device,
                                 self.evaluate,
                                 self.verbose
                                )
        elif self.unlearner_name == 'neggrad':
            unlearner = NegGrad(self.device,
                                 self.evaluate,
                                 self.verbose
                                )
        elif self.unlearner_name == 'neggradplus':
            unlearner = NegGradPlus(self.device,
                                 self.evaluate,
                                 self.verbose
                                )
        elif self.unlearner_name == 'scrub':
            unlearner = SCRUB(self.device,
                                 self.evaluate,
                                 self.verbose
                                )
        elif self.unlearner_name == 'euk':
            k = self.unlearn_params[k]
            unlearner = KUnlearn(
                            k=self.unlearn_params[k],
                            method=self.unlearner_name,
                            device=self.device,
                            evaluate=self.evaluate,
                            verbose=self.verbose
                            )
        elif self.unlearner_name == 'cfk':
            k = self.unlearn_params[k]
            unlearner = KUnlearn(
                            k=self.unlearn_params[k],
                            method=self.unlearner_name,
                            device=self.device,
                            evaluate=self.evaluate,
                            verbose=self.verbose
                            )
        else:
            raise ValueError(f'unlearner_name {self.unlearner_name}'
                             ' not supported.')
        self.unlearner = unlearner
        return unlearner

    def initialize_dataset(self):
        """ Initialize the entire dataset based on dataset name."""
        if self.dataset_name == 'cifar10':
            train_dataset, test_dataset = src_datasets.load_cifar10_datasets()
        elif self.dataset_name == 'cifar5':
            train_dataset, test_dataset = src_datasets.load_cifar5_datasets()
        elif self.dataset_name == 'cifar100':
            train_dataset, test_dataset = src_datasets.load_cifar100_datasets()
        elif self.from_web:
            save_dir = f'./{self.dataset_name}'
            train_dataset, test_dataset = src_datasets.download_dataset_from_web(self.url, save_dir)

        else:
            raise ValueError(f'{self.dataset_name} not supported.')
        return train_dataset, test_dataset

    def save_loaders_to_disk(self, loaders_dict):
        """Save all dataloaders to disk."""
        if not self.save_loaders:
            return

        loaders_dir = os.path.join(self.output_dir, 'loaders')
        os.makedirs(loaders_dir, exist_ok=True)

        save_loaders(path=loaders_dir, **loaders_dict)
        print(f"Saved loaders to {loaders_dir}")

    def load_loaders_from_disk(self):
        """Load dataloaders from disk if available."""
        loaders_dir = os.path.join(self.output_dir, 'loaders')
        if not os.path.exists(loaders_dir):
            return None

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
            checkpoint = torch.load(model_path, map_location=self.device)
            # Initialize appropriate model architecture
            model = self.initialize_model()
            # Load state dict
            model.load_state_dict(checkpoint['state_dict'])
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

            # DEBUGGING
            class_counts = {}
            labels = []
            for i in range(len(train_dataset)):
                label = train_dataset[i][1]  # Assuming labels are at index 1
                labels.append(label)
                if label in class_counts:
                    class_counts[label] += 1
                else:
                    class_counts[label] = 1
            # print(f"Class counts in train_data_subset: {class_counts}")
            print(f'len(labels) {len(labels)}')
            print(f'max: {max(class_counts.values())}')

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

        # Step 3: Initialize or load a pre-trained model, evaluate and save.
        original_model = self.initialize_model()
        if isinstance(data_dict.get('val', None), torch.utils.data.DataLoader):
            val_loss, val_accuracy = Trainer(original_model).evaluate_model(data_dict['val'])
            print(f"Pre-unlearning Validation Loss: {val_loss:.4f}")
            print(f"Pre-unlearning Validation Accuracy: {val_accuracy:.2f}%\n")
        retain_loss, retain_accuracy = Trainer(original_model).evaluate_model(data_dict['retain'])
        forget_loss, forget_accuracy = Trainer(original_model).evaluate_model(data_dict['forget'])
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
        self._extract_unlearner_params()
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
            val_loss, val_accuracy = Trainer(unlearned_model).evaluate_model(data_dict['val'])
            print(f"Post-unlearning Validation Loss: {val_loss:.4f}")
            print(f"Post-unlearning Validation Accuracy: {val_accuracy:.2f}%\n")
        retain_loss, retain_accuracy = Trainer(unlearned_model).evaluate_model(data_dict['retain'])
        forget_loss, forget_accuracy = Trainer(unlearned_model).evaluate_model(data_dict['forget'])
        print(f"Pre-unlearning Retain Loss: {retain_loss:.4f}")
        print(f"Pre-unlearning Retain Accuracy: {retain_accuracy:.2f}%\n")
        print(f"Pre-unlearning Forget Loss: {forget_loss:.4f}")
        print(f"Pre-unlearning Forget Accuracy: {forget_accuracy:.2f}%\n")

        if self.evaluate:
            plot(losses)

        return unlearned_model


if __name__ == '__main__':
    app = UnlearnApp()
    app.run()
    # Run pip install -e .
    # Run python src/unlearning/main.py --config_path src/experiments/simple_exp.yaml
