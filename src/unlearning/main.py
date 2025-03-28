import argparse
import yaml
import timm
import torch
import torch.nn as nn
from unlearning.utils import save_model, set_seed, setup_device
import os
from datasets.preprocessing import get_all_loaders, save_loaders, load_loaders
from unlearning.finetune import FinetuneUnlearner
from unlearning.scrub import SCRUB
from unlearning.kunlearn import KUnlearn
from unlearning.neggrad import NegGrad, NegGradPlus
from unlearning.trainer import Trainer
from models.cnns import AllCNN, CNN
from datasets import load_datasets as src_datasets
import os

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
        self.device = setup_device()
        self.seed = config.get('seed', DEFAULT_SEED)
        set_seed(self.seed)

        model_config= config['model']
        self.model_name = model_config['name']
        self.model_save_name = model_config['save_name']
        self.pretrained = model_config['pretrained']
        self.num_classes = model_config['num_classes']
        self.train_cfg = model_config['train_cfg']

        # Process unlearner-specific parameters
        self.unlearner_name = config['unlearner']['name']
        self.evaluate = config['unlearner']['evaluate']
        self.unlearn_params = config['unlearner']['cfg']

        # Process dataset parameters
        self.dataset_name = config['dataset']['name']
        self.val_ratio = config['dataset']['val_ratio']
        self.save_data = config['dataset']['save_data']
        self.save_loaders = config['dataset']['save_loaders']
        self.save_dir = config['dataset']['save_dir']
        self.dataset_cfg = config['dataset']['cfg']

        # Process forget method parameters
        self.forget_method = config['forget_method']['name']
        self.forget_params = config['forget_method']['parameters']

        # Output directory
        self.output_dir = config.get('output_dir', 'output/')
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
        # TODO replace this algorithm with model_init.py from Jebastin
        if 'cnn' in self.model_name.lower():
            # Create CNN or AllCNN model
            if self.model_name.lower() == 'cnn':
                model = CNN(
                    in_channels=self.model_params.get('in_channels', 3),
                    filters=self.model_params.get('filters', [64, 64, 128, 128, 256, 256]),
                    num_classes=self.num_classes,
                    dropout_rate=self.model_params.get('dropout_rate', 0.3),
                    use_batch_norm=self.model_params.get('use_batch_norm', True),
                    downsample_every=self.model_params.get('downsample_every', 3)
                )
            elif self.model_name.lower() == 'allcnn':
                model = AllCNN(
                    filters=self.model_params.get('filters', [32, 32, 64, 64, 128]),
                    num_classes=self.num_classes,
                    dropout_rates=self.model_params.get('dropout_rates', [0.1, 0.1, 0.1]),
                    use_batchnorm=self.model_params.get('use_batch_norm', True),
                    downsample_every=self.model_params.get('downsample_every', 2)
                )
        else:
            # Use timm or custom resnet implementations
            if 'resnet' in self.model_name.lower():
                # # Check for custom ResNet implementations
                # if self.model_name == 'resnet18':
                #     model = resnets.get_resnet18(num_classes=self.num_classes)
                # elif self.model_name == 'resnet34':
                #     model = resnets.get_resnet34(num_classes=self.num_classes)
                # elif self.model_name == 'resnet50':
                #     model = resnets.get_resnet50(num_classes=self.num_classes)
                # else:
                #     # Fall back to timm for other ResNet variants
                #     model = timm.create_model(
                #         model_name=self.model_name,
                #         pretrained=self.pretrained,
                #         num_classes=self.num_classes
                #     )
            # else:
                # Use timm for other model architectures
                model = timm.create_model(
                    model_name=self.model_name,
                    pretrained=self.pretrained,
                    num_classes=self.num_classes
                )

        return model

    def initialize_unlearner(self):
        """ Initialize the correct unlearner from user specification."""
        if self.unlearner_name == 'finetune':
            unlearner = FinetuneUnlearner(self.device,
                                 self.evaluate
                                )
        elif self.unlearner_name == 'neggrad':
            unlearner = NegGrad(self.device,
                                 self.evaluate
                                )
        elif self.unlearner_name == 'neggradplus':
            unlearner = NegGradPlus(self.device,
                                 self.evaluate
                                )
        elif self.unlearner_name == 'scrub':
            unlearner = SCRUB(self.device,
                                 self.evaluate
                                )
        elif self.unlearner_name == 'euk':
            k = self.unlearn_params[k]
            unlearner = KUnlearn(
                            k=self.unlearn_params[k],
                            method=self.unlearner_name,
                            device=self.device,
                            evaluate=self.evaluate
                            )
        elif self.unlearner_name == 'cfk':
            k = self.unlearn_params[k]
            unlearner = KUnlearn(
                            k=self.unlearn_params[k],
                            method=self.unlearner_name,
                            device=self.device,
                            evaluate=self.evaluate
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
            print(f"Error loading loaders: {e}")
            return None

    def save_model_to_disk(self, model, model_path):
        """Save model to disk."""
        model_dir = os.path.join(self.output_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)

        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': self.model_name,
            'unlearner': self.unlearner_name,
            'num_classes': self.num_classes
        }, model_path)

        print(f"Saved model to {model_path}")
        return model_path

    def load_model_from_disk(self, model_path):
        """Load model from disk."""
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found.")
            return None
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            # Initialize appropriate model architecture
            model = self.initialize_model()
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            print(f"Loaded model from {model_path}")
            return model
        except Exception as e:
            raise Exception(f'Error loading model: {e}')

    def run(self):
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

        # Step 3: Initialize or load a pre-trained model
        model_path = os.path.join(self.output_dir, 'models', f"{self.model_save_name}.pth")
        print('model path {}'.format(model_path))
        if os.path.exists(model_path) and self.pretrained:
            # Load pre-trained model
            original_model = self.load_model_from_disk(model_path)
            print('model loaded')
        else:
            # Initialize new model
            original_model = self.initialize_model()
            print('model initialized')

            # Train the model if it's newly initialized
            if not self.pretrained:
                original_model = Trainer.train_model(
                    model=original_model,
                    epochs=,
                    criterion=,
                    optimizer=,
                    train_dataloader=data_dict['train'])#, 
                #     data_dict['val']
                # )
                # Save the trained model
                self.save_model_to_disk(original_model, model_path)
                print('model trained and saved')

            # Step 4: Evaluate the original model before unlearning TODO: Add evaluation

        unlearner = self.initialize_unlearner()
        print('unlearner initialized')
        for key, param in self.unlearn_params.items():
            print(key, ' , ', param, )
            print(type(param))
        unlearned_model, losses = unlearner.unlearn(original_model, data_dict, **self.unlearn_params)
        print('model unlearned')

        # Step 6: Save the unlearned model
        unlearned_model_path = os.path.join(self.output_dir, 'models', f"{self.model_save_name}_{self.unlearner_name}.pth")
        self.save_model_to_disk(unlearned_model, unlearned_model_path)
        print('model saved')

        # Save losses if available
        if self.evaluate:
            self.save_losses_to_disk(losses)
            print('losses saved')

         # Step 7: Evaluate the model after unlearning TODO: Add evaluation

        return unlearned_model
    


if __name__ == '__main__':
    app = UnlearnApp()
    app.run()
    # Run python src/unlearning/main.py --config_path src/experiments/unlearn_test.yaml
