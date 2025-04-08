import argparse
import yaml
import timm
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from attacks.utils import set_seed, setup_device
import os
from datasets.cifar10 import get_cifar10_test_transform
from datasets.cifar100 import get_cifar100_test_transform
from datasets.imagenet import get_imagenet_test_transform
from attacks.config_validation import InputValidator #change this later 
from attacks.GGL.final_ggl import GGLReconstructor
from pytorch_pretrained_biggan import BigGAN, BigGANConfig
#from attacks.DGL.dgl import DGLReconstructor
#from attacks.InverseGrad.inversegrad import InverseGradReconstructor


class ReconstructorApp(InputValidator):
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
                raise FileNotFoundError('.yaml file not found.')

        # Perform input validation first
        super().__init__(config)

        self.device = setup_device()
        print(f'Using device: {self.device}')
        self.seed = config['experiment']['seed']
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

    def initialize_reconstructor(self, unlearned_model, original_model):
        """ Initialize the correct unlearner from user specification."""
        loss_fn = nn.CrossEntropyLoss()
        if self.reconstructor_name == 'ggl':
            reconstructor = GGLReconstructor(
                lr = self.reconstructor_lr,
                exp_name = self.experiment_name,
                original_model = original_model,
                target_model=unlearned_model, 
                loss_fn = loss_fn,
                **self.reconstructor_params
            )

        elif self.reconstructor_name == 'inversegrad':
            reconstructor = InverseGradReconstructor(
                self.device,
                lr = self.reconstructor_lr,
                exp_name = self.experiment_name,
                original_model = original_model,
                target_model=unlearned_model,
                loss_fn = loss_fn, #not sure if you're using the sae loss?
                **self.reconstructor_params #reconstructor specific parameters
                ### you can add any parameters defined in config_validation here, I think if we need unlearning params then we can also use **self.unlearning_params
            )
        else:
            raise ValueError(f'unlearner_name {self.unlearner_name}'
                             ' not supported.')
        self.reconstructor = reconstructor
        return reconstructor





    def save_results(self):
        pass

    def calculate_metrics(self):
        pass

    def run(self):
        print('running...')
        # Step 1 : read yaml files
        # Step 2 : load unlearned model 
        unlearned_model = self.load_model_from_disk(self.unlearned_weights)

        # Step 3: Initialize the pretrained model
        original_model = self.load_model_from_disk(self.oiginal_weights)


        # Step 4: Reconstruction
        reconstructor = self.initialize_reconstructor(unlearned_model, original_model )
        print('unlearner initialized')
        #reconstruction, losses = reconstructor.reconstruct(original_model, 
                                                    #unlearned_model,
                                                    #verbose=self.verbose,
                                                    #**self.reconstructor_params)
        print('model unlearned')

        # Step 5: Save the unlearned model
        self.save_results()

        self.calculate_metrics()

        return unlearned_model


if __name__ == '__main__':
    app = ReconstructorApp()
    app.run()
    # Run pip install -e .
    # Run python src/unlearning/main.py --config_path src/unlearning/experiments/unlearn_simple.yaml
