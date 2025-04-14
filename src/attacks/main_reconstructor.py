import os
import time
import timm
import torch
import torch.nn as nn
import torchvision

import hydra
from omegaconf import OmegaConf, DictConfig
from omegaconf.errors import MissingMandatoryValue
import wandb

from attacks.utils import set_seed, setup_device, safe_dataclass_load, SaveImage
from datasets.cifar10 import CIFAR10_MEAN, CIFAR10_STD
from datasets.cifar100 import CIFAR100_MEAN, CIFAR_100_STD
from datasets.imagenet import IMAGENET_MEAN, IMAGENET_STD
from attacks.config_validation import InputValidator  
from attacks.GGL.final_ggl import GGLReconstructor
from attacks.InverseGrad.reconstructor import InverseGradReconstructor,InverseGradConfig

DEFAULT_SEED = 42

class ReconstructorApp(InputValidator):
    def __init__(self, config: DictConfig):
        # Perform input validation first
        config = OmegaConf.to_container(config, resolve=True)
        super().__init__(config)

        self.device = setup_device()
        print(config.keys())
        print(f'Using device: {self.device}')
        self.seed = config['seed']
        set_seed(self.seed)

        # Output directory
        self.output_dir = config.get('output_dir', './artifacts/')
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
            checkpoint = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            model.load_state_dict(checkpoint)
            model = model.to(self.device)
            print(f"Loaded model from {model_path}")
            return model

        except Exception as e:
            raise Exception(f'Error loading model: {e}')

    def initalise_image_params(self):
        if self.dataset_name == 'cifar10':
            self.image_mean = CIFAR10_MEAN
            self.image_std = CIFAR10_STD
            self.image_size = [3, 32, 32] 
        elif self.dataset_name == 'cifar100':
            self.image_mean = CIFAR100_MEAN
            self.image_std = CIFAR_100_STD
            self.image_size = [3, 32, 32]
        elif self.dataset_name == 'imagenet':
            self.image_mean = IMAGENET_MEAN
            self.image_std = IMAGENET_STD
            self.image_size = [3, 224, 224]
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported.")
        

    def initialize_reconstructor(self, unlearned_model, original_model):
        """ Initialize the correct unlearner from user specification."""

        loss_fn = nn.CrossEntropyLoss()
        if self.reconstructor_name == 'ggl':
            reconstructor = GGLReconstructor(
                labels = self.labels,
                lr = self.reconstructor_lr,
                exp_name = self.experiment_name,
                original_model = original_model,
                target_model=unlearned_model, 
                loss_fn = loss_fn,
                **self.reconstructor_params
            )

        elif self.reconstructor_name == 'inversegrad':
            reconstructor = InverseGradReconstructor(
                device = self.device,
                original_model = original_model,
                unlearned_model = unlearned_model,
                config = safe_dataclass_load(InverseGradConfig, self.reconstructor_params)
            )
        else:
            raise ValueError(f'reconstructor {self.reconstructor_name}'
                             ' not supported.')
        self.reconstructor = reconstructor
        return reconstructor



    def save_results(self):
        saver = SaveImage(self.reconstructor_name,
                          self.seed,
                          self.experiment_name,
                          self.image_mean,
                          self.image_std,
                          self.output_dir)
        return saver

    def calculate_metrics(self):
        pass

    def run(self):
        print('running...')
        # Step 1 : read yaml files
        # Step 2 : load unlearned model 
        unlearned_model = self.load_model_from_disk(self.unlearned_weights)

        # Step 3: Initialize the pretrained model
        original_model = self.load_model_from_disk(self.original_weights)

        # Step 3: Initialize wandb
        config = self.reconstructor_params.copy() 
        config['type'] = self.reconstructor_name
        config['reconstructor_lr'] = self.reconstructor_lr  
        config.update(self.extra_config)
        
        wandb.init(
            project = self.wandb_project,
            config = config,
            group = self.experiment_name
            )
        

        # Step 4: Reconstruction
        self.initalise_image_params()
        
        reconstructor = self.initialize_reconstructor(unlearned_model, original_model)
        print('reconstructor initialized')

        start_time = time.time()
        print(self.reconstructor_name)
        if self.reconstructor_name == 'ggl':
            reconstruction, z_res, losses = reconstructor.reconstruct()
        elif self.reconstructor_name == 'inversegrad':
            reconstruction, losses = reconstructor.reconstruct(labels = self.labels,
                                                            image_size= self.image_size,
                                                            image_mean= self.image_mean,
                                                            image_std=self.image_std,
                                                            lr= self.reconstructor_lr,
                                                            verbose=self.verbose)
        total_time = time.time() - start_time
        if self.verbose:
            print(f"Reconstruction completed in {total_time:.2f} seconds")
            print(f"Reconstruction Losses: {losses}")


        wandb.log({
            'reconstruction_time': total_time,
            'losses': losses,
            'reconstruction': wandb.Image(reconstruction),
        })

        # # Step 5: Save the reconstructed image
        image_saver = self.save_results()
        image_saver.save_png(reconstruction,
                             normalize=False)
        
        self.calculate_metrics()


@hydra.main(version_base=None,
            config_path="config",
            config_name="config")
def main(cfg: DictConfig):
    # Print the config for the user first
    print('============ Run Configuration ============')
    print(OmegaConf.to_yaml(cfg))
    print('============================================')
    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise MissingMandatoryValue(
            'Missing the following required arguments in the configuration: '
            f'{missing_keys}. \n'
            'Hint: python file.py key=value sets the appropriate value.')
    app = ReconstructorApp(cfg)
    app.run()

if __name__ == '__main__':
    main()

    # Run pip install -e .
    # Run python src/attacks/main_reconstructor.py experiment=reconstruction dataset=cifar10 reconstructor=inversegrad 
