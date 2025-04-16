import torch
from unlearning.utils import ConfigError
from train.utils import setup_device
import os
import sys
import importlib
import wandb
import torchvision
import timm


class ImportModel:
    """ Perform pretraining, or model loading and saving, for a model."""
    def __init__(
                    self,
                    load_method: str,
                    model_name: str,
                    init_path=None,
                    model_ckpt_path=None,
                    model_kwargs=None,
                    num_classes=10,
                    device=None
                ):
        if device is None:
            self.device = setup_device()
        else:
            self.device = device

        self.load_method = load_method
        self.init_path = init_path
        self.model_name = model_name
        self.model_ckpt_path = model_ckpt_path
        self.model_kwargs = model_kwargs or {}  # Handle None
        self.num_classes = num_classes

        self.load_model()

    def load_model(self):
        """
        Load the model based on the specified loading method from config.

        Supported methods:
        - class: Load by importing a class from a module
        - torchhub: Load from torch hub
        - torchvision: Load model from torchvision.models
        - timm: Load model from timm.create_model
        """
        try:
            # Load the model based on the specified method
            if self.load_method == 'class':
                self._init_from_class(**self.model_kwargs)
            elif self.load_method == 'torchhub':
                self._init_from_torch_hub()
            elif self.load_method == 'torchvision':
                self._init_from_torchvision()
            elif self.load_method == 'timm':
                self._init_from_timm()
            else:
                raise ConfigError(
                    "Unknown initialisation method: "
                    f"{self.load_method}"
                )
            # Load weights if specified
            if self.model_ckpt_path is not None:
                self._load_weights()

            # Finalize model setup
            self.model = self.model.to(self.device)
            self.model.eval()
            print(
                    f"Model loaded successfully and set to evaluation mode "
                    f"on {self.device}"
                )

            return self.model

        except Exception as e:
            raise ConfigError(f"Error in load_model: {str(e)}")

    def _init_from_class(self):
        """
        Load model by importing a class from a module and initializing it.
        """
        try:
            module = self._import_module(self.init_path)
            model_init = getattr(module, self.model_name)
            # Initialize the model
            self.model = model_init(**self.model_kwargs)
            print(f"Initialized model using {self.model_name}")

        except ImportError as e:
            raise ConfigError(
                f"Could not import module {self.init_path}: {str(e)}"
                )
        except AttributeError:
            raise ConfigError(
                f"{self.model_name} not found in module {self.init_path}"
                )
        except Exception as e:
            raise ConfigError(f"Failed to initialize model: {str(e)}")

    def _import_module(self, module_path):
        """Import a module from either a file path or module name."""
        # Check if it's a file path or module name
        if (
            module_path.startswith('./')
            or module_path.startswith('/')
            or '/' in module_path
        ):
            abs_path = os.path.abspath(module_path)
            # Get the directory and module name
            directory = os.path.dirname(abs_path)
            module_name = os.path.basename(abs_path)
            sys.path.insert(0, directory)
            try:
                return importlib.import_module(module_name)
            finally:
                sys.path.pop(0)
        else:
            return importlib.import_module(module_path)

    def _init_from_torch_hub(self):
        """Load model from torch hub."""
        try:
            pretrained = (self.model_ckpt_path is None)

            # Try with 'weights' parameter first
            try:
                weights_param = 'DEFAULT' if pretrained else None
                self.model = torch.hub.load(
                    self.init_path, 
                    self.model_name, 
                    weights=weights_param,
                    trust_repo="check"
                )
            except (TypeError, ValueError):
                # Fall back to 'pretrained' parameter
                self.model = torch.hub.load(
                    self.init_path, 
                    self.model_name, 
                    pretrained=pretrained,
                    trust_repo="check"
                )

            print(
                f"Loaded model from torch.hub: "
                f"{self.init_path}/{self.model_name}"
            )

        except Exception as e:
            raise ConfigError(f"Failed to load model from torch.hub: {str(e)}")

    def _init_from_torchvision(self):
        try:
            weights = (self.model_ckpt_path is None)
            self.model = torchvision.models.get_model(
                        self.model_name,
                        weights=weights,
                    )
            # Adjust the last layer based on model type
            if hasattr(self.model, "fc"):  # ResNet-style
                self.model.fc = torch.nn.Linear(
                    self.model.fc.in_features, self.num_classes
                )
            elif hasattr(self.model, "classifier"):
                # e.g MobileNet, EfficientNet, VGG, DenseNet
                if isinstance(self.model.classifier, torch.nn.Sequential):
                    # Handle cases like MobileNet where
                    # classifier is Sequential
                    last_layer_idx = len(self.model.classifier) - 1
                    self.model.classifier[last_layer_idx] = torch.nn.Linear(
                        self.model.classifier[last_layer_idx].in_features,
                        self.num_classes
                    )
                else:
                    self.model.classifier = torch.nn.Linear(
                        self.model.classifier.in_features, self.num_classes
                    )
            else:
                raise AttributeError(
                    f"Unknown classification layer for {self.model_name}"
                )
        except Exception as e:
            raise ConfigError(f"Failed to load torchvision model: {str(e)}")

    def _init_from_timm(self):
        try:
            num_classes = self.model_kwargs.get('num_classes')
            self.model = timm.create_model(
                self.model_name,
                num_classes=num_classes
            )
        except Exception as e:
            raise ConfigError(f"Failed to load timm model: {str(e)}")

    def _load_weights(self):
        """Load weights from a checkpoint file."""
        try:
            print(f"Loading weights from {self.model_ckpt_path}")
            if not os.path.exists(self.model_ckpt_path):
                raise ConfigError(
                    f"Weight file not found: {self.model_ckpt_path}"
                    )

            # Load checkpoint and handle different formats
            checkpoint = torch.load(self.model_ckpt_path, 
                                    map_location=self.device)

            # Extract state dict
            if (
                isinstance(checkpoint, dict)
                and 'model_state_dict' in checkpoint
            ):
                state_dict = checkpoint['model_state_dict']
                print("Found model_state_dict key in checkpoint")
            else:
                state_dict = checkpoint
                print("Using checkpoint directly as state dict")
            # Load the state dict
            self.model.load_state_dict(state_dict)
            print(f"Successfully loaded weights from {self.model_ckpt_path}")

        except Exception as e:
            raise ConfigError(f"Failed to load weights: {str(e)}")

    def save_model(self, save_name=None, output_dir=None):
        """
        Save the model to the specified path and optionally to wandb artifacts.
        """
        if self.model is None:
            raise ConfigError("Model must be loaded before saving.")

        if output_dir is None:
            if save_name is None:
                if self.model_name:
                    save_name = self.model_name.split('.')[-1]
                else:
                    save_name = "model"
            output_dir = os.path.join(output_dir, f"{save_name}.pt") 

        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        torch.save(self.model.state_dict(), output_dir)
        print(f"Model saved to {output_dir}")

        return output_dir
