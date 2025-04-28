import importlib
import os
import sys

import timm
import torch
import torch.nn as nn
import torchvision

from utils import ConfigError, setup_device


class ImportModel:
    """
    A class that handles the loading, saving, and pretraining of machine learning models.

    The class supports multiple model loading methods, including:
    - Class-based model import
    - Torch Hub
    - torchvision
    - timm

    Attributes:
        device (str): The device to use for model inference (e.g., 'cuda', 'cpu').
        load_method (str): The method to use for model loading.
        init_path (str): Path to the model initialization module.
        model_name (str): The name of the model to load.
        model_ckpt_path (str, optional): Path to a pre-trained model checkpoint.
        model_kwargs (dict): Additional keyword arguments to initialize the model.
        num_classes (int): The number of output classes for the model.
        from_pretrained (bool): Flag indicating whether to load a pre-trained model.
        model (nn.Module): The loaded model object.
    """

    def __init__(
        self,
        load_method: str,
        model_name: str,
        num_classes: int,
        init_path=None,
        model_ckpt_path=None,
        model_kwargs=None,
        from_pretrained=True,
    ):
        self.device = setup_device()
        self.load_method = load_method
        self.init_path = init_path
        self.model_name = model_name
        self.model_ckpt_path = model_ckpt_path
        self.model_kwargs = model_kwargs or {}  # Handle None
        self.num_classes = num_classes
        self.from_pretrained = from_pretrained
        self.model = self.load_model()

    def load_model(self):
        """
        Loads the model based on the specified loading method.

        Supported methods:
            - class: Load by importing a class from a module
            - torchhub: Load from torch hub
            - torchvision: Load model from torchvision.models
            - timm: Load model from timm.create_model

        Returns:
            nn.Module: The loaded model.

        Raises:
            ConfigError: If the model fails to load using the specified method.
        """
        try:
            # Load the model based on the specified method
            if self.load_method == "class":
                self._init_from_class(**self.model_kwargs)
            elif self.load_method == "torchhub":
                self._init_from_torch_hub()
            elif self.load_method == "torchvision":
                self._init_from_torchvision()
            elif self.load_method == "timm":
                self._init_from_timm()
            else:
                raise ConfigError(f"Unknown initialization method: {self.load_method}")
            # Load weights if specified
            if self.model_ckpt_path is not None:
                self._load_weights()

            # Finalize model setup
            self.model = self.model.to(self.device)
            self.model.eval()
            print(
                f"Model loaded successfully and set to evaluation mode on {self.device}"
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
            raise ConfigError(f"Could not import module {self.init_path}: {str(e)}")
        except AttributeError:
            raise ConfigError(f"{self.model_name} not found in module {self.init_path}")
        except Exception as e:
            raise ConfigError(f"Failed to initialize model: {str(e)}")

    def _import_module(self, module_path):
        """Import a module from either a file path or module name."""
        # Check if it's a file path or module name
        if (
            module_path.startswith("./")
            or module_path.startswith("/")
            or "/" in module_path
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
            pretrained = self.model_ckpt_path is None and self.from_pretrained

            # Try with 'weights' parameter first
            try:
                weights_param = "DEFAULT" if pretrained else None
                self.model = torch.hub.load(
                    self.init_path,
                    self.model_name,
                    weights=weights_param,
                    trust_repo="check",
                )
            except (TypeError, ValueError):
                # Fall back to 'pretrained' parameter
                self.model = torch.hub.load(
                    self.init_path,
                    self.model_name,
                    pretrained=pretrained,
                    trust_repo="check",
                )

            print(f"Loaded model from torch.hub: {self.init_path}/{self.model_name}")

        except Exception as e:
            raise ConfigError(f"Failed to load model from torch.hub: {str(e)}")

    def _init_from_torchvision(self):
        try:
            if self.model_ckpt_path is None and self.from_pretrained:
                weights = "DEFAULT"
            else:
                weights = None
            self.model = torchvision.models.get_model(
                self.model_name,
                weights=weights,
            )
            if self.num_classes != 1000:  # Not using ImageNet
                print("Replacing default ImageNet classification head.")
                # Adjust the last layer based on model type
                if hasattr(self.model, "fc"):  # ResNet-style
                    self.model.fc = nn.Linear(
                        self.model.fc.in_features, self.num_classes
                    )
                elif hasattr(self.model, "classifier"):
                    # e.g MobileNet, EfficientNet, VGG, DenseNet
                    if isinstance(self.model.classifier, nn.Sequential):
                        # Handle cases like MobileNet where
                        # classifier is Sequential
                        last_layer_idx = len(self.model.classifier) - 1
                        self.model.classifier[last_layer_idx] = nn.Linear(
                            self.model.classifier[last_layer_idx].in_features,
                            self.num_classes,
                        )
                    else:
                        self.model.classifier = nn.Linear(
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
            self.model = timm.create_model(
                self.model_name, num_classes=self.num_classes
            )
        except Exception as e:
            raise ConfigError(f"Failed to load timm model: {str(e)}")

    def _load_weights(self):
        """Load weights from a checkpoint file."""
        try:
            print(f"Loading weights from {self.model_ckpt_path}")
            if not os.path.exists(self.model_ckpt_path):
                raise ConfigError(f"Weight file not found: {self.model_ckpt_path}")

            # Load checkpoint
            checkpoint = torch.load(self.model_ckpt_path, map_location=self.device)

            # Extract state dict smartly
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                    print("Found 'model_state_dict' key in checkpoint.")
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                    print("Found 'state_dict' key in checkpoint.")
                else:
                    state_dict = checkpoint
                    print("Checkpoint looks like a pure state dict.")
            else:
                state_dict = checkpoint
                print("Checkpoint is not a dict, using directly.")

            # Load the state dict
            self.model.load_state_dict(state_dict)
            print(f"Successfully loaded weights from {self.model_ckpt_path}")

        except RuntimeError as e:
            error_msg = str(e)
            if "Missing key(s)" in error_msg or "Unexpected key(s)" in error_msg:
                # Count the number of missing/unexpected keys
                missing_count = error_msg.count("Missing key(s)")
                unexpected_count = error_msg.count("Unexpected key(s)")

                # Create a concise error message
                error_summary = "Model architecture mismatch: "
                if missing_count > 0:
                    error_summary += (
                        f"The checkpoint is missing layers present in your model. "
                    )
                if unexpected_count > 0:
                    error_summary += (
                        f"The checkpoint contains layers not present in your model. "
                    )

                error_summary += "This typically means the model architecture used to create the checkpoint differs from your current model."
                raise ConfigError(error_summary)
            else:
                # Re-raise for other RuntimeErrors
                raise ConfigError(f"Failed to load weights: {str(e)}")
        except Exception as e:
            raise ConfigError(f"Failed to load weights: {str(e)}")
