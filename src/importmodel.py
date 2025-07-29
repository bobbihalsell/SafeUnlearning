import importlib
import os
import sys

import timm
import torch
import torch.nn as nn
import torchvision

from utils import ConfigError, setup_device

# for pgu
import sys
sys.path.append("/vol/bitbucket/blh124/unlpaper/safe-unlearning")


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
                print('importing from class')
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
        except ImportError as e:
            print (f'new_error : {e}')
        try:
            print('module imported')
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
        print(module_path)
        # Check if it's a file path or module name
        if (
            module_path.startswith("./")
            or module_path.startswith("/")
            or "/" in module_path
        ):
            print('enter if')
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
            print('enter else')
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
            checkpoint = torch.load(self.model_ckpt_path, map_location=self.device, weights_only=False)
            print('ckp loaded')
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
                # Re-raise for other RuntimeError
                print('a')
                raise ConfigError(f"Failed to load weights: {str(e)}")
        except Exception as e:
            print('b')
            raise ConfigError(f"Failed to load weights: {str(e)}")



# # UNDO THIS PART ==========
# import os
# import copy
# import math
# import time
# from tqdm import tqdm

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import grad
# import torchvision
# from torchvision import transforms
# from torchvision.models.feature_extraction import create_feature_extractor

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# def get_entropy(model, loader, ignore_first_k=0):
#     with torch.no_grad():
#         model.eval()
#         entropies = []
#         for batch_idx, (images, labels) in enumerate(loader):
#             labels = labels.to(device)
#             images = images.to(device)

#             outputs = model(images)
#             outputs = outputs[:, ignore_first_k:]
#             scores = F.softmax(outputs, dim=1)
#             entropy = torch.log(torch.sum(-scores*torch.log(scores), dim=1))
#             entropies.append(entropy)
            
#         entropies = torch.cat(entropies, dim=0).cpu().numpy().tolist()
#         return entropies

# def compute_svd(model, data_loader, conv_fea_dict=None, linear_fea_dict=None, epochs=1, printer=print):
#     if conv_fea_dict is None:
#         conv_fea_dict = model.conv_fea_dict
#     if linear_fea_dict is None:
#         linear_fea_dict = model.linear_fea_dict
#     start_time = time.time()
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model.eval()
    
#     for md in model.modules():
#         name = md.__class__.__name__
#         if name == 'Dropout':
#             md.training = True

#     fea_dict = {}
#     for val in {**conv_fea_dict, **linear_fea_dict}.values():
#         fea_dict[val] = val

#     printer(fea_dict)
#     tmp_fea_dict = {**fea_dict}
#     if 'input' in fea_dict:
#         features = {'input': []}
#         tmp_fea_dict.pop('input')
#     else:
#         features = {}

#     fea_ext = create_feature_extractor(model, tmp_fea_dict)

#     covar, svd = {}, {}
#     for key in {**conv_fea_dict, **linear_fea_dict}.keys():
#         covar[key] = 0
#     with torch.no_grad():
#         for _ in range(int(epochs)):
#             for batch_idx, (imgs, lbls) in enumerate(tqdm(data_loader)):
#                 imgs, lbls = imgs.to(device), lbls.to(device)
#                 if 'input' in fea_dict:
#                     features['input'] = imgs

#                 feats = fea_ext(imgs)
#                 for fea_name in feats:
#                     features[fea_name] = feats[fea_name].detach()

#                 for layer in conv_fea_dict: 
#                     ks = eval(f'model.{layer}').kernel_size
#                     padding = eval(f'model.{layer}').padding
#                     f = features[conv_fea_dict[layer]]
#                     patch = F.unfold(f, ks, dilation=1, padding=padding, stride=1)
#                     fea_dim = patch.shape[1]
#                     patch = patch.permute(0, 2, 1).reshape(-1, fea_dim).double()
#                     covar[layer] += torch.mm(patch.permute(1, 0), patch)

#                 for layer in linear_fea_dict:
#                     f = features[linear_fea_dict[layer]].double().squeeze()
#                     covar[layer] += torch.mm(f.permute(1, 0), f)
        
#         for layer in covar:
#             stime = time.time()
#             U, S, _ = torch.svd(covar[layer]/epochs)
#             svd[layer] = {'U': U, 'S': torch.sqrt(S)}
#             print(f'Layer: {layer} - SVD time: {time.time() - stime:.06f}')

#     process_time = time.time() - start_time
#     printer(f'Processing time: {process_time:.04f}')
#     return svd

# def compute_retain_svd(full_svd, model, data_loader, conv_fea_dict=None, linear_fea_dict=None, epochs=1, printer=print):
#     if conv_fea_dict is None:
#         conv_fea_dict = model.conv_fea_dict
#     if linear_fea_dict is None:
#         linear_fea_dict = model.linear_fea_dict
#     start_time = time.time()
#     fea_dict = {}
#     for val in {**conv_fea_dict, **linear_fea_dict}.values():
#         fea_dict[val] = val

#     max_dim = 0
#     for k in {**conv_fea_dict, **linear_fea_dict}:
#         w = eval(f'model.{k}.weight')
#         fea_dim = w.numel() // w.shape[0]
#         max_dim = max(max_dim, fea_dim) 

#     # min_epochs = max(epochs, math.ceil(max_dim/len(data_loader.dataset)))
#     min_epochs = epochs

#     tmp_fea_dict = {**fea_dict}
#     if 'input' in fea_dict:
#         features = {'input': []}
#         tmp_fea_dict.pop('input')
#     else:
#         features = {}

#     fea_ext = create_feature_extractor(model, tmp_fea_dict)

#     covar, retain_svd = {}, {}
#     for key in {**conv_fea_dict, **linear_fea_dict}.keys():
#         covar[key] = 0
#     with torch.no_grad():
#         for _ in range(int(min_epochs)):
#             for batch_idx, (imgs, lbls) in enumerate(tqdm(data_loader)):
#                 imgs = imgs.to(device)
#                 lbls = lbls.to(device)
#                 if 'input' in fea_dict:
#                     features['input'] = imgs

#                 feats = fea_ext(imgs)
#                 for fea_name in feats:
#                     features[fea_name] = feats[fea_name].detach()

#                 for layer in conv_fea_dict: 
#                     ks = eval(f'model.{layer}').kernel_size
#                     padding = eval(f'model.{layer}').padding
#                     f = features[conv_fea_dict[layer]]
#                     patch = F.unfold(f, ks, dilation=1, padding=padding, stride=1)
#                     fea_dim = patch.shape[1]
#                     patch = patch.permute(0, 2, 1).reshape(-1, fea_dim).double()
#                     covar[layer] += torch.mm(patch.permute(1, 0), patch)

#                 for layer in linear_fea_dict:
#                     f = features[linear_fea_dict[layer]].double().squeeze()
#                     covar[layer] += torch.mm(f.permute(1, 0), f)
        
#         process_time = time.time() - start_time
#         printer(f'Processing time: {process_time:.04f}')
#         for layer in covar:
#             stime = time.time()
#             U, S = full_svd[layer]['U'].to(device), full_svd[layer]['S'].to(device)
#             M = torch.mm(torch.mm(U, torch.diag(S**2)), U.t())

#             M1 = M - covar[layer]/min_epochs
#             U1_, S1sq_, _ = torch.svd(M1)
#             retain_svd[layer] = {'U': U1_, 'S': torch.sqrt(S1sq_)}
#             printer(f'Layer: {layer} - SVD time: {time.time() - stime:.06f} - M: {M.shape}')

#     process_time = time.time() - start_time
#     printer(f'Processing time: {process_time:.04f}')
#     return retain_svd
      
# def freeze_norm_stats(net):
#     try:
#         for m in net.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.eval()
#             if isinstance(m, nn.BatchNorm1d):
#                 m.eval()

#     except ValueError:  
#         print("error with BatchNorm")
#         return

