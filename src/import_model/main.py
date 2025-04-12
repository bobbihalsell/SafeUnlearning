from train.image_loading import RobustImageFolder
import torch
from torch.utils.data import DataLoader
from datasets.cifar10 import get_cifar10_test_transform
from datasets.cifar100 import get_cifar100_test_transform
from datasets.imagenet import get_imagenet_test_transform
from unlearning.utils import ConfigError
from train.utils import setup_device, set_seed
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue
import torch
from urllib.parse import urlparse
import importlib
import wandb


class LoadModelApp:
    """ Perform pretraining, or model loading and saving, for a model."""
    def __init__(self, config: DictConfig):
        self.config = OmegaConf.to_container(config, resolve=True)
        
        # Set up device and seed for reproducibility
        self.device = setup_device()
        self.seed = config['seed']
        set_seed(self.seed)
        
        # Extract configuration parameters
        self.load_method = config.get('load_method', None)
        self.init_path = config.get('init_path', None)
        self.init_name = config.get('init_name', None)
        self.weight_path = config.get('weight_path', None)
        self.model_kwargs = config.get('model_kwargs', {})
        self.validate = self.config.get('validate', False)
        self.dataset_name = self.config.get('dataset', None)
        self.save_dir = self.config.get('save_dir', None)
        self.model_eval = self.config.get('model_eval', True)
        self.dataset_save_dir = self.config.get('dataset_save_dir', './data')
        self.batch_size = self.config.get('batch_size', 128)
        
        # Validate configuration
        self.validate_config()
        
        # Initialize model and dataset
        self.model = None
        self.dataloaders = None

    def validate_config(self):
        """Validate the configuration settings."""
        if not self.load_method: 
            raise ConfigError("Load method must be specified.")
            
        # Validate load method specific requirements
        
        
        if not self.dataset_name:
            raise ConfigError("Dataset name must be specified.")
        
        if not self.save_dir:
            raise ConfigError("Save directory must be specified.")
            
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

    def load_model(self):
        """ 
        Load the model based on the specified loading method from config.
        
        Supported methods:
        - function: Load by importing a class or calling a function from a module
        - torch: Load from torch hub
        """
        try:
            print(f"Loading model using method: {self.load_method}")
            
            # Get model initialization kwargs from config if provided
            self.model_kwargs = self.config.get('model_kwargs', {})
            # Ensure model_kwargs is a dictionary, even if None was provided in the config
            if self.model_kwargs is None:
                self.model_kwargs = {}
            elif self.model_kwargs:
                print(f"Using model kwargs: {self.model_kwargs}")

            # Initialize model based on method
            if self.load_method == 'function':
                try:
                    # Import the module and get the class/function
                    module = importlib.import_module(self.init_path)
                    model_init = getattr(module, self.init_name)
                    
                    # Initialize the model
                    self.model = model_init(**self.model_kwargs)
                    print(f"Initialized model using {self.init_name}")
                    
                except ImportError:
                    raise ConfigError(f"Could not import module {self.init_path}")
                except AttributeError:
                    raise ConfigError(f"{self.init_name} not found in module {self.init_path}")
                except Exception as e:
                    raise ConfigError(f"Failed to initialize model: {str(e)}")

            elif self.load_method == 'torch':
                pretrained = self.config.get('pretrained', False)
                try:
                    # Try with newer 'weights' parameter first
                    try:
                        weights_param = 'DEFAULT' if pretrained else None
                        self.model = torch.hub.load(self.init_path, self.init_name, weights=weights_param, **self.model_kwargs)
                        print(f"Loaded model from torch.hub using 'weights' parameter")
                    except (TypeError, ValueError):
                        # Fall back to older 'pretrained' parameter
                        self.model = torch.hub.load(self.init_path, self.init_name, pretrained=pretrained, **self.model_kwargs)
                        print(f"Loaded model from torch.hub using 'pretrained' parameter")
                    
                    print(f"Loaded model from torch.hub: {self.init_path}/{self.init_name} (pretrained: {pretrained})")
                    
                except Exception as e:
                    raise ConfigError(f"Failed to load model from torch.hub: {str(e)}")
            
            else:
                raise ConfigError(f"Unknown model loading method: {self.load_method}")
            
            # Load weights if weight_path is provided
            if hasattr(self, 'weight_path') and self.weight_path:
                try:
                    print(f"Loading weights from {self.weight_path}")
                    
                    # Check if file exists
                    if not os.path.exists(self.weight_path):
                        raise ConfigError(f"Weight file not found: {self.weight_path}")
                    
                    # Load the weights
                    state_dict = torch.load(self.weight_path, map_location=self.device)
                    
                    # Handle potential DataParallel wrapping
                    if isinstance(state_dict, dict) and list(state_dict.keys())[0].startswith('module.'):
                        state_dict = {k[7:]: v for k, v in state_dict.items()}
                    
                    # Load the state dict
                    self.model.load_state_dict(state_dict)
                    print(f"Successfully loaded weights from {self.weight_path}")
                    
                except Exception as e:
                    raise ConfigError(f"Failed to load weights: {str(e)}")
            
            # Move model to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully and set to evaluation mode on {self.device}")
            
            return self.model
            
        except Exception as e:
            raise ConfigError(f"Error in load_model: {str(e)}")

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

    def initialize_loaders(self, loader_name='train'):
        """ Initialize dataloaders from the ImageNet dataset folder."""
        transform = self.get_transform()

        # Load datasets for each split
        splits = []
        if self.validate:
            splits.append(loader_name) 
        if self.eval_model:
            splits.append('test') 
        if not splits:
            return None
        
        dataloaders = {}
        for split in splits:
            split_dir = os.path.join(self.dataset_save_dir, self.dataset_name, split)
            if not os.path.exists(split_dir):
                split_dir = os.path.join(self.dataset_save_dir, split)
                if not os.path.exists(split_dir):
                    raise ConfigError(f'Dataset directory not found at {split_dir}. ' 
                                    f'Please ensure the dataset is in ImageFolder format.')

            print(f"Loading {split} dataset from {split_dir}")
            dataset = RobustImageFolder(root=split_dir, transform=transform)
            print(f"Found {len(dataset)} images in {len(dataset.classes)} classes")
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=(split == 'train'),  # Only shuffle train set
                pin_memory=True
            )
    
        return dataloaders

    def validate_model_data_compatibility(self, dataloaders, loader_name='train'):
        """ Validate model and data compatibility. """
        # Check if the model and data are compatible
        loader = dataloaders.get(loader_name)

        if not loader:
            raise ConfigError(f"No {loader_name} dataloader found for compatibility check")
        
        batch = next(iter(loader))
        
        # Handle different batch structures
        if isinstance(batch, tuple) and len(batch) >= 2:
            inputs, labels = batch[0], batch[1]
        elif isinstance(batch, list) and len(batch) >= 2:
            inputs, labels = batch[0], batch[1]
        else:            
            raise ConfigError(f"Unrecognized batch structure: {type(batch)}. "
                            f"Please update the compatibility check to handle this type.")
        
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        
        try:
            # Attempt a forward pass to check compatibility
            outputs = self.model(inputs)
            
            # Optional: Check number of classes for classification models
            if len(outputs.shape) > 1:
                num_classes = outputs.shape[1]
                dataset_classes = len(loader.dataset.classes) if hasattr(loader.dataset, 'classes') else None
                print(f"Model output classes: {num_classes}")
                if dataset_classes:
                    print(f"Dataset classes: {dataset_classes}")
                    if num_classes != dataset_classes:
                        print(f"Warning: Model outputs {num_classes} classes, but dataset has {dataset_classes} classes")
            print("Model and data are compatible!")
            return True

        except Exception as e:
            print(f"Compatibility check failed: {str(e)}")
            print(f"Error details: {e}")
            return False
        
    def eval_model(self, criterion, val_dl):
        self.model.eval()
        device = setup_device()
        val_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for batch in val_dl:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_dl.dataset)
        val_acc = 100.0 * correct / total

        return val_loss, val_acc
    
    def save_model(self, path=None):
        """Save the model to the specified path and optionally to wandb artifacts."""
        if self.model is None:
            raise ConfigError("Model must be loaded before saving.")
            
        if path is None:
            model_name = self.init_name.split('.')[-1] if self.init_name else "model"
            path = os.path.join(self.save_dir, f"{model_name}_{self.dataset_name}.pt") 
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
        
        # Save to wandb if configured
        if 'wandb' in self.config and self.config['wandb'].get('enabled', False):
            if wandb.run is None:
                wandb.init(
                    project=self.config['wandb'].get('project', 'model-eval'),
                    name=self.config['wandb'].get('name', f"{self.init_name}_{self.dataset_name}"),
                    config=self.config
                )
            
            # Log model as artifact
            artifact = wandb.Artifact(
                name=f"model-{self.init_name}-{self.dataset_name}",
                type="model",
                description=f"Trained {self.init_name} model for {self.dataset_name}"
            )
            artifact.add_file(path)
            wandb.log_artifact(artifact)
            
            print(f"Model saved to wandb artifact: {artifact.name}")
        
        return path
    
    def run(self):
        """Run the complete model loading, validation, evaluation, and saving process."""
        print("Starting LoadModelApp run...")
        
        try:
            self.load_model()
            loaders = self.initialize_loaders()
            # Validate model and data compatibility
            if self.validate:
                self.validate_model_data_compatibility(loaders)
            # Evaluate model
            if self.model_eval:
                criterion = torch.nn.CrossEntropyLoss()
                _, test_acc = self.eval_model(criterion=criterion, val_dl=loaders['test'])
                print(f"Test accuracy: {test_acc:.2f}%")
            # Save model
            save_path = self.save_model()
            
            print(f"LoadModelApp run completed successfully. Model saved to {save_path}")
            return True
            
        except Exception as e:
            print(f"Error during LoadModelApp run: {str(e)}")
            raise
        
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

    app = LoadModelApp(config=cfg)
    app.run()

if __name__ == "__main__":
    main()
