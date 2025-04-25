import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module
from omegaconf import DictConfig, OmegaConf, MissingMandatoryValue
import hydra
import json
from typing import Dict, Any, Tuple
from utils import setup_device, set_seed, initialize_dataloaders
from importmodel import ImportModel
from cfg_validator import InputValidator
from collections import defaultdict


class EvaluationValidator(InputValidator):
    def __init__(self, config):
        super().__init__(config)
        self._validate_required_sections(['model', 'dataset'])
        self.check_weights()
        self.verify_1_original()
    
    def _validate_dataset_params(self):
        super()._validate_dataset_params()
        self._require(self.dataset_file, 'cfg')
        self._require(self.cfg, 'batch_sizes', 'data_to_evaluate')
        self.num_workers = self.cfg.get('num_workers', 1)

    def check_weights(self):
        """
        Check if the model weights are available.
        """
        for attr_name in dir(self):
            if attr_name.endswith('_ckpt_path') and not callable(getattr(self, attr_name)):
                if getattr(self, attr_name) is None or getattr(self, attr_name) == '':
                    # delete the attribute
                    delattr(self, attr_name)

    def verify_1_original(self):
        original_models = 0
        for attr_name in dir(self):
            if 'original' in attr_name and not callable(getattr(self, attr_name)):
                original_models += 1
        if original_models > 1:
            raise ValueError(
                "More than one original model found. "
                "Please ensure only one original model is specified."
            )
        self.has_original = (original_models == 1)


class EvaluationApp(EvaluationValidator):
    """
    Evaluate unlearning performance across multiple metrics.
    Compares original model vs unlearned model.
    """
    def __init__(self, config: DictConfig):
        # Perform input validation first
        config = OmegaConf.to_container(config, resolve=True)
        super().__init__(config)

        self.device = setup_device()
        print(f'Using device: {self.device}')
        self.seed = config['seed']
        set_seed(self.seed)

        # Output directory
        self.output_dir = config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Storage for models
        self.models = {}
        
        # Initialize results container
        self.results = {}
        self.results["config"] = config

    def discover_model_checkpoints(self) -> Dict[str, str]:
        """Discover all model checkpoint paths from class attributes."""
        ckpt_paths = {}
        for attr_name in dir(self):
            # Check if the attribute name ends with _ckpt_path and isn't a method
            if attr_name.endswith('_ckpt_path') and not callable(getattr(self, attr_name)):
                model_key = attr_name[:-10]  # Remove '_ckpt_path' suffix
                ckpt_paths[model_key] = getattr(self, attr_name)
        return ckpt_paths

    def initialize_models(self) -> Dict[str, Module]:
        """Initialize all models from checkpoint paths."""
        # Get all checkpoint paths
        ckpt_paths = self.discover_model_checkpoints()
        
        # Load each model
        for model_key, ckpt_path in ckpt_paths.items():
            model_import = ImportModel(
                self.load_method,
                self.model_name,
                self.num_classes,
                self.init_path,
                ckpt_path,
                self.model_kwargs,
            )
            self.models[model_key] = model_import.model.to(self.device)
            if self.verbose:
                print(f"Initialized model '{model_key}' from {ckpt_path}")
        return self.models
    
    def get_dataloaders(self):
        """Initialize dataloaders from the dataset folder."""
        dataloaders = initialize_dataloaders(
            splits=self.data_to_evaluate.keys(),
            batch_sizes=self.data_to_evaluate,
            num_workers=self.num_workers,
            dataset_name=self.dataset_name,
            dataset_save_dir=self.dataset_save_dir,
        )
        return dataloaders
    
    def evaluate_accuracy(self, dataloaders: Dict[str, torch.utils.data.DataLoader]) -> Dict[str, Dict[str, float]]:
        """Evaluate model accuracy on different data splits."""
        results = {}
        
        for model_name, model in self.models.items():
            model.eval()
            model_results = {}
            
            for split_name, dataloader in dataloaders.items():
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for inputs, targets in dataloader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()
                
                accuracy = 100.0 * correct / total if total > 0 else 0.0
                model_results[split_name] = accuracy
            
            results[model_name] = model_results
            if self.verbose:
                print(f"{model_name} model accuracies:", end=" ")
                print(", ".join([f"{key}: {value:.2f}%" for key, value in model_results.items()]))  
        self.results["accuracy"] = results
        return results
    
    def evaluate_model_pairs(self, dataloaders: Dict[str, torch.utils.data.DataLoader]) -> Dict[str, Dict[str, Any]]:
        """Run comparative evaluations on all model pairs."""
        results = {}
        orig_model = self.models['original']
        for model_key, unl_model in self.models.items():
            if model_key != 'original':
                unl_results = {}
                if self.verbose:
                    print(f"Evaluating models: {model_key} vs original")                
                # Basic L2 distance
                unl_results["l2_distance"] = self.compute_l2_distance(orig_model, unl_model)
                # Per-parameter normalized distance
                unl_results["normalized_distance"] = self.compute_normalized_distance(orig_model, unl_model)
                # Normalized L2 distance
                unl_results["normalized_l2_distance"] = self.compute_normalized_l2_distance(orig_model, unl_model)
                # KL divergence on each data split
                unl_results["kl_divergence"] = self.compute_kl_divergence(orig_model, unl_model, dataloaders)
                results[model_key] = unl_results
        self.results["unl_distances"] = results
        return results
    
    def compute_normalized_distance(self, model_a: Module, model_b: Module) -> float:
        """Compute normalized distance between two models."""
        distance = 0
        normalization = 0
        
        for (p1, p2) in zip(model_a.parameters(), model_b.parameters()):
            current_dist = (p1.data - p2.data).pow(2).sum().item()
            current_norm = p1.data.pow(2).sum().item()
            distance += current_dist
            normalization += current_norm
        d = 1.0 * np.sqrt(distance / normalization)
        if self.verbose:
            print(f"Normalized distance: {d:.4f}")
        return d
    
    def compute_l2_distance(self, model_a: Module, model_b: Module) -> float:
        """Calculate the L2 distance (Euclidean distance) between the weights of two models."""
        distance = 0.0
        for p1, p2 in zip(model_a.parameters(), model_b.parameters()):
            if p1.data.nelement() == p2.data.nelement():
                diff = p1.data - p2.data
                distance += torch.norm(diff, p=2) ** 2
            else:
                raise ValueError("Models have different architectures or parameter configurations.")
        d = torch.sqrt(distance).item()
        if self.verbose:
            print(f"L2 distance: {d:.4f}")
        return d
    
    def model_l2_norm(self, model: Module) -> float:
        """Compute the L2 norm of the model's parameters."""
        distance = torch.zeros(1, device=self.device)
        for param in model.parameters():
            distance += torch.norm(param, p=2) ** 2
        return torch.sqrt(distance).item()
    
    def compute_normalized_l2_distance(self, model_a: Module, model_b: Module) -> float:
        """Compute the L2 norm of the difference between two normalized models' parameters."""
        norm_a = self.model_l2_norm(model_a)
        norm_b = self.model_l2_norm(model_b)
        
        distance = torch.zeros(1, device=self.device)
        for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
            normalized_param_a = param_a / norm_a
            normalized_param_b = param_b / norm_b
            distance += torch.norm(normalized_param_a - normalized_param_b, p=2) ** 2
        d = torch.sqrt(distance).item()
        if self.verbose:
            print(f"Normalized L2 distance: {d:.4f}")
        return d
    
    def compute_kl_divergence(self, model_a: Module, model_b: Module, dataloaders: Dict[str, torch.utils.data.DataLoader]) -> Dict[str, float]:
        """
        Compute KL divergence between two model outputs.
        KL(model_a || model_b)
        """
        results = {}
        
        for split_name, dataloader in dataloaders.items():
            total_kl = 0.0
            sample_count = 0
            
            model_a.eval()
            model_b.eval()
            
            with torch.no_grad():
                for inputs, _ in dataloader:
                    inputs = inputs.to(self.device)
                    
                    logits_a = model_a(inputs)
                    logits_b = model_b(inputs)
                    
                    # Apply softmax to get probabilities
                    probs_a = F.softmax(logits_a, dim=1)
                    probs_b = F.softmax(logits_b, dim=1)
                    
                    # Compute KL divergence: KL(p||q)
                    kl_div = F.kl_div(
                        probs_b.log(),  # q (target distribution) in log space
                        probs_a,         # p (source distribution)
                        reduction='batchmean'
                    )
                    
                    total_kl += kl_div.item() * inputs.size(0)
                    sample_count += inputs.size(0)
            
            avg_kl = total_kl / sample_count if sample_count > 0 else 0.0
            results[split_name] = avg_kl
            if self.verbose:
                print(f"KL Divergence ({split_name}): {avg_kl:.4f}")
        
        return results
    
    def format_results(self) -> Dict[str, Any]:
        """Format results for nice presentation and reporting."""
        formatted = {
            "model_accuracy": {},
            "unl_distances": {},
            "config": self.results["config"],
        }
        
        # Add model accuracy
        if "accuracy" in self.results:
            formatted["model_accuracy"] = self.results["accuracy"]
        
        # Add model pair comparison results
        if "unl_distances" in self.results:
            formatted["unl_distances"] = self.results["unl_distances"]
        return formatted
    
    def run(self) -> Dict[str, Any]:
        """Run evaluations based on available models and pairs."""
        print("Initializing models...")
        # logging.info("Initializing models...")

        self.initialize_models()
        
        # logging.info("Initializing dataloaders...")
        print("Initializing dataloaders...") 
        dataloaders = self.get_dataloaders()
        
        # Always evaluate accuracy for all models
        # logging.info("Evaluating accuracy...")
        print("Evaluating accuracy...")
        self.evaluate_accuracy(dataloaders)
        
        # Only run pair comparisons if we have pairs
        if self.has_original and len(self.models) > 1:
            print("Found original model. Running comparative evaluations...")
            self.evaluate_model_pairs(dataloaders)
        else:
            if not self.has_original:
                print("No original model found. Skipping comparative evaluations.")
            elif len(self.models) <= 1:
                print("Only one model found. Skipping comparative evaluations.")
        
        # Format results for nice presentation
        formatted_results = self.format_results()
        
        # Save results to JSON file
        output_path = os.path.join(self.output_dir, "evaluation_results.json")
        with open(output_path, 'w') as f:
            json.dump(formatted_results, f, indent=2)
        
        # logging.info(f"Results saved to {output_path}")
        print(f"Results saved to {output_path}")

        # Print a summary of results
        print("\n============ Evaluation Summary ============")
        if "accuracy" in self.results:
            print("\nModel Accuracy:")
            for model_name, splits in self.results["accuracy"].items():
                print(f"  {model_name}:")
                for split_name, acc in splits.items():
                    print(f"    {split_name}: {acc:.2f}%")
        
        if self.has_original and len(self.models) > 1:
            print("\nModel Comparisons:")
            for unl_model, metrics in self.results["unl_distances"].items():
                print(f"  Unlearned Model '{unl_model}':")
                print(f"    L2 Distance: {metrics['l2_distance']:.4f}")
                print(f"    Normalized Distance: {metrics['normalized_distance']:.4f}")
                print(f"    Normalized L2 Distance: {metrics['normalized_l2_distance']:.4f}")
                print(f"    KL Divergence (val): {metrics['kl_divergence']['val']:.4f}")
        
        print("============================================\n")
        
        return formatted_results


@hydra.main(version_base=None,
            config_path="../../configs",
            config_name="evaluate")
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
    
    app = EvaluationApp(cfg)
    results = app.run()
    
    return results


if __name__ == "__main__":
    main()