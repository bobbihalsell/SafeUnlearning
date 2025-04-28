import json
import os
from typing import Any, Dict

import hydra
import torch
from eval_utils import (
    accuracy,
    kld,
    l2_distance,
    normed_distance,
    normed_l2_distance,
    visualize_evaluation_results,
)
from omegaconf import DictConfig, MissingMandatoryValue, OmegaConf
from torch.nn import Module

from cfg_validator import InputValidator
from importmodel import ImportModel
from utils import initialize_dataloaders, set_seed, setup_device


class EvaluationValidator(InputValidator):
    def __init__(self, config):
        super().__init__(config)
        self._validate_required_sections(["model", "dataset"])
        self.check_weights()
        self.verify_1_original()
        self.filter_data()

    def _validate_dataset_params(self):
        super()._validate_dataset_params()
        self._require(self.dataset_file, "splits")
        self.splits = self.splits.split()
        self._require(self.dataset_file, "cfg")
        self._require(self.cfg, "batch_sizes")
        self.num_workers = self.cfg.get("num_workers", 1)

    def check_weights(self):
        """
        Check if the model weights are available.
        """
        for attr_name in dir(self):
            if attr_name.endswith("_ckpt_path") and not callable(
                getattr(self, attr_name)
            ):
                if getattr(self, attr_name) is None or getattr(self, attr_name) == "":
                    # delete the attribute
                    delattr(self, attr_name)

    def verify_1_original(self):
        original_models = 0
        for attr_name in dir(self):
            if "original" in attr_name and not callable(getattr(self, attr_name)):
                original_models += 1
        if original_models > 1:
            raise ValueError(
                "More than one original model found. "
                "Please ensure only one original model is specified."
            )
        self.has_original = original_models == 1

    def filter_data(self):
        self.data_to_evaluate = {}
        for split in self.splits:
            if split in self.cfg["batch_sizes"]:
                self.data_to_evaluate[split] = self.cfg["batch_sizes"][split]
            else:
                raise ValueError(f"Batch size for split '{split}' not found.")


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
        print(f"Using device: {self.device}")
        self.seed = config["seed"]
        set_seed(self.seed)

        # Output directory
        self.output_dir = config["output_dir"]
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
            if attr_name.endswith("_ckpt_path") and not callable(
                getattr(self, attr_name)
            ):
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

    def evaluate_accuracies(
        self, dataloaders: Dict[str, torch.utils.data.DataLoader]
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate accuracy of all models on the given dataloaders."""
        results = {}
        for model_key, model in self.models.items():
            acc_results = accuracy(model, dataloaders, self.device)
            results[model_key] = acc_results
            if self.verbose:
                print(f"{model_key} model accuracies:", end=" ")
                print(
                    ", ".join(
                        [f"{key}: {value:.2f}%" for key, value in acc_results.items()]
                    )
                )
        self.results["accuracy"] = results
        return results

    def evaluate_model_pairs(
        self, dataloaders: Dict[str, torch.utils.data.DataLoader]
    ) -> Dict[str, Dict[str, Any]]:
        """Run comparative evaluations on all model pairs."""
        results = {}
        orig_model = self.models["original"]
        for model_key, unl_model in self.models.items():
            if model_key != "original":
                unl_results = {}
                if self.verbose:
                    print(f"Evaluating models: {model_key} vs original")
                # Basic L2 distance
                unl_results["l2_distance"] = l2_distance(orig_model, unl_model)
                if self.verbose:
                    print(f"L2 distance: {unl_results['l2_distance']:.4f}")
                # Per-parameter normalized distance
                unl_results["normalized_distance"] = normed_distance(
                    orig_model, unl_model
                )
                if self.verbose:
                    print(
                        f"Normalized distance: {unl_results['normalized_distance']:.4f}"
                    )
                # Normalized L2 distance
                unl_results["normalized_l2_distance"] = normed_l2_distance(
                    orig_model, unl_model, self.device
                )
                if self.verbose:
                    print(
                        f"Normalized L2 distance: {unl_results['normalized_l2_distance']:.4f}"
                    )
                # KL divergence on each data split
                unl_results["kl_divergence"] = kld(
                    orig_model, unl_model, dataloaders, self.device
                )
                print(
                    f"kl_divergence between {model_key} model and the original:",
                    end=" ",
                )
                print(
                    ", ".join(
                        [
                            f"{key}: {value:.4f}"
                            for key, value in unl_results["kl_divergence"].items()
                        ]
                    )
                )
                results[model_key] = unl_results
        self.results["unl_distances"] = results
        return results

    def format_results(self) -> Dict[str, Any]:
        """Format results for nice presentation and reporting."""
        formatted = {
            "model_accuracy": {},
            "unl_distances": {},
            "config": self.results["config"],
        }

        formatted["model_accuracy"] = self.results["accuracy"]
        if "unl_distances" in self.results:
            formatted["unl_distances"] = self.results["unl_distances"]
        return formatted

    def run(self) -> Dict[str, Any]:
        """Run evaluations based on available models and pairs."""
        print("Initializing models...")
        self.initialize_models()

        print("Initializing dataloaders...")
        dataloaders = self.get_dataloaders()

        # Always evaluate accuracy for all models
        print("Evaluating accuracy...")
        self.evaluate_accuracies(dataloaders)

        # Only run pair comparisons if we have original model and more than one model
        if self.has_original and len(self.models) > 1:
            self.evaluate_model_pairs(dataloaders)
        else:
            if not self.has_original:
                print("No original model found. Skipping comparative evaluations.")
            elif len(self.models) <= 1:
                print("Only one model found. Skipping comparative evaluations.")

        formatted_results = self.format_results()

        if self.id is None:
            self.id = ""
        output_path = os.path.join(self.output_dir, f"evaluation_results{self.id}.json")
        with open(output_path, "w") as f:
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
                print(
                    f"    Normalized L2 Distance: {metrics['normalized_l2_distance']:.4f}"
                )
                print(f"    KL Divergence:")
                for model_name, splits in self.results["accuracy"].items():
                    print(f"    {model_name}:")
                    for split_name, acc in splits.items():
                        print(f"        {split_name}: {acc:.4f}%")

        print("============================================\n")
        visualize_evaluation_results(formatted_results, self.output_dir)
        return formatted_results


@hydra.main(version_base=None, config_path="../../configs", config_name="evaluate")
def main(cfg: DictConfig):
    # Print the config for the user first
    print("============ Run Configuration ============")
    print(OmegaConf.to_yaml(cfg))
    print("============================================")

    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise MissingMandatoryValue(
            "Missing the following required arguments in the configuration: "
            f"{missing_keys}. \n"
            "Hint: python file.py key=value sets the appropriate value."
        )

    app = EvaluationApp(cfg)
    results = app.run()

    return results


if __name__ == "__main__":
    main()
