import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from lira_utils import get_loaders_from_indices, get_retain_forget_val_indices
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from unlearning.main import UnlearnApp


class UnlearnAppForLiRA(UnlearnApp):
    """Extension of UnlearnApp specifically to train shadow models for LiRA"""

    def __init__(self, config: DictConfig, unlearner_name: str):
        """
        Initialize the UnlearnApp

        Args:
            config: Configuration for the unlearning application.
            unlearner_name: Name of the unlearning method to be used.
        """
        if not isinstance(config, dict):
            config = OmegaConf.to_container(config, resolve=True)

        config["model"]["pretrained"] = False
        config["unlearner"]["name"] = unlearner_name
        super().__init__(OmegaConf.create(config))
        self.unlearner_name = unlearner_name

    def run(
        self, dataloaders: Dict[str, DataLoader], unlearning: bool = True
    ) -> nn.Module:
        """Executes the unlearning or finetuning process.

        Args:
            dataloaders: Dictionary containing data loaders for finetuning/unlearning.
            unlearning: If True, performs unlearning; otherwise, performs finetuning.
                Defaults to True.

        Returns:
            nn.Module: The resulting model after unlearning or finetuning.
        """
        original_model = self.load_model()
        unlearner = self.initialize_unlearner()

        print(f"{'unlearner' if unlearning else 'finetuner'} initialized")
        unlearned_model, _ = unlearner.unlearn(
            original_model,
            data_dict=dataloaders,
            verbose=self.verbose,
            **self.unlearn_params,
        )
        print(f"model {'unlearned' if unlearning else 'finetuned'}")
        return unlearned_model


def train_models(
    config: DictConfig,
    model_name: str,
    unlearner: str,
    num_splits: int,
    num_forgets: int,
    output_dir: Path,
) -> None:
    """Trains multiple shadow models with different forget sets for LiRA.

    This function handles the training process for multiple model variations, including
    original models, naive retraining, and various unlearning methods.

    Args:
        config: Configuration for model training.
        model_name: Name of the model.
        unlearner: Name of the unlearning method used.
        num_splits: Number of dataset splits.
        num_forgets: Number of forgets per split.
        output_dir: Directory to save the output predictions.

    Returns:
        None. Models are saved to disk at {output_dir}/{unlearner}/models/.
    """
    models_path = output_dir / unlearner / "models"
    models_path.mkdir(parents=True, exist_ok=True)

    unlearner_name = "finetune" if unlearner in ["original", "naive"] else unlearner
    print(f"Unlearner: {unlearner_name}")

    app = UnlearnAppForLiRA(config, unlearner_name)
    print("dataset       ", hasattr(app, "dataset_name"))

    total_models = num_splits * num_forgets
    model_num = 0

    for split_ndx in range(num_splits):
        for forget_ndx in range(num_forgets):
            model_num += 1
            target_model_path = (
                output_dir
                / unlearner
                / "models"
                / f"{model_name}_{split_ndx}_{forget_ndx}.pth"
            )

            if os.path.exists(target_model_path):
                print(
                    f"Model {model_name}_{split_ndx}_{forget_ndx}.pth already exists. Skipping."
                )
                continue

            print(
                f"Split {split_ndx + 1}/{num_splits}, Forget {forget_ndx + 1}/{num_forgets}, "
                f"Model {model_num}/{total_models}"
            )

            try:
                retain, forget, val = get_retain_forget_val_indices(
                    lira_path=output_dir / "splits",
                    split_ndx=split_ndx,
                    forget_ndx=forget_ndx,
                )
                if unlearner == "original":
                    retain = np.concatenate([retain, forget])

                loaders = get_loaders_from_indices(
                    app.dataset_name,
                    app.dataset_save_dir,
                    [retain, forget, val],
                    app.batch_sizes,
                    app.num_workers,
                )

                unlearning = unlearner not in ["original", "naive"]
                if unlearning:
                    original_model_path = (
                        output_dir
                        / "original"
                        / "models"
                        / f"{model_name}_{split_ndx}_{forget_ndx}.pth"
                    )

                    if not os.path.exists(original_model_path):
                        print(
                            f"Warning: Original model {original_model_path} not found. Skipping."
                        )
                        continue

                    app.model_ckpt_path = original_model_path

                unlearned_model = app.run(loaders, unlearning=unlearning)
                torch.save(
                    {"model_state_dict": unlearned_model.state_dict()},
                    target_model_path,
                )

            except Exception as e:
                print(
                    f"Error processing model {model_name}_{split_ndx}_{forget_ndx}: {str(e)}"
                )
