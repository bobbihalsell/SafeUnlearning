import os
from typing import Dict
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import softmax
from torch.utils.data import ConcatDataset, DataLoader

from importmodel import ImportModel
from lira_utils import load_train_test_datasets
from datasets import DATASETS_TO_TRANSFORM


def generate_predictions(
    dataset_name: str,
    dataset_cfg: Dict,
    load_method: str,
    model_name: str,
    num_classes: int,
    init_path: str,
    model_kwargs: Dict,
    unlearner: str,
    num_splits: int,
    num_forgets: int,
    save_path: str,
    output_dir: Path,
    device: str,
) -> None:
    """Generates and saves model predictions for a dataset across multiple model splits and forget iterations.

    This function loads models trained with different forget iterations and splits, then generates
    predictions for both training and test datasets combined. The predictions are saved as numpy arrays.

    Args:
        dataset_name: Name of the dataset to generate predictions for.
        dataset_cfg: Configuration dictionary containing dataset parameters.
        load_method: Method to load the model architecture.
        model_name: Name of the model.
        num_classes: Number of classes in the dataset.
        init_path: Path to model initialization weights.
        model_kwargs: Additional model configuration parameters.
        unlearner: Name of the unlearning method used.
        num_splits: Number of dataset splits.
        num_forgets: Number of forgets per split.
        save_path: Path to load the dataset.
        output_dir: Directory to save the output predictions.
        device: Device to run the model on ('cuda' or 'cpu').

    Returns:
        None. Predictions are saved to disk at {output_dir}/{unlearner}/predictions/.
    """
    models_path = output_dir / unlearner / "models"
    predictions_path = output_dir / unlearner / "predictions"
    predictions_path.mkdir(parents=True, exist_ok=True)

    if os.listdir(predictions_path):
        print("Probabilities are already computed. Skipping.")
        return

    transform = DATASETS_TO_TRANSFORM[dataset_name]()

    train, test = load_train_test_datasets(dataset_name, save_path, transform)
    dataset = ConcatDataset([train, test])
    loader = DataLoader(
        dataset,
        batch_size=dataset_cfg["batch_sizes"]["val"],
        shuffle=False,
        num_workers=dataset_cfg["num_workers"],
    )

    for split_ndx in range(num_splits):
        for forget_ndx in range(num_forgets):
            ckpt_path = models_path / f"{model_name}_{split_ndx}_{forget_ndx}.pth"
            importmodel = ImportModel(
                load_method,
                model_name, 
                num_classes,
                init_path, 
                ckpt_path,
                model_kwargs, 
            )
            model = importmodel.model
            model.to(device)
            model.eval()
            predictions = []

            for images, targets in loader:
                images = images.to(device)
                with torch.no_grad():
                    logits = model(images)
                    probas = softmax(logits, dim=1).cpu().numpy()
                predictions.append(probas)

            output_path = predictions_path / f"{model_name}_{split_ndx}_{forget_ndx}.npy"
            np.save(output_path, np.concatenate(predictions))
