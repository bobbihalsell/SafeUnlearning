import os
from typing import Dict
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import softmax
from torch.utils.data import ConcatDataset, DataLoader

from attacks.lira.utils import load_model
from datasets.load_datasets import load_train_val_test_datasets
from datasets import DATASETS_TO_TRANSFORM


def run(
    dataset_name: str,
    dataset_cfg: Dict,
    model_name: str,
    num_classes: int,
    unlearner: str,
    num_splits: int,
    num_forgets: int,
    save_path: str,
    output_dir: Path,
    device: str,
) -> None:
    models_path = output_dir / unlearner / "models"
    predictions_path = output_dir / unlearner / "predictions"
    predictions_path.mkdir(parents=True, exist_ok=True)

    if os.listdir(predictions_path):
        print("Probabilities are already computed. Skipping.")
        return

    transform = DATASETS_TO_TRANSFORM[dataset_name]()

    train, _, test = load_train_val_test_datasets(
        dataset_name, 1, 0, save_path, "", transform
    )
    dataset = ConcatDataset([train, test])
    loader = DataLoader(
        dataset,
        batch_size=dataset_cfg["batch_sizes"]["val"],
        shuffle=False,
        num_workers=dataset_cfg["num_workers"],
    )

    for split_ndx in range(num_splits):
        for forget_ndx in range(num_forgets):
            model = load_model(
                model_name,
                num_classes,
                models_path / f"{model_name}_{split_ndx}_{forget_ndx}.pth",
            )
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
