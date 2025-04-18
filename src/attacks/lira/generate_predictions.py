import numpy as np
from pathlib import Path
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.utils.data import ConcatDataset, DataLoader

from attacks.lira.utils import load_model
from datasets.load_datasets import load_train_val_test_datasets
from datasets.cifar10 import get_cifar10_test_transform


def run(config: DictConfig, root: Path, unlearner: str) -> None:
    dataset = config.dataset.name
    model_name = config.model.name
    num_splits = config.attack.cfg.num_splits
    num_forgets = config.attack.cfg.num_forgets
    save_dir = config.dataset.save_path
    transform = get_cifar10_test_transform()

    train, val, test = load_train_val_test_datasets(
        dataset, 1, config.dataset.val_ratio, save_dir, "", transform
    )
    dataset = ConcatDataset([train, val, test])
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    for split_ndx in range(num_splits):
        for forget_ndx in range(num_forgets):
            model = load_model(
                model_name,
                config.model.num_classes,
                root
                / unlearner
                / "models"
                / f"{model_name}_{split_ndx}_{forget_ndx}.pth",
            )
            model.to(config.device)
            model.eval()
            predictions = []

            for images, targets in loader:
                images = images.to(config.device)
                with torch.no_grad():
                    logits = model(images)
                    probas = softmax(logits, dim=1).cpu().numpy()
                predictions.append(probas)

            output_path = (
                root
                / unlearner
                / "predictions"
                / f"{model_name}_{split_ndx}_{forget_ndx}.npy"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, np.concatenate(predictions))
