from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.nn.functional import softmax
from torch.utils.data import ConcatDataset, DataLoader

from src.datasets.load_datasets import load_train_val_test_datasets
from src.attacks.lira.utils import load_model


def extract_target_and_outputs(
    model: nn.Module, loader: DataLoader, device: torch.device
):
    """Extract the true labels and predictions from a model and a data loader

    Args:
        model (nn.Module): Model that outputs logits
        loader (DataLoader): Dataloader to obtain predictions from
        device (torch.device): Device to compute on

    Returns:
        typ.Tuple[np.ndarray, np.ndarray]: (True labels, Predicted labels)
    """
    num_entries = len(loader.dataset)
    output_size = model((loader.dataset[0][0]).unsqueeze(0).to(device)).shape[-1]
    predictions = np.zeros(shape=(num_entries, output_size))
    y_true = np.zeros(num_entries, dtype=np.int64)
    batch_size = loader.batch_size
    for batch_ndx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        start = batch_ndx * batch_size
        end = start + batch_size
        predictions[start:end] = model(images).cpu().numpy()
        y_true[start:end] = targets.cpu().numpy()
    return y_true, predictions


def run(config, device):
    lira_root = config.root
    dataset = config.dataset.name
    model_name = config.model.name
    unlearner = config.unlearner.name
    num_splits = config.dataset.num_splits
    num_forgets = config.dataset.num_forgets
    save_dir = config.dataset.save_dir

    train, test = load_train_val_test_datasets(dataset, 1, 0, '', save_dir)
    dataset = ConcatDataset([train, test])
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
    )

    for split_ndx in range(num_splits):
        for forget_ndx in range(num_forgets):
            model = load_model(model_name, lira_root / unlearner / f"{model_name}_{config.seed}_{split_ndx}_{forget_ndx}.pth")
            model.to(device)
            model.eval()
            with torch.no_grad():
                extracted = extract_target_and_outputs(model, loader, device=device)
            _, logits = extracted
            probas = softmax(torch.Tensor(logits), dim=1).numpy()
            output_path = (
                lira_root / unlearner / f"{model_name}_{config.seed}_{split_ndx}_{forget_ndx}.npy"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, probas)
