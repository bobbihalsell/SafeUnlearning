from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.nn.functional import softmax
from torch.utils.data import DataLoader


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
    model.eval()
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


def run(lira_model_root: Path, model_name: str, unlearner: str, num_splits: int,
        num_forgets: int, output_dir: Path, device: str):
    cifar_complete, _ = get_dataset_and_lengths(
        Path("datasets"), "cifar10", transform=get_cifar10_test_transform()
    )
    loader = DataLoader(
        cifar_complete, batch_size=1024, shuffle=False, num_workers=4
    )

    for split_ndx in range(num_splits):
        for forget_ndx in range(num_forgets):
            model = torchvision.models.resnet18(weights=None, num_classes=10)
            weights = torch.load(
                lira_model_root / unlearner / f"{model_name}_{split_ndx}_{forget_ndx}.pth",
                map_location=device,
            )
            model.load_state_dict(weights)
            model.to(device)
            model.eval()
            with torch.no_grad():
                extracted = extract_target_and_outputs(model, loader, device=device)
            _, logits = extracted
            probas = softmax(torch.Tensor(logits), dim=1).numpy()
            output_path = (
                output_dir / unlearner / f"{model_name}_{split_ndx}_{forget_ndx}.npy"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, probas)
