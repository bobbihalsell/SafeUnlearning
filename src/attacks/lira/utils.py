from pathlib import Path
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray as Array
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import timm


def get_retain_forget_val_indices(
    lira_path: Path, split_ndx: int, forget_ndx: int
) -> Tuple[Array, Array, Array]:
    retains = np.load(lira_path / str(split_ndx) / "retains.npy")
    forgets = np.load(lira_path / str(split_ndx) / "forgets.npy")
    vals = np.load(lira_path / "val_matrices.npy")

    retain_indices = retains[forget_ndx]
    forget_indices = forgets[forget_ndx]
    val_indices = vals

    assert set(retain_indices) & set(forget_indices) == set()
    assert set(retain_indices) & set(val_indices) == set()
    assert set(forget_indices) & set(val_indices) == set()
    return retain_indices, forget_indices, val_indices


def get_loaders_from_indices(
    root: Path,
    indices: List[Array],
    config: DictConfig,
) -> List[DataLoader]:
    dataset_root = root / "datasets"
    dataset_name = dataset_cfg.name
    non_augmented_dataset, _ = get_dataset_and_lengths(
        dataset_root, dataset_name, get_train_transform(dataset_name)
    )
    augmented_dataset, _ = get_dataset_and_lengths(
        dataset_root, dataset_name, get_test_transform(dataset_name)
    )

    retain_indices, forget_indices, val_indices = indices
    loaders = []
    for split_name, indices in zip(
        ["retain", "forget", "val"], [retain_indices, forget_indices, val_indices]
    ):
        data_set = get_dataset_based_on_split_state(
            non_augmented_dataset,
            augmented_dataset,
            getattr(unlearner_cfg.loaders, split_name).state,
            indices,
        )
        loader = DataLoader(
            data_set,
            batch_size=unlearner_cfg.batch_size,
            shuffle=getattr(unlearner_cfg.loaders, split_name).shuffle,
            num_workers=get_num_workers_from_shuffle(
                getattr(unlearner_cfg.loaders, split_name).shuffle
            ),
            pin_memory=False,
        )
        loaders.append(loader)

    return loaders


def load_model(model_name: str, num_classes: int, checkpoint_path: Path) -> nn.Module:
    if hasattr(torchvision.models, model_name):
        model = torchvision.models.get_model(model_name, weights=None)

        # Adjust the last layer based on model type
        if hasattr(model, "fc"):  # ResNet-style
            model.fc = nn.Linear(
                model.fc.in_features,
                num_classes
            )
        elif hasattr(model, "classifier"):
            # MobileNet, EfficientNet, VGG, DenseNet
            if isinstance(model.classifier, nn.Sequential):
                # Handle cases where classifier is Sequential
                last_layer_idx = len(model.classifier) - 1
                model.classifier[last_layer_idx] = nn.Linear(
                    model.classifier[last_layer_idx].in_features,
                    num_classes
                )
            else:
                model.classifier = nn.Linear(
                    model.classifier.in_features,
                    num_classes
                )
        else:
            raise AttributeError("Unknown classification layer "
                                 f'for {model_name}')

    else:
        print(f'Could not find {model_name} in torchvision.'
              ' Looking in timm.')
        try:
            model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=num_classes,
            )
        except Exception:
            raise AttributeError(f"{model_name} not found.")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    return model
