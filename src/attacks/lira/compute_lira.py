import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray as Array
from scipy.stats import norm
from torch.utils.data import ConcatDataset

from lira_utils import load_train_test_datasets


class NeverAndForgotten:
    """A class to store indices of samples that were in test and forget set.

    Args:
        never: List of tuples containing (split_index, forget_index) for test samples.
        forgotten: List of tuples containing (split_index, forget_index) for forgotten samples.
    """

    def __init__(
        self,
        never: Optional[List[Tuple[int, int]]] = None,
        forgotten: Optional[List[Tuple[int, int]]] = None,
    ):
        self.never = never if never is not None else []
        self.forgotten = forgotten if forgotten is not None else []

    def assert_no_overlap(self):
        unique_never = set(self.never)
        unique_forgotten = set(self.forgotten)
        assert len(unique_never.intersection(unique_forgotten)) == 0

    def __repr__(self):
        rep = f"Never {len(self.never)} {self.never[:10]}, Forgotten "
        rep += f"{len(self.forgotten)} {self.forgotten[:10]}"
        return rep


class ProbasNeverAndForgotten:
    """A class to store probabilities for test and forgotten samples.

    Args:
        never: List of probabilities for test samples.
        forgotten: List of probabilities for forgotten samples.
    """

    def __init__(
        self,
        never: Optional[List[float]] = None,
        forgotten: Optional[List[float]] = None,
    ):
        self.never = never if never is not None else []
        self.forgotten = forgotten if forgotten is not None else []

    def __repr__(self):
        n_mean = f"{np.mean(self.never):.3f}"
        n_std = f"{np.std(self.never):.3f}"
        f_mean = f"{np.mean(self.forgotten):.3f}"
        f_std = f"{np.std(self.forgotten):.3f}"
        rep = f"Probas Never {n_mean}+-{n_std} Forgotten {f_mean}+-{f_std}"
        return rep


def predicted_membership_probability(
    z: float,
    forget_mean: float,
    forget_sigma: float,
    never_mean: float,
    never_sigma: float,
) -> float:
    """Computes the membership probability using gaussian distributions.

    Args:
        z: The input value to compute probability for.
        forget_mean: Mean of the forgotten samples distribution.
        forget_sigma: Standard deviation of the forgotten samples distribution.
        never_mean: Mean of the test samples distribution.
        never_sigma: Standard deviation of the test samples distribution.

    Returns:
        float: The computed membership probability.
    """
    numerator = norm.pdf(z, forget_mean, forget_sigma + 1e-30)
    denominator = norm.pdf(z, forget_mean, forget_sigma + 1e-30) \
                + norm.pdf(z, never_mean, never_sigma + 1e-30)
    return numerator / denominator


def compute_membership_probabilities(
    id_to_correct_probas: Dict[int, ProbasNeverAndForgotten],
) -> Dict[int, List[float]]:
    """Computes membership probabilities for all samples in forget set.

    Args:
        id_to_correct_probas: Dictionary mapping sample IDs to their probabilities.

    Returns:
        Dictionary mapping sample IDs to their membership probabilities.
    """
    ndx_to_membership = defaultdict(list)

    for ndx in id_to_correct_probas:
        forget_mean = np.mean(id_to_correct_probas[ndx].forgotten)
        forget_std = np.std(id_to_correct_probas[ndx].forgotten)
        never_mean = np.mean(id_to_correct_probas[ndx].never)
        never_std = np.std(id_to_correct_probas[ndx].never)

        for proba in id_to_correct_probas[ndx].forgotten:
            membership_prob = predicted_membership_probability(
                proba, forget_mean, forget_std, never_mean, never_std
            )
            ndx_to_membership[ndx].append(membership_prob)
    return ndx_to_membership


def extract_correct_probabilities(
    probas: Array,
    targets: Array,
    indices: Dict[int, NeverAndForgotten]
) -> Dict[int, ProbasNeverAndForgotten]:
    """Extracts correct class probabilities for test and forgotten samples.

    Args:
        probas: Array of probabilities.
        targets: Array of target labels.
        indices: Dictionary mapping sample IDs to their never/forgotten status.

    Returns:
        Dictionary mapping sample IDs to their correct class probabilities.
    """
    indices_to_correct_probas_nvr_and_fgtn = {}
    for ndx in sorted(indices):
        indices_to_correct_probas_nvr_and_fgtn[ndx] = ProbasNeverAndForgotten()
        forgotten = indices[ndx].forgotten
        for split_ndx, forget_ndx in forgotten:
            proba = probas[ndx, split_ndx, forget_ndx]
            correct_proba = proba[targets[ndx]]
            indices_to_correct_probas_nvr_and_fgtn[ndx].forgotten.append(
                correct_proba
            )
        for split_ndx, forget_ndx in indices[ndx].never:
            proba = probas[ndx, split_ndx, forget_ndx]
            correct_proba = proba[targets[ndx]]
            indices_to_correct_probas_nvr_and_fgtn[ndx].never.append(
                correct_proba
            )
    return indices_to_correct_probas_nvr_and_fgtn


def reconstruct_split_and_forget(root: Path, num_splits: int) -> Array:
    """Reconstructs the split and forget data from saved files.

    Args:
        root: Root directory containing the splits.
        num_splits: Number of splits used for the attack.

    Returns:
        Array containing reconstructed split and forget data.
    """
    reconstructed = []
    for split_ndx in range(num_splits):
        forgets = root / "splits" / str(split_ndx) / "forgets.npy"
        data = np.load(forgets)
        reconstructed.append(data)
    return np.stack(reconstructed)


def get_preds(
    root: Path,
    model: str,
    unlearner: str,
    num_splits: int,
    num_forgets: int,
) -> Array:
    """Retrieves probabilities from saved files.

    Args:
        root: Root directory containing probabilities.
        model: Model name.
        unlearner: Unlearning method used.
        num_splits: Number of splits.
        num_forgets: Number of forgets per split.

    Returns:
        Array containing all probabilities.
    """
    storage = []
    for split_ndx in range(num_splits):
        forgets = []
        for forget_ndx in range(num_forgets):
            fname = f"{model}_{split_ndx}_{forget_ndx}.npy"
            preds = np.load(root / unlearner / "predictions" / fname,
                            allow_pickle=True)
            forgets.append(preds)
        storage.append(np.stack(forgets))
    res = np.transpose(np.stack(storage), (2, 0, 1, 3))
    return res


def lira_score(
    dataset_name: str,
    model_name: str,
    unlearner: str,
    num_splits: int,
    num_forgets: int,
    save_path: str,
    output_dir: Path,
) -> None:
    """Computes LiRA scores for predicting membership inference.

    Args:
        dataset_name: Name of the dataset.
        model_name: Name of the model.
        unlearner: Name of the unlearning method.
        num_splits: Number of splits.
        num_forgets: Number of forgets per split.
        save_path: Path to load data from.
        output_dir: Directory to store output files.

    Raises:
        ValueError: If a forgotten index is not found in test indices.
    """
    test_indices = np.load(output_dir / "splits" / "test_matrices.npy")

    train, test = load_train_test_datasets(dataset_name, save_path)
    dataset = ConcatDataset([train, test])
    targets = np.array([label for (image, label) in dataset])

    forgets_splits_and_indices = reconstruct_split_and_forget(
        output_dir, num_splits
    )
    id_to_forgotten_never_seen = {}
    for split_ndx, row in enumerate(test_indices):
        for value in row:
            if value not in id_to_forgotten_never_seen:
                id_to_forgotten_never_seen[value] = NeverAndForgotten()
            id_to_forgotten_never_seen[value].never.extend(
                [(split_ndx, forget_ndx) for forget_ndx in range(num_forgets)]
            )
    for split_ndx in range(num_splits):
        for forget_ndx in range(num_forgets):
            for value in forgets_splits_and_indices[split_ndx, forget_ndx]:
                if value not in id_to_forgotten_never_seen:
                    raise ValueError(
                        f"Forget index {value} (split {split_ndx}, "
                        f"forget {forget_ndx}) was not found in test indices. "
                        f"Try increasing the number of splits or ensuring test "
                        f"indices cover all forgotten samples."
                    )
                id_to_forgotten_never_seen[value].forgotten.append(
                    (split_ndx, forget_ndx)
                )

    storage = get_preds(output_dir, model_name, unlearner, 
                        num_splits, num_forgets)
    id_to_correct_probas = extract_correct_probabilities(
        storage, targets, id_to_forgotten_never_seen
    )
    ndx_to_membership = compute_membership_probabilities(id_to_correct_probas)
    with open(output_dir / f"{unlearner}_membership.npy", "wb") as f:
        pickle.dump(obj=ndx_to_membership, file=f)
