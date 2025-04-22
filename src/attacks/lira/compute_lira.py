import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray as Array
from scipy.stats import norm
from torch.utils.data import ConcatDataset

from datasets.load_datasets import load_train_val_test_datasets


class NeverAndForgotten:
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
    numerator = norm.pdf(z, forget_mean, forget_sigma)
    denominator = norm.pdf(z, forget_mean, forget_sigma) + norm.pdf(
        z, never_mean, never_sigma
    )
    return numerator / denominator


def compute_membership_probabilities(
    id_to_correct_probas: Dict[int, ProbasNeverAndForgotten],
) -> Dict[int, List[float]]:
    ndx_to_membership = defaultdict(list)

    # For every sample ndx in the dataset, we have computed the
    # confidence of the correct label
    for ndx in id_to_correct_probas:
        # Obtain the mean (mu1) and standard deviation (sigma1) for
        # the 'forgotten' vector (A)
        forget_mean = np.mean(id_to_correct_probas[ndx].forgotten)
        forget_std = np.std(id_to_correct_probas[ndx].forgotten)

        # Obtain the mean (mu0) and standard deviation (sigma0) for the
        # 'never' vector (B)
        never_mean = np.mean(id_to_correct_probas[ndx].never)
        never_std = np.std(id_to_correct_probas[ndx].never)

        # For each predicted probability in the 'forgotten' vector (A),
        # compute the membership probability
        for proba in id_to_correct_probas[ndx].forgotten:
            membership_prob = predicted_membership_probability(
                proba, forget_mean, forget_std, never_mean, never_std
            )
            ndx_to_membership[ndx].append(membership_prob)
    return ndx_to_membership


def extract_correct_probabilities(
    probas: Array, targets: Array, indices: Dict[int, NeverAndForgotten]
) -> Dict[int, ProbasNeverAndForgotten]:
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
    test_indices = np.load(output_dir / "splits" / "test_matrices.npy")

    train, _, test = load_train_val_test_datasets(dataset_name, 1, 0, 
                                                  save_path, "")
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
