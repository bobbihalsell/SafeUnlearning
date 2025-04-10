import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray as Array
from scipy.stats import norm


def predicted_membership_probability(
    z, forget_mean, forget_sigma, never_mean, never_sigma
):
    numerator = norm.pdf(z, forget_mean, forget_sigma)
    denominator = norm.pdf(z, forget_mean, forget_sigma) + norm.pdf(
        z, never_mean, never_sigma
    )
    return numerator / denominator


def compute_membership_probabilities(id_to_correct_probas):
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
            membership_prob = predicted_membership_probability(proba, forget_mean, forget_std, never_mean, never_std)
            ndx_to_membership[ndx].append(membership_prob)
    return ndx_to_membership


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
        rep = f"Probas Never {np.mean(self.never):.3f}+-{np.std(self.never):.3f} "
        rep += f"Forgotten  {np.mean(self.forgotten):.3f}+-{np.std(self.forgotten):.3f}"
        return rep


def extract_correct_probabilities(
    probas: Array, targets: Array, indices: Dict[int, NeverAndForgotten]
):
    indices_to_correct_probas_never_and_forgotten = {}
    for ndx in sorted(indices):
        indices_to_correct_probas_never_and_forgotten[ndx] = ProbasNeverAndForgotten()
        forgotten = indices[ndx].forgotten
        for split_ndx, forget_ndx in forgotten:
            proba = probas[ndx, split_ndx, forget_ndx]
            correct_proba = proba[targets[ndx]]
            indices_to_correct_probas_never_and_forgotten[ndx].forgotten.append(
                correct_proba
            )
        for split_ndx, forget_ndx in indices[ndx].never:
            proba = probas[ndx, split_ndx, forget_ndx]
            correct_proba = proba[targets[ndx]]
            indices_to_correct_probas_never_and_forgotten[ndx].never.append(
                correct_proba
            )
    return indices_to_correct_probas_never_and_forgotten


def reconstruct_split_and_forget(
    lira_root: Path,
    num_splits: int = 64,
    num_forgets: int = 10,
    num_elements: int = 2375,
):
    reconstructed = np.zeros((num_splits, num_forgets, num_elements), dtype=int)
    for split_ndx in range(num_splits):
        forgets = lira_root / str(split_ndx) / "forgets.npy"
        data = np.load(forgets)
        reconstructed[split_ndx] = data
    return reconstructed


# TODO: fix hardcoded values
def get_preds(lira_preds: Path, unlearner: str):
    storage = np.zeros(shape=(64, 10, 60_000, 10))
    unlearner_dir = lira_preds / unlearner
    ndx = 0
    for split_ndx in range(64):
        for forget_ndx in range(10):
            model = f"resnet18_0_{split_ndx}_{forget_ndx}.npy"
            preds = np.load(unlearner_dir / model)
            assert preds.shape == (60_000, 10)
            storage[split_ndx][forget_ndx] = preds
            ndx += 1
    res = np.transpose(storage, (2, 0, 1, 3))
    return res


def run(args):
    unlearner = args.unlearner
    lira_root = args.lira_root
    lira_preds = args.lira_preds
    num_splits = args.num_splits
    num_forgets = args.num_forgets
    num_elements = args.num_elements
    test_indices = np.load(args.lira_root / "test_matrices.npy")

    # TODO: fix hardcoded dataset
    cifar_complete, _ = get_dataset_and_lengths(
        Path("datasets"), "cifar10", transform=get_cifar10_test_transform()
    )
    targets = np.concatenate([cifar_complete.datasets[ndx].targets for ndx in range(2)])

    forgets_splits_and_forget_indices = reconstruct_split_and_forget(lira_root, num_splits, num_forgets, num_elements)

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
            for value in forgets_splits_and_forget_indices[split_ndx, forget_ndx]:
                id_to_forgotten_never_seen[value].forgotten.append(
                    (split_ndx, forget_ndx)
                )

    storage = get_preds(lira_preds, unlearner)
    id_to_correct_probas = extract_correct_probabilities(
        storage, targets, id_to_forgotten_never_seen
    )
    ndx_to_membership = compute_membership_probabilities(id_to_correct_probas)
    with open(f"lira/{unlearner}_membership.npy", "wb") as out_fo:
        pickle.dump(obj=ndx_to_membership, file=out_fo)
