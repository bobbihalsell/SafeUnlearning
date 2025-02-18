from torch.utils.data import Subset


def remove_samples_by_indices(dataset, forget_set_indices, 
                              return_forget=True, verbose=False):
    """ Remove the forget set from the dataset by indices.

    Args:
        dataset (torch.utils.data.Dataset): Usually a training dataset.
        forget_set_indices (list[int]): A list containing the indices of the forget set.
        return_forget (bool, optional): Whether to return the forget set.
        verbose (bool, optional): Whether to print dataset sizes.

    Returns:
        torch.utils.data.Dataset or tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
            - If `return_forget=True`: Returns `(retain_set, forget_set)`, where:
                - `retain_set` is the dataset after removal.
                - `forget_set` contains only the removed samples.
            - If `return_forget=False`: Returns only `retain_set`.
    """
    remaining_indices = list(set(range(len(dataset))) - set(forget_set_indices))
    retain_set = Subset(dataset, remaining_indices)
    forget_set = Subset(dataset, forget_set_indices)

    if verbose:
        print(f"Original dataset size: {len(dataset)}")
        print(f"Retained dataset size: {len(retain_set)}")
        print(f"Forget set size: {len(forget_set)}")

    return (retain_set, forget_set) if return_forget else retain_set


def remove_classes(dataset, forget_labels,
                   return_forget=True, verbose=False):
    """ Remove certain labels from the dataset. These labels constitute the forget set.

    Args:
        dataset (torch.utils.data.Dataset): Usually a training dataset.
        forget_set_indices (list[int]): A list containing the labels of the classes to remove.
        return_forget (bool, optional): Whether to return the forget set.
        verbose (bool, optional): Whether to print dataset sizes.

    Returns:
        torch.utils.data.Dataset or tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
            - If `return_forget=True`: Returns `(retain_set, forget_set)`, where:
                - `retain_set` is the dataset after removal.
                - `forget_set` contains only the removed samples.
            - If `return_forget=False`: Returns only `retain_set`.
    """
    remaining_indices = [i for i, (_, label) in enumerate(dataset) if label not in forget_labels]
    forget_indices = [i for i, (_, label) in enumerate(dataset) if label in forget_labels]
    retain_set = Subset(dataset, remaining_indices)
    forget_set = Subset(dataset, forget_indices)

    if verbose:
        print(f"Original dataset size: {len(dataset)}")
        print(f"Retained dataset size: {len(retain_set)}")
        print(f"Forget set size: {len(forget_set)}")

    return (retain_set, forget_set) if return_forget else retain_set
