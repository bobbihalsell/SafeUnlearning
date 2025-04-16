from torch.utils.data import Subset
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
import pickle


def remove_samples_by_indices(dataset: torch.utils.data.Dataset,
                              forget_set_indices: list[int],
                              return_forget: bool = True,
                              verbose: bool = False):
    """ Remove the forget set from the dataset by indices.

    Args:
        dataset (torch.utils.data.Dataset): Usually a training dataset.
        forget_set_indices (list[int]): A list containing the indices of the 
        forget set.
        return_forget (bool, optional): Whether to return the forget set.
        verbose (bool, optional): Whether to print dataset sizes.

    Returns:
        torch.utils.data.Dataset or tuple[torch.utils.data.Dataset, 
        torch.utils.data.Dataset]:
            - If `return_forget=True`: Returns `(retain_set, forget_set)`, 
            where:
                - `retain_set` is the dataset after removal.
                - `forget_set` contains only the removed samples.
            - If `return_forget=False`: Returns only `retain_set`.
    """
    remaining_indices = list(
                            set(range(len(dataset))) - set(forget_set_indices)
                            )
    retain_set = Subset(dataset, remaining_indices)
    forget_set = Subset(dataset, forget_set_indices)

    if verbose:
        print(f"Original dataset size: {len(dataset)}")
        print(f"Retained dataset size: {len(retain_set)}")
        print(f"Forget set size: {len(forget_set)}")

    return (retain_set, forget_set) if return_forget else retain_set


def remove_samples_by_class(train_data: torch.utils.data.Dataset, 
                            classes_to_forget: list[int], 
                            num_to_forget: int,
                            return_forget: bool = True,
                            verbose: bool = False):
    """
    Perform instance-based removal and create retain/forget data loaders for 
    multiple classes. Train_data is a dataset not a loader
    
    Args:
        train_data: The original training dataset
        classes_to_forget: List of class indices to forget
        num_to_forget: Number of samples to forget per class
        return_forget (bool, optional): Whether to return the forget set.
        verbose (bool, optional): Whether to print dataset sizes.
        
    Returns:
        torch.utils.data.Dataset or tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
            - If `return_forget=True`: Returns `(retain_set, forget_set)`, where:
                - `retain_set` is the dataset after removal.
                - `forget_set` contains only the removed samples.
            - If `return_forget=False`: Returns only `retain_set`.
    """
    # Get the targets as a numpy array
    if isinstance(train_data, torch.utils.data.Subset):
        if hasattr(train_data.dataset, "targets"):
            targets = np.array(train_data.dataset.targets)[train_data.indices]
        else:
            targets = np.array([train_data.dataset[i][1] 
                                for i in train_data.indices])
    else:
        if hasattr(train_data, "targets"):
            targets = np.array(train_data.targets) 
        else: 
            targets = np.array([train_data[i][1] for i in range(len(train_data))])

    forget_indices = sorted([
        idx 
        for class_idx in classes_to_forget
        for idx in np.random.choice(
            np.where(targets == class_idx)[0], num_to_forget, replace=False
        )
    ])
    return remove_samples_by_indices(train_data, 
                                     forget_indices, 
                                     return_forget=return_forget, 
                                     verbose=verbose)


def remove_classes(dataset: torch.utils.data.Dataset,
                   forget_labels: list[int],
                   return_forget: bool = True, 
                   verbose: bool = False):
    """ 
    Remove certain labels from the dataset. These labels constitute the 
    forget set.

    Args:
        dataset (torch.utils.data.Dataset): Usually a training dataset.
        forget_labels (list[int]): A list containing the labels of the classes 
            to remove.
        return_forget (bool, optional): Whether to return the forget set.
        verbose (bool, optional): Whether to print dataset sizes.

    Returns:
        torch.utils.data.Dataset or tuple[torch.utils.data.Dataset, 
        torch.utils.data.Dataset]:
            - If `return_forget=True`: Returns `(retain_set, forget_set)`, 
            where:
                - `retain_set` is the dataset after removal.
                - `forget_set` contains only the removed samples.
            - If `return_forget=False`: Returns only `retain_set`.
    """
    remaining_indices = [i for i, (_, label) in enumerate(dataset) 
                         if label not in forget_labels]
    forget_indices = [i for i, (_, label) in enumerate(dataset) 
                      if label in forget_labels]
    retain_set = Subset(dataset, remaining_indices)
    forget_set = Subset(dataset, forget_indices)

    if verbose:
        print(f"Original dataset size: {len(dataset)}")
        print(f"Retained dataset size: {len(retain_set)}")
        print(f"Forget set size: {len(forget_set)}")

    return (retain_set, forget_set) if return_forget else retain_set


def save_loaders(path, **loaders):
    """
    Save datasets and their configuration settings.

    Args:
        path (str): Directory where datasets and configs will be saved.
        **loaders: Named DataLoaders in the form `name=loader`.
    """
    os.makedirs(path, exist_ok=True)
    
    # We need to save the actual configurations alongside the loaders
    # because DataLoader doesn't expose all settings as attributes
    loader_configs = {}
    
    # Save each dataset
    for name, loader in loaders.items():
        file_path = f"{path}/{name}_dataset.pkl"

        if name == 'val' and loader is None:
            continue

        # Save the dataset
        with open(file_path, "wb") as f:
            pickle.dump(loader.dataset, f)

        # Store batch_size (the only reliably accessible attribute)
        loader_configs[name] = {"batch_size": loader.batch_size}

    # Save the complete config file with a special name
    with open(f"{path}/loader_configs.pkl", "wb") as f:
        pickle.dump(loader_configs, f)
    print(f"Datasets saved to {path}")


def load_loaders(datapath, 
                 shuffle_settings=None, 
                 num_workers=4, 
                 pin_memory=True):
    """
    Load datasets and create DataLoaders with specified settings.

    Args:
        datapath (str): Directory where datasets are stored.
        shuffle_settings (dict, optional): Dictionary of shuffle settings by 
            loader name.
        num_workers (int, optional): Number of worker processes.
        pin_memory (bool, optional): Whether to pin memory.

    Returns:
        dict: A dictionary mapping dataset names to DataLoaders.
    """
    # Default shuffle settings if none provided
    if shuffle_settings is None:
        shuffle_settings = {
            'train': True,
            'retain': True,
            'forget': True,
            'val': False,
            'test': False
        }
    
    loaders = {}
    # Load configurations
    try:
        with open(f"{datapath}/loader_configs.pkl", "rb") as f:
            loader_configs = pickle.load(f)
    except FileNotFoundError:
        loader_configs = {}
    # Find all dataset files
    dataset_files = [f for f in os.listdir(datapath) 
                     if f.endswith("_dataset.pkl")]
    
    for file in dataset_files:
        name = file.replace("_dataset.pkl", "")
        # Load dataset
        with open(f"{datapath}/{file}", "rb") as f:
            dataset = pickle.load(f)
        # Get saved batch size or use default
        config = loader_configs.get(name, {})
        batch_size = config.get("batch_size", 32)
        # Get shuffle setting for this loader 
        shuffle = shuffle_settings[name]
        # Create DataLoader
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    return loaders


def train_val_split(train_dataset,
                    val_ratio,
                    verbose=True):
    """ Perform a train/validation dataset split based on val ratio.

    Args:
        train_dataset (Dataset)
        val_ratio (float): Validation ratio

    Returns:
        train_data_subset, val_dataset
    """
    if verbose:
        print(f'Creating validation set with {val_ratio*100:.1f}% '
              ' of training data.')

    train_size = len(train_dataset)
    indices = list(range(train_size))

    np.random.shuffle(indices)

    val_size = int(val_ratio * train_size)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    # Create subsets
    val_dataset = Subset(train_dataset, val_indices)
    train_data_subset = Subset(train_dataset, train_indices)

    return train_data_subset, val_dataset


def get_all_loaders(train_data, 
                    test_data, 
                    method='instances', 
                    val_ratio=0.0, 
                    batch_sizes=None, 
                    shuffle_settings=None, 
                    random_seed=42,
                    **kwargs):
    """
    Complete pipeline to load data, create data splits, and save loaders.

    Args:
        train_path: Path to the training data file
        test_path: Path to the test data file
        method: Method to use for splitting 
            ('instances', 'class_instances', or 'class')
        batch_sizes: Dictionary containing the batch sizes for each loader type
                    Default: {'forget': 64, 'retain': 64, 'val': 64, 
                                'train': 128, 'test': 128}
        shuffle_settings: Dictionary specifying shuffle settings for each 
            loader type
                         Default: {'forget': True, 'retain': True, 
                         'train': True, 'test': False}
        **kwargs: Additional arguments including:
                 - return_forget: Whether to return forget dataset 
                    (default: True)
                 - verbose: Whether to print progress information 
                    (default: False)
                 - forget_set_indices: Indices to forget for 'instances' 
                    method (default: [0])
                 - forget_labels: Labels to forget for class-based methods 
                    (default: [0])
                 - num_to_forget: Number of instances to forget per class 
                    (default: 1)
                 - num_workers: Number of worker processes for data loading 
                    (default: 4)

    Returns:
        Tuple containing (forget_loader, retain_loader, train_loader, 
                            test_loader)
    """
    # Default configurations
    default_batch_sizes = {'forget': 64, 'retain': 64, 'val': 64, 
                           'train': 128, 'test': 128}
    
    default_shuffle = {'forget': True, 'retain': True, 'train': True, 
                       'val': False, 'test': False}
    
    # Use provided configs or defaults
    batch_sizes = batch_sizes or default_batch_sizes
    shuffle_settings = shuffle_settings or default_shuffle
    
    # Extract other kwargs with defaults
    return_forget = kwargs.get('return_forget', True)
    verbose = kwargs.get('verbose', False)
    num_workers = kwargs.get('num_workers', 4)

    train_data_subset, val_dataset = train_val_split(train_data, 
                                                     val_ratio, 
                                                     verbose=verbose)
    if len(val_dataset) == 0:
        val_dataset = None

    # Create forget/retain splits based on specified method
    if method == 'instances':
        forget_set_indices = kwargs['forget_set_indices']
        if verbose:
            print(f"Removing {len(forget_set_indices)} instances by indices")
        retain_dataset = remove_samples_by_indices(train_data_subset, 
                                                   forget_set_indices, 
                                                   return_forget, 
                                                   verbose)

    elif method == 'class_instances':
        forget_labels = kwargs['forget_labels']
        num_to_forget = kwargs['num_to_forget']
        if verbose:
            print(
                  f"Removing {num_to_forget} instances from each of classes "
                  f"{forget_labels}"
                  )

        retain_dataset = remove_samples_by_class(train_data_subset, 
                                                 forget_labels, 
                                                 num_to_forget, 
                                                 return_forget, 
                                                 verbose)

    elif method == 'class':
        forget_labels = kwargs['forget_labels']
        if verbose:
            print(f"Removing all instances of classes {forget_labels}")
        retain_dataset = remove_classes(train_data_subset, 
                                        forget_labels, 
                                        return_forget, 
                                        verbose)

    else:
        raise ValueError(
                         f"Unknown method: {method}. Choose from 'instances', "
                         f" 'class_instances', or 'class'."
                         )

    # Separate retain and forget datasets if both were returned
    if (return_forget 
        and isinstance(retain_dataset, tuple) 
        and len(retain_dataset) == 2):
        retain_dataset, forget_dataset = retain_dataset
    elif return_forget:
        raise ValueError("Expected return_forget=True to return a tuple of "
                         "(retain, forget) datasets")

    # Create loader configurations
    loader_configs = {
        'forget': (forget_dataset, batch_sizes['forget'], 
                   shuffle_settings['forget']),
        'retain': (retain_dataset, batch_sizes['retain'], 
                   shuffle_settings['retain']),
        'train': (train_data_subset, batch_sizes['train'], 
                  shuffle_settings['train']),
        'test': (test_data, batch_sizes['test'], 
                 shuffle_settings['test'])
    }

    # Add validation loader if validation set exists
    if val_dataset is not None:
        loader_configs['val'] = (val_dataset,
                                 batch_sizes.get('val', 
                                                 default_batch_sizes['val']), 
                                 shuffle_settings.get('val', 
                                                      default_shuffle['val']))

    loaders = {}
    for name, (dataset, bs, shuffle) in loader_configs.items():
        loaders[name] = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )

    return  (
            loaders['forget'], 
            loaders['retain'], 
            loaders['train'], 
            loaders.get('val'),  # May be None if val_ratio=0
            loaders['test']
            )
