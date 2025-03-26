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


def remove_samples_by_class(train_data: torch.utils.data.Dataset, 
                                            classes_to_forget: list[int], 
                                            num_to_forget: int,
                                            return_forget: bool = True,
                                            verbose: bool = False):
    """
    Perform instance-based removal and create retain/forget data loaders for multiple classes.
    train_data is a dataset not a loader
    
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
    targets = np.array(train_data.targets) if isinstance(train_data.targets, list) else train_data.targets

    forget_indices = sorted([
        idx 
        for class_idx in classes_to_forget
        for idx in np.random.choice(
            np.where(targets == class_idx)[0], num_to_forget, replace=False
        )
    ])
    return remove_samples_by_indices(train_data, forget_indices, return_forget=return_forget, verbose=verbose)


def remove_classes(dataset: torch.utils.data.Dataset,
                   forget_labels: list[int],
                   return_forget: bool = True, 
                   verbose: bool = False):
    """ Remove certain labels from the dataset. These labels constitute the forget set.

    Args:
        dataset (torch.utils.data.Dataset): Usually a training dataset.
        forget_labels (list[int]): A list containing the labels of the classes to remove.
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


def get_loaders(num_workers=4, pin_memory=True, **datasets):
    """
    Create DataLoaders for multiple datasets with arbitrary names.

    Args:
        num_workers (int): Number of workers for data loading.
        pin_memory (bool): Use pinned memory for faster GPU transfer.
        **datasets: Datasets in the form `name=(dataset, batch_size, shuffle)`.
        
    Returns:
        dict: A dictionary mapping dataset names to DataLoaders.

    Example:
        loaders = get_loaders(
                    forget=(forget_dataset, 32, True),   # Batch size = 32, shuffle
                    retain=(retain_dataset, 64, True),   # Batch size = 64, shuffle
                    test1=(test_dataset1, 128, False)    # Batch size = 128, no shuffle
        return {
            'forget': DataLoader(forget_dataset, batch_size=32, shuffle=True),
            'retain': DataLoader(retain_dataset, batch_size=64, shuffle=True),
            'test1': DataLoader(test_dataset1, batch_size=128, shuffle=False)
        }
)
    """
    loader_args = {'num_workers': num_workers, 'pin_memory': pin_memory}
    
    loaders = {}
    for name, (dataset, batch_size, shuffle) in datasets.items():
        loaders[name] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **loader_args)
    
    return loaders


def save_loaders(path, **loaders):
    """
    Save multiple datasets and DataLoader configurations to a specified path.

    Args:
        path (str): Directory where datasets and configs will be saved.
        **loaders: Named DataLoaders in the form `name=loader`.
    """
    os.makedirs(path, exist_ok=True)

    # Save each dataset and loader config
    for name, loader in loaders.items():
        # Save the dataset
        with open(f"{path}/{name}_dataset.pkl", "wb") as f:
            pickle.dump(loader.dataset, f)

        # Save loader config
        config = {
            "batch_size": loader.batch_size,
            "shuffle": loader.shuffle,
            "num_workers": loader.num_workers,
            "pin_memory": loader.pin_memory if hasattr(loader, 'pin_memory') else False,
        }
        
        with open(f"{path}/{name}_config.pkl", "wb") as f:
            pickle.dump(config, f)

    print(f"Datasets and configurations saved to {path}")


def load_loaders(datapath, num_workers=None):
    """
    Load multiple DataLoaders from saved datasets and configurations.

    Args:
        datapath (str): Directory where datasets and configs are stored.
        num_workers (int, optional): Override saved num_workers setting.

    Returns:
        dict: A dictionary mapping dataset names to DataLoaders.
    """
    loaders = {}

    # Find all dataset files
    dataset_files = [f for f in os.listdir(datapath) if f.endswith("_dataset.pkl")]
    
    for file in dataset_files:
        name = file.replace("_dataset.pkl", "")
        config_file = f"{datapath}/{name}_config.pkl"
        
        # Load dataset
        with open(f"{datapath}/{file}", "rb") as f:
            dataset = pickle.load(f)
        
        # Load config
        with open(config_file, "rb") as f:
            config = pickle.load(f)
        
        # Override num_workers if specified
        if num_workers is not None:
            config["num_workers"] = num_workers
            
        # Create DataLoader with loaded config
        loaders[name] = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=config["shuffle"],
            num_workers=config["num_workers"],
            pin_memory=config.get("pin_memory", False),
        )

    return loaders


def get_all_loaders(train_path, test_path, save_path=None, method='instances', 
                   batch_sizes=None, shuffle_settings=None, **kwargs):
    """
    Complete pipeline to load data, create forget/retain splits, and save loaders.
    
    Args:
        train_path: Path to the training data file
        test_path: Path to the test data file
        save_path: Path to save the loaders (if None, loaders won't be saved)
        method: Method to use for splitting ('instances', 'class_instances', or 'class')
        batch_sizes: Dictionary containing the batch sizes for each loader type
                    Default: {'forget': 64, 'retain': 64, 'train': 128, 'test': 128}
        shuffle_settings: Dictionary specifying shuffle settings for each loader type
                         Default: {'forget': True, 'retain': True, 'train': True, 'test': False}
        **kwargs: Additional arguments including:
                 - return_forget: Whether to return forget dataset (default: True)
                 - verbose: Whether to print progress information (default: False)
                 - forget_set_indices: Indices to forget for 'instances' method (default: [0])
                 - forget_labels: Labels to forget for class-based methods (default: [0])
                 - num_to_forget: Number of instances to forget per class (default: 1)
                 - num_workers: Number of worker processes for data loading (default: 4)
        
    Returns:
        Tuple containing (forget_loader, retain_loader, train_loader, test_loader)
    """
    # Default configurations
    default_batch_sizes = {'forget': 64, 'retain': 64, 'train': 128, 'test': 128}
    default_shuffle = {'forget': True, 'retain': True, 'train': True, 'test': False}
    
    # Use provided configs or defaults
    batch_sizes = batch_sizes or default_batch_sizes
    shuffle_settings = shuffle_settings or default_shuffle
    
    # Extract other kwargs with defaults
    return_forget = kwargs.get('return_forget', True)
    verbose = kwargs.get('verbose', False)
    num_workers = kwargs.get('num_workers', 4)
    
    # Load data
    if verbose:
        print(f"Loading data from {train_path} and {test_path}")
        
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    # Create forget/retain splits based on specified method
    if method == 'instances':
        forget_set_indices = kwargs.get('forget_set_indices', [0])
        if verbose:
            print(f"Removing {len(forget_set_indices)} instances by indices")
        retain_dataset = remove_samples_by_indices(train_data, forget_set_indices, return_forget, verbose)
        
    elif method == 'class_instances':
        forget_labels = kwargs.get('forget_labels', [0])
        num_to_forget = kwargs.get('num_to_forget', 1)
        if verbose:
            print(f"Removing {num_to_forget} instances from each of classes {forget_labels}")
        retain_dataset = remove_samples_by_class(train_data, forget_labels, num_to_forget, return_forget, verbose)
        
    elif method == 'class':
        forget_labels = kwargs.get('forget_labels', [0])
        if verbose:
            print(f"Removing all instances of classes {forget_labels}")
        retain_dataset = remove_classes(train_data, forget_labels, return_forget, verbose)
        
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'instances', 'class_instances', or 'class'.")
    
    # Separate retain and forget datasets if both were returned
    if return_forget and isinstance(retain_dataset, tuple) and len(retain_dataset) == 2:
        retain_dataset, forget_dataset = retain_dataset
    elif return_forget:
        raise ValueError("Expected return_forget=True to return a tuple of (retain, forget) datasets")
    
    # Create loader configurations
    loader_configs = {
        'forget': (forget_dataset, batch_sizes['forget'], shuffle_settings['forget']),
        'retain': (retain_dataset, batch_sizes['retain'], shuffle_settings['retain']),
        'train': (train_data, batch_sizes['train'], shuffle_settings['train']),
        'test': (test_data, batch_sizes['test'], shuffle_settings['test'])
    }
    
    # # Create DataLoaders
    # if verbose:
    #     print("Creating DataLoaders with configurations:")
    #     for name, (_, bs, shuffle) in loader_configs.items():
    #         print(f"  {name}: batch_size={bs}, shuffle={shuffle}")
            
    loaders = get_loaders(**loader_configs, num_workers=num_workers)
    forget_loader, retain_loader, train_loader, test_loader = loaders
    
    # Save loaders if path is provided
    if save_path:
        if verbose:
            print(f"Saving loaders to {save_path}")
        save_loaders(path=save_path, forget = forget_loader, retain = retain_loader, train = train_loader, test = test_loader)
    
    return forget_loader, retain_loader, train_loader, test_loader
