from attacks.GLiR.utils import create_subset, create_matched_subset
import os
import torch
import random
from train.image_loading import RobustImageFolder
from datasets.cifar10 import get_cifar10_test_transform
from datasets.cifar100 import get_cifar100_test_transform
from datasets.imagenet import get_imagenet_test_transform
from unlearning.utils import ConfigError


class DataHandler:
    def __init__(self, dataset_name, dataset_save_dir, background_ratio, test_size):
        """
        Initialize the data handler with configuration parameters.
        
        Args:
            cfg: Configuration object containing dataset parameters
        """
        self.dataset_name = dataset_name
        self.dataset_save_dir = dataset_save_dir
        # self.batch_sizes = cfg.dataset.cfg.batch_sizes
        # self.num_workers = cfg.dataset.cfg.num_workers
        self.background_ratio = background_ratio
        self.test_size = test_size
        
    def get_transform(self):
        """
        Get the appropriate transform based on the dataset name.
        """
        if self.dataset_name == 'cifar10':
            return get_cifar10_test_transform()
        elif self.dataset_name == 'cifar100':
            return get_cifar100_test_transform()
        elif self.dataset_name == 'imagenet':
            return get_imagenet_test_transform()
        else:
            raise ConfigError(
                f'dataset_name {self.dataset_name} not supported.'
                )

    def initialize_datasets(self):
        """ 
        Initialize dataloaders from the dataset folder in ImageFolder format.
        Returns a dictionary of dataloaders for retain, forget, and val splits.
        """
        transform = self.get_transform()

        # Load datasets for each split
        splits = ['forget', 'test']
        datasets = {}

        for split in splits:
            split_dir = os.path.join(self.dataset_save_dir, split)
            if not os.path.exists(split_dir):
                raise Exception(f'{split_dir} does not exist. '
                                'Is the dataset in ImageFolder format?')

            dataset = RobustImageFolder(root=split_dir, transform=transform)
            datasets[split] = dataset
        return datasets

    def prepare_background_points(self):
        """
        Prepare background points for establishing the baseline.
        
        Args:
            dataloaders: Dictionary containing dataloaders for different splits
            
        Returns:
            background_points (Subset): of data points for establishing 
                baseline
            indices (list): indices used to create subset
        """
        datasets = self.initialize_datasets()
        background_points, indices = create_subset(
                                                   datasets['test'], 
                                                   self.background_ratio, 
                                                   return_indices=True
                                                   )
        self.background_points = background_points
        self.indices = indices
        return background_points, indices
    
    def prepare_querypoints(self):
        """Create the query points"""
        datasets = self.initialize_datasets()
        # use the whole forgetting datasets
        forget_points = create_subset(datasets["forget"], 1)  
        print("In the query points, number of forget points is "
              f"{len(forget_points)}")
        # Ensure that the number of forget data is same as the number of 
        # background points
        forget_size = len(forget_points)

        # Create a background subset with the same size
        test_points = create_matched_subset(
                    datasets['test'], 
                    target_size=forget_size,  # Force matching forget_data size
                    exempt_points=self.indices,
                    test_size = self.test_size
                    )
        print(
            f"In the query points, number of test points is {len(test_points)}"
            )
        # Create labels indicating in the forget set or not
        test_y = torch.zeros(len(test_points))
        forget_y = torch.ones(len(forget_points))
        all_points = test_points + forget_points
        all_labels = torch.cat((test_y, forget_y), dim=0)
        query_points = list(zip(all_points, all_labels))
        random.shuffle(query_points)

        return query_points
