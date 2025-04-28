import random

import torch

from attacks.GLiR.glir_utils import create_matched_subset, create_subset
from utils import initialize_datasets


class DataHandler:
    """
    Handles dataset preparation for GLiR-based unlearning evaluation.

    This class is responsible for preparing the background and query points
    used in membership inference attacks. It ensures consistent dataset
    loading, subset creation, and label assignment for evaluation.

    Args:
        dataset_name (str): Name of the dataset to be used.
        dataset_save_dir (str): Directory where dataset files are stored or cached.
        background_ratio (float): Proportion of test data to use as background.
        test_size (float): Proportion of test data to draw query test points from.
    """

    def __init__(self, dataset_name, dataset_save_dir, background_ratio, test_size):
        """
        Initialize the data handler with configuration parameters.

        Args:
            cfg: Configuration object containing dataset parameters
        """
        self.dataset_name = dataset_name
        self.dataset_save_dir = dataset_save_dir
        self.background_ratio = background_ratio
        self.test_size = test_size

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
        back_data = initialize_datasets(
            splits=["test"],
            dataset_name=self.dataset_name,
            dataset_save_dir=self.dataset_save_dir,
        )
        background_points, indices = create_subset(
            back_data["test"], self.background_ratio, return_indices=True
        )
        self.background_points = background_points
        self.indices = indices
        return background_points, indices

    def prepare_querypoints(self):
        """
        Prepares the query points used in the membership inference attack.

        Loads both the 'forget' and 'test' splits of the dataset.
        - All data from the 'forget' split is used as one half of the query points.
        - A matching number of samples are drawn from the 'test' split (excluding
          any overlapping with background points) to balance the dataset.
        - Labels: 1 for forget points (members), 0 for test points (non-members).
        - The combined dataset is shuffled for unbiased evaluation.

        Returns:
            query_points (list of tuples): List of (data, label) tuples where
            label is 1 if the point belongs to the forget set, 0 otherwise.
        """
        datasets = initialize_datasets(
            splits=["forget", "test"],
            dataset_name=self.dataset_name,
            dataset_save_dir=self.dataset_save_dir,
        )
        # use the whole forgetting datasets
        forget_points = create_subset(datasets["forget"], 1)
        print(f"In the query points, number of forget points is {len(forget_points)}")
        # Ensure that the number of forget data is same as the number of
        # background points
        forget_size = len(forget_points)

        # Create a background subset with the same size
        test_points = create_matched_subset(
            datasets["test"],
            target_size=forget_size,  # Force matching forget_data size
            exempt_points=self.indices,
            test_size=self.test_size,
        )
        print(f"In the query points, number of test points is {len(test_points)}")
        # Create labels indicating in the forget set or not
        test_y = torch.zeros(len(test_points))
        forget_y = torch.ones(len(forget_points))
        all_points = test_points + forget_points
        all_labels = torch.cat((test_y, forget_y), dim=0)
        query_points = list(zip(all_points, all_labels))
        random.shuffle(query_points)

        return query_points
