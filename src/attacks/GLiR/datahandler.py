import torch
import random
from attacks.GLiR.glir_utils import create_subset, create_matched_subset
from utils import initialize_datasets


class DataHandler:
    def __init__(self, dataset_name, dataset_save_dir, 
                 background_ratio, test_size):
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
        back_data = initialize_datasets(splits=['test'],
                                        dataset_name=self.dataset_name,
                                        dataset_save_dir=self.dataset_save_dir)
        background_points, indices = create_subset(
                                                   back_data,
                                                   self.background_ratio, 
                                                   return_indices=True
                                                   )
        self.background_points = background_points
        self.indices = indices
        return background_points, indices
    
    def prepare_querypoints(self):
        """Create the query points"""
        datasets = initialize_datasets(splits=['forget', 'test'],
                                       dataset_name=self.dataset_name,
                                       dataset_save_dir=self.dataset_save_dir)
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
                    test_size=self.test_size
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
