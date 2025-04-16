import unittest
import torch
import numpy as np
import os
import tempfile
import shutil
import pickle
from torch.utils.data import Dataset, DataLoader, Subset
from unittest.mock import patch, MagicMock

# Import the functions to test
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.datasets.preprocessing import (
    remove_samples_by_indices,
    remove_samples_by_class,
    remove_classes,
    save_loaders,
    load_loaders,
    train_val_split,
    get_all_loaders
)

class MockDataset(Dataset):
    """A mock dataset for testing with controllable properties."""
    def __init__(self, num_samples=100, num_classes=10, with_targets=True):
        self.data = torch.randn(num_samples, 3, 32, 32)
        # Create targets as a sequence from 0 to num_classes, repeating
        self.targets = torch.tensor([i % num_classes for i in range(num_samples)])
        self.with_targets = with_targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx].item()

class TestPreprocessing(unittest.TestCase):
    
    def setUp(self):
        """Initialize test datasets and other resources."""
        # Set random seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create mock datasets for testing
        self.dataset = MockDataset(num_samples=100, num_classes=10)
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up resources after tests."""
        shutil.rmtree(self.test_dir)
    
    def test_remove_samples_by_indices(self):
        """Test removing samples by indices."""
        # Test with a few indices to remove
        forget_indices = [0, 10, 20, 30, 40]
        
        # Test with return_forget=True
        retain_set, forget_set = remove_samples_by_indices(
            self.dataset, forget_indices, return_forget=True
        )
        
        # Check sizes
        self.assertEqual(len(retain_set), len(self.dataset) - len(forget_indices))
        self.assertEqual(len(forget_set), len(forget_indices))
        
        # Check that the forget set contains the right indices
        for i, idx in enumerate(forget_indices):
            _, label_original = self.dataset[idx]
            _, label_forget = forget_set[i]
            self.assertEqual(label_original, label_forget)
        
        # Test with return_forget=False
        retain_set_only = remove_samples_by_indices(
            self.dataset, forget_indices, return_forget=False
        )
        
        # Check it's just the retain set
        self.assertEqual(len(retain_set_only), len(self.dataset) - len(forget_indices))
    
    def test_remove_samples_by_class(self):
        """Test removing samples by class."""
        classes_to_forget = [0, 2]
        num_to_forget = 3  # 3 samples per class
        
        # Test with return_forget=True
        retain_set, forget_set = remove_samples_by_class(
            self.dataset, classes_to_forget, num_to_forget, return_forget=True
        )
        
        # Check sizes
        expected_forget_size = len(classes_to_forget) * num_to_forget
        self.assertEqual(len(forget_set), expected_forget_size)
        self.assertEqual(len(retain_set), len(self.dataset) - expected_forget_size)
        
        # Verify that forgotten samples are from the right classes
        forget_classes = set()
        for i in range(len(forget_set)):
            _, label = forget_set[i]
            forget_classes.add(label)
        
        self.assertEqual(forget_classes, set(classes_to_forget))
        
        # Test with a Subset as input
        subset_indices = list(range(0, 50))  # First 50 samples
        dataset_subset = Subset(self.dataset, subset_indices)
        
        retain_subset, forget_subset = remove_samples_by_class(
            dataset_subset, classes_to_forget, 2, return_forget=True
        )
        
        # Check that it works with a subset
        self.assertLess(len(retain_subset), len(dataset_subset))
        self.assertEqual(len(forget_subset), len(classes_to_forget) * 2)
    
    def test_remove_classes(self):
        """Test removing entire classes."""
        forget_labels = [1, 3, 5]
        
        # Count samples with these labels in the original dataset
        forget_count = sum(1 for _, label in self.dataset if label in forget_labels)
        
        # Test with return_forget=True
        retain_set, forget_set = remove_classes(
            self.dataset, forget_labels, return_forget=True
        )
        
        # Check sizes
        self.assertEqual(len(forget_set), forget_count)
        self.assertEqual(len(retain_set), len(self.dataset) - forget_count)
        
        # Verify that retained set doesn't have any of the forget labels
        for i in range(len(retain_set)):
            _, label = retain_set[i]
            self.assertNotIn(label, forget_labels)
        
        # Verify that forget set only has the forget labels
        for i in range(len(forget_set)):
            _, label = forget_set[i]
            self.assertIn(label, forget_labels)
    
    def test_save_and_load_loaders(self):
        """Test saving and loading data loaders."""
        # Create test loaders
        test_loaders = {
            'train': DataLoader(self.dataset, batch_size=32, shuffle=True),
            'test': DataLoader(self.dataset, batch_size=64, shuffle=False)
        }
        
        # Save the loaders
        save_path = os.path.join(self.test_dir, "test_loaders")
        save_loaders(save_path, **test_loaders)
        
        # Check that files were created
        self.assertTrue(os.path.exists(os.path.join(save_path, "train_dataset.pkl")))
        self.assertTrue(os.path.exists(os.path.join(save_path, "test_dataset.pkl")))
        self.assertTrue(os.path.exists(os.path.join(save_path, "loader_configs.pkl")))
        
        # Load the loaders
        loaded_loaders = load_loaders(save_path)
        
        # Check that we got the expected loaders back
        self.assertIn('train', loaded_loaders)
        self.assertIn('test', loaded_loaders)
        
        # Check loader properties
        self.assertEqual(loaded_loaders['train'].batch_size, 32)
        self.assertEqual(loaded_loaders['test'].batch_size, 64)
        
        # Check dataset sizes
        self.assertEqual(len(loaded_loaders['train'].dataset), len(self.dataset))
        self.assertEqual(len(loaded_loaders['test'].dataset), len(self.dataset))
    
    def test_train_val_split(self):
        """Test splitting a dataset into train and validation sets."""
        val_ratio = 0.2
        
        train_subset, val_dataset = train_val_split(
            self.dataset, val_ratio, verbose=False
        )
        
        # Check sizes
        expected_val_size = int(len(self.dataset) * val_ratio)
        expected_train_size = len(self.dataset) - expected_val_size
        
        self.assertEqual(len(val_dataset), expected_val_size)
        self.assertEqual(len(train_subset), expected_train_size)
        
        # Verify that train and val sets have no overlap
        train_indices = set(train_subset.indices)
        val_indices = set(val_dataset.indices)
        
        self.assertEqual(len(train_indices.intersection(val_indices)), 0)
        self.assertEqual(len(train_indices) + len(val_indices), len(self.dataset))
    
    def test_get_all_loaders_instances(self):
        """Test the get_all_loaders function with 'instances' method."""
        test_dataset = MockDataset(num_samples=50, num_classes=10)
        forget_indices = [0, 10, 20, 30]
        
        # Create loaders
        forget_loader, retain_loader, train_loader, val_loader, test_loader = get_all_loaders(
            train_data=self.dataset,
            test_data=test_dataset,
            method='instances',
            val_ratio=0.1,
            forget_set_indices=forget_indices,
            verbose=False
        )
        
        # Check loaders exist and have correct batch sizes
        self.assertEqual(forget_loader.batch_size, 64)
        self.assertEqual(retain_loader.batch_size, 64)
        self.assertEqual(train_loader.batch_size, 128)
        self.assertEqual(val_loader.batch_size, 64)
        self.assertEqual(test_loader.batch_size, 128)
        
        # Check dataset sizes
        self.assertEqual(len(forget_loader.dataset), len(forget_indices))
        self.assertEqual(len(retain_loader.dataset) + len(forget_loader.dataset), 
                         len(train_loader.dataset))
        self.assertLess(len(train_loader.dataset), len(self.dataset))  # Due to val_ratio
        self.assertEqual(len(test_loader.dataset), len(test_dataset))
    
    def test_get_all_loaders_class_instances(self):
        """Test the get_all_loaders function with 'class_instances' method."""
        test_dataset = MockDataset(num_samples=50, num_classes=10)
        forget_labels = [0, 2]
        num_to_forget = 3
        
        # Create loaders
        forget_loader, retain_loader, train_loader, val_loader, test_loader = get_all_loaders(
            train_data=self.dataset,
            test_data=test_dataset,
            method='class_instances',
            val_ratio=0.1,
            forget_labels=forget_labels,
            num_to_forget=num_to_forget,
            verbose=False
        )
        
        # Check forget loader size
        expected_forget_size = len(forget_labels) * num_to_forget
        self.assertEqual(len(forget_loader.dataset), expected_forget_size)
        
        # Check relationship between datasets
        self.assertEqual(len(retain_loader.dataset) + len(forget_loader.dataset), 
                         len(train_loader.dataset))
    
    def test_get_all_loaders_class(self):
        """Test the get_all_loaders function with 'class' method."""
        test_dataset = MockDataset(num_samples=50, num_classes=10)
        forget_labels = [1, 3, 5]
        
        # Count samples with these labels to predict forget_loader size
        forget_count = sum(1 for _, label in self.dataset if label in forget_labels)
        
        # Create loaders
        forget_loader, retain_loader, train_loader, val_loader, test_loader = get_all_loaders(
            train_data=self.dataset,
            test_data=test_dataset,
            method='class',
            val_ratio=0.1,
            forget_labels=forget_labels,
            verbose=False
        )
        
        # Check sizes accounting for validation split
        val_size = int(len(self.dataset) * 0.1)
        train_size_after_val = len(self.dataset) - val_size
        
        # Calculate expected forget count in the training subset
        # This is approximate since the random split affects class distribution
        expected_forget_count = int(forget_count * (train_size_after_val / len(self.dataset)))
        
        # We check with some tolerance due to randomness in the val split
        self.assertAlmostEqual(len(forget_loader.dataset), expected_forget_count, delta=15)
        
        # Check that retain + forget = train
        self.assertEqual(len(retain_loader.dataset) + len(forget_loader.dataset), 
                         len(train_loader.dataset))
    
    def test_get_all_loaders_invalid_method(self):
        """Test that get_all_loaders raises an error for invalid methods."""
        test_dataset = MockDataset(num_samples=50, num_classes=10)
        
        with self.assertRaises(ValueError):
            get_all_loaders(
                train_data=self.dataset,
                test_data=test_dataset,
                method='invalid_method',
                val_ratio=0.1,
                verbose=False
            )

    def test_subset_without_targets(self):
        """Test handling a dataset that doesn't have targets attribute."""
        # Create a dataset without targets
        class DatasetWithoutTargets(Dataset):
            def __init__(self, num_samples=100, num_classes=10):
                self.data = torch.randn(num_samples, 3, 32, 32)
                self.labels = torch.tensor([i % num_classes for i in range(num_samples)])
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx].item()
        
        dataset = DatasetWithoutTargets()
        forget_indices = [0, 10, 20, 30]
        
        # Test remove_samples_by_indices
        retain_set, forget_set = remove_samples_by_indices(dataset, forget_indices)
        self.assertEqual(len(retain_set), len(dataset) - len(forget_indices))
        
        # Test remove_samples_by_class
        # This uses __getitem__ instead of targets
        classes_to_forget = [0, 2]
        num_to_forget = 3
        
        retain_set, forget_set = remove_samples_by_class(dataset, classes_to_forget, num_to_forget)
        
        # Verify size
        expected_forget_size = len(classes_to_forget) * num_to_forget
        self.assertEqual(len(forget_set), expected_forget_size)

if __name__ == '__main__':
    unittest.main()