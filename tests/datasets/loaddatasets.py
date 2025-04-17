import unittest
import torch
from PIL import Image
import sys
import os
import numpy as np
import shutil
import tempfile
from torch.utils.data import Subset, Dataset
from unittest.mock import patch, MagicMock

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    )
from src.datasets.load_datasets import (_get_stratified_split,
                                        load_train_val_test_datasets,
                                        _save_as_imagefolder
                                        )
                                        

class MockDataset(Dataset):
    """Mock dataset with predefined targets for testing."""
    def __init__(self, num_samples=100, num_classes=10):
        self.data = torch.randn(num_samples, 3, 32, 32)
        # Create balanced classes
        self.targets = []
        for i in range(num_classes):
            self.targets.extend([i] * (num_samples // num_classes))
        self.targets = torch.tensor(self.targets)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx].item()


class TestDatasetUtils(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test datasets
        self.test_dir = tempfile.mkdtemp()
        self.dataset = MockDataset(num_samples=100, num_classes=10)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_get_stratified_split(self):
        """Test stratified split function."""
        # Test with different proportions
        for proportion in [0.2, 0.5, 0.8]:
            subset1, subset2 = _get_stratified_split(self.dataset, proportion)
            
            # Check lengths are correct
            expected_len1 = int(len(self.dataset) * proportion)
            expected_len2 = len(self.dataset) - expected_len1
            self.assertEqual(len(subset1), expected_len1)
            self.assertEqual(len(subset2), expected_len2)
            
            # Check class distribution is maintained
            orig_classes = np.array([self.dataset[i][1] 
                                     for i in range(len(self.dataset))])
            subset1_classes = np.array([self.dataset[i][1] 
                                        for i in subset1.indices])
            subset2_classes = np.array([self.dataset[i][1] 
                                        for i in subset2.indices])
            
            for cls in range(10):
                orig_count = np.sum(orig_classes == cls)
                subset1_count = np.sum(subset1_classes == cls)
                subset2_count = np.sum(subset2_classes == cls)
                
                # Check if proportions are roughly maintained (allow 1 sample 
                # difference due to rounding)
                self.assertAlmostEqual(subset1_count / orig_count, 
                                       proportion, delta=0.2)
                self.assertAlmostEqual(subset2_count / orig_count, 
                                       1 - proportion, delta=0.2)
                
            # Check that indices don't overlap
            self.assertEqual(
                len(set(subset1.indices).intersection(set(subset2.indices))), 
                0)

    def test_get_stratified_split_error_handling(self):
        """Test that _get_stratified_split raises appropriate errors."""
        # Test with invalid dataset (no targets attribute)
        class DatasetWithoutTargets(Dataset):
            def __init__(self):
                self.data = torch.randn(10, 3, 32, 32)
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], 0
        
        invalid_dataset = DatasetWithoutTargets()
        
        # Test that it raises TypeError for dataset without targets
        with self.assertRaises(TypeError):
            _get_stratified_split(invalid_dataset, 0.5)
        
        # Test with invalid proportion values        
        with self.assertRaises(ValueError):
            _get_stratified_split(self.dataset, 2)  # proportion > 1
        
        with self.assertRaises(ValueError):
            _get_stratified_split(self.dataset, -0.5)  # proportion < 0
    
    def test_save_as_imagefolder(self):
        """Test saving dataset as ImageFolder format."""
        subset = Subset(self.dataset, indices=range(10))
        save_path = os.path.join(self.test_dir, "imagefolder_test")
        
        _save_as_imagefolder(subset, save_path, "train")
        
        # Check directory structure
        for i in range(10):
            if i < len(subset):
                _, label = subset[i]
                class_dir = os.path.join(save_path, "train", str(label))
                self.assertTrue(os.path.exists(class_dir))
                self.assertTrue(
                    os.path.exists(os.path.join(class_dir, f"{i}.png"))
                    )
        
        # Test with tensor data
        tensor_dataset = [(torch.randn(3, 32, 32), i % 5) for i in range(5)]
        _save_as_imagefolder(tensor_dataset, save_path, "tensor_test")
        
        # Check directory structure for tensor data
        for i in range(5):
            _, label = tensor_dataset[i]
            class_dir = os.path.join(save_path, "tensor_test", str(label))
            self.assertTrue(os.path.exists(class_dir))
            self.assertTrue(
                os.path.exists(os.path.join(class_dir, f"{i}.png"))
                )

    def test_load_train_val_test_datasets_cifar10(self):
        """Test loading CIFAR10 dataset."""
        # Use self to store the datasets
        self.saved_datasets = {}
        
        def side_effect(dataset, name):
            self.saved_datasets[name] = dataset
            print(f"Mock _save_as_imagefolder called with {name}")
        
        mock_dataset = MockDataset(num_samples=100, num_classes=10)
        
        # Use patch to mock the necessary functions
        with patch('src.datasets.load_datasets.datasets.CIFAR10', 
                   return_value=mock_dataset), \
             patch(
                 'src.datasets.load_datasets._save_as_imagefolder'
                  ) as mock_save:
            
            # Set up the side effect to capture saved datasets
            mock_save.side_effect = side_effect
            
            # Run the function
            load_train_val_test_datasets(
                dataset_name='cifar10',
                proportion=0.5,
                val_ratio=0.2,
                dataset_load_dir=self.test_dir,
                dataset_save_dir=self.test_dir
            )
            
            # Check that the mock was called
            mock_save.assert_called()
            
            # Check saved datasets
            self.assertIn('train', self.saved_datasets)
            self.assertIn('val', self.saved_datasets)
            self.assertIn('test', self.saved_datasets)
            
            # Calculate expected sizes
            expected_train_size = int(100 * 0.5 * 0.8) 
            expected_val_size = int(100 * 0.5 * 0.2)  
            
            self.assertEqual(len(self.saved_datasets['train']), 
                             expected_train_size)
            self.assertEqual(len(self.saved_datasets['val']), 
                             expected_val_size)
    
    def test_load_train_val_test_datasets_cifar100(self):
        """Test loading CIFAR100 dataset."""
        self.saved_datasets = {}
        
        def side_effect(dataset, root_path, name):
            self.saved_datasets[name] = dataset
            print(f"Mock _save_as_imagefolder called with {name}")
        
        mock_dataset = MockDataset(num_samples=1000, num_classes=100)
        
        # Use patch to mock the necessary functions
        with patch('src.datasets.load_datasets.datasets.CIFAR100', 
                   return_value=mock_dataset), \
             patch(
                 'src.datasets.load_datasets._save_as_imagefolder'
                  ) as mock_save:
            
            # Set up the side effect to capture saved datasets
            mock_save.side_effect = side_effect
            
            # Run the function
            load_train_val_test_datasets(
                dataset_name='cifar100',
                proportion=0.5,
                val_ratio=0.2,
                dataset_load_dir=self.test_dir,
                dataset_save_dir=self.test_dir
            )
            
            # Check that the mock was called
            mock_save.assert_called()
            
            # Check saved datasets
            self.assertIn('train', self.saved_datasets)
            self.assertIn('val', self.saved_datasets)
            self.assertIn('test', self.saved_datasets)
            
            # Calculate expected sizes
            # 100 samples * proportion * (1-val_ratio)
            expected_train_size = int(1000 * 0.5 * 0.8) 
            # 100 samples * proportion * val_ratio
            expected_val_size = int(1000 * 0.5 * 0.2)  
            
            self.assertEqual(len(self.saved_datasets['train']), 
                             expected_train_size)
            self.assertEqual(len(self.saved_datasets['val']), 
                             expected_val_size)


# Mock the RobustImageFolder class that is used for ImageNet
class MockRobustImageFolder:
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = [(None, i % 1000) for i in range(1000)]
        self.targets = [i % 1000 for i in range(1000)]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Create a dummy image
        img = Image.new(
                        'RGB', (32, 32), 
                        color=(idx % 255, (idx * 2) % 255, 
                               (idx * 3) % 255)
                               )
        return img, self.samples[idx][1]


# Additional test for ImageNet dataset
class TestImageNetLoading(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.saved_datasets = {}
        
        # Create directories for ImageNet structure with proper class folders
        train_dir = os.path.join(self.test_dir, "imagenet", "train")
        val_dir = os.path.join(self.test_dir, "imagenet", "val")
        
        # Create class directories (integers as folder names)
        for i in range(5):
            os.makedirs(os.path.join(train_dir, str(i)), exist_ok=True)
            os.makedirs(os.path.join(val_dir, str(i)), exist_ok=True)
            
            # Create sample images
            for j in range(3):
                img = Image.new('RGB', (8, 8), color=(i*50, j*50, 100))
                img.save(os.path.join(train_dir, str(i), f"img_{j}.jpg"))
                img.save(os.path.join(val_dir, str(i), f"img_{j}.jpg"))
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir)
    
    def test_load_imagenet(self):
        """Test loading ImageNet dataset."""
        # Create side effect function to capture saved datasets
        def save_side_effect(dataset, root_path, name):
            self.saved_datasets[name] = dataset
        
        # Setup the mock with side effect
        mock_save = MagicMock(side_effect=save_side_effect)
        
        # Patch RobustImageFolder class and _save_as_imagefolder function
        with patch('src.datasets.load_datasets.RobustImageFolder', 
                   return_value=MockDataset(1000, 5)), \
             patch(
                 'src.datasets.load_datasets._save_as_imagefolder', 
                 mock_save
                 ):
            
            # Run the function
            load_train_val_test_datasets(
                dataset_name='imagenet',
                proportion=0.1,
                val_ratio=0.15,
                dataset_load_dir=os.path.join(self.test_dir, 
                                              "imagenet"),
                dataset_save_dir=os.path.join(self.test_dir, 
                                              "processed_imagenet")
            )
            
            # Verify the mock was called
            mock_save.assert_called()
            
            # Check saved datasets
            self.assertIn('train', self.saved_datasets)
            self.assertIn('val', self.saved_datasets)
            self.assertIn('test', self.saved_datasets)
    
    def test_load_imagenet_no_val(self):
        """Test loading ImageNet dataset without validation directory."""
        # Remove val directory
        val_dir = os.path.join(self.test_dir, "imagenet", "val")
        if os.path.exists(val_dir):
            shutil.rmtree(val_dir)
        
        # Create side effect function to capture saved datasets
        def save_side_effect(dataset, root_path, name):
            self.saved_datasets[name] = dataset
        
        # Setup the mock with side effect
        mock_save = MagicMock(side_effect=save_side_effect)
        
        # Patch RobustImageFolder class and _save_as_imagefolder function
        with patch('src.datasets.load_datasets.RobustImageFolder', 
                   return_value=MockDataset(1000, 5)), \
             patch(
                 'src.datasets.load_datasets._save_as_imagefolder', 
                 mock_save), \
             patch('os.path.exists', lambda path: 'val' not in path):
            
            load_train_val_test_datasets(
                dataset_name='imagenet',
                proportion=0.1,
                val_ratio=0.15,
                dataset_load_dir=os.path.join(self.test_dir, 
                                              "imagenet"),
                dataset_save_dir=os.path.join(self.test_dir, 
                                              "processed_imagenet")
            )
            # Verify the mock was called
            mock_save.assert_called()

            # Check saved datasets - should have train and val but no test
            self.assertIn('train', self.saved_datasets)
            self.assertIn('val', self.saved_datasets)
            self.assertNotIn('test', self.saved_datasets)


if __name__ == '__main__':
    unittest.main()