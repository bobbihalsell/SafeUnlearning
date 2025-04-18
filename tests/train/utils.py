import unittest
from unittest.mock import patch
import random
import numpy as np
import torch
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    )

from src.train.utils import set_seed, setup_device


class TestDeviceSetup(unittest.TestCase):
    """Test cases for setup_device and set_seed functions."""

    @patch('torch.cuda.is_available')
    @patch('torch.mps.is_available')
    def test_setup_device_cuda(self, mock_mps_available, mock_cuda_available):
        """Test that 'cuda' is returned when CUDA is available."""
        mock_cuda_available.return_value = True
        mock_mps_available.return_value = False
        
        device = setup_device()
        self.assertEqual(device, 'cuda')

    @patch('torch.cuda.is_available')
    @patch('torch.mps.is_available')
    def test_setup_device_mps(self, mock_mps_available, mock_cuda_available):
        """Test that 'mps' is returned when MPS is available and CUDA is not."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = True
        
        device = setup_device()
        self.assertEqual(device, 'mps')

    @patch('torch.cuda.is_available')
    @patch('torch.mps.is_available')
    def test_setup_device_cpu(self, mock_mps_available, mock_cuda_available):
        """Test that 'cpu' is returned when neither CUDA nor MPS is available."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False
        
        device = setup_device()
        self.assertEqual(device, 'cpu')

    @patch('random.seed')
    @patch('numpy.random.seed')
    @patch('torch.manual_seed')
    @patch('torch.cuda.manual_seed_all')
    def test_set_seed(self, mock_cuda_seed, mock_torch_seed, mock_np_seed, mock_random_seed):
        """Test that set_seed calls all the seed functions with the correct value."""
        test_seed = 123
        set_seed(test_seed)
        
        mock_random_seed.assert_called_once_with(test_seed)
        mock_np_seed.assert_called_once_with(test_seed)
        mock_torch_seed.assert_called_once_with(test_seed)
        mock_cuda_seed.assert_called_once_with(test_seed)

    @patch('random.seed')
    @patch('numpy.random.seed')
    @patch('torch.manual_seed')
    @patch('torch.cuda.manual_seed_all')
    def test_set_seed_default(self, mock_cuda_seed, mock_torch_seed, mock_np_seed, mock_random_seed):
        """Test that set_seed uses the default seed (42) when no seed is provided."""
        set_seed()
        
        mock_random_seed.assert_called_once_with(42)
        mock_np_seed.assert_called_once_with(42)
        mock_torch_seed.assert_called_once_with(42)
        mock_cuda_seed.assert_called_once_with(42)

    def test_set_seed_reproducibility(self):
        """Test that set_seed makes random operations reproducible."""
        # Set a seed
        set_seed(42)
        
        # Get random values
        random_val1 = random.random()
        torch_val1 = torch.rand(1).item()
        np_val1 = np.random.random()
        
        # Reset the seed to the same value
        set_seed(42)
        
        # Get random values again
        random_val2 = random.random()
        torch_val2 = torch.rand(1).item()
        np_val2 = np.random.random()
        
        # Values should be the same if seed works correctly
        self.assertEqual(random_val1, random_val2)
        self.assertEqual(torch_val1, torch_val2)
        self.assertEqual(np_val1, np_val2)


if __name__ == '__main__':
    unittest.main()