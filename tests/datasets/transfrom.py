import unittest
import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    )

from src.datasets import cifar10,  cifar100, imagenet


class TestTransformations(unittest.TestCase):
    
    def setUp(self):
        """Create a test image for use in the tests."""
        # Create a simple RGB test image (3x100x100)
        self.test_image = Image.new('RGB', (100, 100), color=(128, 128, 128))
        
    def test_cifar10_transform_composition(self):
        """Test that CIFAR-10 transform has the correct components."""
        transform = cifar10.get_cifar10_test_transform()
        
        # Check that it's a Compose object
        self.assertIsInstance(transform, transforms.Compose)
        
        # Check that it has exactly 3 transforms
        self.assertEqual(len(transform.transforms), 3)
        
        # Check each transform type and parameters
        resize_transform = transform.transforms[0]
        self.assertIsInstance(resize_transform, transforms.Resize)
        self.assertEqual(
                        resize_transform.size, 
                        (cifar10.CIFAR10_IMAGE_SIZE, 
                         cifar10.CIFAR10_IMAGE_SIZE)
                        )
        
        self.assertIsInstance(transform.transforms[1], transforms.ToTensor)
        
        normalize_transform = transform.transforms[2]
        self.assertIsInstance(normalize_transform, transforms.Normalize)
    
    def test_cifar100_transform_composition(self):
        """Test that CIFAR-100 transform has the correct components."""
        transform = cifar100.get_cifar100_test_transform()
        
        # Check that it's a Compose object
        self.assertIsInstance(transform, transforms.Compose)
        
        # Check that it has exactly 3 transforms
        self.assertEqual(len(transform.transforms), 3)
        
        # Check each transform type and parameters
        resize_transform = transform.transforms[0]
        self.assertIsInstance(resize_transform, transforms.Resize)
        self.assertEqual(
                         resize_transform.size, 
                         (cifar100.CIFAR100_IMAGE_SIZE, 
                          cifar100.CIFAR100_IMAGE_SIZE)
                          )
        
        self.assertIsInstance(transform.transforms[1], transforms.ToTensor)
        
        normalize_transform = transform.transforms[2]
        self.assertIsInstance(normalize_transform, transforms.Normalize)
        
    def test_imagenet_transform_composition(self):
        """Test that ImageNet transform has the correct components."""
        transform = imagenet.get_imagenet_test_transform()
        
        # Check that it's a Compose object
        self.assertIsInstance(transform, transforms.Compose)
        
        # Check that it has exactly 4 transforms
        self.assertEqual(len(transform.transforms), 3)
        
        # Check each transform type and parameters
        resize_transform = transform.transforms[0]
        self.assertIsInstance(resize_transform, transforms.Resize)
        self.assertEqual(resize_transform.size, (224, 224))
        
        self.assertIsInstance(transform.transforms[1], transforms.ToTensor)
        
        normalize_transform = transform.transforms[2]
        self.assertIsInstance(normalize_transform, transforms.Normalize)
    
    def test_cifar10_transform_output(self):
        """Test that CIFAR-10 transform produces the expected output format."""
        transform = cifar10.get_cifar10_test_transform()
        output = transform(self.test_image)
        
        # Check output is a tensor with the right shape and type
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(
                         output.shape, 
                         (3, cifar10.CIFAR10_IMAGE_SIZE, 
                          cifar10.CIFAR10_IMAGE_SIZE)
                          )
        self.assertEqual(output.dtype, torch.float32)
        
        # Test normalization - original image had 128 for all channels
        # After ToTensor, it becomes 128/255 = 0.502
        # After normalization, it should be (0.502 - mean) / std 
        # for each channel
        expected_channel0 = ((0.502 - cifar10.CIFAR10_MEAN[0]) / 
                             cifar10.CIFAR10_STD[0])
        # Check that values are close to expected
        self.assertAlmostEqual(
                               float(output[0, 0, 0]), 
                               expected_channel0, 
                               places=3
                               )
    
    def test_cifar100_transform_output(self):
        """
        Test that CIFAR-100 transform produces the expected output format.
        """
        transform = cifar100.get_cifar100_test_transform()
        output = transform(self.test_image)
        
        # Check output is a tensor with the right shape and type
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(
                         output.shape, 
                         (3, cifar100.CIFAR100_IMAGE_SIZE, 
                          cifar100.CIFAR100_IMAGE_SIZE)
                          )
        self.assertEqual(output.dtype, torch.float32)
        
        # Test normalization - original image had 128 for all channels
        # After ToTensor, it becomes 128/255 = 0.502
        # After normalization, should be (0.502 - mean) / std for each channel
        expected_channel0 = ((0.502 - cifar100.CIFAR100_MEAN[0]) / 
                             cifar100.CIFAR_100_STD[0])
        # Check that values are close to expected 
        self.assertAlmostEqual(
                               float(output[0, 0, 0]), 
                               expected_channel0, 
                               places=3
                               )
    
    def test_imagenet_transform_output(self):
        """Test that ImageNet transform produces the expected output format."""
        transform = imagenet.get_imagenet_test_transform()
        output = transform(self.test_image)
        
        # Check output is a tensor with the right shape and type
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(
            output.shape, 
            (3, 224, 224)  # ImageNet typically uses 224x224
        )
        self.assertEqual(output.dtype, torch.float32)

        # Test normalization values
        expected_channel0 = ((0.502 - imagenet.IMAGENET_MEAN[0]) / 
                             imagenet.IMAGENET_STD[0])

        # Check that values are close to expected 
        self.assertAlmostEqual(
                               float(output[0, 0, 0]), 
                               expected_channel0, 
                               places=3
                               )


if __name__ == '__main__':
    unittest.main()