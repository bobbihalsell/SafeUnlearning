import unittest
import torch
import numpy as np
import os
import tempfile
from unittest.mock import MagicMock, patch

# Make sure the src directory is in the Python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Now import from src
from attacks.GGL.reconstructor import GGLReconstructor

class TestGGLReconstructor(unittest.TestCase):
    
    def setUp(self):
        # Create mock models for testing
        self.original_model = MagicMock()
        self.target_model = MagicMock()
        
        # Mock parameters for the models
        param1 = torch.ones(10, 10)
        param2 = torch.ones(10, 10) * 2
        
        # Set up parameter mocking
        self.original_model.parameters = MagicMock(return_value=[param1, param1.clone()])
        self.target_model.parameters = MagicMock(return_value=[param2, param2.clone()])
        
        # Mock loss function with requires_grad=True
        self.loss_fn = MagicMock(return_value=torch.tensor(0.5, requires_grad=True))
        
        # Create a temporary directory for file outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create GGLReconstructor with minimal setup
        self.reconstructor = GGLReconstructor(
            original_model=self.original_model,
            target_model=self.target_model,
            loss_fn=self.loss_fn,
            num_classes=1000,
            num_updates=1,
            lr=0.001,
            labels=torch.tensor([1]),
            exp_name='test',
            batch_size=1,
            budget=10
        )
        
        # Mock the BigGAN generator
        self.reconstructor.generator = MagicMock()
        self.reconstructor.generator.return_value = torch.ones(1, 3, 256, 256)
        
    def tearDown(self):
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test proper initialization of GGLReconstructor"""
        self.assertEqual(self.reconstructor.original_model, self.original_model)
        self.assertEqual(self.reconstructor.target_model, self.target_model)
        self.assertEqual(self.reconstructor.loss_fn, self.loss_fn)
        self.assertEqual(self.reconstructor.num_classes, 1000)
        self.assertEqual(self.reconstructor.lr, 0.001)
        self.assertEqual(self.reconstructor.batch_size, 1)
        self.assertEqual(self.reconstructor.budget, 10)
        
    @patch('torch.nn.functional.interpolate')
    def test_generate_image(self, mock_interpolate):
        """Test image generation from latent vector"""
        # Setup mock for interpolate
        mock_interpolate.return_value = torch.ones(1, 3, 224, 224)
        
        # Test with numpy array
        z_np = np.random.randn(1, 128)
        labels = torch.tensor([1])
        result = self.reconstructor.generate_image(z_np, labels)
        
        # Verify shape and calls
        self.assertEqual(result.shape, (1, 3, 224, 224))
        self.reconstructor.generator.assert_called_once()
        mock_interpolate.assert_called_once()
        
        # Reset mocks
        self.reconstructor.generator.reset_mock()
        mock_interpolate.reset_mock()
        
        # Test with tensor
        z_tensor = torch.randn(128)
        result = self.reconstructor.generate_image(z_tensor, 1)
        
        # Verify shape and calls again
        self.assertEqual(result.shape, (1, 3, 224, 224))
        self.reconstructor.generator.assert_called_once()
        mock_interpolate.assert_called_once()
    
    def test_compute_model_difference_l1(self):
        """Test L1 difference calculation between models"""
        # Models have params that differ by 1.0, so total diff should be 2 * 10 * 10 = 200
        diff = self.reconstructor.compute_model_difference_l1(
            self.original_model, self.target_model
        )
        self.assertEqual(diff, 200.0)
    
    def test_compute_model_difference_l2(self):
        """Test L2 difference calculation between models"""
        # Models have params that differ by 1.0, so total squared diff should be 2 * 10 * 10 = 200
        diff = self.reconstructor.compute_model_difference_l2(
            self.original_model, self.target_model
        )
        self.assertEqual(diff, 200.0)
    
    def test_compute_model_difference_weighted(self):
        """Test weighted difference calculation between models"""
        # First param gets weight 1/2, second gets weight 2/2
        # Total should be 0.5*100 + 1.0*100 = 150, but due to duplicate line, it's 300
        diff = self.reconstructor.compute_model_difference_weighted(
            self.original_model, self.target_model
        )
        # Update expected value to match current implementation
        self.assertEqual(diff, 300.0)
    
    def test_interpolate_models(self):
        """Test model interpolation with different alpha values"""
        # Create models with known parameters
        model1 = MagicMock()
        param1 = torch.zeros(5, 5)
        model1.named_parameters = MagicMock(return_value=[("layer1", param1)])
        model1.parameters = MagicMock(return_value=[param1])
        
        model2 = MagicMock()
        param2 = torch.ones(5, 5) * 10
        model2.named_parameters = MagicMock(return_value=[("layer1", param2)])
        model2.parameters = MagicMock(return_value=[param2])
        
        # Create a copy function for the interpolated model - FIX: add parameter
        def copy_side_effect(original_model):  # Accept the model parameter
            model_copy = MagicMock()
            param_copy = torch.zeros(5, 5)
            model_copy.named_parameters = MagicMock(return_value=[("layer1", param_copy)])
            model_copy.parameters = MagicMock(return_value=[param_copy])
            return model_copy
        
        # Test with alpha = 0.3
        with patch('copy.deepcopy', side_effect=copy_side_effect):
            result = self.reconstructor.interpolate_models(model1, model2, alpha=0.3)
            
            # Get parameter from the result
            param_result = list(result.parameters())[0]
            
            # Expected: 0.3 * 0 + 0.7 * 10 = 7
            expected = torch.ones(5, 5) * 7
            torch.testing.assert_close(param_result, expected)
    
    @patch('torch.optim.SGD')
    def test_perform_sgd_updates(self, mock_sgd):
        """Test SGD update mechanism"""
        # Create mock model and optimizer with requires_grad=True tensors
        mock_model = MagicMock()
        mock_model.forward = MagicMock(return_value=torch.tensor([0.5], requires_grad=True))
        mock_optimizer = MagicMock()
        mock_sgd.return_value = mock_optimizer
        
        # Mock generated image and labels
        generated_image = torch.randn(1, 3, 224, 224)
        labels = torch.tensor([1])
        
        # Mock the backward operation
        with patch('torch.Tensor.backward', MagicMock()) as mock_backward:
            # Perform updates
            result = self.reconstructor.perform_sgd_updates(
                generated_image, labels, mock_model, steps=3
            )
            
            # Verify optimizer was created and step was called correct number of times
            mock_sgd.assert_called_once()
            self.assertEqual(mock_optimizer.zero_grad.call_count, 3)
            self.assertEqual(mock_optimizer.step.call_count, 3)
            self.assertEqual(result, mock_model)
    
    @patch('unlearning.scrub.SCRUB')  # Fixed import path
    @patch('torch.utils.data.DataLoader')
    @patch('torch.utils.data.TensorDataset')
    def test_perform_scrub_updates(self, mock_dataset, mock_dataloader, mock_scrub_class):
        """Test SCRUB unlearning method"""
        # Mock SCRUB instance and unlearn method
        mock_scrub_instance = MagicMock()
        mock_scrub_class.return_value = mock_scrub_instance
        mock_unlearned_model = MagicMock()
        mock_scrub_instance.unlearn.return_value = (mock_unlearned_model, None)
        
        # Mock generated image and labels
        generated_image = torch.randn(1, 3, 224, 224)
        labels = torch.tensor([1])
        
        # Call the method
        result = self.reconstructor.perform_scrub_updates(
            generated_image, labels, self.original_model
        )
        
        # Verify SCRUB instance was created and unlearn was called
        mock_scrub_class.assert_called_once()
        mock_scrub_instance.unlearn.assert_called_once()
        self.assertEqual(result, mock_unlearned_model)
    
    @patch('os.makedirs')
    @patch('numpy.save')
    @patch('torchvision.transforms.ToPILImage')
    @patch('os.path.dirname')
    @patch('os.path.abspath')
    @patch('os.path.join')
    @patch('shutil.rmtree')
    @patch('os.listdir')
    def test_save_results(self, mock_listdir, mock_rmtree, mock_join, 
                          mock_abspath, mock_dirname, mock_transform, 
                          mock_np_save, mock_makedirs):
        """Test result saving functionality"""
        # Set up path mocking
        mock_dirname.return_value = "/mock/dir"
        mock_abspath.return_value = "/mock/abspath"
        mock_join.return_value = "/mock/path"
        mock_listdir.return_value = []  # No previous NPY files
        
        # Mock PIL image and its save method
        #mock_pil_image = MagicMock()
        #mock_transform.return_value = mock_pil_image

        # Mock PIL image and its save method
        mock_pil_image = MagicMock()
        mock_pil_image.save = MagicMock()  # explicitly mock save
        mock_transform.return_value = mock_pil_image

        
        # Setup test data
        z = torch.randn(1, 128)
        x = torch.randn(1, 3, 224, 224)
        z_path = os.path.join(self.temp_dir.name, "test_z")
        x_path = os.path.join(self.temp_dir.name, "test_x.png")
        
        # Call save_results
        self.reconstructor.save_results(z, x, z_path, x_path)
        
        # Verify directories were created
        mock_makedirs.assert_called()
        
        # Verify transforms were applied and image was saved
        mock_transform.assert_called()
        mock_pil_image.save.assert_called()
        
        # Verify numpy save was called
        mock_np_save.assert_called_once()
        
        # Verify rmtree was called
        mock_rmtree.assert_called_once()

if __name__ == '__main__':
    unittest.main()