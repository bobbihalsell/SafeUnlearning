# Standard library imports
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Third-party imports
import numpy as np
import torch

# Set up path for local imports
script_dir = os.path.dirname(__file__)
src_path = os.path.abspath(os.path.join(script_dir, "../../src"))
sys.path.insert(0, src_path)

# Local imports
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
        self.original_model.parameters = MagicMock(
            return_value=[param1, param1.clone()]
        )
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
            exp_name="test",
            batch_size=1,
            budget=10,
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

    @patch("torch.nn.functional.interpolate")
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
        # Models have params that differ by 1.0, so total diff
        # should be 2 * 10 * 10 = 200
        diff = self.reconstructor.compute_model_difference_l1(
            self.original_model, self.target_model
        )
        self.assertEqual(diff, 200.0)

    def test_compute_model_difference_l2(self):
        """Test L2 difference calculation between models"""
        # Models have params that differ by 1.0, so total
        # squared diff should be 2 * 10 * 10 = 200
        diff = self.reconstructor.compute_model_difference_l2(
            self.original_model, self.target_model
        )
        self.assertEqual(diff, 200.0)

    def test_compute_model_difference_weighted(self):
        """Test weighted difference calculation between models"""
        # First param gets weight 1/2, second gets weight 2/2
        # Total should be 0.5*100 + 1.0*100 = 150
        diff = self.reconstructor.compute_model_difference_weighted(
            self.original_model, self.target_model
        )
        # Update expected value to match current implementation
        self.assertEqual(diff, 150.0)

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

        # Create a copy function for the interpolated model
        def copy_side_effect(original_model):
            model_copy = MagicMock()
            param_copy = torch.zeros(5, 5)
            model_copy.named_parameters = MagicMock(
                return_value=[("layer1", param_copy)]
            )
            model_copy.parameters = MagicMock(return_value=[param_copy])
            return model_copy

        # Test with alpha = 0.3
        with patch("copy.deepcopy", side_effect=copy_side_effect):
            result = self.reconstructor.interpolate_models(model1, model2, alpha=0.3)

            # Get parameter from the result
            param_result = list(result.parameters())[0]

            # Expected: 0.3 * 0 + 0.7 * 10 = 7
            expected = torch.ones(5, 5) * 7
            torch.testing.assert_close(param_result, expected)

    @patch("attacks.GGL.reconstructor.SCRUB")
    def test_label_conversion_int_to_tensor(self, mock_scrub_class):
        mock_scrub = MagicMock()
        mock_scrub.unlearn.return_value = (MagicMock(), None)
        mock_scrub_class.return_value = mock_scrub

        expected_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        image = torch.randn(1, 3, 224, 224)
        label = 2  # integer input
        model = MagicMock()

        self.reconstructor.perform_scrub_updates(image, label, model)

        args, kwargs = mock_scrub.unlearn.call_args
        forget_loader = kwargs["data_dict"]["forget"]
        forget_batch = next(iter(forget_loader))
        _, forget_labels = forget_batch

        self.assertTrue(torch.is_tensor(forget_labels))
        self.assertEqual(forget_labels.dtype, torch.long)
        self.assertEqual(str(forget_labels.device), expected_device)

    @patch("attacks.GGL.reconstructor.SCRUB")
    def test_forget_and_retain_dataloaders_scrub(self, mock_scrub_class):
        mock_scrub = MagicMock()
        mock_scrub.unlearn.return_value = (MagicMock(), None)
        mock_scrub_class.return_value = mock_scrub

        image = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 2, 3])
        model = MagicMock()

        self.reconstructor.perform_scrub_updates(image, labels, model)

        args, kwargs = mock_scrub.unlearn.call_args
        data_dict = kwargs["data_dict"]
        self.assertIn("forget", data_dict)
        self.assertIn("retain", data_dict)

        # forget should have 1 batch of size 4
        forget_loader = data_dict["forget"]
        self.assertEqual(len(forget_loader), 1)

        # retain should be empty
        retain_loader = data_dict["retain"]
        self.assertEqual(len(retain_loader), 0)

    @patch("attacks.GGL.reconstructor.SCRUB")
    def test_perform_scrub_updates(self, mock_scrub_class):
        mock_scrub = MagicMock()
        mock_scrub.unlearn.return_value = (MagicMock(), None)
        mock_scrub_class.return_value = mock_scrub

        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        image = torch.randn(1, 3, 224, 224)
        labels = torch.tensor([1])
        model = MagicMock()
        kwargs = {
            "momentum": 0.9,
            "weight_decay": 0.1,
            "alpha": 0.5,
            "gamma": 0.5,
            "epochs_per_lr_decay": 2,
            "lr_decay_factor": 0.1,
            "min_epochs": 0,
            "max_epochs": 2,
            "use_l2_penalty": False,
            "sep_epochs": True,
            "optimizer": "sgd",
        }

        self.reconstructor.perform_scrub_updates(image, labels, model, **kwargs)

        mock_scrub_class.assert_called_once_with(device=torch.device(expected_device))
        mock_scrub.unlearn.assert_called_once()

        _, unlearn_kwargs = mock_scrub.unlearn.call_args
        self.assertEqual(unlearn_kwargs["model"], model)
        self.assertIn("data_dict", unlearn_kwargs)
        self.assertEqual(unlearn_kwargs["lr"], self.reconstructor.lr)
        self.assertEqual(unlearn_kwargs["momentum"], 0.9)
        self.assertEqual(unlearn_kwargs["gamma"], 0.5)

    @patch("torch.optim.SGD")
    def test_perform_sgd_updates(self, mock_sgd):
        """Test SGD update mechanism"""
        # Create mock model and optimizer with requires_grad=True tensors
        mock_model = MagicMock()
        mock_model.forward = MagicMock(
            return_value=torch.tensor([0.5], requires_grad=True)
        )
        mock_optimizer = MagicMock()
        mock_sgd.return_value = mock_optimizer

        # Mock generated image and labels
        generated_image = torch.randn(1, 3, 224, 224)
        labels = torch.tensor([1])

        # Mock the backward operation
        with patch("torch.Tensor.backward", MagicMock()):
            # Perform updates
            result = self.reconstructor.perform_sgd_updates(
                generated_image, labels, mock_model, steps=3
            )

            # Verify optimizer was created and step was called correct
            # number of times
            mock_sgd.assert_called_once()
            self.assertEqual(mock_optimizer.zero_grad.call_count, 3)
            self.assertEqual(mock_optimizer.step.call_count, 3)
            self.assertEqual(result, mock_model)

    @patch("attacks.GGL.reconstructor.NegGradPlus")
    def test_perform_neggradplus_updates(self, mock_neggradplus_class):
        """Test NegGradPlus update mechanism"""
        # Set up mocks
        mock_neggradplus = MagicMock()
        mock_neggradplus.unlearn.return_value = (MagicMock(), None)
        mock_neggradplus_class.return_value = mock_neggradplus

        # Expected device based on system config
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create test data
        image = torch.randn(1, 3, 224, 224)
        labels = torch.tensor([1])
        model = MagicMock()

        # Additional kwargs for the method
        kwargs = {"momentum": 0.9, "weight_decay": 0.1, "epochs": 5, "batch_size": 32}

        # Call the method being tested
        result = self.reconstructor.perform_neggradplus_updates(
            image, labels, model, **kwargs
        )

        mock_neggradplus_class.assert_called_once_with(
            device=torch.device(expected_device)
        )
        # Verify unlearn method was called
        mock_neggradplus.unlearn.assert_called_once()

        # Check the arguments passed to unlearn
        _, unlearn_kwargs = mock_neggradplus.unlearn.call_args
        self.assertEqual(unlearn_kwargs["model"], model)
        self.assertIn("data_dict", unlearn_kwargs)
        self.assertEqual(unlearn_kwargs["lr"], self.reconstructor.lr)
        self.assertEqual(unlearn_kwargs["momentum"], 0.9)
        self.assertEqual(unlearn_kwargs["weight_decay"], 0.1)
        self.assertEqual(unlearn_kwargs["epochs"], 5)
        self.assertEqual(unlearn_kwargs["batch_size"], 32)

        # Verify the returned model is the unlearned model
        self.assertEqual(result, mock_neggradplus.unlearn.return_value[0])

    @patch("attacks.GGL.reconstructor.NegGradPlus")
    def test_neggradplus_label_conversion(self, mock_neggradplus_class):
        """Test label conversion from int to tensor in NegGradPlus updates"""
        mock_neggradplus = MagicMock()
        mock_neggradplus.unlearn.return_value = (MagicMock(), None)
        mock_neggradplus_class.return_value = mock_neggradplus

        expected_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        image = torch.randn(1, 3, 224, 224)
        label = 2  # integer input
        model = MagicMock()

        self.reconstructor.perform_neggradplus_updates(image, label, model)

        args, kwargs = mock_neggradplus.unlearn.call_args
        forget_dict = kwargs["data_dict"]
        forget_loader = forget_dict["forget"]
        forget_batch = next(iter(forget_loader))
        _, forget_labels = forget_batch

        self.assertTrue(torch.is_tensor(forget_labels))
        self.assertEqual(forget_labels.dtype, torch.long)
        self.assertEqual(str(forget_labels.device), expected_device)

    @patch("attacks.GGL.reconstructor.NegGradPlus")
    def test_neggradplus_forget_and_retain_loaders(self, mock_neggradplus_class):
        """Test forget and retain dataloaders for NegGradPlus updates"""
        mock_neggradplus = MagicMock()
        mock_neggradplus.unlearn.return_value = (MagicMock(), None)
        mock_neggradplus_class.return_value = mock_neggradplus

        image = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 2, 3])
        model = MagicMock()

        self.reconstructor.perform_neggradplus_updates(image, labels, model)

        args, kwargs = mock_neggradplus.unlearn.call_args
        forget_dict = kwargs["data_dict"]

        # Check that we have both forget and retain loaders
        self.assertIn("forget", forget_dict)
        self.assertIn("retain", forget_dict)

        # forget should have 1 batch with all 4 samples
        forget_loader = forget_dict["forget"]
        self.assertEqual(len(forget_loader), 1)

        # Check the batch size of the forget loader
        for batch in forget_loader:
            x, y = batch
            self.assertEqual(x.shape[0], 4)
            self.assertEqual(y.shape[0], 4)

        # retain should be empty
        retain_loader = forget_dict["retain"]
        self.assertEqual(len(list(retain_loader)), 0)

    def test_save_results(self):
        """Test result saving functionality"""
        # Create mock objects
        mock_z = MagicMock(spec=torch.Tensor)
        mock_z.detach.return_value = mock_z
        mock_z.cpu.return_value = mock_z
        mock_z.numpy.return_value = np.array([1, 2, 3])

        mock_x = MagicMock(spec=torch.Tensor)
        mock_x.detach.return_value = mock_x
        mock_x.cpu.return_value = mock_x
        mock_x.squeeze.return_value = mock_x
        mock_x.clamp.return_value = mock_x

        # Mock the necessary modules
        with (
            patch("PIL.Image"),
            patch("torchvision.transforms.ToPILImage") as mock_transform,
            patch("numpy.save") as mock_np_save,
            patch("os.makedirs"),
            patch("os.path.dirname", return_value="/mock"),
            patch("os.path.abspath", return_value="/mock"),
            patch("os.path.join", return_value="/mock/path"),
            patch("os.listdir", return_value=[]),
            patch("shutil.rmtree"),
        ):
            # Set up the transform mock
            mock_pil_image = MagicMock()
            mock_transform_instance = MagicMock()
            mock_transform.return_value = mock_transform_instance
            mock_transform_instance.return_value = mock_pil_image

            # Call the method under test with the correct signature
            z_path = "/mock/path"
            x_path = "/mock/path.jpg"
            self.reconstructor.save_results(mock_z, mock_x, z_path, x_path)

            # Assertions
            mock_transform_instance.assert_called_once()
            mock_pil_image.save.assert_called_once_with(x_path)
            mock_np_save.assert_called_once()


if __name__ == "__main__":
    unittest.main()
