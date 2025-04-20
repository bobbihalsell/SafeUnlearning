import unittest
import torch
from unittest.mock import patch, MagicMock
import sys
import os


sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    )


from src.attacks.InvertGrad.reconstructor import InvertGradConfig, InvertGradReconstructor  


## Dummy model for testing
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.dummy_layer(x)

    def eval(self):
        return self
    
    
class TestInvertGradConfig(unittest.TestCase):

    def test_default_config_values(self):
        config = InvertGradConfig()
        self.assertEqual(config.grad_diff_lr, 1e-4)
        self.assertFalse(config.signed)
        self.assertEqual(config.scoring_choice, 'loss')


    def test_total_variation_cast_to_float(self):
        config = InvertGradConfig(total_variation="0.05")
        self.assertIsInstance(config.total_variation, float)
        self.assertAlmostEqual(config.total_variation, 0.05)

    def test_config_value_type(self):
        config = InvertGradConfig()
        self.assertIsInstance(config.grad_diff_lr, float)
        self.assertIsInstance(config.boxed, bool)
        self.assertIsInstance(config.cost_fn, str)
        self.assertIsInstance(config.init, str)
        self.assertIsInstance(config.filter, (str, type(None)))


class TestInvertGradReconstructor(unittest.TestCase):
    """
    Test the InvertGradReconstructor class. Does a mock
    reconstruction and checks the output shape and type.
    """

    def setUp(self):
        self.device = torch.device("cpu")
        self.model = DummyModel()
        self.config = InvertGradConfig()

    def test_initialization(self):
        reconstructor = InvertGradReconstructor(
            device=self.device,
            original_model=self.model,
            unlearned_model=self.model,
            config=self.config
        )
        self.assertEqual(reconstructor.config.cost_fn, 'sim')
        self.assertIs(reconstructor.original_model, self.model)

    @patch.object(InvertGradReconstructor, "_run_trial")
    @patch.object(InvertGradReconstructor, "_score_trial")
    def test_reconstruct_runs_with_mocked_trial(self, mock_score_trial, mock_run_trial):
        # Mock trial and score
        mock_run_trial.return_value = (torch.rand(1, 3, 32, 32), torch.tensor([1]))
        mock_score_trial.return_value = torch.tensor(0.2)

        reconstructor = InvertGradReconstructor(
            device=self.device,
            original_model=self.model,
            unlearned_model=self.model,
            config=self.config
        )

        output, score = reconstructor.reconstruct(
            labels=torch.tensor([1]),
            num_images=1,
            image_size=[3, 32, 32],
            verbose=False
        )

        self.assertEqual(output.shape, (1, 3, 32, 32))
        self.assertIsInstance(score, float)


class TestInvertGradValueErrors(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = DummyModel()

    def test_pixelmean_raises_for_multiple_images(self):
        config = InvertGradConfig(scoring_choice="pixelmean")
        reconstructor = InvertGradReconstructor(
            device=self.device,
            original_model=self.model,
            unlearned_model=self.model,
            config=config
        )

        with self.assertRaises(ValueError):
            reconstructor.reconstruct(
                labels=torch.tensor([1, 2]), 
                num_images=2,
                image_size=[3, 32, 32]
            )

    def test_invalid_optim(self):
        config = InvertGradConfig(optim="invalid_optim")
        reconstructor = InvertGradReconstructor(
            device=self.device,
            original_model=self.model,
            unlearned_model=self.model,
            config=config
        )
        with self.assertRaises(ValueError):
            reconstructor.reconstruct(
                labels=torch.tensor([1]),
                num_images=1,
                image_size=[3, 32, 32]
            )

class TestGradientDifference(unittest.TestCase):
    def make_shifted_model(self, original_model, shift=1.0):
        shifted_model = DummyModel()
        for p1, p2 in zip(original_model.parameters(), shifted_model.parameters()):
            p2.data = p1.data + shift
        return shifted_model
    
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = DummyModel()
        self.model_shifted = self.make_shifted_model(self.model)
        self.config = InvertGradConfig()

    def test_zero_gradient_difference(self):
        reconstructor = InvertGradReconstructor(
            device=self.device,
            original_model=self.model,
            unlearned_model=self.model,
            config=self.config
        )

        model_params_shape = [
            param.shape for param in self.model.parameters()
        ]
        # Calculate gradient difference
        grad_diff = reconstructor._gradient_difference()
        self.assertIsInstance(grad_diff, list)
        self.assertIsInstance(grad_diff[0], torch.Tensor)
        self.assertEqual(grad_diff[0].shape, model_params_shape[0])

        # Check if gradient difference is all zeros
        self.assertTrue(torch.all(grad_diff[0] == 0).item())
        # Check if gradient difference is not NaN
        self.assertFalse(torch.any(torch.isnan(grad_diff[0])).item())
        # Check if gradient difference is not Inf
        self.assertFalse(torch.any(torch.isinf(grad_diff[0])).item())

    def test_one_gradient_difference(self):
        reconstructor = InvertGradReconstructor(
            device=self.device,
            original_model=self.model,
            unlearned_model=self.model_shifted,
            config=self.config
        )

        # Calculate gradient difference
        grad_diff = reconstructor._gradient_difference(grad_lr=1)

        self.assertTrue(torch.allclose(grad_diff[0], torch.ones_like(grad_diff[0])))

