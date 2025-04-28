import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)
from attacks.metrics import apply_normalization, apply_resizing, mse_image_space, psnr


class TestImageComparisonFunctions(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        self.batch_size = 2
        self.channels = 3
        self.height = 32
        self.width = 32
        self.dataset_size = (self.channels, self.height, self.width)

        # Create random image batches for testing
        torch.manual_seed(42)
        self.img_batch = [
            torch.rand(self.channels, 24, 24) for _ in range(self.batch_size)
        ]
        self.ref_batch = [
            torch.rand(self.channels, 24, 24) for _ in range(self.batch_size)
        ]

        # Dataset stats for normalization
        self.dataset_mean = [0.5, 0.5, 0.5]
        self.dataset_std = [0.5, 0.5, 0.5]

    def test_apply_resizing(self):
        """Test if images are correctly resized to target dimensions."""
        resized_img, resized_ref = apply_resizing(
            self.img_batch, self.ref_batch, self.dataset_size
        )

        # Check output shapes
        self.assertEqual(
            resized_img.shape, (self.batch_size, self.channels, self.height, self.width)
        )
        self.assertEqual(
            resized_ref.shape, (self.batch_size, self.channels, self.height, self.width)
        )

        # Test with different sized input images
        varied_img_batch = [
            torch.rand(self.channels, 16, 16),
            torch.rand(self.channels, 48, 48),
        ]
        varied_ref_batch = [
            torch.rand(self.channels, 64, 64),
            torch.rand(self.channels, 32, 32),
        ]

        resized_varied_img, resized_varied_ref = apply_resizing(
            varied_img_batch, varied_ref_batch, self.dataset_size
        )

        # Check all images were resized to the target size
        self.assertEqual(
            resized_varied_img.shape,
            (self.batch_size, self.channels, self.height, self.width),
        )
        self.assertEqual(
            resized_varied_ref.shape,
            (self.batch_size, self.channels, self.height, self.width),
        )

    def test_apply_normalization(self):
        """Test if normalization is correctly applied to images."""
        # First apply resizing to have consistent tensors
        resized_img, resized_ref = apply_resizing(
            self.img_batch, self.ref_batch, self.dataset_size
        )

        # Now apply normalization
        norm_img, norm_ref = apply_normalization(
            resized_img, resized_ref, self.dataset_mean, self.dataset_std
        )

        # Check shapes didn't change
        self.assertEqual(norm_img.shape, resized_img.shape)
        self.assertEqual(norm_ref.shape, resized_ref.shape)

        # Check the values are normalized
        # For mean 0.5 and std 0.5, values should be in range [-1, 1]
        self.assertTrue((norm_img >= -1).all() and (norm_img <= 1).all())

        # Manually normalize and compare
        manual_norm = transforms.Normalize(self.dataset_mean, self.dataset_std)
        expected_norm = manual_norm(resized_img[0])
        self.assertTrue(torch.allclose(norm_img[0], expected_norm))

    def test_psnr_comprehensive(self):
        """Test PSNR calculation for both identical and different images."""

        # PART 1: Test with identical images
        # Create a test image
        test_img = (
            torch.ones(self.channels, self.height, self.width) * 0.5
        )  # Gray image

        # Create identical reference images
        identical_ref = test_img.clone()

        # Calculate PSNR between identical images
        identical_psnr = self._calculate_psnr(test_img, identical_ref)
        print(f"PSNR for identical images: {identical_psnr}")

        # Verify PSNR is very high or infinite for identical images
        self.assertTrue(
            identical_psnr > 100.0 or np.isinf(identical_psnr),
            f"PSNR for identical images should be very high, got {identical_psnr}",
        )

        # Test the psnr function with identical images
        identical_results = psnr([test_img], [identical_ref], self.dataset_size)

        # Verify the returned PSNR value is very high or infinite
        self.assertTrue(
            identical_results[0][0] > 100.0 or np.isinf(identical_results[0][0]),
            f"PSNR for identical images should be very high, got {identical_results[0][0]}",
        )

        # Verify the correct reference image index (should be 0 since there's only one reference)
        self.assertEqual(identical_results[0][1], 0)

        # PART 2: Test with different images
        # Create test image - a black image (all zeros)
        black_img = torch.zeros(self.channels, self.height, self.width)

        # Create two reference images with different levels of similarity to the test image
        slightly_diff_ref = (
            torch.ones(self.channels, self.height, self.width) * 0.01
        )  # Almost black
        very_diff_ref = torch.ones(self.channels, self.height, self.width) * 0.5  # Gray

        # Put them in batches
        test_batch = [black_img]
        ref_batch = [slightly_diff_ref, very_diff_ref]

        # Calculate PSNR
        results = psnr(test_batch, ref_batch, self.dataset_size)

        # Manually calculate expected PSNR values
        psnr_to_slightly_diff = self._calculate_psnr(
            black_img, slightly_diff_ref
        )  # Higher
        psnr_to_very_diff = self._calculate_psnr(black_img, very_diff_ref)  # Lower

        print(f"PSNR with slightly different image: {psnr_to_slightly_diff}")
        print(f"PSNR with very different image: {psnr_to_very_diff}")

        # Verify the slightly different reference has higher PSNR
        self.assertTrue(
            psnr_to_slightly_diff > psnr_to_very_diff,
            f"PSNR for slightly different image ({psnr_to_slightly_diff}) should be higher than very different image ({psnr_to_very_diff})",
        )

        # Determine expected best match (should be index 0, the slightly different image)
        expected_best_idx = 0 if psnr_to_slightly_diff > psnr_to_very_diff else 1

        # Verify the function returns the reference index with higher PSNR
        self.assertEqual(
            results[0][1],
            expected_best_idx,
            f"Expected best match index {expected_best_idx}, got {results[0][1]}",
        )

        # Also verify the actual PSNR value matches our calculation
        best_psnr_value = max(psnr_to_slightly_diff, psnr_to_very_diff)
        self.assertAlmostEqual(
            results[0][0],
            best_psnr_value,
            places=5,
            msg=f"Expected PSNR value {best_psnr_value}, got {results[0][0]}",
        )

    def _calculate_psnr(self, img1, img2):
        """Helper method to calculate PSNR between two images."""
        mse = ((img1 - img2) ** 2).mean()
        if mse > 0 and torch.isfinite(mse):
            return 10 * torch.log10(1.0**2 / mse)
        elif not torch.isfinite(mse):
            return float("nan")
        else:
            return float("inf")

    def test_mse_image_space(self):
        """Test MSE calculation in image space."""
        # Create identical images (should give zero MSE)
        identical_img = torch.ones(self.channels, self.height, self.width)
        identical_batch = [identical_img, identical_img]

        results = mse_image_space(identical_batch, identical_batch, self.dataset_size)

        # Check correct number of results
        self.assertEqual(len(results), len(identical_batch))

        # For identical images, MSE should be zero
        for mse_val, idx in results:
            self.assertAlmostEqual(mse_val, 0.0, places=5)

        # Test with different images - create images with clear differences
        img1 = torch.zeros(self.channels, self.height, self.width)
        img2 = torch.ones(self.channels, self.height, self.width)

        # Create a reference batch with images in the opposite order
        # and add a random image for more testing
        torch.manual_seed(42)
        random_img = torch.rand(self.channels, self.height, self.width)
        ref_batch = [img2, img1, random_img]

        results = mse_image_space([img1, img2], ref_batch, self.dataset_size)

        # Manually calculate which index should have minimum MSE
        mse_img1_to_ref = []
        for ref_img in ref_batch:
            mse = ((img1 - ref_img) ** 2).mean().item()
            mse_img1_to_ref.append(mse)

        mse_img2_to_ref = []
        for ref_img in ref_batch:
            mse = ((img2 - ref_img) ** 2).mean().item()
            mse_img2_to_ref.append(mse)

        expected_idx_1 = mse_img1_to_ref.index(min(mse_img1_to_ref))
        expected_idx_2 = mse_img2_to_ref.index(min(mse_img2_to_ref))

        # Check if best index is correctly identified
        self.assertEqual(results[0][1], expected_idx_1)
        self.assertEqual(results[1][1], expected_idx_2)

        # Check MSE values are positive
        for mse_val, _ in results:
            self.assertGreaterEqual(mse_val, 0.0)


if __name__ == "__main__":
    unittest.main()
