import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch

from src.attacks.utils import SaveImage


class TestSaveImage(unittest.TestCase):
    def setUp(self):
        self.attack_name = "test_attack"
        self.seed = "12345"
        self.experiment_name = "test_experiment"
        self.output_dir = tempfile.mkdtemp()

    def test_save_png_creates_directory(self):
        # Create a SaveImage instance
        save_image = SaveImage(
            attack_name=self.attack_name,
            seed=self.seed,
            experiment_name=self.experiment_name,
            output_dir=self.output_dir,
        )

        # Mock images
        images = torch.randn(1, 3, 64, 64)

        # Call the save_png method
        save_image.save_png(images)

        # Check if the directory was created
        self.assertTrue(os.path.exists(save_image.directory))

    def test_save_png_creates_files(self):
        # Create a SaveImage instance
        save_image = SaveImage(
            attack_name=self.attack_name,
            seed=self.seed,
            experiment_name=self.experiment_name,
            output_dir=self.output_dir,
        )

        # Mock images
        images = torch.randn(2, 3, 32, 32)  # 2 images of size 32x32

        # Call the save_png method
        save_image.save_png(images)

        # Check if files were created
        file_path = os.path.join(
            self.output_dir,
            f"reconstruction/{self.attack_name}/{self.attack_name}_{self.seed}_{self.experiment_name}.png",
        )
        self.assertTrue(os.path.exists(file_path))

    def test_save_png_with_custom_filename(self):
        # Create a SaveImage instance
        save_image = SaveImage(
            attack_name=self.attack_name,
            seed=self.seed,
            experiment_name=self.experiment_name,
            output_dir=self.output_dir,
        )

        # Mock images
        images = torch.randn(1, 3, 64, 64)

        # Call the save_png method with a custom filename
        save_image.save_png(images, filename="custom_test.png")

        # Check if the file was created
        file_path = os.path.join(
            self.output_dir, f"reconstruction/{self.attack_name}/custom_test.png"
        )
        self.assertTrue(os.path.exists(file_path))

    def test_save_png_with_invalid_images(self):
        # Create a SaveImage instance
        save_image = SaveImage(
            attack_name=self.attack_name,
            seed=self.seed,
            experiment_name=self.experiment_name,
            output_dir=self.output_dir,
        )

        # Mock invalid images (not a list)
        images = "invalid_images"

        # Call the save_png method and check for exception
        with self.assertRaises(TypeError):
            save_image.save_png(images)


class TestLoadSaveTensor(unittest.TestCase):
    def setUp(self):
        self.attack_name = "test_attack"
        self.seed = "12345"
        self.experiment_name = "test_experiment"
        self.output_dir = tempfile.mkdtemp()

    def test_save_tensor_creates_directory(self):
        # Create a SaveImage instance
        save_image = SaveImage(
            attack_name=self.attack_name,
            seed=self.seed,
            experiment_name=self.experiment_name,
            output_dir=self.output_dir,
        )

        # Mock tensor
        tensor = torch.randn(3, 32, 32)

        # Call the save_tensor method
        save_image.save_tensor(tensor)

        # Check if the directory was created
        self.assertTrue(os.path.exists(self.output_dir))
        # Check if the tensor file was created
        file_path = os.path.join(
            self.output_dir,
            f"reconstruction/{self.attack_name}/{self.attack_name}_{self.seed}_{self.experiment_name}.pt",
        )
        self.assertTrue(os.path.exists(file_path))

    def test_load_tensor(self):
        # Create a SaveImage instance
        save_image = SaveImage(
            attack_name=self.attack_name,
            seed=self.seed,
            experiment_name=self.experiment_name,
            output_dir=self.output_dir,
        )

        # Mock tensor
        tensor = torch.randn(3, 32, 32)

        # Call the save_tensor method
        save_image.save_tensor(tensor)

        filepath = os.path.join(
            self.output_dir,
            f"reconstruction/{self.attack_name}/{self.attack_name}_{self.seed}_{self.experiment_name}.pt",
        )

        # Load the tensor back
        loaded_tensor = save_image.load_tensor(filepath)

        # Check if the loaded tensor is the same as the original tensor
        self.assertTrue(torch.equal(tensor, loaded_tensor))

    def test_load_tensor_invalid_file(self):
        # Create a SaveImage instance
        save_image = SaveImage(
            attack_name=self.attack_name,
            seed=self.seed,
            experiment_name=self.experiment_name,
            output_dir=self.output_dir,
        )

        # Mock invalid file path
        invalid_file_path = os.path.join(self.output_dir, "invalid.pt")

        # Check if loading raises a FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            save_image.load_tensor(filepath=invalid_file_path)
