import os
import shutil
import sys
import tempfile
import unittest

import torch
from PIL import Image
from torchvision import transforms

from train.image_loading import RobustImageFolder


class TestRobustImageFolder(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory structure with test images
        self.temp_dir = tempfile.mkdtemp()

        # Create class directories (using integers as names)
        self.class_dirs = []
        for i in range(3):
            class_dir = os.path.join(self.temp_dir, str(i))
            os.makedirs(class_dir)
            self.class_dirs.append(class_dir)

        # Create valid images
        for i, class_dir in enumerate(self.class_dirs):
            for j in range(3):
                # Create a small colored image
                img = Image.new("RGB", (10, 10), color=(i * 50, j * 50, 100))
                img_path = os.path.join(class_dir, f"valid_{j}.jpg")
                img.save(img_path)

        # Create a corrupted image in class 0
        with open(os.path.join(self.class_dirs[0], "corrupted.jpg"), "w") as f:
            f.write("This is not a valid image file")

        # Setup transformation
        self.transform = transforms.Compose(
            [transforms.Resize((8, 8)), transforms.ToTensor()]
        )

    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_load_valid_images(self):
        # Test loading all valid images
        dataset = RobustImageFolder(self.temp_dir, transform=self.transform)

        # Check if classes are correctly loaded
        self.assertEqual(len(dataset.classes), 3)
        self.assertEqual(dataset.class_to_idx, {"0": 0, "1": 1, "2": 2})

        # Test accessing items
        for i in range(len(dataset)):
            img, label = dataset[i]
            self.assertIsInstance(img, torch.Tensor)
            self.assertEqual(img.shape, (3, 8, 8))  # Check dimensions after transform
            self.assertIn(label, [0, 1, 2])

    def test_corrupted_image_handling(self):
        # Redirect stdout to suppress the print message about skipping corrupted images
        import sys
        from io import StringIO

        original_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            dataset = RobustImageFolder(self.temp_dir, transform=self.transform)

            # If we have a corrupted image, the __getitem__ should handle it and return a valid image
            for i in range(len(dataset)):
                img, label = dataset[i]
                self.assertIsInstance(img, torch.Tensor)
        finally:
            # Check if "Skipping corrupted image" was printed
            output = sys.stdout.getvalue()
            sys.stdout = original_stdout

            self.assertIn("Skipping corrupted image", output)

    def test_find_classes(self):
        dataset = RobustImageFolder(self.temp_dir)

        # Test the find_classes method
        classes, class_to_idx = dataset.find_classes(self.temp_dir)
        self.assertEqual(classes, ["0", "1", "2"])
        self.assertEqual(class_to_idx, {"0": 0, "1": 1, "2": 2})

        # Test with invalid class names (non-integer directory names)
        invalid_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(invalid_dir, "not_a_number"))

        with self.assertRaises(RuntimeError):
            dataset.find_classes(invalid_dir)

        shutil.rmtree(invalid_dir)


if __name__ == "__main__":
    unittest.main()
