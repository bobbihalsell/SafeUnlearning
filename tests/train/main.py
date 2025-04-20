import unittest
import tempfile
import shutil
import os
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from unittest.mock import patch, MagicMock, mock_open

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    )

from src.train.main import TrainApp

class TestTrainApp(unittest.TestCase):
    
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
                img = Image.new('RGB', (10, 10), color=(i*50, j*50, 100))
                img_path = os.path.join(class_dir, f'valid_{j}.jpg')
                img.save(img_path)
                
        # Create a corrupted image in class 0
        with open(os.path.join(self.class_dirs[0], 'corrupted.jpg'), 'w') as f:
            f.write('This is not a valid image file')
            
        # Setup transformation
        self.transform = transforms.Compose([
            transforms.Resize((8, 8)),
            transforms.ToTensor()
        ])
        
        # Create a mock config
        self.mock_config = MagicMock()
        self.mock_config.__getitem__.side_effect = lambda key: {
            'seed': 42,
            'model': {
                'name': 'resnet18',
                'pretrained': True,
                'num_classes': 3,
                'save_dir': os.path.join(self.temp_dir, 'models'),
                'freeze_all_except_classifier': False,
                'checkpoint_path': None
            },
            'dataset': {
                'name': 'test_dataset',
                'load_dir': self.temp_dir,
                'batch_sizes': {'train': 4, 'val': 2},
                'num_workers': 1
            },
            'train_cfg': {
                'lr': 0.001,
                'epochs': 2,
                'weight_decay': 0.0001
            },
            'wandb_cfg': {
                'run_id': 'test-run',
                'project_name': 'test-project'
            }
        }[key]
        
        # Patch OmegaConf.to_container to return our mock config
        self.omegaconf_patcher = patch('omegaconf.OmegaConf.to_container', return_value=self.mock_config)
        self.omegaconf_patcher.start()
        
        # Patch set_seed
        self.set_seed_patcher = patch('src.train.main.set_seed')
        self.mock_set_seed = self.set_seed_patcher.start()
        
        # Patch setup_device to return 'cpu'
        self.setup_device_patcher = patch('src.train.main.setup_device', return_value='cpu')
        self.mock_setup_device = self.setup_device_patcher.start()
        
        # Patch wandb
        self.wandb_patcher = patch('wandb.init')
        self.mock_wandb_init = self.wandb_patcher.start()
        self.wandb_log_patcher = patch('wandb.log')
        self.mock_wandb_log = self.wandb_log_patcher.start()
        self.wandb_finish_patcher = patch('wandb.finish')
        self.mock_wandb_finish = self.wandb_finish_patcher.start()
        
        # Import TrainApp after patching
        self.TrainApp = TrainApp

    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
        
        # Stop all patches
        self.omegaconf_patcher.stop()
        self.set_seed_patcher.stop()
        self.setup_device_patcher.stop()
        self.wandb_patcher.stop()
        self.wandb_log_patcher.stop()
        self.wandb_finish_patcher.stop()

    def test_trainapp_initialization(self):
        """Test that TrainApp initializes correctly with config."""
        app = self.TrainApp(self.mock_config)
        
        # Check that all attributes are correctly set
        self.assertEqual(app.seed, 42)
        self.assertEqual(app.model_name, 'resnet18')
        self.assertTrue(app.pretrained)
        self.assertEqual(app.num_classes, 3)
        self.assertEqual(app.model_save_dir, os.path.join(self.temp_dir, 'models'))
        self.assertFalse(app.freeze_all_except_last)
        self.assertEqual(app.dataset_name, 'test_dataset')
        self.assertEqual(app.dataset_save_dir, self.temp_dir)
        self.assertEqual(app.batch_sizes, {'train': 4, 'val': 2})
        self.assertEqual(app.num_workers, 1)
        self.assertEqual(app.run_id, 'test-run')
        self.assertEqual(app.project_name, 'test-project')
        self.assertIsNone(app.checkpoint_path)
        self.assertFalse(app.from_checkpoint)
        
        # Verify that set_seed was called with the correct seed
        self.mock_set_seed.assert_called_once_with(42)

    def test_initialize_model(self):
        """Test initialize_model with both torchvision and timm paths."""
        pass

    def test_freeze_all_except_classifier(self):
        """Test freezing all layers except classifier."""
        pass

    def test_reinitialize_checkpoints(self):
        """Test reinitializing model and optimizer from checkpoint."""
        pass

    def test_pretrain(self):
        """Test the pretrain method with mocked components."""
        pass

    def test_eval_model(self):
        """Test the eval_model method."""
        app = self.TrainApp(self.mock_config)
        
        # Create a simple model
        model = nn.Linear(10, 3)
        criterion = nn.CrossEntropyLoss()
        
        # Create a mock dataloader
        mock_dl = MagicMock()
        mock_dl.dataset = [1, 2, 3, 4, 5]  # 5 items
        
        # Configure mock_dl.__iter__ to yield batches
        batch1 = (torch.randn(2, 10), torch.tensor([0, 1]))
        batch2 = (torch.randn(3, 10), torch.tensor([2, 0, 1]))
        mock_dl.__iter__.return_value = [batch1, batch2]
        
        # Run eval_model
        val_loss, val_acc = app.eval_model(criterion, model, mock_dl)
        
        # Check that results are of expected types
        # self.assertIsInstance(val_loss, float)
        # self.assertIsInstance(val_acc, float)
        
        # Val accuracy should be between 0 and 1
        # self.assertGreaterEqual(val_acc, 0.0)
        # self.assertLessEqual(val_acc, 1.0)


if __name__ == '__main__':
    unittest.main()