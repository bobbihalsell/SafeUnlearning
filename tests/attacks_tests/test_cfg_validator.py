import os
import unittest
import tempfile
from attacks.config_validation import ReconstructorValidator
from cfg_validator import ConfigError

class TestGGLValidation(unittest.TestCase):
    def setUp(self):
        # Set up a sample config for testing
        self.config = {
            'model': {
                'load_method': 'torchvision',
                'model_name': 'resnet18',
                'original_model_ckpt_path': 'path/to/original/model.pth',
                'unlearned_model_ckpt_path': 'path/to/unlearned/model.pth',
            },
            'dataset': {
                'name': 'cifar10',  
                'num_classes': 10,
                'save_path': 'path/to/dataset',
                'cfg': {
                    'batch_sizes': {
                        'retain': 128,
                        'forget': 128,
                        'val': 128,
                    },
                    'num_workers': 4,
                },
            },
            'reconstructor': {
                'type': 'ggl',
                'lr': 0.01,
                'unlearned_labels': [0, 1],
                'cfg': {
                    'num_updates': 10,
                    'unlearning_method': 'sample_method',
                    'batch_size': 32,
                    'loss_models': ['model1', 'model2'],
                    'budget': 100,
                    'search_dim': 5,
                    'use_tanh': True,
                    'gp_optim': False,
                    'use_scheduler': True,
                    'initial_lr': 0.001,
                },
            },
        }

    def test_validate_required_sections(self):
        # Test if required sections are validated correctly
        required_sections = ['model', 'dataset', 'reconstructor']
        for section in required_sections:
            with self.assertRaises(ConfigError):
                del self.config[section]
                ReconstructorValidator(self.config)
    
    def test_validate_reconstructor_params(self):
        # Test if reconstructor parameters are validated correctly
        recon_config = self.config['reconstructor']
        recon_config['type'] = 'ggl'
        validator = ReconstructorValidator(self.config)
        
        # Check if the recon_params are set correctly
        self.assertIn('num_updates', validator.recon_params)
        self.assertIn('unlearning_method', validator.recon_params)
        
        # Check if the labels are loaded correctly
        self.assertEqual(validator.labels, [0, 1])
        
        # Test with an unsupported unlearning method
        recon_config['type'] = 'unsupported_method'
        with self.assertRaises(ConfigError):
            ReconstructorValidator(self.config)

        # Test with missing required parameters
        del recon_config['cfg']['num_updates']
        with self.assertRaises(ConfigError):
            ReconstructorValidator(self.config)

        # Test with missing reconstructor name
        del recon_config['type']
        with self.assertRaises(ConfigError):
            ReconstructorValidator(self.config)
    
    
class TestInverseGradValidation(unittest.TestCase):
    def setUp(self):
        # Set up a sample config for testing
        self.config = {
            'model': {
                'load_method': 'torchvision',
                'model_name': 'resnet18',
                'original_model_ckpt_path': 'path/to/original/model.pth',
                'unlearned_model_ckpt_path': 'path/to/unlearned/model.pth',
            },
            'dataset': {
                'name': 'cifar10',  
                'num_classes': 10,
                'save_path': 'path/to/dataset',
                'cfg': {
                    'batch_sizes': {
                        'retain': 128,
                        'forget': 128,
                        'val': 128,
                    },
                    'num_workers': 4,
                },
            },
            'reconstructor': {
                'type': 'invertgrad',
                'lr': 0.01,
                'unlearned_labels': [0, 1],
            'cfg': {
                    'num_images': None,        
                    'grad_diff_lr': 0.01,
                    'signed': True,
                    'boxed': True,
                    'cost_fn': 'sim',
                    'indices': 'def',
                    'weights': 'equal',
                    'optim': 'adamw',
                    'num_runs': 1,
                    'recon_iterations': 5000,
                    'total_variation': 1e-2,
                    'init': 'randn',
                    'lr_decay': True,
                    'scoring_choice': 'loss',
                    'eval': True,
                    'filter': None
                    },
        }}
    def test_validate_required_sections(self):
        # Test if required sections are validated correctly
        required_sections = ['model', 'dataset', 'reconstructor']
        for section in required_sections:
            with self.assertRaises(ConfigError):
                del self.config[section]
                ReconstructorValidator(self.config)
    
    def test_validate_reconstructor_params(self):
        test_config = self.config['reconstructor']
        test_config['type'] = 'invertgrad'
        validator = ReconstructorValidator(self.config)

        # Check if the recon_params are set correctly
        self.assertIn('grad_diff_lr', validator.recon_params)
        self.assertIn('signed', validator.recon_params)
        self.assertIn('boxed', validator.recon_params)

        # Check if the labels are loaded correctly
        self.assertEqual(validator.labels, [0, 1])

    def test_no_labels_and_num_images(self):
        # Test if the config raises an error when both labels and num_images are None
        self.config['reconstructor']['unlearned_labels'] = None
        self.config['reconstructor']['num_images'] = None
        with self.assertRaises(ConfigError):
            ReconstructorValidator(self.config)

    def test_num_images_only(self):
        self.config['reconstructor']['unlearned_labels'] = None
        self.config['reconstructor']['num_images'] = 10
        validator = ReconstructorValidator(self.config)
        self.assertEqual(validator.num_images, 10)
        self.assertIsNone(validator.labels)

    def test_missing_required_params(self):
        # Test if the config raises an error when required parameters are missing
        del self.config['reconstructor']['cfg']['grad_diff_lr']
        with self.assertRaises(ConfigError):
            ReconstructorValidator(self.config)

    def test_load_labels_from_list(self):
        labels = [0, 1, 2]
        self.config['reconstructor']['unlearned_labels'] = [0, 1, 2]
        validator = ReconstructorValidator(self.config)
        self.assertEqual(validator.labels, labels)

    def test_load_labels_from_file(self):
         # Create a temp file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp_file:
            tmp_file.write("0\n1\n2\n")
            tmp_path = tmp_file.name

        try:
            self.config['reconstructor']['unlearned_labels'] = tmp_path
            validator = ReconstructorValidator(self.config)
            self.assertEqual(validator.labels, [0, 1, 2])
        finally:
            os.remove(tmp_path)

    def test_load_labels_invalid_file(self):
        # Create a temp file with invalid content
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp_file:
            tmp_file.write("invalid\ncontent\n")
            tmp_path = tmp_file.name

        try:
            self.config['reconstructor']['unlearned_labels'] = tmp_path
            with self.assertRaises(ConfigError):
                ReconstructorValidator(self.config)
        finally:
            os.remove(tmp_path)
    
    def test_load_labels_empty_file(self):
        # Create a empty temp file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            self.config['reconstructor']['unlearned_labels'] = tmp_path
            with self.assertRaises(ConfigError):
                ReconstructorValidator(self.config)
        finally:
            os.remove(tmp_path)
    
    def test_load_labels_invalid_input(self):
        # Test with invalid input 
        self.config['reconstructor']['unlearned_labels'] = 12345
        with self.assertRaises(ConfigError):
            ReconstructorValidator(self.config)

if __name__ == '__main__':
    unittest.main()