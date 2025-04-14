from attacks.utils import ConfigError
import os


class InputValidator:
    """ Validates most inputs to the configuration YAML file."""
    def __init__(self, config):
        assert isinstance(config, dict)

        self.experiment_name = config['experiment_name']
        self.original_weights = config['original_weights']
        self.unlearned_weights = config['unlearned_weights']

        self.seed = config['seed']
        self.verbose = config['verbose']

        self.wandb = config['wandb']

        data_config = config['data']
        self.model_name = data_config['model_name']
        self.labels = self._load_labels(data_config['labels'])
        self.dataset_name = data_config['dataset_name']
        self.num_classes = data_config['num_classes']
        self.data_root = data_config['data_root']

        self.reconstructor_name = config['reconstructor']['type']
        self.reconstructor_lr = config['reconstructor']['lr']
        self.reconstructor_params = config['reconstructor']['cfg']

        self.wandb_project = config['wandb']['project']
        self.extra_config = self.wandb.get('extra_config', {})


        
        self._validate_experiment_params()
        self._validate_dataset_params()

        self._validate_reconstructor_params()
        
    def _load_labels(self, labels_cfg):
        if isinstance(labels_cfg, str) and os.path.isfile(labels_cfg):
            with open(labels_cfg, 'r') as f:
                return [int(line.strip()) for line in f if line.strip()]
        elif isinstance(labels_cfg, list):
            return labels_cfg
        else:
            raise ValueError("Unsupported label format: must be a list or path to a file.")        

    def _validate_reconstructor_params(self):
        """
        Validate parameters required by each unlearner type.
        Raises exception if required parameters are missing.
        """
        # common parameters for unlearners
        if self.reconstructor_name is None:
            raise ConfigError('Reconstruction method must be provided.')
        
        if self.reconstructor_lr is None:
            raise ConfigError('Learning rate for the unlearning '
                              'reconstructor must be provided.')


        # Add specific parameters based on unlearner type
        elif self.reconstructor_name == 'ggl':
            # Check for required SCRUB-specific parameters
            required_ggl_params = ['num_updates', 'unlearning_method',
                                     'alpha', 'gamma', 'min_epochs',
                                     'max_epochs', 'batch_size',
                                     'loss_models', 'budget',
                                     'search_dim', 'use_tanh', 
                                     'gp_optim', 'use_scheduler', 
                                     'initial_lr']
            missing_ggl_params = []
            for param in required_ggl_params:
                if param not in self.reconstructor_params:
                    missing_ggl_params.append(param)
            if missing_ggl_params:
                raise ConfigError('Missing required params for GGL reconstruction:'
                                  f' {', '.join(missing_ggl_params)}')
            
        elif self.reconstructor_name == 'inversegrad':
            print("checking inverse grad params")
            # Check for required Inverse Grad-specific parameters
            required_igrad_params = ['grad_diff_lr', 'signed', 'boxed',
                                   'cost_fn', 'indices', 'weights',
                                   'optim', 'num_runs', 'recon_iterations',
                                   'total_variation', 'init', 'lr_decay',
                                   'scoring_choice', 'eval', 'filter']
            missing_igrad_params = []
            for param in required_igrad_params:
                if param not in self.reconstructor_params:
                    missing_igrad_params.append(param)
            if missing_igrad_params:
                raise ConfigError('Missing required params for Inverse Grad reconstruction:'
                                  f' {', '.join(missing_igrad_params)}')
    

    def _validate_dataset_params(self):
        valid_dataset_names = {'cifar5',
                               'cifar10',
                               'cifar100',
                               'imagenet'}
        if self.dataset_name not in valid_dataset_names:
            raise ConfigError('Dataset support only for '
                              f'{', '.join(valid_dataset_names)}. '
                              f'Received {self.dataset_name}')
        if self.data_root is None:
            raise ConfigError('A path to '
                              'load the dataset from must be provided. '
                              'data_root cannot be empty.')
        
        if not isinstance(self.labels, list) or not all(isinstance(label, int) for label in self.labels):
            raise ConfigError('Labels must be a list of integers.')
        
    def _validate_experiment_params(self):
        """ Check some of the model-related config params from the YAML file.

        This validates only some of the model parameters, as it is more
        efficient to use the EAFP approach for model name and
        initialization checking."""
        if not os.path.exists(self.original_weights):
            raise ConfigError("Model weights file not found: "
                              f"{self.original_weights}")

        if not os.path.exists(self.unlearned_weights):
            raise ConfigError("Model weights file not found: "
                              f"{self.unlearned_weights}")
