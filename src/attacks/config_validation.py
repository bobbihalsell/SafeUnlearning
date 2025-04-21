from cfg_validator import InputValidator, ConfigError
# import os


class GLiRValidator(InputValidator):
    """ Validator specifically for unlearning configurations."""
    def __init__(self, config):
        super().__init__(config)
        self._validate_required_sections()
    
    def _validate_required_sections(self):
        """Ensure unlearning-specific sections exist."""
        required_sections = ['model', 'dataset', 'attack']
        for section in required_sections:
            if not hasattr(self, f'{section}_file'):
                raise ConfigError(
                    f"Missing required config section: '{section}'"
                    )


class InputValidator:
    """ Validates most inputs to the configuration YAML file."""
    def __init__(self, config):
        assert isinstance(config, dict)

        self.experiment_name = config['experiment_name']
        self.original_weights = config['original_weights']
        self.unlearned_weights = config['unlearned_weights']

        self.seed = config['seed']
        self.verbose = config['verbose']

        self.wandb = config.get('wandb_cfg', None)

        data_config = config['data']
        self.model_name = data_config['model_name']
        self.labels = self._load_labels(data_config.get('labels', None))
        self.dataset_name = data_config['dataset_name']
        self.num_classes = data_config['num_classes']
        self.data_root = data_config['data_root']

        self.reconstructor_name = config['reconstructor']['type']
        self.reconstructor_lr = config['reconstructor']['lr']
        self.reconstructor_params = config['reconstructor']['cfg']

        if 'unlearner' in config['reconstructor']:
            self.unlearner_params = config['reconstructor']['unlearner']
        else:
            self.unlearner_params = {}
                   
        if self.wandb is not None:
            self.wandb_enabled = True
            self.wandb_project = self.wandb['project']
            self.extra_config = self.wandb.get('extra_config', {})
        else:
            self.wandb_enabled = False
        
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
            return None    

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
                                     'batch_size',
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
            
        elif self.reconstructor_name == 'invertgrad':
            print("checking invert grad params")
            # Check for required Invert Grad-specific parameters
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
                raise ConfigError('Missing required params for Invert Grad reconstruction:'
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
        
        if self.labels:
            if not isinstance(self.labels, list) or not all(isinstance(label, int) for label in self.labels):
                raise ConfigError('Labels must be a list of integers.')
        else:
            if not isinstance(self.reconstructor_params['num_images'], int):
                raise ConfigError('Either labels or num_images must be provided.')
        
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
