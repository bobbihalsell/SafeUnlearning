from cfg_validator import InputValidator, ConfigError
import os


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


class ReconstructorValidator(InputValidator):
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
    
    def _validate_attack_params(self):
        recon_config = self.attack_file
        method = self._require(recon_config, 'type', 'attack_name')


        self.labels = self._load_labels(recon_config.get('unlearned_labels', None))

        if self.attack_name == 'ggl':
            self._validate_ggl_params(recon_config)
        elif self.attack_name == 'invertgrad':
            self._validate_invertgrad_params(recon_config)
        else:
            raise ConfigError(f"Unsupported unlearning method: {method}")
        
        self._require(recon_config, 'lr', 'attack_lr')


    def _validate_ggl_params(self, config):
        self.recon_params = self._require(config, 'cfg', 'attack_params')

        required_params = ['num_updates', 'unlearning_method', 
                           'batch_size', 'loss_models', 'budget',
                           'search_dim', 'use_tanh', 
                           'gp_optim', 'use_scheduler', 
                           'initial_lr']
        for param in required_params:
            self._require(self.recon_params, param)

        self.unlearner_params = config.get('unlearner', {})
          
    def _validate_invertgrad_params(self, config):
        self.recon_params = self._require(config, 'cfg', 'attack_params')

        required_params = ['grad_diff_lr', 'signed', 'boxed',
                           'cost_fn', 'indices', 'weights',
                           'optim', 'num_runs', 'recon_iterations',
                           'total_variation', 'init', 'lr_decay',
                           'scoring_choice', 'eval', 'filter']
        
        for param in required_params:
            self._require(self.recon_params, param)
        
        self.num_images = config.get('num_images', None)
        if self.labels is None and self.num_images is None:
            raise ConfigError('Either labels or num_images must be provided.')

    
    def _validate_wandb_params(self):
        self._require(self.wandb_file, 'project_name')
        self.extra_config = self._require(self.wandb_file, 'extra_config') if self.wandb_file.get('extra_config') else {}
        self.run_id = self._require(self.wandb_file, 'run_id') if self.wandb_file.get('run_id') else None


        
    def _load_labels(self, labels_cfg):
        if isinstance(labels_cfg, str) and os.path.isfile(labels_cfg):
            try:
                with open(labels_cfg, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]
                    labels = [int(line) for line in lines]
                if not labels:
                    raise ConfigError(f"Label file {labels_cfg} is empty.")
                return labels
            except ValueError:
                raise ConfigError(f"Invalid label format in {labels_cfg}. Expected integers.")
        
        elif isinstance(labels_cfg, list):
            if all(isinstance(label, int) for label in labels_cfg):
                return labels_cfg
            else:
                raise ConfigError(f"Invalid label format in {labels_cfg}. Expected a list of integers.")
        else:
            return None

<<<<<<< HEAD
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
=======
>>>>>>> dev
